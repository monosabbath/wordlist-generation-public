import os
import time
import uuid
import json
import tempfile
import unicodedata
import re
import logging
import datetime
from functools import cache
from typing import Optional, Dict, Any, List

import torch
import torch.distributed as dist

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Header,
    Query,
    UploadFile,
    File,
    BackgroundTasks,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from huggingface_hub import snapshot_download

# Load environment as early as possible
load_dotenv()

# Torch/backends knobs
torch.backends.cuda.matmul.allow_tf32 = True

from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("vocab-constrained")

app = FastAPI()

# -----------------------
# Config via environment
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-4.6")

# Choose how to place model across GPUs
# - tp: tensor parallel with tp_plan='auto' (requires torch.distributed)
# - device_map: HF device_map path (single Python process)
PARALLEL_MODE = os.getenv("PARALLEL_MODE", "device_map").strip().lower()
if PARALLEL_MODE not in ("tp", "device_map"):
    logger.warning(f"Invalid PARALLEL_MODE={PARALLEL_MODE}, defaulting to 'device_map'")
    PARALLEL_MODE = "device_map"

# If TP is requested but distributed env is not ready, optionally fall back
TP_FALLBACK_TO_DEVICE_MAP = os.getenv("TP_FALLBACK_TO_DEVICE_MAP", "true").lower() == "true"
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")  # used only when PARALLEL_MODE=device_map
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my-secret-token-structured-generation")

# Attention implementation preference
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2").strip()

# Padding/length controls
TOKENIZER_PADDING_SIDE = os.getenv("TOKENIZER_PADDING_SIDE", "left").strip()
PAD_TO_MULTIPLE_OF = int(os.getenv("PAD_TO_MULTIPLE_OF", "64"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))
ALLOWED_MAX_NEW_TOKENS = tuple(
    sorted({int(x) for x in os.getenv("ALLOWED_MAX_NEW_TOKENS", "64,128,256,512").split(",")})
)

STATIC_KV_CACHE = os.getenv("STATIC_KV_CACHE", "false").lower() == "true"

# Constrained vocab prebuild
PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "3000").split(","))
PREBUILD_LANGS = [x.strip() for x in os.getenv("PREBUILD_LANGS", "").split(",") if x.strip()]

# Torch dtype
TORCH_DTYPE_STR = os.getenv("TORCH_DTYPE", "auto")

# Config for Batch Jobs
BATCH_JOB_TEMP_DIR = os.getenv("BATCH_JOB_TEMP_DIR", tempfile.gettempdir())
BATCH_JOB_PIPELINE_SIZE = int(os.getenv("BATCH_JOB_PIPELINE_SIZE", "8"))
logger.info(f"Batch jobs will be stored in: {BATCH_JOB_TEMP_DIR}")
logger.info(f"Batch job pipeline size: {BATCH_JOB_PIPELINE_SIZE}")

# Local staging dir for model (rank-0 will fill this once)
# Default uses nested path so MODEL_NAME with slashes creates a subdir tree.
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", os.path.join("/local_nvme/models", MODEL_NAME))
HF_TOKEN = os.getenv("HF_TOKEN", None)  # optional private hub token

# -----------------------
# Distributed helpers
# -----------------------
def _ddp_env_ready():
    return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))

def is_rank0() -> bool:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

# -----------------------
# Model / tokenizer load
# -----------------------

# Determine dtype from environment variable
if TORCH_DTYPE_STR.lower() in ("bf16", "bfloat16", "torch.bfloat16"):
    TORCH_DTYPE = torch.bfloat16
    logger.info("Using dtype: bfloat16")
elif TORCH_DTYPE_STR.lower() in ("fp16", "float16", "torch.float16"):
    TORCH_DTYPE = torch.float16
    logger.info("Using dtype: float16")
else:
    TORCH_DTYPE = "auto"
    logger.info("Using dtype: auto")

# Detect if distributed is ready for TP
USE_TP = PARALLEL_MODE == "tp"
if USE_TP:
    ddp_ready = dist.is_available() and (dist.is_initialized() or _ddp_env_ready())
    if not ddp_ready:
        msg = "PARALLEL_MODE=tp requested but torch.distributed is not initialized and env vars (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT) are missing."
        if TP_FALLBACK_TO_DEVICE_MAP:
            logger.warning(msg + " Falling back to PARALLEL_MODE=device_map for this run.")
            USE_TP = False
            PARALLEL_MODE = "device_map"
        else:
            raise OSError(
                msg + " Start with torchrun, e.g.: "
                "torchrun --standalone --nproc-per-node=6 --master-port=29500 tp_launcher.py"
            )
    elif not dist.is_initialized():
        # Initialize process group if env is present but not yet initialized
        logger.info("Initializing torch.distributed process group for TP...")
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

# Prepare model init kwargs per mode
model_init_kwargs: Dict[str, Any] = {
    "trust_remote_code": TRUST_REMOTE_CODE,
    "low_cpu_mem_usage": True,
    "local_files_only": True,  # important once staged
}
# Use new 'dtype' param to avoid deprecation warnings if not "auto"
if TORCH_DTYPE != "auto":
    model_init_kwargs["torch_dtype"] = TORCH_DTYPE  # prefer torch_dtype kw in new transformers

if USE_TP:
    model_init_kwargs["tp_plan"] = "auto"
    logger.info("Loading model in Tensor Parallel (tp_plan='auto') mode")
else:
    model_init_kwargs["device_map"] = DEVICE_MAP
    logger.info(f"Loading model with device_map='{DEVICE_MAP}'")

attn_impl = ATTN_IMPLEMENTATION
logger.info(
    f"Preparing to load model '{MODEL_NAME}' "
    f"(Trust Remote Code: {TRUST_REMOTE_CODE}, attn='{attn_impl}')..."
)

# ---------------
# Rank-0 staging
# ---------------
def stage_model_locally():
    """
    Rank-0 downloads/snapshots the model repo into LOCAL_MODEL_DIR.
    Other ranks wait at a barrier (if distributed is initialized).
    After the barrier, all ranks load strictly from local disk.
    """
    # Make sure parent directories exist
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    rank = 0
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if rank == 0:
        logger.info(f"[Staging] Snapshotting '{MODEL_NAME}' to '{LOCAL_MODEL_DIR}' (world_size={world_size})")
        # NOTE:
        # - local_dir_use_symlinks=False ensures a plain copy rather than symlinks
        # - HF_HUB_ENABLE_HF_TRANSFER=1 env speeds up downloads
        # - snapshot_download is idempotent; it won’t redownload if already present
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
        logger.info("[Staging] Snapshot complete.")

    if dist.is_available() and dist.is_initialized():
        logger.info("[Staging] Waiting at distributed barrier...")
        dist.barrier()
        logger.info("[Staging] Barrier complete for all ranks.")

# Stage before calling any from_pretrained
stage_model_locally()

# Now load from LOCAL_MODEL_DIR only
logger.info("Loading model from local staged directory...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        attn_implementation=attn_impl,
        **model_init_kwargs,
    )
except TypeError:
    logger.warning("Model.from_pretrained() did not accept 'attn_implementation'; loading without it.")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        **model_init_kwargs,
    )
    # Try to set after load
    try:
        model.set_attention_implementation(attn_impl)
    except Exception as e:
        logger.warning(f"Could not set attention implementation to '{attn_impl}': {e}. Falling back to SDPA.")
        try:
            model.set_attention_implementation("sdpa")
        except Exception as e2:
            logger.warning(f"Could not set SDPA either: {e2}")

# Optional: static KV cache (advanced; only enable if you can keep shapes consistent)
if STATIC_KV_CACHE:
    try:
        model.generation_config.cache_implementation = "static"
        logger.info("Enabled static KV cache (model.generation_config.cache_implementation='static').")
    except Exception as e:
        logger.warning(f"Static KV cache not available: {e}")

tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR, trust_remote_code=TRUST_REMOTE_CODE, local_files_only=True
)
tokenizer.padding_side = TOKENIZER_PADDING_SIDE  # left-padding for better KV/cache perf
model.eval()

# Ensure a pad token is set to avoid warnings/problems during generation
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# -----------------------
# Helpers
# -----------------------
def move_inputs_to_correct_device(inputs: Dict[str, torch.Tensor], m):
    """
    Always move inputs to the device of the model's first parameter.
    Works for both device_map and TP-sharded models (first param device is fine).
    """
    try:
        device = next(m.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {k: v.to(device) for k, v in inputs.items()}

def tokenizer_encode_for_chat(texts: Any) -> Dict[str, torch.Tensor]:
    # texts can be str or List[str]
    tok_kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "pad_to_multiple_of": PAD_TO_MULTIPLE_OF,
        "max_length": MAX_INPUT_TOKENS,
        "return_token_type_ids": False,  # IMPORTANT: avoid passing token_type_ids to generate()
    }
    return tokenizer(texts, **tok_kwargs)

def strip_unused_model_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove keys that the model/generate doesn't accept, like token_type_ids.
    """
    inputs = dict(inputs)
    inputs.pop("token_type_ids", None)
    return inputs

def normalize_max_new_tokens(requested: Optional[int]) -> int:
    """
    Cap to 512 and round up to the nearest allowed value to stabilize shapes.
    Allowed set from ALLOWED_MAX_NEW_TOKENS.
    """
    if not ALLOWED_MAX_NEW_TOKENS:
        return min(int(requested) if requested else 512, 512)
    target = int(requested) if requested is not None else 512
    target = min(target, 512)
    for a in ALLOWED_MAX_NEW_TOKENS:
        if target <= a:
            return a
    return ALLOWED_MAX_NEW_TOKENS[-1]

# -----------------------
# Create the generation pipeline
# -----------------------
logger.info("Creating text-generation pipeline...")
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=TRUST_REMOTE_CODE,
)
logger.info("Pipeline created.")

# -----------------------
# Helpers for stop tokens
# -----------------------
@cache
def get_stop_ids(tok: AutoTokenizer) -> List[int]:
    stop_ids: List[int] = []
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, int):
            stop_ids.append(tok.eos_token_id)
        elif isinstance(tok.eos_token_id, list):
            stop_ids.extend(tok.eos_token_id)

    common_end_markers = (
        "<end_of_turn>",
        "<|eot_id|>",
        "<|im_end|>",
    )
    for special in common_end_markers:
        try:
            eid = tok.convert_tokens_to_ids(special)
            if eid is not None and eid != tok.unk_token_id and eid != -1:
                stop_ids.append(eid)
        except Exception:
            pass
    return list(set(stop_ids))

# -----------------------
# Regex-constrained vocab (lowercase-only, trie-based)
# -----------------------
def normalizeword(w: str) -> str:
    return unicodedata.normalize("NFC", w.strip()).lower()

class _TrieNode:
    __slots__ = ("children", "end", "min_rank")
    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.end: bool = False
        self.min_rank: int = 10**12

def buildtrie_with_ranks(words: List[str]) -> _TrieNode:
    root = _TrieNode()
    for rank, w in enumerate(words, start=1):
        node = root
        node.min_rank = min(node.min_rank, rank)
        for ch in w:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
            node.min_rank = min(node.min_rank, rank)
        node.end = True
    return root

def escapefor_regex(ch: str) -> str:
    return re.escape(ch)

def trieto_regex(node: _TrieNode, nlimit: int) -> str:
    alts = []
    for ch, child in sorted(node.children.items()):
        if child.min_rank > nlimit:
            continue
        sub = trieto_regex(child, nlimit)
        alts.append(escapefor_regex(ch) + sub)
    if node.end and node.min_rank <= nlimit:
        alts.append("")
    if not alts:
        return ""
    if len(alts) == 1:
        return alts[0]
    return "(?:" + "|".join(alts) + ")"

TRIECACHE: Dict[str, Dict[str, Any]] = {}

def _safe_lang_name(lang: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", lang))

def getor_build_trie(lang: str) -> Optional[Dict[str, Any]]:
    if not _safe_lang_name(lang):
        logger.warning(f"Unsafe or invalid language identifier: {lang}")
        return None
    if lang in TRIECACHE:
        return TRIECACHE[lang]

    filename = f"{lang}.txt"
    if not os.path.exists(filename):
        return None

    try:
        with open(filename, encoding="utf-8") as fin:
            words = [normalizeword(w) for w in fin if w.strip()]
    except Exception as e:
        logger.error(f"Error reading language file {filename}: {e}")
        return None

    if not words:
        return None

    trie = buildtrie_with_ranks(words)
    TRIECACHE[lang] = {"trie": trie}
    return TRIECACHE[lang]

def buildword_regex_for_n(lang: str, n_words: int) -> Optional[str]:
    data = getor_build_trie(lang)
    if data is None:
        return None
    trie: _TrieNode = data["trie"]
    if trie.min_rank > n_words:
        return None
    return trieto_regex(trie, nlimit=n_words)

@cache
def build_regexp_prefix_fn(lang: str, n_words: int):
    logger.info(f"building prefix function for {lang} ({n_words} words)")
    if getor_build_trie(lang) is None:
        logger.warning(f"Language file not found or empty: {lang}.txt. Skipping build.")
        return None

    word_regex = buildword_regex_for_n(lang, n_words)
    if not word_regex:
        logger.warning(f"No words available for {lang} with n={n_words}. Skipping build.")
        return None

    punct_regex = r'[.,!?¿¡…\s]+'
    flexible_grammar = fr'(?:{punct_regex})?(?:{word_regex})(?:{punct_regex}(?:{word_regex}))*(?:{punct_regex})?'
    parser = RegexParser(flexible_grammar)
    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    stop_ids = set(get_stop_ids(tokenizer))

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        return list(allowed | stop_ids)

    return wrapped_prefix_fn

# Startup prebuilds (if requested) — do on rank 0 only
if PREBUILD_PREFIX and is_rank0():
    if PREBUILD_LANGS:
        logger.info("Prebuilding prefix functions (rank 0 only)...")
        for lang in PREBUILD_LANGS:
            if not _safe_lang_name(lang):
                logger.warning(f"Skipping unsafe language name in PREBUILD_LANGS: {lang}")
                continue
            for n_words in PREBUILD_WORD_COUNTS:
                try:
                    build_regexp_prefix_fn(lang, n_words)
                except Exception as e:
                    logger.warning(f"Prebuild failed for {lang}, n={n_words}: {e}")
        logger.info("Prebuild done.")
    else:
        logger.info("PREBUILD_PREFIX is enabled but PREBUILD_LANGS is empty; skipping prebuild.")

# -----------------------
# Auth helper
# -----------------------
def verify_token(
    token: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
):
    supplied = token
    if not supplied and authorization:
        parts = authorization.split()
        if len(parts) >= 2 and parts[0].lower() == "bearer":
            supplied = parts[1]
    if SECRET_TOKEN and supplied != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True

# -----------------------
# Helper for generation kwargs (deterministic beam search)
# -----------------------
def getgen_kwargs(
    max_new_tokens: int,
    stop_ids: Optional[List[int]],
    num_beams: Optional[int] = None,
    length_penalty: float = 1.0,
    prefix_fn=None
):
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,                 # deterministic
        num_beams=int(num_beams or 10),  # override via request
        length_penalty=float(length_penalty),
    )
    if prefix_fn:
        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        else:
            gen_kwargs["pad_token_id"] = stop_ids[0] if isinstance(stop_ids, list) and stop_ids else None
    return gen_kwargs

# -----------------------
# OpenAI-compatible API
# -----------------------
class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    vocab_lang: Optional[str] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = 10
    length_penalty: Optional[float] = 1.0

@app.get("/v1/models")
def list_models(auth_ok: bool = Depends(verify_token)):
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "owner",
            }
        ],
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, auth_ok: bool = Depends(verify_token)):
    # Only use system prompt provided in API request
    system_prompt = ""
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
            break

    messages = []
    if system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    for msg in req.messages:
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if MODEL_NAME.startswith("zai-org/GLM-4.6"):
        logger.info("Applying 'enable_thinking: False' for GLM-4.6.")
        template_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    texts = tokenizer.apply_chat_template(messages, **template_kwargs)

    # Tokenize with left padding, truncation, pad_to_multiple_of
    inputs = tokenizer_encode_for_chat(texts)

    # Drop keys not accepted by model.generate()
    inputs = strip_unused_model_inputs(inputs)

    inputs = move_inputs_to_correct_device(inputs, model)

    input_len = inputs["input_ids"].shape[1]

    prefix_fn = None
    if req.vocab_lang and req.vocab_n_words:
        prefix_fn = build_regexp_prefix_fn(req.vocab_lang, req.vocab_n_words)
        if prefix_fn is None:
            raise HTTPException(status_code=500, detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.")

    max_new_tokens = normalize_max_new_tokens(req.max_tokens)
    stop_ids = get_stop_ids(tokenizer)

    gen_kwargs = getgen_kwargs(
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        num_beams=req.num_beams,
        length_penalty=req.length_penalty if req.length_penalty is not None else 1.0,
        prefix_fn=prefix_fn
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Determine finish reason
    gen_len = int(outputs[0].shape[0] - input_len)
    last_token = int(outputs[0][-1].item())
    finish_reason = "stop" if stop_ids and (last_token in set(stop_ids)) else ("length" if gen_len >= max_new_tokens else "stop")

    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = gen_len

    created = int(time.time())

    resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": req.model or MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp

# ---------------------------------------------------
# BATCH JOB API
# ---------------------------------------------------
JOB_STATUS: Dict[str, Dict[str, Any]] = {}

def process_batch_job(
    job_id: str,
    input_path: str,
    output_path: str,
    job_config: Dict[str, Any],
):
    logger.info(f"[Job {job_id}] Starting processing...")
    try:
        JOB_STATUS[job_id]["status"] = "processing"

        # 1. Read and parse the input file
        prompts = []
        raw_requests = []
        with open(input_path, "r", encoding="utf-8") as f:
            try:
                raw_requests = json.load(f)
                if not isinstance(raw_requests, list):
                    raise ValueError("Input file must contain a JSON list.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}")

        # 2. Prepare all prompts
        logger.info(f"[Job {job_id}] Preparing {len(raw_requests)} prompts...")
        for i, req_data in enumerate(raw_requests):
            try:
                req = ChatCompletionRequest(**req_data)

                system_prompt = ""
                for msg in req.messages:
                    if msg.role == "system":
                        system_prompt = msg.content
                        break

                messages = []
                if system_prompt != "":
                    messages.append({"role": "system", "content": system_prompt})
                for msg in req.messages:
                    if msg.role != "system":
                        messages.append({"role": msg.role, "content": msg.content})

                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if MODEL_NAME.startswith("zai-org/GLM-4.6"):
                    template_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

                text = tokenizer.apply_chat_template(messages, **template_kwargs)
                prompts.append(text)
            except ValidationError as e:
                logger.warning(f"[Job {job_id}] Skipping request {i}: Invalid format. {e}")
            except Exception as e:
                logger.warning(f"[Job {job_id}] Skipping request {i}: Error processing. {e}")

        # 3. Prepare shared generation kwargs for the entire batch job
        stop_ids = get_stop_ids(tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = stop_ids[0] if stop_ids else tokenizer.eos_token_id

        max_new_tokens = normalize_max_new_tokens(job_config.get("max_tokens", 512))
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic
            num_beams=job_config.get("num_beams", 10),
            length_penalty=job_config.get("length_penalty", 1.0),
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Build prefix function if vocab is constrained
        vocab_lang = job_config.get("vocab_lang")
        vocab_n_words = job_config.get("vocab_n_words")
        if vocab_lang and vocab_n_words:
            logger.info(f"[Job {job_id}] Building constrained vocab for {vocab_lang} ({vocab_n_words} words)")
            prefix_fn = build_regexp_prefix_fn(vocab_lang, vocab_n_words)
            if prefix_fn:
                generation_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
                logger.info(f"[Job {job_id}] Successfully added prefix function.")
            else:
                raise ValueError(f"Constrained vocabulary config failed for lang '{vocab_lang}'. Check server logs.")

        # 4. Run the pipeline with padding/truncation controls
        logger.info(f"[Job {job_id}] Running pipeline (Batch Size: {BATCH_JOB_PIPELINE_SIZE})...")
        results = []
        for output in text_gen_pipeline(
            prompts,
            batch_size=BATCH_JOB_PIPELINE_SIZE,
            return_full_text=False,
            padding=True,
            truncation=True,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
            max_length=MAX_INPUT_TOKENS,         # tokenizer truncation length
            return_token_type_ids=False,          # IMPORTANT: avoid token_type_ids in pipeline
            **generation_kwargs,
        ):
            results.append(output[0]["generated_text"])

        # 5. Format and save the output file
        logger.info(f"[Job {job_id}] Formatting and saving results...")
        created = int(time.time())
        final_output = []
        for i, text in enumerate(results):
            resp = {
                "id": f"chatcmpl-batch-{job_id}-{i}",
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            }
            final_output.append(resp)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)

        JOB_STATUS[job_id]["status"] = "completed"
        logger.info(f"[Job {job_id}] Processing complete.")
    except Exception as e:
        logger.error(f"[Job {job_id}] Processing FAILED: {e}")
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["error"] = str(e)
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"[Job {job_id}] Cleaned up input file.")
        except Exception as e:
            logger.warning(f"[Job {job_id}] Failed to clean up input file: {e}")

# Endpoint 1: Submit a new batch job
@app.post("/v1/batch/jobs")
def create_batch_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auth_ok: bool = Depends(verify_token),
    max_tokens: int = 512,
    num_beams: int = 10,
    length_penalty: float = 1.0,
    vocab_lang: Optional[str] = None,
    vocab_n_words: Optional[int] = None,
):
    job_id = str(uuid.uuid4())
    input_path = os.path.join(BATCH_JOB_TEMP_DIR, f"{job_id}_input.json")
    output_path = os.path.join(BATCH_JOB_TEMP_DIR, f"{job_id}_output.json")

    try:
        with open(input_path, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {e}")

    job_config = {
        "max_tokens": max_tokens,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "vocab_lang": vocab_lang,
        "vocab_n_words": vocab_n_words,
    }

    JOB_STATUS[job_id] = {
        "status": "pending",
        "input_path": input_path,
        "output_path": output_path,
        "submitted_at": int(time.time()),
        "config": job_config,
    }

    background_tasks.add_task(
        process_batch_job, job_id, input_path, output_path, job_config
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Batch job accepted and queued for processing.",
    }

# Endpoint 2: Check job status
@app.get("/v1/batch/jobs/{job_id}")
def get_batch_job_status(job_id: str, auth_ok: bool = Depends(verify_token)):
    job = JOB_STATUS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "submitted_at": job["submitted_at"],
        "error": job.get("error"),
    }

# Endpoint 3: Get job results
@app.get("/v1/batch/jobs/{job_id}/results")
def get_batch_job_results(job_id: str, auth_ok: bool = Depends(verify_token)):
    job = JOB_STATUS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "completed":
        output_path = job["output_path"]
        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=500, detail="Job completed but output file is missing."
            )
        return FileResponse(
            path=output_path,
            media_type="application/json",
            filename=f"{job_id}_output.json",
        )
    elif job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {job.get('error', 'Unknown error')}",
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job['status']}",
        )
