import os
import time
import uuid
import json
import tempfile
import unicodedata
import re
import logging
from functools import cache
from typing import Optional, Dict, Any, List

import torch
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

# Device map (single process)
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")  # used for model placement

# If specific models use custom chat templates, you may need TRUST_REMOTE_CODE=true
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

# Prepare model init kwargs
model_init_kwargs: Dict[str, Any] = {
    "trust_remote_code": TRUST_REMOTE_CODE,
    "low_cpu_mem_usage": True,
    "local_files_only": False,  # allow download into default HF cache
    "device_map": DEVICE_MAP,
}

# Use new 'dtype' param to avoid deprecation warnings if not "auto"
if TORCH_DTYPE != "auto":
    model_init_kwargs["torch_dtype"] = TORCH_DTYPE  # prefer torch_dtype kw in new transformers

attn_impl = ATTN_IMPLEMENTATION
logger.info(
    f"Preparing to load model '{MODEL_NAME}' "
    f"(Trust Remote Code: {TRUST_REMOTE_CODE}, attn='{attn_impl}', device_map='{DEVICE_MAP}')..."
)

# Load from hub/cache directly
logger.info("Loading model from hub/cache...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation=attn_impl,
        **model_init_kwargs,
    )
except TypeError:
    logger.warning("Model.from_pretrained() did not accept 'attn_implementation'; loading without it.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
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
    MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE, local_files_only=False
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
    Works for device_map-sharded models (first param device is fine).
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

    # Base EOS
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, int):
            stop_ids.append(tok.eos_token_id)
        elif isinstance(tok.eos_token_id, list):
            stop_ids.extend(tok.eos_token_id)

    # Common end markers
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

    # Cohere end markers (safe: only added if tokenizer knows them)
    for special in ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"):
        try:
            eid = tok.convert_tokens_to_ids(special)
            if eid is not None and eid != tok.unk_token_id and eid != -1:
                stop_ids.append(eid)
        except Exception:
            pass

    return list(set(stop_ids))

def get_cohere_control_ids(tok: AutoTokenizer) -> List[int]:
    names = [
        "<|START_OF_TURN_TOKEN|>",
        "<|END_OF_TURN_TOKEN|>",
        "<|CHATBOT_TOKEN|>",
        "<|START_RESPONSE|>",
        "<|END_RESPONSE|>",
        "<|START_THINKING|>",
        "<|END_THINKING|>",
    ]
    out: List[int] = []
    for n in names:
        try:
            i = tok.convert_tokens_to_ids(n)
            if i is not None and i != tok.unk_token_id and i != -1:
                out.append(i)
        except Exception:
            pass
    return list(set(out))

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
    cohere_ids = set(get_cohere_control_ids(tokenizer)) if "coherelabs/command-a-reasoning-08-2025" == MODEL_NAME.strip().lower() else set()

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        # Always allow stop + Cohere control ids so the model can open/close turns/thinking/response
        return list(allowed | stop_ids | cohere_ids)

    return wrapped_prefix_fn

# Startup prebuilds (if requested)
if PREBUILD_PREFIX:
    if PREBUILD_LANGS:
        logger.info("Prebuilding prefix functions (single process/device_map)...")
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
# Model-specific helpers
# -----------------------
def is_cohere_reasoning_model(model_name: str) -> bool:
    return model_name.strip().lower() == "coherelabs/command-a-reasoning-08-2025"

def build_template_kwargs_for_model() -> Dict[str, Any]:
    # Base kwargs used everywhere in this server
    kwargs: Dict[str, Any] = {"add_generation_prompt": True}

    # For CohereLabs/command-a-reasoning-08-2025: use tokenized chat template and disable reasoning by default
    if is_cohere_reasoning_model(MODEL_NAME):
        kwargs.update({
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "reasoning": False,  # set True if you want thinking spans
        })
    else:
        kwargs["tokenize"] = False

    return kwargs

def truncate_inputs_to_max_length(inputs: Dict[str, torch.Tensor], max_length: int) -> Dict[str, torch.Tensor]:
    """
    Left-truncate input_ids and attention_mask to max_length tokens (keeping the last max_length tokens).
    """
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    if input_ids.dim() == 1:
        length = input_ids.size(0)
        if length > max_length:
            input_ids = input_ids[-max_length:]
            if attn is not None:
                attn = attn[-max_length:]
    elif input_ids.dim() == 2:
        # batch case
        bsz, seqlen = input_ids.size()
        if seqlen > max_length:
            input_ids = input_ids[:, -max_length:]
            if attn is not None:
                attn = attn[:, -max_length:]
    out = {"input_ids": input_ids}
    if attn is not None:
        out["attention_mask"] = attn
    # pass through any other keys unchanged
    for k, v in inputs.items():
        if k not in out:
            out[k] = v
    return out

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

    # Build base chat template
    template_kwargs = build_template_kwargs_for_model()

    # Cohere path: use tokenized chat template directly
    if is_cohere_reasoning_model(MODEL_NAME):
        inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
        # Optional truncate to MAX_INPUT_TOKENS
        inputs = truncate_inputs_to_max_length(inputs, MAX_INPUT_TOKENS)
        inputs = strip_unused_model_inputs(inputs)
        inputs = move_inputs_to_correct_device(inputs, model)
        input_len = inputs["input_ids"].shape[1]
    else:
        # Other models: render to text and then tokenize
        prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)  # tokenize=False here
        # For GLM-4.6 only: prefill the assistant turn with empty think tags
        if "glm-4.6" in MODEL_NAME.lower():
            prompt_text = prompt_text + "<think></think>"
        inputs = tokenizer_encode_for_chat(prompt_text)
        inputs = strip_unused_model_inputs(inputs)
        inputs = move_inputs_to_correct_device(inputs, model)
        input_len = inputs["input_ids"].shape[1]

    # Build constrained prefix function (validate early)
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

        messages_list: List[List[Dict[str, str]]] = []
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

                messages_list.append(messages)

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
        pf = None
        if vocab_lang and vocab_n_words:
            logger.info(f"[Job {job_id}] Building constrained vocab for {vocab_lang} ({vocab_n_words} words)")
            pf = build_regexp_prefix_fn(vocab_lang, vocab_n_words)
            if pf:
                generation_kwargs["prefix_allowed_tokens_fn"] = pf
                logger.info(f"[Job {job_id}] Successfully added prefix function.")
            else:
                raise ValueError(f"Constrained vocabulary config failed for lang '{vocab_lang}'. Check server logs.")

        # 4. Run generation
        logger.info(f"[Job {job_id}] Running generation...")

        results: List[str] = []

        if is_cohere_reasoning_model(MODEL_NAME):
            # Cohere path: build tokenized inputs via chat template (batched)
            encoded_list = []
            input_lengths = []
            for messages in messages_list:
                tkwargs = build_template_kwargs_for_model()  # tokenize=True for Cohere
                enc = tokenizer.apply_chat_template(messages, **tkwargs)
                # Truncate per-sample to MAX_INPUT_TOKENS (left-truncate)
                if enc["input_ids"].size(1) > MAX_INPUT_TOKENS:
                    enc = truncate_inputs_to_max_length(enc, MAX_INPUT_TOKENS)
                # Convert single-sample tensors to lists for tokenizer.pad
                encoded_list.append({
                    "input_ids": enc["input_ids"][0].tolist(),
                    "attention_mask": enc["attention_mask"][0].tolist() if "attention_mask" in enc else [1] * enc["input_ids"].size(1),
                })
                input_lengths.append(len(encoded_list[-1]["input_ids"]))

            # Pad to batch
            batch = tokenizer.pad(
                encoded_list,
                padding=True,
                pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
                return_tensors="pt",
            )
            batch = strip_unused_model_inputs(batch)
            batch = move_inputs_to_correct_device(batch, model)

            with torch.inference_mode():
                outputs = model.generate(**batch, **generation_kwargs)

            # Decode each item using its own input length
            bsz = outputs.size(0)
            for i in range(bsz):
                in_len = input_lengths[i]
                out_ids = outputs[i][in_len:]
                text = tokenizer.decode(out_ids, skip_special_tokens=True)
                results.append(text)
        else:
            # Other models: use pipeline with string prompts
            prompts: List[str] = []
            for messages in messages_list:
                tkwargs = build_template_kwargs_for_model()  # tokenize=False for non-Cohere
                text = tokenizer.apply_chat_template(messages, **tkwargs)
                # For GLM-4.6 only: prefill the assistant turn with empty think tags
                if "glm-4.6" in MODEL_NAME.lower():
                    text = text + "<think></think>"
                prompts.append(text)

            for output in text_gen_pipeline(
                prompts,
                batch_size=BATCH_JOB_PIPELINE_SIZE,
                return_full_text=False,
                padding=True,
                truncation=True,
                pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
                max_length=MAX_INPUT_TOKENS,         # tokenizer truncation length (pipeline interprets as gen max length)
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
