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
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda")  # e.g., "cuda" or "auto"
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my-secret-token-structured-generation")

PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(
    int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "3000").split(",")
)
# Optional: languages to prebuild, comma-separated. If empty, skip prebuild.
PREBUILD_LANGS = [x.strip() for x in os.getenv("PREBUILD_LANGS", "").split(",") if x.strip()]

TORCH_DTYPE_STR = os.getenv("TORCH_DTYPE", "auto")

# Config for Batch Jobs
BATCH_JOB_TEMP_DIR = os.getenv("BATCH_JOB_TEMP_DIR", tempfile.gettempdir())
BATCH_JOB_PIPELINE_SIZE = int(os.getenv("BATCH_JOB_PIPELINE_SIZE", "8"))

logger.info(f"Batch jobs will be stored in: {BATCH_JOB_TEMP_DIR}")
logger.info(f"Batch job pipeline size: {BATCH_JOB_PIPELINE_SIZE}")

# -----------------------
# Model / tokenizer load
# -----------------------
# Determine torch_dtype from environment variable
if TORCH_DTYPE_STR.lower() in ("bf16", "bfloat16", "torch.bfloat16"):
    TORCH_DTYPE = torch.bfloat16
    logger.info("Using torch_dtype: bfloat16")
elif TORCH_DTYPE_STR.lower() in ("fp16", "float16", "torch.float16"):
    TORCH_DTYPE = torch.float16
    logger.info("Using torch_dtype: float16")
else:
    TORCH_DTYPE = "auto"
    logger.info("Using torch_dtype: auto")

model_init_kwargs = {
    "device_map": DEVICE_MAP,
    "trust_remote_code": TRUST_REMOTE_CODE,
    "torch_dtype": TORCH_DTYPE,
}
logger.info(f"Loading model '{MODEL_NAME}' (Trust Remote Code: {TRUST_REMOTE_CODE})...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    **model_init_kwargs,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)
model.eval()

# -----------------------
# Helpers
# -----------------------
def is_sharded_model(m) -> bool:
    # True when loaded with accelerate dispatch (device_map="auto" etc.)
    return getattr(m, "hf_device_map", None) is not None

def move_inputs_to_correct_device(inputs: Dict[str, torch.Tensor], m):
    # Universal safe rule:
    # - If model is sharded (multi-GPU dispatch), keep inputs on CPU; dispatch will handle it.
    # - Otherwise, move to the model's parameter device.
    if is_sharded_model(m):
        return inputs
    # Non-sharded model case
    try:
        device = next(m.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {k: v.to(device) for k, v in inputs.items()}

def tokenizer_encode_for_chat(text: str) -> Dict[str, torch.Tensor]:
    # Conditionally disable token_type_ids for GLM-4.6-FP8 only
    tok_kwargs = {"return_tensors": "pt"}
    if MODEL_NAME == "zai-org/GLM-4.6-FP8":
        tok_kwargs["return_token_type_ids"] = False
    return tokenizer(text, **tok_kwargs)

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
            if eid is not None and eid != tok.unk_token_id:
                stop_ids.append(eid)
        except Exception:
            pass
    # unique
    return list(set(stop_ids))

# -----------------------
# Regex-constrained vocab (lowercase-only, trie-based)
# -----------------------
def normalizeword(w: str) -> str:
    # Normalize Unicode and keep accents; enforce lowercase
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
    # Returns a regex for all suffixes from this node, pruned by n_limit.
    alts = []
    # Deterministic order for stable patterns
    for ch, child in sorted(node.children.items()):
        if child.min_rank > nlimit:
            continue
        sub = trieto_regex(child, nlimit)
        alts.append(escapefor_regex(ch) + sub)
    # If this node can terminate a word within the n_limit, allow empty suffix.
    if node.end and node.min_rank <= nlimit:
        alts.append("")
    if not alts:
        return ""
    if len(alts) == 1:
        return alts[0]
    return "(?:" + "|".join(alts) + ")"

TRIECACHE: Dict[str, Dict[str, Any]] = {}

def _safe_lang_name(lang: str) -> bool:
    # Prevent path traversal; only allow simple alphanum, underscore, dash
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
    # Store only the trie to reduce memory footprint
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
    # Punctuation between words; includes spaces/newlines and common punctuation
    punct_regex = r'[.,!?¿¡…\s]+'
    # Full-string grammar (not anchored, per user preference)
    flexible_grammar = fr'(?:{punct_regex})?(?:{word_regex})(?:{punct_regex}(?:{word_regex}))*(?:{punct_regex})?'
    parser = RegexParser(flexible_grammar)
    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    stop_ids = set(get_stop_ids(tokenizer))
    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        # Ensure EOS is always allowed
        return list(allowed | stop_ids)
    return wrapped_prefix_fn

# Startup prebuilds (if requested)
if PREBUILD_PREFIX:
    if PREBUILD_LANGS:
        logger.info("Prebuilding prefix functions...")
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
        if len(parts) == 2 and parts[0].lower() == "bearer":
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
        do_sample=False,                 # always deterministic
        num_beams=int(num_beams or 10),  # default to 10 beams (can be overridden)
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

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True
    }
    if MODEL_NAME == "zai-org/GLM-4.6-FP8":
        logger.info("Applying 'enable_thinking: False' for GLM-4.6-FP8.")
        template_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    texts = tokenizer.apply_chat_template(messages, **template_kwargs)

    # Conditionally suppress token_type_ids only for GLM-4.6-FP8
    tok_kwargs = {"return_tensors": "pt"}
    if MODEL_NAME == "zai-org/GLM-4.6-FP8":
        tok_kwargs["return_token_type_ids"] = False
    inputs = tokenizer(texts, **tok_kwargs)

    # Universal sharding-safe device handling
    inputs = move_inputs_to_correct_device(inputs, model)

    input_len = inputs["input_ids"].shape[1]

    prefix_fn = None
    if req.vocab_lang and req.vocab_n_words:
        prefix_fn = build_regexp_prefix_fn(req.vocab_lang, req.vocab_n_words)
        if prefix_fn is None:
            raise HTTPException(status_code=500, detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.")

    max_new_tokens = int(req.max_tokens) if req.max_tokens is not None else 512
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
                if MODEL_NAME == "zai-org/GLM-4.6-FP8":
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

        generation_kwargs = dict(
            max_new_tokens=job_config.get("max_tokens", 512),
            do_sample=False,  # always deterministic
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

        # 4. Run the pipeline
        logger.info(f"[Job {job_id}] Running pipeline (Batch Size: {BATCH_JOB_PIPELINE_SIZE})...")

        # Conditionally suppress token_type_ids only for GLM-4.6-FP8
        pipeline_kwargs = {"return_full_text": False}
        if MODEL_NAME == "zai-org/GLM-4.6-FP8":
            pipeline_kwargs["return_token_type_ids"] = False

        results = []
        for output in text_gen_pipeline(
            prompts,
            batch_size=BATCH_JOB_PIPELINE_SIZE,
            **pipeline_kwargs,
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
        # Cleanup input file to save space
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

    # 1. Save the uploaded file
    try:
        with open(input_path, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input file: {e}")

    # 2. Store job config
    job_config = {
        "max_tokens": max_tokens,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "vocab_lang": vocab_lang,
        "vocab_n_words": vocab_n_words,
    }

    # 3. Register the job
    JOB_STATUS[job_id] = {
        "status": "pending",
        "input_path": input_path,
        "output_path": output_path,
        "submitted_at": int(time.time()),
        "config": job_config,
    }

    # 4. Schedule the background task
    background_tasks.add_task(
        process_batch_job, job_id, input_path, output_path, job_config
    )

    # 5. Return 202 Accepted immediately
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
