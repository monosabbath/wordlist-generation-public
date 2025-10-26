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

import torch  # required for building input tensors
import yaml
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

# Load env
load_dotenv()

# TensorRT-LLM
from tensorrt_llm import LLM
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime import SamplingConfig as RuntimeSamplingConfig

# lm-format-enforcer integration for TensorRT-LLM
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.trtllm import build_trtllm_logits_processor

# For chat templates only; we do NOT use transformers for inference
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("vocab-constrained-trtllm")
app = FastAPI()

# -----------------------
# Config via environment
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my-secret-token-structured-generation")

# Constrained vocab prebuild config
PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(
    int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "3000").split(",")
)
PREBUILD_LANGS = [x.strip() for x in os.getenv("PREBUILD_LANGS", "").split(",") if x.strip()]

# TRT-LLM memory caps (keep modest to avoid OOM; adjust via env as needed)
MAX_BEAM_WIDTH = int(os.getenv("MAX_BEAM_WIDTH", "10"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1"))
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "512"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "1024"))
MAX_NUM_TOKENS = int(os.getenv("MAX_NUM_TOKENS", str(MAX_SEQ_LEN)))
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto").lower()
PARALLEL_CONFIG_PATH = os.getenv("PARALLEL_CONFIG_PATH", None)

# Batch jobs
BATCH_JOB_TEMP_DIR = os.getenv("BATCH_JOB_TEMP_DIR", tempfile.gettempdir())
BATCH_JOB_PIPELINE_SIZE = int(os.getenv("BATCH_JOB_PIPELINE_SIZE", "8"))
logger.info(f"Batch jobs will be stored in: {BATCH_JOB_TEMP_DIR}")
logger.info(f"Batch job pipeline size (unused by TRT-LLM scheduler): {BATCH_JOB_PIPELINE_SIZE}")

# -----------------------
# Model / tokenizer load
# -----------------------
logger.info(f"Loading TRT-LLM model '{MODEL_NAME}' ...")
kv_cfg = None
if KV_CACHE_DTYPE == "fp8":
    kv_cfg = KvCacheConfig(dtype="fp8")
elif KV_CACHE_DTYPE not in ("auto", ""):
    logger.warning(
        f"KV_CACHE_DTYPE={KV_CACHE_DTYPE} is not a supported override; accepted: 'fp8' or 'auto'. Using 'auto'."
    )

llm_kwargs: Dict[str, Any] = dict(
    model=MODEL_NAME,
    enable_trtllm_sampler=True,  # harmless here; we use runtime API for generation
    max_beam_width=MAX_BEAM_WIDTH,
    disable_overlap_scheduler=True,
    cuda_graph_config=None,
    trust_remote_code=True,
    # Memory limiting (critical to avoid OOM)
    max_batch_size=MAX_BATCH_SIZE,
    max_input_len=MAX_INPUT_LEN,
    max_seq_len=MAX_SEQ_LEN,
    max_num_tokens=MAX_NUM_TOKENS,
)
if kv_cfg is not None:
    llm_kwargs["kv_cache_config"] = kv_cfg

# Map parallel_config.yaml to supported LLM kwargs (TP/PP/CP/MoE/attention DP/etc.)
if PARALLEL_CONFIG_PATH and os.path.exists(PARALLEL_CONFIG_PATH):
    logger.info(f"Using parallel config: {PARALLEL_CONFIG_PATH}")
    try:
        with open(PARALLEL_CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        tp = cfg.get("tensor_parallel_size") or cfg.get("tp_size") or cfg.get("tp") or 1
        pp = cfg.get("pipeline_parallel_size") or cfg.get("pp_size") or cfg.get("pp") or 1
        cp = cfg.get("context_parallel_size") or cfg.get("cp_size") or cfg.get("cp") or 1
        moe_tp = cfg.get("moe_tensor_parallel_size") or cfg.get("moe_tp")
        moe_ep = cfg.get("moe_expert_parallel_size") or cfg.get("moe_ep")
        gpn = cfg.get("gpus_per_node")
        enable_attention_dp = bool(cfg.get("enable_attention_dp", False))
        llm_kwargs.update(
            tensor_parallel_size=int(tp),
            pipeline_parallel_size=int(pp),
            context_parallel_size=int(cp),
            enable_attention_dp=enable_attention_dp,
        )
        if moe_tp is not None:
            llm_kwargs["moe_tensor_parallel_size"] = int(moe_tp)
        if moe_ep is not None:
            llm_kwargs["moe_expert_parallel_size"] = int(moe_ep)
        if gpn is not None:
            llm_kwargs["gpus_per_node"] = int(gpn)
        logger.info(
            f"LLM parallel settings -> TP={llm_kwargs.get('tensor_parallel_size', 1)}, "
            f"PP={llm_kwargs.get('pipeline_parallel_size', 1)}, "
            f"CP={llm_kwargs.get('context_parallel_size', 1)}, "
            f"MoE-TP={llm_kwargs.get('moe_tensor_parallel_size')}, "
            f"MoE-EP={llm_kwargs.get('moe_expert_parallel_size')}, "
            f"Attention-DP={llm_kwargs.get('enable_attention_dp', False)}, "
            f"GPN={llm_kwargs.get('gpus_per_node')}"
        )
    except Exception as e:
        logger.warning(f"Failed to parse {PARALLEL_CONFIG_PATH}: {e}. Falling back to defaults.")
else:
    logger.info("No parallel_config provided. TRT-LLM will use defaults (single node).")

llm = LLM(**llm_kwargs)

# -----------------------
# Warmup helper (to create runtime once)
# -----------------------
def ensure_trt_runtime_ready():
    # Lazily initialize the TRT-LLM runtime so llm.runtime_context is not None.
    if getattr(llm, "runtime_context", None) is None or getattr(llm.runtime_context, "runtime", None) is None:
        try:
            # Try convenience signature first
            llm.generate(" ", max_new_tokens=1)
            logger.info("TRT-LLM warmup: runtime initialized using convenience generate()")
        except TypeError:
            # Fall back to explicit SamplingParams for older/newer versions
            try:
                from tensorrt_llm.llmapi import SamplingParams as LLMAPISamplingParams  # type: ignore
                llm.generate(" ", LLMAPISamplingParams(max_tokens=1))
                logger.info("TRT-LLM warmup: runtime initialized using LLMAPISamplingParams")
            except Exception as e2:
                logger.warning(f"TRT-LLM warmup via LLMAPISamplingParams failed: {e2}")
        except Exception as e:
            logger.warning(f"TRT-LLM warmup failed (runtime may still be uninitialized): {e}")

# Proactively warm up runtime to avoid first-request latency and None runtime_context
ensure_trt_runtime_ready()
logger.info(
    "TRT-LLM limits -> batch=%s, beam=%s, input=%s, seq=%s, num_tokens=%s, kv_override=%s",
    MAX_BATCH_SIZE, MAX_BEAM_WIDTH, MAX_INPUT_LEN, MAX_SEQ_LEN, MAX_NUM_TOKENS, KV_CACHE_DTYPE
)

# HF tokenizer only to render chat templates (no weights loaded)
chat_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if chat_tokenizer.pad_token_id is None and chat_tokenizer.eos_token_id is not None:
    chat_tokenizer.pad_token_id = chat_tokenizer.eos_token_id

# -----------------------
# Helpers for stop tokens
# -----------------------
def _runtime_tokenizer():
    # Prefer TRT-LLM runtime tokenizer if runtime is initialized;
    # otherwise fall back to the HF tokenizer.
    ctx = getattr(llm, "runtime_context", None)
    if ctx is not None and getattr(ctx, "tokenizer", None) is not None:
        return ctx.tokenizer
    return chat_tokenizer

@cache
def get_eos_id() -> int:
    tok = _runtime_tokenizer()
    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is None:
        eos_id = chat_tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 2
    return int(eos_id)

@cache
def get_pad_id() -> int:
    tok = _runtime_tokenizer()
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = chat_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = get_eos_id()
    return int(pad_id)

# -----------------------
# Regex-constrained vocab (lowercase-only, trie-based)
# -----------------------
def normalize_word(w: str) -> str:
    return unicodedata.normalize("NFC", w.strip()).lower()

class _TrieNode:
    __slots__ = ("children", "end", "min_rank")
    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.end: bool = False
        self.min_rank: int = 10**12

def build_trie_with_ranks(words: List[str]) -> _TrieNode:
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

def escape_for_regex(ch: str) -> str:
    return re.escape(ch)

def trie_to_regex(node: _TrieNode, nlimit: int) -> str:
    alts = []
    for ch, child in sorted(node.children.items()):
        if child.min_rank > nlimit:
            continue
        sub = trie_to_regex(child, nlimit)
        alts.append(escape_for_regex(ch) + sub)
    if node.end and node.min_rank <= nlimit:
        alts.append("")
    if not alts:
        return ""
    if len(alts) == 1:
        return alts[0]
    return "(?:" + "|".join(alts) + ")"

TRIE_CACHE: Dict[str, Dict[str, Any]] = {}

def safe_lang_name(lang: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9-]+", lang))

def get_or_build_trie(lang: str) -> Optional[Dict[str, Any]]:
    if not safe_lang_name(lang):
        logger.warning(f"Unsafe or invalid language identifier: {lang}")
        return None
    if lang in TRIE_CACHE:
        return TRIE_CACHE[lang]
    filename = f"{lang}.txt"
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, encoding="utf-8") as fin:
            words = [normalize_word(w) for w in fin if w.strip()]
    except Exception as e:
        logger.error(f"Error reading language file {filename}: {e}")
        return None
    if not words:
        return None
    trie = build_trie_with_ranks(words)
    TRIE_CACHE[lang] = {"trie": trie}
    return TRIE_CACHE[lang]

def build_word_regex_for_n(lang: str, n_words: int) -> Optional[str]:
    data = get_or_build_trie(lang)
    if data is None:
        return None
    trie: _TrieNode = data["trie"]
    if trie.min_rank > n_words:
        return None
    return trie_to_regex(trie, nlimit=n_words)

@cache
def build_regex_grammar(lang: str, n_words: int) -> Optional[str]:
    word_regex = build_word_regex_for_n(lang, n_words)
    if not word_regex:
        return None
    punct_regex = r'[.,!?¿¡…\s]+'
    return fr'(?:{punct_regex})?(?:{word_regex})(?:{punct_regex}(?:{word_regex}))*(?:{punct_regex})?'

def build_logits_processor_for(lang: str, n_words: int):
    # Ensure runtime is ready so we prefer TRT tokenizer if available
    ensure_trt_runtime_ready()
    grammar = build_regex_grammar(lang, n_words)
    if not grammar:
        return None
    parser = RegexParser(grammar)
    tok = _runtime_tokenizer()  # TRT tokenizer if available; HF fallback otherwise
    return build_trtllm_logits_processor(tok, parser)

# Startup prebuilds (optional)
if PREBUILD_PREFIX:
    if PREBUILD_LANGS:
        logger.info("Prebuilding regex grammars...")
        for lang in PREBUILD_LANGS:
            if not safe_lang_name(lang):
                logger.warning(f"Skipping unsafe language name in PREBUILD_LANGS: {lang}")
                continue
            for n_words in PREBUILD_WORD_COUNTS:
                try:
                    _ = build_regex_grammar(lang, n_words)
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
# Generation helpers (runtime API only, constrained decoding)
# -----------------------
def apply_chat_template(messages: List[Dict[str, str]]) -> str:
    system_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
            break
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for msg in messages:
        if msg["role"] != "system":
            msgs.append({"role": msg["role"], "content": msg["content"]})
    text = chat_tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text

def build_runtime_sampling_config(
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
) -> RuntimeSamplingConfig:
    eos = get_eos_id()
    pad = get_pad_id()
    # Enforce TRT-LLM requirement: request beam width == configured max_beam_width
    nb = MAX_BEAM_WIDTH
    # Beam search: enforce temperature 0.0; greedy (nb=1) uses temperature 1.0
    temperature = 0.0 if nb > 1 else 1.0
    top_k = 1
    top_p = 0.0
    return RuntimeSamplingConfig(
        end_id=eos,
        pad_id=pad,
        max_new_tokens=int(max_new_tokens),
        num_beams=nb,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        length_penalty=float(length_penalty),
        early_stopping=1,
        use_beam_hyps=True,
    )

def trt_runtime_generate_texts(
    prompts: List[str],
    runtime_sampling_config: RuntimeSamplingConfig,
    logits_processor,
) -> List[str]:
    # Ensure TRT-LLM runtime is ready before low-level generate
    ensure_trt_runtime_ready()
    rt_ctx = getattr(llm, "runtime_context", None)
    if rt_ctx is None or getattr(rt_ctx, "runtime", None) is None:
        raise RuntimeError("TRT-LLM runtime is not initialized. See logs for warmup failures.")
    tokenizer = _runtime_tokenizer()
    # Encode batch
    enc = tokenizer.batch_encode_plus(prompts)
    batch_enc = enc["input_ids"]
    inputs = torch.LongTensor(batch_enc)
    # Call the low-level runtime API with logits_processor
    out = rt_ctx.runtime.generate(
        inputs,
        sampling_config=runtime_sampling_config,
        logits_processor=logits_processor,
    )
    # Decode only the newly generated tokens per item (beam 0)
    texts: List[str] = []
    output_ids = out["output_ids"]
    for i, inp_ids in enumerate(batch_enc):
        seq_ids = output_ids[i][0].tolist()
        new_ids = seq_ids[len(inp_ids):]
        texts.append(tokenizer.decode(new_ids))
    return texts

# -----------------------
# OpenAI-compatible API (constrained only)
# -----------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    vocab_lang: str
    vocab_n_words: int
    # Default to MAX_BEAM_WIDTH; if provided and mismatched, we will coerce to MAX_BEAM_WIDTH.
    num_beams: Optional[int] = None
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
    # Ensure runtime is ready early so eos/pad IDs come from TRT tokenizer
    ensure_trt_runtime_ready()

    # Require constrained config
    if not req.vocab_lang or not req.vocab_n_words:
        raise HTTPException(status_code=400, detail="vocab_lang and vocab_n_words are required for constrained decoding.")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text_prompt = apply_chat_template(messages)

    # Build logits processor from LMFE
    logits_processor = build_logits_processor_for(req.vocab_lang, int(req.vocab_n_words))
    if logits_processor is None:
        raise HTTPException(
            status_code=500,
            detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.",
        )

    max_new_tokens = int(req.max_tokens) if req.max_tokens is not None else 128

    # Resolve num_beams; coerce to MAX_BEAM_WIDTH to satisfy TRT-LLM assertion
    requested_beams = int(req.num_beams) if req.num_beams is not None else MAX_BEAM_WIDTH
    if requested_beams != MAX_BEAM_WIDTH:
        logger.warning(
            f"Request num_beams={requested_beams} differs from MAX_BEAM_WIDTH={MAX_BEAM_WIDTH}. "
            f"Coercing to {MAX_BEAM_WIDTH} to satisfy TRT-LLM."
        )
    runtime_cfg = build_runtime_sampling_config(
        max_new_tokens=max_new_tokens,
        num_beams=MAX_BEAM_WIDTH,
        length_penalty=float(req.length_penalty if req.length_penalty is not None else 1.0),
    )

    start = time.time()
    outputs = trt_runtime_generate_texts([text_prompt], runtime_cfg, logits_processor=logits_processor)
    text = outputs[0] if outputs else ""
    elapsed = time.time() - start
    logger.info(f"Generated (runtime+LMFE) in {elapsed:.2f}s")

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
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }
    return resp

# ---------------------------------------------------
# BATCH JOB API (constrained only)
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

        with open(input_path, "r", encoding="utf-8") as f:
            raw_requests = json.load(f)
        if not isinstance(raw_requests, list):
            raise ValueError("Input file must contain a JSON list.")

        logger.info(f"[Job {job_id}] Preparing {len(raw_requests)} prompts...")
        prompts: List[str] = []
        for i, req_data in enumerate(raw_requests):
            try:
                req = ChatCompletionRequest(**req_data)
                messages = [{"role": m.role, "content": m.content} for m in req.messages]
                text = apply_chat_template(messages)
                prompts.append(text)
            except ValidationError as e:
                logger.warning(f"[Job {job_id}] Skipping request {i}: Invalid format. {e}")
            except Exception as e:
                logger.warning(f"[Job {job_id}] Skipping request {i}: Error processing. {e}")

        if not prompts:
            raise ValueError("No valid requests in batch.")

        max_tokens = int(job_config.get("max_tokens", 128))
        requested_num_beams = int(job_config.get("num_beams", MAX_BEAM_WIDTH))
        if requested_num_beams != MAX_BEAM_WIDTH:
            logger.warning(
                f"[Job {job_id}] Requested num_beams={requested_num_beams} differs from MAX_BEAM_WIDTH={MAX_BEAM_WIDTH}. "
                f"Coercing to {MAX_BEAM_WIDTH} to satisfy TRT-LLM."
            )
        length_penalty = float(job_config.get("length_penalty", 1.0))
        vocab_lang = job_config.get("vocab_lang")
        vocab_n_words = job_config.get("vocab_n_words")
        if not vocab_lang or not vocab_n_words:
            raise ValueError("vocab_lang and vocab_n_words are required for constrained decoding batch jobs.")

        # Ensure runtime is ready early so eos/pad IDs come from TRT tokenizer
        ensure_trt_runtime_ready()

        runtime_cfg = build_runtime_sampling_config(max_tokens, MAX_BEAM_WIDTH, length_penalty)

        logger.info(f"[Job {job_id}] Building constrained vocab for {vocab_lang} ({vocab_n_words} words)")
        logits_processor = build_logits_processor_for(vocab_lang, int(vocab_n_words))
        if logits_processor is None:
            raise ValueError(f"Constrained vocabulary config failed for lang '{vocab_lang}'. Check server logs.")

        logger.info(f"[Job {job_id}] Generating {len(prompts)} responses with TRT-LLM runtime + LMFE ...")
        results = trt_runtime_generate_texts(prompts, runtime_cfg, logits_processor=logits_processor)

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

@app.post("/v1/batch/jobs")
def create_batch_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auth_ok: bool = Depends(verify_token),
    max_tokens: int = 128,
    # Default num_beams to MAX_BEAM_WIDTH to satisfy TRT-LLM assertion
    num_beams: int = MAX_BEAM_WIDTH,
    length_penalty: float = 1.0,
    vocab_lang: Optional[str] = None,
    vocab_n_words: Optional[int] = None,
):
    if not vocab_lang or not vocab_n_words:
        raise HTTPException(status_code=400, detail="vocab_lang and vocab_n_words are required for constrained decoding batch jobs.")

    job_id = str(uuid.uuid4())
    input_path = os.path.join(BATCH_JOB_TEMP_DIR, f"{job_id}_input.json")
    output_path = os.path.join(BATCH_JOB_TEMP_DIR, f"{job_id}_output.json")
    os.makedirs(BATCH_JOB_TEMP_DIR, exist_ok=True)

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
    background_tasks.add_task(process_batch_job, job_id, input_path, output_path, job_config)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Batch job accepted and queued for processing.",
    }

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
