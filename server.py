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

import torch  # noqa: F401
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
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

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
    enable_trtllm_sampler=True,
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
trt_tokenizer = llm.tokenizer
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
@cache
def get_eos_id() -> int:
    eos_id = getattr(trt_tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = chat_tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 2
    return int(eos_id)

@cache
def get_pad_id() -> int:
    pad_id = getattr(trt_tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = chat_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = get_eos_id()
    return int(pad_id)

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
def build_regex_grammar(lang: str, n_words: int) -> Optional[str]:
    word_regex = buildword_regex_for_n(lang, n_words)
    if not word_regex:
        return None
    punct_regex = r'[.,!?¿¡…\s]+'
    return fr'(?:{punct_regex})?(?:{word_regex})(?:{punct_regex}(?:{word_regex}))*(?:{punct_regex})?'

def _choose_enforcer_tokenizer():
    """
    Prefer TRT tokenizer; fallback to HF tokenizer if len(...) is missing,
    because lm-format-enforcer calls len(tokenizer).
    """
    try:
        _ = len(trt_tokenizer)  # may raise NotImplementedError
        return trt_tokenizer
    except Exception:
        logger.warning("TRT tokenizer lacks len(); using HF tokenizer for constrained decoding.")
        return chat_tokenizer

def build_logits_processor_for(lang: str, n_words: int):
    grammar = build_regex_grammar(lang, n_words)
    if not grammar:
        return None
    parser = RegexParser(grammar)
    tok_for_enforcer = _choose_enforcer_tokenizer()
    return build_trtllm_logits_processor(tok_for_enforcer, parser)

# Startup prebuilds (optional)
if PREBUILD_PREFIX:
    if PREBUILD_LANGS:
        logger.info("Prebuilding regex grammars...")
        for lang in PREBUILD_LANGS:
            if not _safe_lang_name(lang):
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
# Generation helpers
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

def build_sampling_params(
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
) -> SamplingParams:
    eos = get_eos_id()
    pad = get_pad_id()

    # Clamp to engine caps
    max_tokens = int(max(1, min(max_new_tokens or 1, MAX_NUM_TOKENS)))
    nb_req = int(num_beams or 1)
    if nb_req > MAX_BEAM_WIDTH:
        logger.warning("Requested num_beams=%s exceeds engine max_beam_width=%s; clamping.",
                       nb_req, MAX_BEAM_WIDTH)
    nb = max(1, min(nb_req, MAX_BEAM_WIDTH))

    kwargs = dict(
        end_id=eos,
        pad_id=pad,
        max_tokens=max_tokens,           # TRT-LLM 1.0 uses max_tokens
        length_penalty=float(length_penalty),
    )

    if nb > 1:
        # Beam search: best_of == requested beams (clamped)
        kwargs.update(
            use_beam_search=True,
            best_of=nb,
            n=1,                         # return 1 sequence
            temperature=0.0,             # typical for beams; ignored by beam search
        )
    else:
        # Greedy
        kwargs.update(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
        )

    return SamplingParams(**kwargs)

def _extract_text_from_generate_output(o: Any) -> str:
    # Try common TRT-LLM output shapes
    try:
        # Most common: o.outputs is a list, each has .text
        if hasattr(o, "outputs") and o.outputs:
            cand = o.outputs[0]
            if hasattr(cand, "text"):
                return cand.text
        # Sometimes direct .text exists
        if hasattr(o, "text"):
            return o.text
        # Fallback: token_ids -> detokenize
        if hasattr(o, "token_ids") and o.token_ids:
            return trt_tokenizer.decode(o.token_ids)
    except Exception:
        pass
    return ""

def trt_generate_texts(
    prompts: List[str],
    sampling_params: SamplingParams,
    logits_processor=None,
) -> List[str]:
    # In TRT-LLM 1.0, logits processors are set on SamplingParams
    if logits_processor is not None:
        sampling_params.logits_processor = logits_processor

    outs = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )
    texts: List[str] = []
    for o in outs:
        texts.append(_extract_text_from_generate_output(o))
    return texts

# -----------------------
# OpenAI-compatible API
# -----------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    vocab_lang: Optional[str] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = 1
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
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text_prompt = apply_chat_template(messages)

    logits_processor = None
    if req.vocab_lang and req.vocab_n_words:
        logits_processor = build_logits_processor_for(req.vocab_lang, req.vocab_n_words)
        if logits_processor is None:
            raise HTTPException(
                status_code=500,
                detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.",
            )

    max_new_tokens = int(req.max_tokens) if req.max_tokens is not None else 128
    sampling_params = build_sampling_params(
        max_new_tokens=max_new_tokens,
        num_beams=int(req.num_beams or 1),
        length_penalty=float(req.length_penalty if req.length_penalty is not None else 1.0),
    )

    start = time.time()
    outputs = trt_generate_texts([text_prompt], sampling_params, logits_processor=logits_processor)
    text = outputs[0] if outputs else ""
    elapsed = time.time() - start
    logger.info(f"Generated in {elapsed:.2f}s")

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
        num_beams = int(job_config.get("num_beams", 1))
        length_penalty = float(job_config.get("length_penalty", 1.0))
        sampling_params = build_sampling_params(max_tokens, num_beams, length_penalty)

        vocab_lang = job_config.get("vocab_lang")
        vocab_n_words = job_config.get("vocab_n_words")
        logits_processor = None
        if vocab_lang and vocab_n_words:
            logger.info(f"[Job {job_id}] Building constrained vocab for {vocab_lang} ({vocab_n_words} words)")
            logits_processor = build_logits_processor_for(vocab_lang, int(vocab_n_words))
            if logits_processor is None:
                raise ValueError(f"Constrained vocabulary config failed for lang '{vocab_lang}'. Check server logs.")

        logger.info(f"[Job {job_id}] Generating {len(prompts)} responses with TRT-LLM...")
        results = trt_generate_texts(prompts, sampling_params, logits_processor=logits_processor)

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
    num_beams: int = 1,
    length_penalty: float = 1.0,
    vocab_lang: Optional[str] = None,
    vocab_n_words: Optional[int] = None,
):
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
