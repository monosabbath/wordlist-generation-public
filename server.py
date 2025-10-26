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
from fastapi import Depends, FastAPI, HTTPException, Header, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# Load env
load_dotenv()

# TRT-LLM + LMFE
from tensorrt_llm import LLM, SamplingParams
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.trtllm import build_trtllm_logits_processor

# Optionally keep HF tokenizer for chat templates
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

# Beam/LLM config
TLLM_MAX_BEAM_WIDTH = int(os.getenv("TLLM_MAX_BEAM_WIDTH", "10"))
TLLM_DISABLE_OVERLAP_SCHEDULER = os.getenv("TLLM_DISABLE_OVERLAP_SCHEDULER", "true").lower() == "true"
TLLM_DISABLE_CUDA_GRAPHS = os.getenv("TLLM_DISABLE_CUDA_GRAPHS", "true").lower() == "true"

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"

# Constrained vocab prebuild
PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "3000").split(","))
PREBUILD_LANGS = [x.strip() for x in os.getenv("PREBUILD_LANGS", "").split(",") if x.strip()]

# Batch jobs
BATCH_JOB_TEMP_DIR = os.getenv("BATCH_JOB_TEMP_DIR", tempfile.gettempdir())
BATCH_JOB_PIPELINE_SIZE = int(os.getenv("BATCH_JOB_PIPELINE_SIZE", "8"))
logger.info(f"Batch jobs will be stored in: {BATCH_JOB_TEMP_DIR}")
logger.info(f"Batch job pipeline size: {BATCH_JOB_PIPELINE_SIZE}")

# -----------------------
# Load model (TRT-LLM)
# -----------------------
logger.info(f"Loading TRT-LLM model '{MODEL_NAME}' ...")
llm = LLM(
    model=MODEL_NAME,
    # Beam search requirements:
    max_beam_width=TLLM_MAX_BEAM_WIDTH,
    disable_overlap_scheduler=TLLM_DISABLE_OVERLAP_SCHEDULER,
    cuda_graph_config=None if TLLM_DISABLE_CUDA_GRAPHS else "default",
)
# Use HF tokenizer for chat templates; decode with the same tokenizer for consistency
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)
if hf_tokenizer.pad_token_id is None:
    hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id

# -----------------------
# Helpers for stop tokens
# -----------------------
@cache
def get_stop_ids(tok) -> List[int]:
    stop_ids: List[int] = []
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, int):
            stop_ids.append(tok.eos_token_id)
        elif isinstance(tok.eos_token_id, list):
            stop_ids.extend(tok.eos_token_id)
    for special in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
        try:
            eid = tok.convert_tokens_to_ids(special)
            if eid is not None and eid != tok.unk_token_id:
                stop_ids.append(eid)
        except Exception:
            pass
    return list(set(stop_ids))

# -----------------------
# Regex-constrained vocab (trie) — unchanged
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

def trieto_regex(node: _TrieNode, nlimit: int) -> str:
    alts = []
    for ch, child in sorted(node.children.items()):
        if child.min_rank > nlimit:
            continue
        sub = trieto_regex(child, nlimit)
        alts.append(re.escape(ch) + sub)
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

# Build a TRT-LLM logits processor to enforce the regex at generation time
@cache
def build_trtllm_vocab_logits_processor(lang: str, n_words: int):
    logger.info(f"building TRT-LLM logits processor for {lang} ({n_words} words)")
    if getor_build_trie(lang) is None:
        logger.warning(f"Language file not found or empty: {lang}.txt")
        return None
    word_regex = buildword_regex_for_n(lang, n_words)
    if not word_regex:
        logger.warning(f"No words available for {lang} with n={n_words}.")
        return None
    punct_regex = r'[.,!?¿¡…\s]+'
    grammar = fr'(?:{punct_regex})?(?:{word_regex})(?:{punct_regex}(?:{word_regex}))*(?:{punct_regex})?'
    parser = RegexParser(grammar)
    # Use TRT-LLM tokenizer. It is compatible with HF tokenization rules for the same model.
    tllm_tok = llm.runtime_context.tokenizer
    return build_trtllm_logits_processor(tllm_tok, parser)

# Startup prebuild (unchanged semantics; just builds/warms cache)
if PREBUILD_PREFIX and PREBUILD_LANGS:
    for lang in PREBUILD_LANGS:
        if not _safe_lang_name(lang):
            logger.warning(f"Skipping unsafe language name in PREBUILD_LANGS: {lang}")
            continue
        for n in PREBUILD_WORD_COUNTS:
            try:
                build_trtllm_vocab_logits_processor(lang, n)
            except Exception as e:
                logger.warning(f"Prebuild failed for {lang}, n={n}: {e}")

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
# API models
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
    num_beams: Optional[int] = 10
    length_penalty: Optional[float] = 1.0

@app.get("/v1/models")
def list_models(auth_ok: bool = Depends(verify_token)):
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "owner"}],
    }

def build_sampling_params(max_new_tokens: int, num_beams: int, length_penalty: float, stop_ids: List[int]) -> SamplingParams:
    use_beam = (num_beams or 1) > 1
    sp = SamplingParams(
        max_new_tokens=int(max_new_tokens),
        temperature=0.0,  # deterministic
        top_k=1,
        top_p=0.0,
        use_beam_search=use_beam,
        best_of=(num_beams if use_beam else 1),  # beam width
        n=1,  # we return only the top sequence
        length_penalty=float(length_penalty),
        end_id=hf_tokenizer.eos_token_id or 0,
        pad_id=hf_tokenizer.pad_token_id or (hf_tokenizer.eos_token_id or 0),
        stop_words_list=[[sid] for sid in stop_ids] if stop_ids else None,
    )
    return sp

def apply_chat_template(messages: List[Dict[str,str]]) -> str:
    # Use HF tokenizer to build a single prompt string
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Example: for GLM-4.6-FP8 you might disable token_type_ids; not needed here.
    return hf_tokenizer.apply_chat_template(messages, **template_kwargs)

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, auth_ok: bool = Depends(verify_token)):
    # Build messages (system-first, preserve only the provided system prompt)
    system_prompt = ""
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
            break
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in req.messages:
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    text = apply_chat_template(messages)
    encoded = hf_tokenizer(text, return_tensors="pt", padding=False)
    input_ids = encoded["input_ids"]  # [1, seq]
    input_len = int(input_ids.shape[1])

    max_new_tokens = int(req.max_tokens) if req.max_tokens is not None else 512
    stop_ids = get_stop_ids(hf_tokenizer)
    sampling_params = build_sampling_params(
        max_new_tokens=max_new_tokens,
        num_beams=req.num_beams or 10,
        length_penalty=req.length_penalty if req.length_penalty is not None else 1.0,
        stop_ids=stop_ids,
    )

    logits_processor = None
    if req.vocab_lang and req.vocab_n_words:
        logits_processor = build_trtllm_vocab_logits_processor(req.vocab_lang, req.vocab_n_words)
        if logits_processor is None:
            raise HTTPException(status_code=500, detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.")

    # Use TRT-LLM runtime to generate with optional logits processor
    with torch.inference_mode():
        out = llm.runtime_context.runtime.generate(
            input_ids,
            sampling_params=sampling_params,
            logits_processor=logits_processor,
        )

    # TRT-LLM returns output_ids shaped [batch, beams, seq]; we use first beam of first batch
    out_ids = out["output_ids"][0][0]
    gen_ids = out_ids[input_len:]
    # If constrained, ensure we trim partial token if needed
    if logits_processor:
        gen_ids = logits_processor._trim(gen_ids)
    gen_ids = gen_ids.tolist()
    text_out = hf_tokenizer.decode(gen_ids, skip_special_tokens=True)

    gen_len = len(gen_ids)
    last_token = gen_ids[-1] if gen_ids else None
    finish_reason = "stop" if (stop_ids and last_token in set(stop_ids)) else ("length" if gen_len >= max_new_tokens else "stop")

    created = int(time.time())
    resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": req.model or MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text_out},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": int(input_len),
            "completion_tokens": int(gen_len),
            "total_tokens": int(input_len + gen_len),
        },
    }
    return resp

# ---------------------------------------------------
# BATCH JOB API (rewrite pipeline part to TRT-LLM)
# ---------------------------------------------------
JOB_STATUS: Dict[str, Dict[str, Any]] = {}

def process_batch_job(job_id: str, input_path: str, output_path: str, job_config: Dict[str, Any]):
    logger.info(f"[Job {job_id}] Starting processing...")
    try:
        JOB_STATUS[job_id]["status"] = "processing"
        with open(input_path, "r", encoding="utf-8") as f:
            raw_requests = json.load(f)
            if not isinstance(raw_requests, list):
                raise ValueError("Input file must contain a JSON list.")

        prompts = []
        for i, req_data in enumerate(raw_requests):
            try:
                req = ChatCompletionRequest(**req_data)
                system_prompt = ""
                for msg in req.messages:
                    if msg.role == "system":
                        system_prompt = msg.content
                        break
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                for msg in req.messages:
                    if msg.role != "system":
                        messages.append({"role": msg.role, "content": msg.content})
                prompts.append(apply_chat_template(messages))
            except Exception as e:
                logger.warning(f"[Job {job_id}] Skipping request {i}: {e}")

        stop_ids = get_stop_ids(hf_tokenizer)
        sampling_params = build_sampling_params(
            max_new_tokens=int(job_config.get("max_tokens", 512)),
            num_beams=int(job_config.get("num_beams", 10)),
            length_penalty=float(job_config.get("length_penalty", 1.0)),
            stop_ids=stop_ids,
        )

        logits_processor = None
        vocab_lang = job_config.get("vocab_lang")
        vocab_n_words = job_config.get("vocab_n_words")
        if vocab_lang and vocab_n_words:
            logger.info(f"[Job {job_id}] Building constrained vocab for {vocab_lang} ({vocab_n_words} words)")
            logits_processor = build_trtllm_vocab_logits_processor(vocab_lang, vocab_n_words)
            if not logits_processor:
                raise ValueError(f"Constrained vocabulary config failed for lang '{vocab_lang}'.")

        # Batch in chunks
        results = []
        bs = int(BATCH_JOB_PIPELINE_SIZE)
        for start in range(0, len(prompts), bs):
            chunk = prompts[start:start+bs]
            enc = hf_tokenizer(chunk, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"]
            input_lens = [int(l) for l in (enc["attention_mask"].sum(dim=1).tolist() if "attention_mask" in enc else [input_ids.shape[1]]*len(chunk))]
            out = llm.runtime_context.runtime.generate(
                input_ids,
                sampling_params=sampling_params,
                logits_processor=logits_processor,
            )
            # Collect outputs
            for i in range(len(chunk)):
                out_ids = out["output_ids"][i][0]
                gen_ids = out_ids[input_lens[i]:]
                if logits_processor:
                    gen_ids = logits_processor._trim(gen_ids)
                text_out = hf_tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
                results.append(text_out)

        created = int(time.time())
        final_output = []
        for i, text in enumerate(results):
            final_output.append({
                "id": f"chatcmpl-batch-{job_id}-{i}",
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            })
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
        except Exception:
            pass

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
    background_tasks.add_task(process_batch_job, job_id, input_path, output_path, job_config)
    return {"job_id": job_id, "status": "pending", "message": "Batch job accepted and queued for processing."}

@app.get("/v1/batch/jobs/{job_id}")
def get_batch_job_status(job_id: str, auth_ok: bool = Depends(verify_token)):
    job = JOB_STATUS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"], "submitted_at": job["submitted_at"], "error": job.get("error")}

@app.get("/v1/batch/jobs/{job_id}/results")
def get_batch_job_results(job_id: str, auth_ok: bool = Depends(verify_token)):
    job = JOB_STATUS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "completed":
        output_path = job["output_path"]
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Job completed but output file is missing.")
        return FileResponse(path=output_path, media_type="application/json", filename=f"{job_id}_output.json")
    elif job["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.get('error', 'Unknown error')}")
    else:
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {job['status']}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
