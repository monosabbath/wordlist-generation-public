# --- Load environment ASAP ---
from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
import uuid
import asyncio
import signal
import subprocess
from typing import Literal, Optional, List, Dict, Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, ConfigDict

# =========================
# Fallback request schemas
# =========================
try:
    from common import GenerateRequest
except ImportError:
    print("Warning: common.GenerateRequest not found. Using placeholder definition.")
    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 100
        repetition_penalty: Optional[float] = 1.0  # not used by vLLM OpenAI API
        length_penalty: Optional[float] = 1.0      # not used by vLLM OpenAI API
        num_beams: Optional[int] = 1
        vocab_lang: Optional[Literal["en", "es"]] = None
        vocab_n_words: Optional[int] = None

# =========================
# FastAPI app
# =========================
app = FastAPI()

# =========================
# Config via environment
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

# vLLM server spawn / connection
START_VLLM_SERVER = os.getenv("START_VLLM_SERVER", "true").lower() == "true"
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8011"))
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

# vLLM engine args
DTYPE_STR = os.getenv("TORCH_DTYPE", "auto").lower()
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", 0.90))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", 0))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"

# HF tokens and cache
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
HF_HOME = os.getenv("HF_HOME", None)

# Auth for this proxy server
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my-secret-token-structured-generation")

# Default system prompts
DEFAULT_SYSTEM_PROMPT_EN = os.getenv("DEFAULT_SYSTEM_PROMPT_EN", "")
DEFAULT_SYSTEM_PROMPT_ES = os.getenv("DEFAULT_SYSTEM_PROMPT_ES", "")

# Prebuild constrained vocab regex parsers at startup
PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(
    int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "500,1000,5000").split(",")
)

# Optional: server-level guided decoding backend (we still set per-request too)
GUIDED_DECODING_BACKEND = os.getenv("GUIDED_DECODING_BACKEND", "lm-format-enforcer")

# =========================
# Wordlist-based Regex Builder (no LMFE import required)
# =========================
def _here_file(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, *parts)

def _load_words(lang: Literal["en", "es"], n_words: int) -> list[str]:
    filename = _here_file(f"{lang}.txt")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Language file not found: {filename}")
    with open(filename, encoding="utf-8") as fin:
        words = [w.strip() for w in fin if w.strip()]
    return words[:n_words]

# Simple cache for patterns
_regex_cache: Dict[tuple[str, int], Optional[str]] = {}

def get_cached_wordlist_regex(
    lang: Optional[Literal["en", "es"]],
    n_words: Optional[int],
) -> Optional[str]:
    """
    Build a case-flexible wordlist regex with optional punctuation between words.
    Grammar (prefix-friendly):
      (SEP)? (WORD) (SEP WORD)* (SEP)?
    """
    if not (lang and n_words):
        return None
    key = (lang, n_words)
    if key in _regex_cache:
        return _regex_cache[key]
    try:
        words = _load_words(lang, n_words)
    except FileNotFoundError as e:
        print(str(e))
        _regex_cache[key] = None
        return None
    if not words:
        print(f"Warning: {lang}.txt is empty or contains no valid words.")
        _regex_cache[key] = None
        return None
    alts = []
    for w in words:
        if not w:
            continue
        first = w[0]
        rest = w[1:]
        if first.isalpha():
            # Build an alt where just the first letter is case-insensitive
            esc_rest = re.escape(rest)
            lc, uc = first.lower(), first.upper()
            alts.append(f"(?:[{lc}{uc}]{esc_rest})")
        else:
            alts.append(f"(?:{re.escape(w)})")
    if not alts:
        _regex_cache[key] = None
        return None
    word_alt = "|".join(alts)
    sep_re = r"[-.,!?():;¿¡\"'“”‘’\s]+"
    # No ^/$ anchors to keep it prefix-friendly for guided decoding
    pattern = f"(?:{sep_re})?(?:{word_alt})(?:{sep_re}(?:{word_alt}))*(?:{sep_re})?"
    _regex_cache[key] = pattern
    return pattern

if PREBUILD_PREFIX:
    print("Prebuilding regex patterns...")
    for lang in ("es",):
        for n in PREBUILD_WORD_COUNTS:
            _ = get_cached_wordlist_regex(lang, n)
    print("Done prebuilding patterns.")

# =========================
# Auth helper
# =========================
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

# =========================
# Pydantic Models (OpenAI)
# =========================
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[List[str] | str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    # Custom parameters
    vocab_lang: Optional[Literal["en", "es"]] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = None
    early_stopping: Optional[bool] = None  # forwarded via extra_body

class BatchGenerateRequest(BaseModel):
    requests: List[GenerateRequest]

# =========================
# Utilities
# =========================
def _normalize_stop(stop_in):
    if stop_in is None:
        return None
    if isinstance(stop_in, str):
        s = stop_in.strip()
        return [s] if s else None
    if isinstance(stop_in, list):
        out = [s for s in (x.strip() for x in stop_in) if s]
        return out or None
    raise HTTPException(
        status_code=400, detail="stop must be a string or a list of strings"
    )

def _build_vllm_payload_for_chat(req: ChatCompletionRequest) -> Dict[str, Any]:
    # Beam search handling
    num_beams = int(req.num_beams or 1)
    use_beam_search = num_beams > 1
    n = int(req.n or 1)
    if use_beam_search and n > num_beams:
        raise HTTPException(
            status_code=400,
            detail=f"n ({n}) cannot exceed num_beams ({num_beams}) when using beam search.",
        )

    # Regex pattern from wordlist (if requested)
    guided_regex = get_cached_wordlist_regex(req.vocab_lang, req.vocab_n_words)

    # Temperature/top_p deterministic under beam search
    temperature = 0.0 if use_beam_search else (req.temperature if req.temperature is not None else 1.0)
    top_p = 1.0 if use_beam_search else (req.top_p if req.top_p is not None else 1.0)

    payload: Dict[str, Any] = {
        "model": req.model or MODEL_NAME,
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens or 100,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stop": _normalize_stop(req.stop),
        "presence_penalty": req.presence_penalty if req.presence_penalty is not None else 0.0,
        "frequency_penalty": req.frequency_penalty if req.frequency_penalty is not None else 0.0,
        "extra_body": {
            # Core: enable LMFE on vLLM server
            "guided_decoding_backend": GUIDED_DECODING_BACKEND,
            # Only include guided_regex if we actually built one
            **({"guided_regex": guided_regex} if guided_regex else {}),
            # Beam search extras for vLLM OpenAI server
            **({"use_beam_search": True, "best_of": num_beams} if use_beam_search else {}),
            # Optional early_stopping passthrough (vLLM accepts it in extra_body)
            **({"early_stopping": req.early_stopping} if req.early_stopping is not None else {}),
        },
    }
    return payload

def _build_vllm_payload_for_prompt(
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    vocab_lang: Optional[Literal["en","es"]],
    vocab_n_words: Optional[int],
    system_msg_en: str,
    system_msg_es: str,
) -> Dict[str, Any]:
    messages = []
    system_msg = system_msg_es if vocab_lang == "es" else system_msg_en
    if system_msg != "":
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(**m) for m in messages],
        max_tokens=max_new_tokens,
        num_beams=num_beams,
        vocab_lang=vocab_lang,
        vocab_n_words=vocab_n_words,
        temperature=1.0,
        top_p=1.0,
        n=1,
    )
    return _build_vllm_payload_for_chat(req)

# =========================
# vLLM server process management
# =========================
_vllm_proc: Optional[subprocess.Popen] = None
_http_client: Optional[httpx.AsyncClient] = None

async def _wait_for_vllm_ready(timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{VLLM_BASE_URL}/v1/models")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError("Timed out waiting for vLLM OpenAI server to become ready.")

def _spawn_vllm_server():
    global _vllm_proc
    if _vllm_proc is not None and _vllm_proc.poll() is None:
        return  # already running

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--guided-decoding-backend", GUIDED_DECODING_BACKEND,
    ]
    if TRUST_REMOTE_CODE:
        cmd += ["--trust-remote-code"]
    if DTYPE_STR:
        cmd += ["--dtype", DTYPE_STR]
    if MAX_MODEL_LEN > 0:
        cmd += ["--max-model-len", str(MAX_MODEL_LEN)]
    if HF_HOME:
        cmd += ["--download-dir", HF_HOME]
    # Helpful for UI testing across origins
    cmd += ["--cors-origins", "*"]

    env = os.environ.copy()
    if HF_TOKEN:
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    print("Starting vLLM OpenAI server:", " ".join(cmd))
    _vllm_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

@app.on_event("startup")
async def on_startup():
    global _http_client
    if START_VLLM_SERVER:
        _spawn_vllm_server()
        # Wait for readiness
        await _wait_for_vllm_ready()
        print(f"vLLM OpenAI server ready at {VLLM_BASE_URL}")
    _http_client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=120.0)

@app.on_event("shutdown")
async def on_shutdown():
    global _http_client, _vllm_proc
    if _http_client:
        await _http_client.aclose()
        _http_client = None
    if _vllm_proc and _vllm_proc.poll() is None:
        print("Stopping vLLM OpenAI server...")
        try:
            _vllm_proc.send_signal(signal.SIGTERM)
            try:
                _vllm_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                _vllm_proc.kill()
        except Exception:
            pass
        _vllm_proc = None

# =========================
# Proxy helpers
# =========================
async def _post_vllm(path: str, json: Dict[str, Any]) -> httpx.Response:
    if not _http_client:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")
    return await _http_client.post(path, json=json)

async def _get_vllm(path: str) -> httpx.Response:
    if not _http_client:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")
    return await _http_client.get(path)

# =========================
# Legacy constrained API
# =========================
@app.post("/generate")
async def generate(
    request: GenerateRequest, auth_ok: bool = Depends(verify_token)
) -> str:
    try:
        payload = _build_vllm_payload_for_prompt(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            num_beams=int(request.num_beams or 1),
            vocab_lang=request.vocab_lang,
            vocab_n_words=request.vocab_n_words,
            system_msg_en=DEFAULT_SYSTEM_PROMPT_EN,
            system_msg_es=DEFAULT_SYSTEM_PROMPT_ES,
        )
        r = await _post_vllm("/v1/chat/completions", payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        if data.get("choices"):
            return data["choices"][0]["message"]["content"]
        return ""
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Explicit Batch API
# =========================
@app.post("/generate_batch")
async def generate_batch(
    batch_request: BatchGenerateRequest, auth_ok: bool = Depends(verify_token)
):
    tasks = []
    for i, request in enumerate(batch_request.requests):
        payload = _build_vllm_payload_for_prompt(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            num_beams=int(request.num_beams or 1),
            vocab_lang=request.vocab_lang,
            vocab_n_words=request.vocab_n_words,
            system_msg_en=DEFAULT_SYSTEM_PROMPT_EN,
            system_msg_es=DEFAULT_SYSTEM_PROMPT_ES,
        )
        tasks.append(_post_vllm("/v1/chat/completions", payload))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs = []
    for r in results:
        if isinstance(r, Exception):
            outputs.append({"success": False, "error": str(r), "text": None})
        else:
            try:
                data = r.json()
                if r.status_code != 200:
                    outputs.append({"success": False, "error": r.text, "text": None})
                elif data.get("choices"):
                    outputs.append({"success": True, "error": None, "text": data["choices"][0]["message"]["content"]})
                else:
                    outputs.append({"success": True, "error": "Generation produced no output.", "text": ""})
            except Exception as e:
                outputs.append({"success": False, "error": str(e), "text": None})
    return {"results": outputs}

# =========================
# OpenAI-compatible API (proxied)
# =========================
@app.get("/v1/models")
async def list_models(auth_ok: bool = Depends(verify_token)):
    # Prefer to proxy vLLM for exact metadata
    try:
        r = await _get_vllm("/v1/models")
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback minimal response
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "owner"}],
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest, auth_ok: bool = Depends(verify_token)
):
    # 1) Ensure a system prompt if none supplied and env default exists
    has_system = any(m.role == "system" for m in req.messages)
    if not has_system:
        system_prompt = DEFAULT_SYSTEM_PROMPT_EN
        if system_prompt != "":
            req.messages = [ChatMessage(role="system", content=system_prompt)] + req.messages

    # 2) Build vLLM payload (adds guided_regex + beam search in extra_body)
    payload = _build_vllm_payload_for_chat(req)

    # 3) Proxy to vLLM OpenAI server
    r = await _post_vllm("/v1/chat/completions", payload)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()
