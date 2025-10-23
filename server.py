# --- Load environment ASAP ---
from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
import uuid
import asyncio
import subprocess
import signal
import sys
from contextlib import asynccontextmanager
from typing import Literal, Optional, List, Any, Dict

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
        repetition_penalty: Optional[float] = 1.0
        length_penalty: Optional[float] = 1.0
        num_beams: Optional[int] = 1
        vocab_lang: Optional[Literal["en", "es"]] = None
        vocab_n_words: Optional[int] = None

# =========================
# FastAPI app
# =========================
# We will start the vLLM OpenAI server as a background subprocess during lifespan.
# All requests will be proxied to that server to use LMFE via guided_regex.
# This preserves your existing simple launch commands.
vllm_process: Optional[subprocess.Popen] = None
http_client: Optional[httpx.AsyncClient] = None

# App will be created at the end after defining lifespan
app: FastAPI

# =========================
# Config via environment
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

DTYPE_STR = os.getenv("TORCH_DTYPE", "auto").lower()
QUANTIZATION = os.getenv("QUANTIZATION", None)
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", 0.90))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", 0))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"

SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my-secret-token-structured-generation")
DEFAULT_SYSTEM_PROMPT_EN = os.getenv("DEFAULT_SYSTEM_PROMPT_EN", "")
DEFAULT_SYSTEM_PROMPT_ES = os.getenv("DEFAULT_SYSTEM_PROMPT_ES", "")

PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(
    int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "500,1000,5000").split(",")
)

# Optional HF token for gated models/tokenizers
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# vLLM OpenAI server runtime
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8011"))
LAUNCH_VLLM_SERVER = os.getenv("LAUNCH_VLLM_SERVER", "true").lower() == "true"
VLLM_LOG_LEVEL = os.getenv("VLLM_LOG_LEVEL", "info")

# =========================
# Wordlist-based regex builder (no LMFE objects here)
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

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

@cache
def get_cached_regex_pattern(
    lang: Literal["en", "es"], n_words: int
) -> Optional[str]:
    """
    Build a case-flexible wordlist regex with optional leading/trailing punctuation and separators.
    Grammar (prefix-friendly):
      (SEP)? (WORD) (SEP WORD)* (SEP)?
    where:
      WORD matches any word from the list with only the initial letter case-insensitive.
      SEP  matches punctuation/whitespace between words.
    Note: ^ and $ anchors are avoided for prefix-friendly decoding.
    """
    try:
        words = _load_words(lang, n_words)
    except FileNotFoundError as e:
        print(str(e))
        return None
    if not words:
        print(f"Warning: {lang}.txt is empty or contains no valid words.")
        return None

    alts = []
    for w in words:
        if not w:
            continue
        first = w[0]
        rest = w[1:]
        if first.isalpha():
            esc_rest = re.escape(rest)
            lc, uc = first.lower(), first.upper()
            alts.append(f"(?:[{lc}{uc}]{esc_rest})")
        else:
            alts.append(f"(?:{re.escape(w)})")
    word_alt = "|".join(alts)
    sep_re = r"[-.,!?():;¿¡\"'“”‘’\s]+"
    pattern = f"(?:{sep_re})?(?:{word_alt})(?:{sep_re}(?:{word_alt}))*(?:{sep_re})?"
    return pattern

if PREBUILD_PREFIX:
    print("Prebuilding regex patterns...")
    for lang in ("es",):  # only Spanish prebuild as requested
        for n_words in PREBUILD_WORD_COUNTS:
            get_cached_regex_pattern(lang, n_words)
    print("Done prebuilding.")

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
# Pydantic Models (OpenAI-compatible)
# =========================
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    # Allow extra keys to pass through if clients send them
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[list[str] | str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stream: Optional[bool] = False
    # Custom parameters
    vocab_lang: Optional[Literal["en", "es"]] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    # vLLM OpenAI server accepts "extra_body" for sampling params and guided decoding
    extra_body: Optional[dict] = None

class BatchGenerateRequest(BaseModel):
    requests: List[GenerateRequest]

# =========================
# Utility helpers
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
    raise HTTPException(status_code=400, detail="stop must be a string or a list of strings")

def _add_default_system_prompt(messages: list[ChatMessage], vocab_lang: Optional[str]) -> list[dict]:
    has_system = any(m.role == "system" for m in messages)
    # Choose default prompt based on vocab_lang if provided
    default_system = DEFAULT_SYSTEM_PROMPT_ES if vocab_lang == "es" else DEFAULT_SYSTEM_PROMPT_EN
    out: list[dict] = []
    if not has_system and default_system != "":
        out.append({"role": "system", "content": default_system})
    for m in messages:
        out.append({"role": m.role, "content": m.content})
    return out

def _build_extra_body_for_request(
    req: ChatCompletionRequest
) -> Dict[str, Any]:
    extra = dict(req.extra_body or {})

    # Beam search mapping: vLLM requires use_beam_search=True and best_of=beam_width; n <= best_of
    if req.num_beams and req.num_beams > 1:
        beam_width = int(req.num_beams)
        n_val = int(req.n or 1)
        if n_val > beam_width:
            raise HTTPException(
                status_code=400,
                detail=f"n ({n_val}) cannot exceed num_beams ({beam_width}) when using beam search."
            )
        extra["use_beam_search"] = True
        extra["best_of"] = beam_width
        # Encourage deterministic decoding under beam search:
        # We'll also set temperature/top_p in the top-level payload accordingly.

    # Penalties and others (vLLM OpenAI server forwards these into SamplingParams)
    if req.repetition_penalty is not None:
        extra["repetition_penalty"] = req.repetition_penalty
    if req.length_penalty is not None:
        extra["length_penalty"] = req.length_penalty
    if req.early_stopping is not None:
        extra["early_stopping"] = req.early_stopping
    if req.presence_penalty is not None:
        extra["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        extra["frequency_penalty"] = req.frequency_penalty

    # Guided decoding via regex if both params provided
    if req.vocab_lang and req.vocab_n_words:
        pattern = get_cached_regex_pattern(req.vocab_lang, req.vocab_n_words)
        if pattern is None:
            raise HTTPException(
                status_code=400,
                detail=f"Dictionary file {req.vocab_lang}.txt not found or empty."
            )
        # Only set guided_regex if not already provided by caller
        if "guided_regex" not in extra:
            extra["guided_regex"] = pattern
        # Ensure LMFE backend is selected
        if "guided_decoding_backend" not in extra:
            extra["guided_decoding_backend"] = "lm-format-enforcer"

    return extra

# =========================
# vLLM OpenAI server management
# =========================
def _build_vllm_server_cmd() -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        # NOTE: newer vLLM uses --uvicorn-log-level (not --log-level)
        "--uvicorn-log-level",
        VLLM_LOG_LEVEL,
        # We remove the deprecated server-wide flag and rely on per-request
        # guided_decoding_backend="lm-format-enforcer" that we inject in extra_body.
        # "--guided-decoding-backend", "lm-format-enforcer",
    ]
    if TENSOR_PARALLEL_SIZE and TENSOR_PARALLEL_SIZE > 1:
        cmd += ["--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE)]
    if DTYPE_STR:
        cmd += ["--dtype", DTYPE_STR]
    if QUANTIZATION:
        cmd += ["--quantization", QUANTIZATION]
    if GPU_MEMORY_UTILIZATION is not None:
        cmd += ["--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION)]
    if MAX_MODEL_LEN and MAX_MODEL_LEN > 0:
        cmd += ["--max-model-len", str(MAX_MODEL_LEN)]
    if TRUST_REMOTE_CODE:
        cmd += ["--trust-remote-code"]
    return cmd

async def _wait_for_vllm_ready(timeout_s: float = 7200.0) -> None:
    global http_client
    start = time.monotonic()
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models"
    # Create a short-lived client if not yet created
    local_client = http_client or httpx.AsyncClient(timeout=10.0)
    while True:
        try:
            resp = await local_client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data:
                    return
        except Exception:
            pass
        if time.monotonic() - start > timeout_s:
            raise RuntimeError("Timed out waiting for vLLM OpenAI server to become ready.")
        await asyncio.sleep(0.5)
    # no return — loop exits either on return above or exception due to timeout

@asynccontextmanager
async def lifespan(app_: FastAPI):
    global vllm_process, http_client
    # Start vLLM OpenAI server as a background subprocess (if enabled)
    env = dict(os.environ)
    if LAUNCH_VLLM_SERVER:
        cmd = _build_vllm_server_cmd()
        print(f"Launching vLLM OpenAI server: {' '.join(cmd)}")
        vllm_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        try:
            # Wait for vLLM server readiness
            await _wait_for_vllm_ready()
            print("vLLM OpenAI server is ready.")
        except Exception as e:
            print(f"CRITICAL: vLLM OpenAI server did not become ready: {e}")
            # If it cannot start, stop the process and raise
            if vllm_process and vllm_process.poll() is None:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
                    else:
                        vllm_process.terminate()
                except Exception:
                    pass
            raise

    # Create a shared HTTP client for proxying requests
    http_client = httpx.AsyncClient(timeout=60.0)

    try:
        yield
    finally:
        # Close HTTP client
        if http_client:
            try:
                await http_client.aclose()
            except Exception:
                pass
        # Stop vLLM server
        if vllm_process and vllm_process.poll() is None:
            print("Shutting down vLLM OpenAI server...")
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
                else:
                    vllm_process.terminate()
                try:
                    vllm_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print("vLLM server did not exit in time; killing.")
                    vllm_process.kill()
            except Exception as e:
                print(f"Error while shutting down vLLM server: {e}")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# =========================
# HTTP proxy helpers
# =========================
async def _vllm_chat_completions_proxy(payload: dict) -> dict:
    """
    Sends the given OpenAI Chat Completions payload to the local vLLM OpenAI server
    and returns its JSON response.
    """
    global http_client
    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized.")
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"
    try:
        resp = await http_client.post(url, json=payload)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach vLLM server: {e}")
    if resp.status_code != 200:
        detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()

async def _vllm_list_models_proxy() -> dict:
    global http_client
    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized.")
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models"
    try:
        resp = await http_client.get(url)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach vLLM server: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()

# =========================
# Legacy constrained API
# =========================
@app.post("/generate")
async def generate(
    request: GenerateRequest, auth_ok: bool = Depends(verify_token)
) -> str:
    # Build messages (system prompt can be selected by vocab_lang)
    system_msg = DEFAULT_SYSTEM_PROMPT_ES if request.vocab_lang == "es" else DEFAULT_SYSTEM_PROMPT_EN
    messages = []
    if system_msg != "":
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": request.prompt})

    # Build extra_body for guided regex + penalties + beam
    fake_req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
        max_tokens=request.max_new_tokens,
        temperature=0.0 if (request.num_beams and request.num_beams > 1) else 1.0,
        top_p=1.0,
        n=1,
        stop=None,
        vocab_lang=request.vocab_lang,
        vocab_n_words=request.vocab_n_words,
        num_beams=int(request.num_beams or 1),
        repetition_penalty=request.repetition_penalty,
        length_penalty=request.length_penalty,
    )
    extra_body = _build_extra_body_for_request(fake_req)

    # Prepare payload for vLLM OpenAI server
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": request.max_new_tokens,
        "temperature": 0.0 if (request.num_beams and request.num_beams > 1) else 1.0,
        "top_p": 1.0 if (request.num_beams and request.num_beams > 1) else 1.0,
        "n": 1,
        "extra_body": extra_body,
    }

    data = await _vllm_chat_completions_proxy(payload)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""

# =========================
# Explicit Batch API
# =========================
@app.post("/generate_batch")
async def generate_batch(
    batch_request: BatchGenerateRequest, auth_ok: bool = Depends(verify_token)
):
    async def _one(req: GenerateRequest):
        try:
            system_msg = DEFAULT_SYSTEM_PROMPT_ES if req.vocab_lang == "es" else DEFAULT_SYSTEM_PROMPT_EN
            messages = []
            if system_msg != "":
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": req.prompt})

            fake_req = ChatCompletionRequest(
                model=MODEL_NAME,
                messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
                max_tokens=req.max_new_tokens,
                temperature=0.0 if (req.num_beams and req.num_beams > 1) else 1.0,
                top_p=1.0,
                n=1,
                stop=None,
                vocab_lang=req.vocab_lang,
                vocab_n_words=req.vocab_n_words,
                num_beams=int(req.num_beams or 1),
                repetition_penalty=req.repetition_penalty,
                length_penalty=req.length_penalty,
            )
            extra_body = _build_extra_body_for_request(fake_req)

            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": req.max_new_tokens,
                "temperature": 0.0 if (req.num_beams and req.num_beams > 1) else 1.0,
                "top_p": 1.0 if (req.num_beams and req.num_beams > 1) else 1.0,
                "n": 1,
                "extra_body": extra_body,
            }
            data = await _vllm_chat_completions_proxy(payload)
            text = data["choices"][0]["message"]["content"]
            return {"success": True, "error": None, "text": text}
        except HTTPException as he:
            return {"success": False, "error": str(he.detail), "text": None}
        except Exception as e:
            return {"success": False, "error": str(e), "text": None}

    tasks = [asyncio.create_task(_one(r)) for r in batch_request.requests]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return {"results": results}

# =========================
# OpenAI-compatible API (proxied to vLLM server with guided decoding)
# =========================
@app.get("/v1/models")
async def list_models(auth_ok: bool = Depends(verify_token)):
    # Pass-through to the vLLM server
    return await _vllm_list_models_proxy()

@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest, auth_ok: bool = Depends(verify_token)
):
    # Ensure messages include default system prompt if none was provided
    messages_out = _add_default_system_prompt(req.messages, req.vocab_lang)

    # Merge extra_body with guided decoding/beam/penalties logic
    extra_body = _build_extra_body_for_request(req)

    # Beam search implies deterministic sampling
    use_beam = bool(req.num_beams and req.num_beams > 1)
    temperature = 0.0 if use_beam else (req.temperature if req.temperature is not None else 1.0)
    top_p = 1.0 if use_beam else (req.top_p if req.top_p is not None else 1.0)

    # n cannot exceed best_of when beam search is enabled (already enforced in _build_extra_body_for_request)
    n_val = int(req.n or 1)

    # Prepare payload to vLLM OpenAI server
    payload = {
        "model": req.model or MODEL_NAME,
        "messages": messages_out,
        "max_tokens": req.max_tokens or 100,
        "temperature": temperature,
        "top_p": top_p,
        "n": n_val,
        "stop": _normalize_stop(req.stop),
        # Pass extra params into SamplingParams via extra_body
        "extra_body": extra_body,
        # We disable streaming passthrough in this proxy for simplicity.
        # If a client sets stream=True, you can either reject or convert to non-stream here.
        "stream": False,
    }

    data = await _vllm_chat_completions_proxy(payload)
    # Pass through vLLM response unchanged for maximum OpenAI compatibility
    return data
