# --- Load environment ASAP ---
from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
import uuid
import asyncio
from typing import Literal, Optional, List

from fastapi import Depends, FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, ConfigDict

# lm-format-enforcer
try:
    from lmformatenforcer import RegexParser
except ImportError:
    # Fallback import path in some installations
    from lmformatenforcer.regexparser import RegexParser

# Use the vLLM integration for tokenizer data + logits processor
from lmformatenforcer.integrations.vllm import (
    build_vllm_logits_processor,
    build_vllm_token_enforcer_tokenizer_data,
)

# vLLM
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput

# HF tokenizer (only for chat template convenience)
from transformers import AutoTokenizer

# functools.cache fallback for older Python
try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

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
app = FastAPI()

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

# Optional: enable lm-format-enforcer analyzer
LOGITS_ANALYZE = os.getenv("LOGITS_ANALYZE", "false").lower() == "true"

# =========================
# Initialize vLLM engine
# =========================
print(f"Initializing vLLM Async Engine for {MODEL_NAME}...")
print(
    f"TP Size: {TENSOR_PARALLEL_SIZE}, Quant: {QUANTIZATION}, Dtype: {DTYPE_STR}, Trust Remote: {TRUST_REMOTE_CODE}"
)

engine_args_kwargs = {
    "model": MODEL_NAME,
    "dtype": DTYPE_STR,
    "quantization": QUANTIZATION,
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "trust_remote_code": TRUST_REMOTE_CODE,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
}
if MAX_MODEL_LEN > 0:
    engine_args_kwargs["max_model_len"] = MAX_MODEL_LEN
    print(f"Overriding Max Model Length to: {MAX_MODEL_LEN}")

try:
    engine_args = AsyncEngineArgs(**engine_args_kwargs)
    llm = AsyncLLMEngine.from_engine_args(engine_args)
except Exception as e:
    print(f"CRITICAL: Failed to initialize vLLM engine: {e}")
    print(
        f"Ensure you have {TENSOR_PARALLEL_SIZE} GPUs available and the configuration is supported."
    )
    raise

# =========================
# Tokenizer (synchronous, for chat template)
# =========================
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE, token=HF_TOKEN
    )
except Exception as e:
    print(
        f"CRITICAL: Failed to load AutoTokenizer synchronously ({e}). Cannot initialize format enforcer."
    )
    raise

# Build tokenizer_data for vLLM logits processor once (use vLLM integration)
try:
    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm)
except Exception as e:
    print(f"CRITICAL: Failed to build vLLM tokenizer data for enforcer: {e}")
    raise

# =========================
# Wordlist-based RegexParser
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


@cache
def get_cached_regex_parser(
    lang: Literal["en", "es"], n_words: int
) -> Optional[RegexParser]:
    """
    Build a case-flexible wordlist regex parser with optional trailing punctuation.
    Grammar (prefix-friendly):
      (SEP)? (WORD) (SEP WORD)* (SEP)?
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
    # Allow typical punctuation and spaces between words (includes quotes)
    sep_re = r"[-.,!?():;¿¡\"'“”‘’\s]+"
    # Allow optional leading SEP and optional trailing SEP (no ^/$ anchors)
    pattern = f"(?:{sep_re})?(?:{word_alt})(?:{sep_re}(?:{word_alt}))*(?:{sep_re})?"
    return RegexParser(pattern)


if PREBUILD_PREFIX:
    print("Prebuilding regex parsers...")
    for lang in ("es",):  # Prebuild Spanish as requested
        for n_words in PREBUILD_WORD_COUNTS:
            get_cached_regex_parser(lang, n_words)
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
        if len(parts) >= 2 and parts[0].lower() == "bearer":
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
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[ChatMessage]
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
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None  # if supported by vLLM


class BatchGenerateRequest(BaseModel):
    requests: List[GenerateRequest]


# =========================
# Inference helpers
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


def _build_logits_processors(vocab_lang, vocab_n_words):
    if not (vocab_lang and vocab_n_words):
        return None, None
    parser = get_cached_regex_parser(vocab_lang, vocab_n_words)
    if parser is None:
        raise HTTPException(
            status_code=400,
            detail=f"Dictionary file {vocab_lang}.txt not found or empty.",
        )
    logits_proc = build_vllm_logits_processor(
        tokenizer_data, parser, analyze=LOGITS_ANALYZE
    )
    return parser, [logits_proc]


def _sp_has_field(field: str) -> bool:
    # SamplingParams is a Pydantic v2 BaseModel; use model_fields instead of inspect.signature
    try:
        return field in getattr(SamplingParams, "model_fields", {})
    except Exception:
        return False


def _create_sampling_params(request) -> SamplingParams:
    # Map generic fields
    if isinstance(request, GenerateRequest):
        params = {
            "max_tokens": request.max_new_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "repetition_penalty": request.repetition_penalty or 1.0,
            "length_penalty": request.length_penalty
            if request.length_penalty is not None
            else 1.0,
            "num_beams": int(request.num_beams or 1),
            "vocab_lang": request.vocab_lang,
            "vocab_n_words": request.vocab_n_words,
            "n": 1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": None,
            "early_stopping": None,
        }
    elif isinstance(request, ChatCompletionRequest):
        params = {
            "max_tokens": request.max_tokens or 100,
            "temperature": 0.0
            if (request.num_beams and request.num_beams > 1)
            else (request.temperature if request.temperature is not None else 1.0),
            "top_p": 1.0
            if (request.num_beams and request.num_beams > 1)
            else (request.top_p if request.top_p is not None else 1.0),
            "repetition_penalty": request.repetition_penalty or 1.0,
            "length_penalty": request.length_penalty
            if request.length_penalty is not None
            else 1.0,
            "num_beams": int(request.num_beams or 1),
            "vocab_lang": request.vocab_lang,
            "vocab_n_words": request.vocab_n_words,
            "stop": _normalize_stop(request.stop),
            "n": int(request.n or 1),
            "presence_penalty": request.presence_penalty
            if request.presence_penalty is not None
            else 0.0,
            "frequency_penalty": request.frequency_penalty
            if request.frequency_penalty is not None
            else 0.0,
            "early_stopping": request.early_stopping,
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid request type")

    # Build logits processors if constraints requested
    parser, logits_processors = _build_logits_processors(
        params["vocab_lang"], params["vocab_n_words"]
    )

    # Beam search handling
    use_beam_search = params["num_beams"] > 1
    if use_beam_search:
        beam_width = params["num_beams"]
        if params["n"] > beam_width:
            raise HTTPException(
                status_code=400,
                detail=f"n ({params['n']}) cannot exceed num_beams ({beam_width}) when using beam search.",
            )
        best_of = beam_width
        # deterministic settings with beam search
        params["temperature"] = 0.0
        params["top_p"] = 1.0
    else:
        best_of = max(1, params["n"])

    # Assemble SamplingParams kwargs
    sp_kwargs = dict(
        max_tokens=params["max_tokens"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        best_of=best_of,
        stop=params["stop"],
        n=params["n"],
    )

    # Optional parameters (guarded by Pydantic model_fields)
    if _sp_has_field("use_beam_search"):
        sp_kwargs["use_beam_search"] = use_beam_search
    elif use_beam_search:
        raise HTTPException(
            status_code=400,
            detail="This vLLM build does not support use_beam_search in SamplingParams. Upgrade vLLM.",
        )

    if _sp_has_field("length_penalty") and params["length_penalty"] is not None:
        sp_kwargs["length_penalty"] = params["length_penalty"]
    if _sp_has_field("early_stopping") and params.get("early_stopping") is not None:
        sp_kwargs["early_stopping"] = params["early_stopping"]
    if _sp_has_field("presence_penalty"):
        sp_kwargs["presence_penalty"] = params.get("presence_penalty", 0.0)
    if _sp_has_field("frequency_penalty"):
        sp_kwargs["frequency_penalty"] = params.get("frequency_penalty", 0.0)

    try:
        sampling_params = SamplingParams(**sp_kwargs)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid sampling parameters: {e}")

    # Set logits processors AFTER construction
    if logits_processors:
        try:
            sampling_params.logits_processors = logits_processors
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="This vLLM build does not support logits_processors on SamplingParams. Upgrade vLLM to use format enforcement.",
            )

    return sampling_params


async def run_inference(
    prompt_text: str, sampling_params: SamplingParams
) -> RequestOutput:
    """
    Run async generation and collect the final RequestOutput from the async iterator.
    """
    request_id = str(uuid.uuid4())
    agen = llm.generate(prompt_text, sampling_params, request_id)
    final_output: Optional[RequestOutput] = None
    async for ro in agen:
        final_output = ro
    return final_output


# =========================
# Explicit Batch API
# =========================
@app.post("/generate_batch")
async def generate_batch(
    batch_request: BatchGenerateRequest, auth_ok: bool = Depends(verify_token)
):
    tasks = []

    async def _create_failing_task(error_msg: str):
        raise Exception(error_msg)

    for i, request in enumerate(batch_request.requests):
        try:
            system_msg = (
                DEFAULT_SYSTEM_PROMPT_ES
                if request.vocab_lang == "es"
                else DEFAULT_SYSTEM_PROMPT_EN
            )
            messages = []
            if system_msg != "":
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": request.prompt})
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception as e:
                print(
                    f"Error applying chat template (idx={i}): {e}. Falling back to raw prompt."
                )
                prompt_text = request.prompt

            sampling_params = _create_sampling_params(request)
            tasks.append(run_inference(prompt_text, sampling_params))

        except Exception as e:
            detail = str(e.detail) if isinstance(e, HTTPException) else str(e)
            print(f"Error preparing request index {i}: {detail}")
            tasks.append(_create_failing_task(f"Preparation Error: {detail}"))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs = []
    for final_output in results:
        if isinstance(final_output, Exception):
            outputs.append(
                {"success": False, "error": str(final_output), "text": None}
            )
        elif isinstance(final_output, RequestOutput) and final_output.outputs:
            outputs.append(
                {
                    "success": True,
                    "error": None,
                    "text": final_output.outputs[0].text,
                }
            )
        else:
            outputs.append(
                {
                    "success": True,
                    "error": "Generation produced no output.",
                    "text": "",
                }
            )
    return {"results": outputs}


# =========================
# OpenAI-compatible API
# =========================
@app.get("/v1/models")
def list_models(auth_ok: bool = Depends(verify_token)):
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "owner"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest, auth_ok: bool = Depends(verify_token)
):
    start_time = time.time()

    # 1) Determine effective system prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT_EN
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
            break

    # 2) Build message list for template
    messages = []
    if system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    for msg in req.messages:
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    # 3) Apply chat template (fallback to last user message)
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as e:
        print(f"Error applying chat template: {e}. Messages: {messages}")
        # Fallback: last non-system message or empty
        prompt_text = ""
        for m in reversed(messages):
            if m["role"] != "system":
                prompt_text = m["content"]
                break
        if not prompt_text:
            raise HTTPException(
                status_code=400,
                detail=f"Error applying chat template and no usable fallback prompt: {e}",
            )

    # 4) Build SamplingParams
    sampling_params = _create_sampling_params(req)

    # 5) Run async inference
    final_output = await run_inference(prompt_text, sampling_params)
    if not final_output:
        raise HTTPException(status_code=500, detail="Generation failed")

    # 6) Format response
    created = int(start_time)
    prompt_tokens = len(final_output.prompt_token_ids)
    choices = []
    for i, completion_output in enumerate(final_output.outputs):
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": completion_output.text},
                "finish_reason": completion_output.finish_reason or "stop",
            }
        )
    completion_tokens = sum(len(o.token_ids) for o in final_output.outputs)
    resp = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": req.model or MODEL_NAME,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
