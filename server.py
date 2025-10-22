# --- Load .env as early as possible ---
from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
import uuid
import asyncio
from functools import cache
from inspect import signature
from typing import Literal, Optional, List, Union

from fastapi import Depends, FastAPI, HTTPException, Header, Query
from pydantic import BaseModel

from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput

from transformers import AutoTokenizer

# ---------------------------------------------------------
# Fallback GenerateRequest if not provided by common.py
# ---------------------------------------------------------
try:
    from common import GenerateRequest
except Exception:
    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 100
        repetition_penalty: Optional[float] = 1.0
        length_penalty: Optional[float] = 1.0
        num_beams: Optional[int] = 1
        vocab_lang: Optional[Literal["en", "es"]] = None
        vocab_n_words: Optional[int] = None

app = FastAPI()

# -----------------------
# Config via environment
# -----------------------
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

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

# -----------------------
# Model / tokenizer load (vLLM)
# -----------------------
print(f"Initializing vLLM Async Engine for {MODEL_NAME}...")
print(f"TP Size: {TENSOR_PARALLEL_SIZE}, Quant: {QUANTIZATION}, Dtype: {DTYPE_STR}, Trust Remote: {TRUST_REMOTE_CODE}")

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
    raise

# Tokenizer (sync) for chat templates and tokenizer_data
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE, token=HF_TOKEN
    )
except Exception as e:
    print(f"CRITICAL: Failed to load AutoTokenizer synchronously ({e}). Cannot initialize format enforcer.")
    raise

tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

# -----------------------
# Regex-constrained vocab (Spanish prebuild only per your requirement)
# -----------------------
def _words_path(lang: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, f"{lang}.txt")

@cache
def get_cached_regex_parser(lang: Literal["en", "es"], n_words: int) -> Optional[RegexParser]:
    print(f"building regex parser for {lang} ({n_words} words)")
    path = _words_path(lang)
    if not os.path.exists(path):
        print(f"Warning: Dictionary file {lang}.txt not found at {path}.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as fin:
            words = [word.strip().lower() for word in fin if word.strip()]
    except Exception as e:
        print(f"Error reading dictionary file {lang}.txt: {e}")
        return None

    words = words[:n_words]
    if not words:
        print(f"Warning: No valid words found in {lang}.txt.")
        return None

    # Your original grammar (kept as-is):
    # - Case-insensitive first char
    # - punctuation/whitespace class includes newlines
    word_regexp = "|".join(
        "[" + w[0].lower() + w[0].upper() + "]" + w[1:] for w in words if w
    )
    word_regexp = "(" + word_regexp + ")"
    punct_regexp = "[-.,!?():;¿!¡\\s]+"
    flexible_grammar = f"({word_regexp}{punct_regexp})+"

    parser = RegexParser(flexible_grammar)
    return parser

if PREBUILD_PREFIX:
    print("prebuilding regex parsers (Spanish only)...")
    for n_words in PREBUILD_WORD_COUNTS:
        get_cached_regex_parser("es", n_words)
    print("done")

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
# Pydantic Models
# -----------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[List[str], str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    # Custom parameters
    vocab_lang: Optional[Literal["en", "es"]] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None

class BatchGenerateRequest(BaseModel):
    requests: List[GenerateRequest]

# -----------------------
# Inference Helpers
# -----------------------
def _create_sampling_params(request) -> SamplingParams:
    # Extract normalized parameters
    if isinstance(request, GenerateRequest):
        params = {
            "max_tokens": getattr(request, "max_new_tokens", 100),
            "temperature": 1.0,
            "top_p": 1.0,
            "repetition_penalty": (request.repetition_penalty or 1.0),
            "length_penalty": (request.length_penalty or 1.0),
            "num_beams": (request.num_beams or 1),
            "vocab_lang": request.vocab_lang,
            "vocab_n_words": request.vocab_n_words,
            "n": 1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": None,
        }
    elif isinstance(request, ChatCompletionRequest):
        params = {
            "max_tokens": request.max_tokens or 100,
            "temperature": 1.0 if request.temperature is None else request.temperature,
            "top_p": 1.0 if request.top_p is None else request.top_p,
            "repetition_penalty": request.repetition_penalty or 1.0,
            "length_penalty": request.length_penalty or 1.0,
            "num_beams": request.num_beams or 1,
            "vocab_lang": request.vocab_lang,
            "vocab_n_words": request.vocab_n_words,
            "stop": request.stop,
            "n": request.n or 1,
            "presence_penalty": request.presence_penalty or 0.0,
            "frequency_penalty": request.frequency_penalty or 0.0,
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid request payload")

    # Normalize stop to list[str] or None
    stop = params.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif stop is not None and not isinstance(stop, list):
        raise HTTPException(status_code=400, detail="stop must be a string or a list of strings")

    # Build logits processors (constraints)
    logits_processors = None
    if params.get("vocab_lang") and params.get("vocab_n_words"):
        parser = get_cached_regex_parser(params["vocab_lang"], params["vocab_n_words"])
        if parser is None:
            raise HTTPException(status_code=400, detail=f"Dictionary file {params['vocab_lang']}.txt not found or empty.")
        logits_processors = [build_vllm_logits_processor(tokenizer_data, parser)]

    # Beam search handling
    use_beam_search = params["num_beams"] > 1
    best_of = params["n"]
    n = params["n"]

    if use_beam_search:
        # vLLM beam search path
        # Enforce beam-search friendly settings
        if params["temperature"] != 0.0:
            print(f"Warning: Beam search requires temperature=0. Overriding {params['temperature']} -> 0.0")
            params["temperature"] = 0.0
        if params["top_p"] != 1.0:
            print(f"Warning: Beam search requires top_p=1.0. Overriding {params['top_p']} -> 1.0")
            params["top_p"] = 1.0

        # In vLLM, best_of acts as beam width in beam search mode
        best_of = max(2, int(params["num_beams"]))
        # Typically return the single best hypothesis; you can later extend to top-n beams.
        n = 1

    # Assemble SamplingParams with conditional kwargs for version compatibility
    sp_sig = signature(SamplingParams.__init__)
    sp_kwargs = dict(
        max_tokens=params["max_tokens"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        best_of=best_of,
        stop=stop,
        n=n,
        logits_processors=logits_processors,
    )
    if "use_beam_search" in sp_sig.parameters:
        sp_kwargs["use_beam_search"] = use_beam_search
    if "length_penalty" in sp_sig.parameters and params.get("length_penalty") is not None:
        sp_kwargs["length_penalty"] = params["length_penalty"]
    if "presence_penalty" in sp_sig.parameters:
        sp_kwargs["presence_penalty"] = params.get("presence_penalty", 0.0)
    if "frequency_penalty" in sp_sig.parameters:
        sp_kwargs["frequency_penalty"] = params.get("frequency_penalty", 0.0)

    try:
        sampling_params = SamplingParams(**sp_kwargs)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid sampling parameters: {e}")
    return sampling_params

async def run_inference(prompt_text: str, sampling_params: SamplingParams) -> RequestOutput:
    """Run async generation and return the final RequestOutput."""
    request_id = str(uuid.uuid4())
    agen = llm.generate(prompt_text, sampling_params, request_id)
    final_output: Optional[RequestOutput] = None
    async for ro in agen:
        final_output = ro
    return final_output

def _apply_chat_template_or_fallback(messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying chat template: {e}. Falling back to raw user prompt.")
        # Fallback: last user/assistant content; prefer last non-system message
        for m in reversed(messages):
            if m.get("role") in ("user", "assistant"):
                return m.get("content", "")
        # Final fallback to empty string
        return ""

# -----------------------
# Legacy constrained API
# -----------------------
@app.post("/generate")
async def generate(request: GenerateRequest, auth_ok: bool = Depends(verify_token)) -> str:
    system_msg = DEFAULT_SYSTEM_PROMPT_ES if request.vocab_lang == "es" else DEFAULT_SYSTEM_PROMPT_EN
    messages = []
    if system_msg != "":
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": request.prompt})

    prompt_text = _apply_chat_template_or_fallback(messages)
    sampling_params = _create_sampling_params(request)
    final_output = await run_inference(prompt_text, sampling_params)

    if final_output and final_output.outputs:
        return final_output.outputs[0].text
    return ""

# -----------------------
# Explicit Batch API
# -----------------------
@app.post("/generate_batch")
async def generate_batch(batch_request: BatchGenerateRequest, auth_ok: bool = Depends(verify_token)):
    tasks = []

    async def _create_failing_task(error_msg: str):
        raise Exception(error_msg)

    for i, request in enumerate(batch_request.requests):
        try:
            system_msg = DEFAULT_SYSTEM_PROMPT_ES if request.vocab_lang == "es" else DEFAULT_SYSTEM_PROMPT_EN
            messages = []
            if system_msg != "":
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": request.prompt})

            prompt_text = _apply_chat_template_or_fallback(messages)
            sampling_params = _create_sampling_params(request)
            tasks.append(run_inference(prompt_text, sampling_params))
        except Exception as e:
            error_detail = str(e.detail) if isinstance(e, HTTPException) else str(e)
            print(f"Error preparing request index {i}: {error_detail}")
            tasks.append(_create_failing_task(f"Preparation Error: {error_detail}"))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    outputs = []
    for final_output in results:
        if isinstance(final_output, Exception):
            outputs.append({"success": False, "error": str(final_output), "text": None})
        elif isinstance(final_output, RequestOutput) and final_output.outputs:
            outputs.append({"success": True, "error": None, "text": final_output.outputs[0].text})
        else:
            outputs.append({"success": True, "error": "Generation produced no output.", "text": ""})

    return {"results": outputs}

# -----------------------
# OpenAI-compatible API
# -----------------------
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
async def chat_completions(req: ChatCompletionRequest, auth_ok: bool = Depends(verify_token)):
    start_time = time.time()

    # Determine the system prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT_EN
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

    prompt_text = _apply_chat_template_or_fallback(messages)

    sampling_params = _create_sampling_params(req)
    final_output = await run_inference(prompt_text, sampling_params)
    if not final_output:
        raise HTTPException(status_code=500, detail="Generation failed")

    created = int(start_time)
    prompt_tokens = len(final_output.prompt_token_ids)
    choices = []
    for i, completion_output in enumerate(final_output.outputs):
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": completion_output.text},
            "finish_reason": completion_output.finish_reason or "stop",
        })
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
