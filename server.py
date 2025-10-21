# --- MODIFIED: Load .env file at the absolute beginning ---
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------

import os
import time
import uuid
import asyncio
from functools import cache
from typing import Literal, Optional, List

from fastapi import Depends, FastAPI, HTTPException, Header, Query
from lmformatenforcer import RegexParser

# Import vLLM Async Engine, SamplingParams, and integrations
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput

# --- MODIFIED IMPORTS START ---
# Import the generic builder from the transformers integration
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

# We still need the VLLM-specific logits processor builder
from lmformatenforcer.integrations.vllm import (
    build_vllm_logits_processor,
    # build_vllm_token_enforcer_tokenizer_data, # <-- REMOVED: Caused async conflict
)
# --- MODIFIED IMPORTS END ---

from pydantic import BaseModel

# We retain AutoTokenizer to ensure chat templates are applied correctly
from transformers import AutoTokenizer

# Assuming common.py contains GenerateRequest definition
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


# .env is now loaded at the top
app = FastAPI()

# -----------------------
# Config via environment
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

# vLLM specific configurations
# 'auto' is generally fine, but 'bfloat16' is recommended for A100/H100 compute.
DTYPE_STR = os.getenv("TORCH_DTYPE", "auto").lower()

# Quantization method (e.g., "fp8", "awq", "gptq", "bitsandbytes").
QUANTIZATION = os.getenv("QUANTIZATION", None)

# For multi-GPU setups (Tensor Parallelism)
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))

# --- NEW: Optimization parameters for VRAM management (Crucial for large models) ---
# Controls the fraction of GPU memory vLLM can use (default 0.90).
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", 0.90))
# Controls the maximum context length. 0 means vLLM uses the model's config default.
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", 0))
# Allows loading models that require custom code (like Kimi-K2). Defaults to False.
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"

# Auth token
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my-secret-token-structured-generation")

# Default system prompts
DEFAULT_SYSTEM_PROMPT_EN = os.getenv("DEFAULT_SYSTEM_PROMPT_EN", "")
DEFAULT_SYSTEM_PROMPT_ES = os.getenv("DEFAULT_SYSTEM_PROMPT_ES", "")

# Prebuild toggles
PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(
    int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "500,1000,5000").split(",")
)

# -----------------------
# Model / tokenizer load (vLLM)
# -----------------------

# Initialize vLLM Async Engine for high throughput
print(f"Initializing vLLM Async Engine for {MODEL_NAME}...")
print(f"TP Size: {TENSOR_PARALLEL_SIZE}, Quant: {QUANTIZATION}, Dtype: {DTYPE_STR}, Trust Remote: {TRUST_REMOTE_CODE}")

# --- BLOCK MODIFIED ---

# Build initialization arguments dynamically
engine_args_kwargs = {
    "model": MODEL_NAME,
    "dtype": DTYPE_STR,
    "quantization": QUANTIZATION,
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "trust_remote_code": TRUST_REMOTE_CODE,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
}

# Only add max_model_len if specified (i.e., > 0) to override the default
if MAX_MODEL_LEN > 0:
    engine_args_kwargs["max_model_len"] = MAX_MODEL_LEN
    print(f"Overriding Max Model Length to: {MAX_MODEL_LEN}")

try:
    # 1. Create AsyncEngineArgs with your parameters
    engine_args = AsyncEngineArgs(**engine_args_kwargs)
    
    # 2. Initialize the AsyncLLMEngine using .from_engine_args()
    llm = AsyncLLMEngine.from_engine_args(engine_args)
except Exception as e:
    print(f"CRITICAL: Failed to initialize vLLM engine: {e}")
    print(f"Ensure you have {TENSOR_PARALLEL_SIZE} GPUs available and the configuration is supported.")
    raise

# --- END MODIFIED BLOCK ---


# --- FIX START: Synchronous Tokenizer Initialization ---
# Load tokenizer using transformers for reliable chat template support
try:
    # Ensure trust_remote_code is used for the tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)
except Exception as e:
    # CRITICAL FIX: We must have a synchronous tokenizer for initialization.
    # The fallback llm.get_tokenizer() is async and cannot be used here synchronously.
    print(f"CRITICAL: Failed to load AutoTokenizer synchronously ({e}). Cannot initialize format enforcer.")
    raise

# Initialize LM Format Enforcer integration data (done once).
# MODIFIED: Use the generic builder and pass the synchronous tokenizer instance directly.
tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
# --- FIX END ---


# -----------------------
# Regex-constrained vocab
# -----------------------
@cache
def get_cached_regex_parser(lang: Literal["en", "es"], n_words: int) -> Optional[RegexParser]:
    """Builds and caches the RegexParser."""
    # We cache the Parser itself. The logits processor must be created fresh per request.
    print(f"building regex parser for {lang} ({n_words} words)")
    try:
        with open(lang + ".txt") as fin:
            words = [word.strip().lower() for word in fin]
    except FileNotFoundError:
         print(f"Warning: Dictionary file {lang}.txt not found.")
         return None

    words = words[:n_words]
    # Case-insensitive first char
    word_regexp = "|".join(
        "[" + w[0].lower() + w[0].upper() + "]" + w[1:] for w in words if w
    )
    word_regexp = "(" + word_regexp + ")"

    punct_regexp = "[-.,!?():;¿!¡\\s]+"

    # Grammar: (Word OR Punctuation)+
    flexible_grammar = f"({word_regexp}{punct_regexp})+"
    parser = RegexParser(flexible_grammar)

    return parser


if PREBUILD_PREFIX:
    print("prebuilding regex parsers...")
    # Ensure both languages are covered for compatibility with other models
    for lang in ("es", "en"):
        for n_words in PREBUILD_WORD_COUNTS:
            get_cached_regex_parser(lang, n_words)
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
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[list[str] | str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    # Custom parameters
    vocab_lang: Optional[Literal["en", "es"]] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None

# Model for Explicit Batching
class BatchGenerateRequest(BaseModel):
    """Accepts a list of individual generation requests to be processed concurrently."""
    requests: List[GenerateRequest]


# -----------------------
# Inference Helpers
# -----------------------

def _create_sampling_params(request) -> SamplingParams:
    """Helper to map API requests to vLLM SamplingParams and integrate the logits processor."""

    # Initialize defaults and extract parameters based on request type
    if isinstance(request, GenerateRequest):
        params = {
            "max_tokens": request.max_new_tokens,
            "temperature": 1.0, # Default temperature
            "top_p": 1.0,
            "repetition_penalty": request.repetition_penalty or 1.0,
            "length_penalty": request.length_penalty or 1.0,
            "num_beams": request.num_beams or 1,
            "vocab_lang": request.vocab_lang,
            "vocab_n_words": request.vocab_n_words,
            "n": 1,
            # Ensure defaults for penalties if not present
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
    elif isinstance(request, ChatCompletionRequest):
        params = {
            "max_tokens": request.max_tokens or 100,
            "temperature": request.temperature if request.temperature is not None else 1.0,
            "top_p": request.top_p if request.top_p is not None else 1.0,
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
        raise ValueError("Invalid request type")

    # Handle Logits Processor (Constrained Generation)
    logits_processors = []
    parser = None
    if params.get("vocab_lang") and params.get("vocab_n_words"):
        parser = get_cached_regex_parser(params["vocab_lang"], params["vocab_n_words"])
        if parser:
            # The logits processor must be created fresh for every new request.
            # We use the VLLM builder, passing the tokenizer_data initialized synchronously.
            logits_processor = build_vllm_logits_processor(tokenizer_data, parser)
            logits_processors.append(logits_processor)

    # Handle Beam Search
    use_beam_search = params["num_beams"] > 1

    if use_beam_search:
        if parser:
            print("Note: Beam search requested with format enforcement.")

        # vLLM requires temperature=0 for beam search
        params["temperature"] = 0.0
        best_of = params["num_beams"]
    else:
        # best_of must be >= n
        best_of = params["n"]

    # Construct SamplingParams
    try:
        sampling_params = SamplingParams(
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            length_penalty=params["length_penalty"],
            use_beam_search=use_beam_search,
            best_of=best_of,
            stop=params.get("stop"),
            n=params["n"],
            logits_processors=logits_processors if logits_processors else None,
            presence_penalty=params.get("presence_penalty"),
            frequency_penalty=params.get("frequency_penalty"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid sampling parameters: {e}")

    return sampling_params


async def run_inference(prompt_text: str, sampling_params: SamplingParams) -> RequestOutput:
    """Runs the asynchronous vLLM generation and waits for the final output."""
    request_id = str(uuid.uuid4())
    
    # NOTE: `llm.generate` in modern vLLM is an async method that returns the
    # final RequestOutput directly, not a generator.
    # If you were using an older vLLM with streaming, the logic below would be different.
    final_output: RequestOutput = await llm.generate(prompt_text, sampling_params, request_id)

    return final_output

# -----------------------
# Legacy constrained API
# -----------------------
@app.post("/generate")
async def generate(request: GenerateRequest, auth_ok: bool = Depends(verify_token)) -> str:
    system_msg = (
        DEFAULT_SYSTEM_PROMPT_ES
        if request.vocab_lang == "es"
        else DEFAULT_SYSTEM_PROMPT_EN
    )

    messages = []
    if system_msg != "":
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": request.prompt})

    # Apply chat template
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        # Handle cases where the template fails (e.g., invalid roles for specific models)
        print(f"Error applying chat template: {e}. Messages: {messages}")
        raise HTTPException(status_code=400, detail=f"Error applying chat template: {e}")


    # Build Sampling Params using the helper
    sampling_params = _create_sampling_params(request)

    # Run async inference. vLLM automatically batches concurrent requests.
    final_output = await run_inference(prompt_text, sampling_params)

    if final_output and final_output.outputs:
        return final_output.outputs[0].text
    return ""

# -----------------------
# Explicit Batch API
# -----------------------

@app.post("/generate_batch")
async def generate_batch(batch_request: BatchGenerateRequest, auth_ok: bool = Depends(verify_token)):
    """
    Processes an explicit batch of generation requests concurrently within a single HTTP request.
    """
    tasks = []

    # Helper function to create a placeholder task that immediately fails.
    async def _create_failing_task(error_msg: str):
        raise Exception(error_msg)

    # 1. Preparation Phase
    for i, request in enumerate(batch_request.requests):
        try:
            # 1a. Prepare Prompt
            system_msg = (
                DEFAULT_SYSTEM_PROMPT_ES
                if request.vocab_lang == "es"
                else DEFAULT_SYSTEM_PROMPT_EN
            )

            messages = []
            if system_msg != "":
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": request.prompt})

            # Apply chat template
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 1b. Prepare SamplingParams
            # CRUCIAL: Must be called inside the loop for fresh logits_processor state.
            sampling_params = _create_sampling_params(request)

            # 1c. Create the async task
            tasks.append(run_inference(prompt_text, sampling_params))

        except Exception as e:
            # Handle errors during preparation
            error_detail = str(e.detail) if isinstance(e, HTTPException) else str(e)
            print(f"Error preparing request index {i}: {error_detail}")
            tasks.append(_create_failing_task(f"Preparation Error: {error_detail}"))

    # 2. Execution Phase
    # Run all tasks concurrently.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Response Formatting Phase
    outputs = []
    for final_output in results:
        if isinstance(final_output, Exception):
            # Handle inference or preparation failures
            outputs.append({"success": False, "error": str(final_output), "text": None})
        elif isinstance(final_output, RequestOutput) and final_output.outputs:
            # Handle success
            outputs.append({"success": True, "error": None, "text": final_output.outputs[0].text})
        else:
            # Handle cases where generation succeeded but produced no output
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

    # 1. Prepare Prompt
    # Determine the effective system prompt: use the first one in the request if present, otherwise use the default.
    system_prompt = DEFAULT_SYSTEM_PROMPT_EN
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
            break

    messages = []
    if system_prompt != "":
        messages.append({"role": "system", "content": system_prompt})
    
    # Add all non-system messages from the request
    for msg in req.messages:
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        # Handle cases where the template fails (e.g., invalid conversation structure)
        print(f"Error applying chat template: {e}. Messages: {messages}")
        raise HTTPException(status_code=400, detail=f"Error applying chat template: {e}")


    # 2. Build Sampling Params
    sampling_params = _create_sampling_params(req)

    # 3. Run async inference
    final_output = await run_inference(prompt_text, sampling_params)

    if not final_output:
        raise HTTPException(status_code=500, detail="Generation failed")

    # 4. Format the response
    created = int(start_time)
    # Token usage accounting (vLLM provides this directly)
    prompt_tokens = len(final_output.prompt_token_ids)

    choices = []
    # Loop through the generated outputs (handles the 'n' parameter)
    for i, completion_output in enumerate(final_output.outputs):
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": completion_output.text},
            "finish_reason": completion_output.finish_reason or "stop",
        })

    # Sum tokens if n > 1
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
