import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import cache
from typing import Literal, Optional

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Header, Query
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from common import GenerateRequest

# Optional: only needed if you plan to use 8/4-bit quantization
try:
    from transformers import BitsAndBytesConfig  # noqa: F401
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

# Load .env as early as possible
load_dotenv()

# -----------------------
# Config via environment
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda")

# Quantization / dtype
DTYPE_STR = os.getenv("TORCH_DTYPE", "float32").lower()
_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
TORCH_DTYPE = _DTYPE_MAP.get(DTYPE_STR, torch.float32)

LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
if (LOAD_IN_8BIT or LOAD_IN_4BIT) and not HAVE_BNB:
    raise RuntimeError(
        "bitsandbytes not installed but LOAD_IN_8BIT/LOAD_IN_4BIT requested. "
        "Install it or disable quantization."
    )

# Auth token
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "my-secret-token-structured-generation")

# Default system prompts (can be empty strings)
DEFAULT_SYSTEM_PROMPT_EN = os.getenv("DEFAULT_SYSTEM_PROMPT_EN", "")
DEFAULT_SYSTEM_PROMPT_ES = os.getenv("DEFAULT_SYSTEM_PROMPT_ES", "")

# Prebuild toggles for constrained vocab
PREBUILD_PREFIX = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
PREBUILD_WORD_COUNTS = tuple(
    int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "500,1000,5000").split(",")
)

# ----------------------------------------------------
# Lifespan Manager for Startup and Shutdown Events
# ----------------------------------------------------

# Create a dictionary to hold the model and tokenizer
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code to run on startup ---
    print("INFO:     Application startup...")
    
    quant_config = None
    model_init_kwargs = {
        "device_map": DEVICE_MAP,
    }

    if LOAD_IN_8BIT or LOAD_IN_4BIT:
        # Lazy import only when needed
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_8bit=LOAD_IN_8BIT,
            load_in_4bit=LOAD_IN_4BIT,
            bnb_4bit_compute_dtype=torch.bfloat16 if TORCH_DTYPE == torch.bfloat16 else torch.float16,
        )
        model_init_kwargs["quantization_config"] = quant_config
    else:
        # The 'torch_dtype' argument is deprecated and will be removed in a future version of Transformers.
        # It is recommended to use the 'dtype' argument instead.
        model_init_kwargs["dtype"] = TORCH_DTYPE

    # Load the model and tokenizer into the ml_models dictionary
    ml_models["model"] = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_init_kwargs,
    )
    ml_models["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)

    if PREBUILD_PREFIX:
        print("prebuilding prefix functions...")
        for lang in ("es",):
            for n_words in PREBUILD_WORD_COUNTS:
                build_regexp_prefix_fn(lang, n_words, ml_models["tokenizer"])
        print("done")
    
    print("INFO:     Application startup complete.")
    
    # Let the application run
    yield
    
    # --- Code to run on shutdown ---
    print("INFO:     Application shutdown...")
    ml_models.clear()


# Initialize FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

# -----------------------
# Regex-constrained vocab
# -----------------------
@cache
def build_regexp_prefix_fn(lang: Literal["en", "es"], n_words: int, tokenizer):
    """Builds the prefix function for regex-constrained generation."""
    print(f"building prefix function for {lang} ({n_words} words)")
    with open(lang + ".txt") as fin:
        words = [word.strip().lower() for word in fin]
    words = words[:n_words]
    # Case-insensitive first char
    word_regexp = "|".join(
        "[" + w[0].lower() + w[0].upper() + "]" + w[1:] for w in words if w
    )
    word_regexp = "(" + word_regexp + ")"
    punct_regexp = "[-.,!?():;¿!¡\\s]+"
    flexible_grammar = f"({word_regexp}|{punct_regexp})+"
    parser = RegexParser(flexible_grammar)

    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return prefix_fn

# -----------------------
# Auth helper
# -----------------------
def verify_token(
    token: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
):
    """Verifies the authentication token."""
    supplied = token
    if not supplied and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            supplied = parts[1]
    if SECRET_TOKEN and supplied != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True

# -----------------------
# Legacy constrained API
# -----------------------
@app.post("/generate")
def generate(request: GenerateRequest, auth_ok: bool = Depends(verify_token)) -> str:
    """Generates text using the legacy API format."""
    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]

    system_msg = (
        DEFAULT_SYSTEM_PROMPT_ES
        if request.vocab_lang == "es"
        else DEFAULT_SYSTEM_PROMPT_EN
    )

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": request.prompt})

    texts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(texts, return_tensors="pt").to(model.device)
    input_len = input_ids["input_ids"].shape[1]
    prefix_fn = build_regexp_prefix_fn(request.vocab_lang, request.vocab_n_words, tokenizer)

    generated_ids = model.generate(
        **input_ids,
        max_new_tokens=request.max_new_tokens,
        prefix_allowed_tokens_fn=prefix_fn,
        num_beams=request.num_beams,
        do_sample=True,
        temperature=1.0,
        repetition_penalty=request.repetition_penalty,
        length_penalty=request.length_penalty,
    )
    result = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    return result

# -----------------------
# OpenAI-compatible API
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

@app.get("/v1/models")
def list_models(auth_ok: bool = Depends(verify_token)):
    """Lists the available models."""
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
    """Handles chat completion requests in an OpenAI-compatible format."""
    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]
    
    system_prompt = DEFAULT_SYSTEM_PROMPT_EN
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

    texts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(texts, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Generation params
    max_new_tokens = req.max_tokens if req.max_tokens is not None else 100
    do_sample = (req.temperature is None) or (req.temperature > 0)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p if req.top_p is not None else 1.0,
    )

    if req.vocab_lang and req.vocab_n_words:
        prefix_fn = build_regexp_prefix_fn(req.vocab_lang, req.vocab_n_words, tokenizer)
        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
    if req.num_beams:
        gen_kwargs["num_beams"] = req.num_beams
    if req.repetition_penalty:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty
    if req.length_penalty:
        gen_kwargs["length_penalty"] = req.length_penalty

    outputs = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Token usage accounting
    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = int(outputs[0].shape[0] - input_len)
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp
