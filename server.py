import os
import time
import uuid
from functools import cache
from typing import Literal, Optional

from dotenv import load_dotenv
import torch
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

# Optional: only needed if you plan to use 8/4-bit quantization
try:
    from transformers import BitsAndBytesConfig  # noqa: F401
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

from common import GenerateRequest

# Load .env as early as possible
load_dotenv()

app = FastAPI()

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

# -----------------------
# Model / tokenizer load
# -----------------------
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
    model_init_kwargs["torch_dtype"] = TORCH_DTYPE

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    **model_init_kwargs,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------
# Regex-constrained vocab
# -----------------------
@cache
def build_regexp_prefix_fn(lang: Literal["en", "es"], n_words: int):
    print(f"building prefix function for {lang} ({n_words} words)")
    with open(lang + ".txt") as fin:
        words = [word.strip().lower() for word in fin]
    words = words[:n_words]
    # Case-insensitive first char
    word_regexp = "|".join(
        "[" + w[0].lower() + w[0].upper() + "]" + w[1:] for w in words if w
    )
    word_regexp = "(" + word_regexp + ")"

    # 1. Update to include whitespace (including newlines) using \\s
    punct_regexp = "[-.,!?():;¿!¡\\s]+"

    # 2. Update grammar to allow flexible interleaving and natural EOS
    # Grammar: (Word OR Punctuation)+
    flexible_grammar = f"({word_regexp}|{punct_regexp})+"
    parser = RegexParser(flexible_grammar)

    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return prefix_fn


if PREBUILD_PREFIX:
    print("prebuilding prefix functions...")
    for lang in ("es",):
        for n_words in PREBUILD_WORD_COUNTS:
            build_regexp_prefix_fn(lang, n_words)
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
        # Expect "Bearer <token>"
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
    system_msg = (
        DEFAULT_SYSTEM_PROMPT_ES
        if request.vocab_lang == "es"
        else DEFAULT_SYSTEM_PROMPT_EN
    )

    messages = []
    if system_msg != "":
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": request.prompt})

    texts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(texts, return_tensors="pt").to(model.device)
    input_len = input_ids["input_ids"].shape[1]
    prefix_fn = build_regexp_prefix_fn(request.vocab_lang, request.vocab_n_words)

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
# OpenAI-compatible API (optional but useful)
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
    
    # Move these from extra to direct parameters
    vocab_lang: Optional[Literal["en", "es"]] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None


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
    # Merge system prompt(s)
    system_prompt = DEFAULT_SYSTEM_PROMPT_EN
    # If any system message provided by client, prefer the first
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
            break

    # Build messages for the model
    messages = []
    if system_prompt != "":
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

    # Access parameters directly from req instead of req.extra
    if req.vocab_lang and req.vocab_n_words:
        prefix_fn = build_regexp_prefix_fn(req.vocab_lang, req.vocab_n_words)
        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
    if req.num_beams:
        gen_kwargs["num_beams"] = req.num_beams
    if req.repetition_penalty:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty
    if req.length_penalty:
        gen_kwargs["length_penalty"] = req.length_penalty

    outputs = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Token usage accounting (approx)
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
