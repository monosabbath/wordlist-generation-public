import os
from dotenv import load_dotenv

# Load .env as early as possible
load_dotenv()

# Optional: install uvloop if available
try:
    import uvloop
    uvloop.install()
except Exception:
    pass

import re
import time
import uuid
from functools import cache
from typing import Literal, Optional

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

# Assuming 'common.py' contains GenerateRequest. If not, define a placeholder.
try:
    from common import GenerateRequest
except ImportError:
    class GenerateRequest(BaseModel):
        prompt: str
        vocab_lang: Literal["en", "es"] = "en"
        vocab_n_words: int = 1000
        max_new_tokens: int = 100
        num_beams: int = 1
        repetition_penalty: float = 1.0
        length_penalty: float = 1.0


app = FastAPI()


# -----------------------
# Config via environment
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"

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

# Total token cap (prompt + generated) to control KV cache/VRAM
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "1024"))

# Attention backend override
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "").strip()

# Hugging Face auth
HF_TOKEN = os.getenv("HF_TOKEN")


# -----------------------
# Model / tokenizer load
# -----------------------
quant_config = None
model_init_kwargs = {
    "device_map": DEVICE_MAP,
    "trust_remote_code": TRUST_REMOTE_CODE,
    "low_cpu_mem_usage": True,
}
if ATTN_IMPLEMENTATION:
    model_init_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

if LOAD_IN_8BIT or LOAD_IN_4BIT:
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_8bit=LOAD_IN_8BIT,
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=torch.bfloat16 if TORCH_DTYPE == torch.bfloat16 else torch.float16,
    )
    model_init_kwargs["quantization_config"] = quant_config
else:
    model_init_kwargs["torch_dtype"] = TORCH_DTYPE

print(f"Loading model '{MODEL_NAME}' (Trust Remote Code: {TRUST_REMOTE_CODE})...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    **model_init_kwargs,
)
model.eval()
torch.set_grad_enabled(False)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=TRUST_REMOTE_CODE,
)
# Ensure pad_token_id is set to avoid generate() warnings
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# -----------------------
# Helpers for stop tokens (Generalized for various models)
# -----------------------
def get_stop_ids(tok: AutoTokenizer) -> list[int]:
    stop_ids: list[int] = []
    # 1. Add the tokenizer's defined EOS token(s)
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, int):
            stop_ids.append(tok.eos_token_id)
        elif isinstance(tok.eos_token_id, list):
            stop_ids.extend(tok.eos_token_id)
    # 2. Add common special tokens, if they exist
    common_end_markers = (
        "<end_of_turn>",  # Gemma
        "<|eot_id|>",     # Llama 3.x
        "<|im_end|>",     # ChatML (Kimi, Qwen)
    )
    for special in common_end_markers:
        try:
            eid = tok.convert_tokens_to_ids(special)
            if eid is not None and eid != tok.unk_token_id:
                stop_ids.append(eid)
        except Exception:
            pass
    return list(set(stop_ids))  # unique IDs


# -----------------------
# Regex-constrained vocab
# -----------------------
@cache
def build_regexp_prefix_fn(lang: Literal["en", "es"], n_words: int):
    print(f"building prefix function for {lang} ({n_words} words)")
    filename = lang + ".txt"
    if not os.path.exists(filename):
        print(f"Warning: Language file not found: {filename}. Skipping build.")
        return None

    try:
        with open(filename, encoding="utf-8") as fin:
            words = [w.strip() for w in fin if w.strip()]
    except Exception as e:
        print(f"Error reading language file {filename}: {e}. Skipping build.")
        return None

    words = words[:n_words]
    if not words:
        print(f"Warning: Vocabulary file {filename} is empty or contains no valid words. Skipping build.")
        return None

    alts = []
    for w in words:
        first, rest = w[0], w[1:]
        if first.isalpha():
            esc_rest = re.escape(rest)
            lc, uc = first.lower(), first.upper()
            alts.append(f"(?:[{lc}{uc}]{esc_rest})")
        else:
            alts.append(f"(?:{re.escape(w)})")
    word_alt = "|".join(alts)

    # Include quotes; allow optional leading/trailing separators
    sep_re = r"[-.,!?():;¿¡\"'“”‘’\s]+"
    pattern = f"^(?:{sep_re})?(?:{word_alt})(?:{sep_re}(?:{word_alt}))*(?:{sep_re})?$"

    parser = RegexParser(pattern)
    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    stop_ids = set(get_stop_ids(tokenizer))

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        # Always allow EOS/stop tokens to prevent dead-ends at the final step.
        return list(allowed | stop_ids)

    return wrapped_prefix_fn

if PREBUILD_PREFIX:
    print("prebuilding prefix functions...")
    # Only Spanish prebuild as requested
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
        if len(parts) >= 2 and parts[0].lower() == "bearer":
            supplied = parts[1]
    if SECRET_TOKEN and supplied != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True


# -----------------------
# Helper for generation kwargs
# -----------------------
def _get_gen_kwargs(
    max_new_tokens,
    stop_ids,
    temperature=1.0,
    top_p=1.0,
    num_beams=None,
    repetition_penalty=None,
    length_penalty=None,
    prefix_fn=None,
):
    do_sample = (temperature is not None) and (temperature > 0)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if temperature is not None else 1.0,
        top_p=top_p if top_p is not None else 1.0,
        use_cache=True,
    )
    if prefix_fn:
        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
    if num_beams and num_beams > 1:
        gen_kwargs["num_beams"] = num_beams
    if repetition_penalty:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if length_penalty:
        gen_kwargs["length_penalty"] = length_penalty
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        else:
            gen_kwargs["pad_token_id"] = stop_ids[0] if isinstance(stop_ids, list) and stop_ids else None
    return gen_kwargs

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
    # Extra parameters
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


# Optional: simple token-based stopping on stop strings for OpenAI endpoint
from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: E402


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: list[list[int]]):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        for stop_seq in self.stop_token_ids:
            L = len(stop_seq)
            if L > 0 and len(seq) >= L and seq[-L:] == stop_seq:
                return True
        return False


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, auth_ok: bool = Depends(verify_token)):
    # Determine system prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT_EN
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

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Truncate prompt to cap tokens
    inputs = tokenizer(
        text, return_tensors="pt", max_length=MAX_TOTAL_TOKENS, truncation=True
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Calculate max_new_tokens to respect the total token cap
    max_new_from_request = req.max_tokens if req.max_tokens is not None else 100
    allowed_new_tokens = max(0, MAX_TOTAL_TOKENS - input_len)
    if allowed_new_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail="No room for generation: input reached MAX_TOTAL_TOKENS. Reduce prompt length or increase MAX_TOTAL_TOKENS.",
        )
    max_new_tokens = min(max_new_from_request, allowed_new_tokens)

    # Constrained vocab (optional)
    prefix_fn = None
    if req.vocab_lang and req.vocab_n_words:
        prefix_fn = build_regexp_prefix_fn(req.vocab_lang, req.vocab_n_words)
        if prefix_fn is None:
            raise HTTPException(
                status_code=500,
                detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'. Check server logs for missing/empty .txt files.",
            )

    stop_ids = get_stop_ids(tokenizer)

    # Optional stop strings (token-based)
    stopping_criteria = None
    if req.stop:
        stops = req.stop if isinstance(req.stop, list) else [req.stop]
        stop_token_ids = []
        for s in stops:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_token_ids.append(ids)
        if stop_token_ids:
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    gen_kwargs = _get_gen_kwargs(
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        temperature=req.temperature,
        top_p=req.top_p,
        num_beams=req.num_beams,
        repetition_penalty=req.repetition_penalty,
        length_penalty=req.length_penalty,
        prefix_fn=prefix_fn,
    )
    if stopping_criteria is not None:
        gen_kwargs["stopping_criteria"] = stopping_criteria

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    text_out = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

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
                "message": {"role": "assistant", "content": text_out},
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
