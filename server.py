from functools import cache
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import GenerateRequest

app = FastAPI()

model_name = "Qwen/Qwen3-4B-Instruct-2507"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


@cache
def build_regexp_prefix_fn(lang: Literal["en", "es"], n_words: int):
    print(f"building prefix function for {lang} ({n_words} words)")
    with open(lang + ".txt") as fin:
        words = [word.strip().lower() for word in fin]
    words = words[:n_words]
    word_regexp = "|".join(
        "[" + w[0].lower() + w[0].upper() + "]" + w[1:] for w in words
    )
    word_regexp = "(" + word_regexp + ")"
    punct_regexp = "[- ,.!?():;¿!]+"
    parser = RegexParser(f"({word_regexp}{punct_regexp})+")
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return prefix_fn


print("prebuilding prefix functions...")
for lang in ("en", "es"):
    for n_words in (500, 1000, 5000):
        build_regexp_prefix_fn(lang, n_words)
print("done")


@app.post("/generate")
def generate(request: GenerateRequest, token: str) -> str:
    if token != "my-secret-token-structured-generation":
        raise HTTPException(status_code=403)

    system_msg = "You are a helpful assistant that uses only simple words."
    if request.vocab_lang == "es":
        system_msg = "Eres un asistente útil que utiliza sólo palabras sencillas."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": request.prompt},
    ]
    texts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(texts, return_tensors="pt").to(model.device)
    input_len = len(input_ids[0])
    prefix_fn = build_regexp_prefix_fn(request.vocab_lang, request.vocab_n_words)

    generated_ids = model.generate(
        **input_ids,
        max_new_tokens=request.max_new_tokens,
        prefix_allowed_tokens_fn=prefix_fn,
        num_beams=request.num_beams,
        do_sample=True,
        repetition_penalty=request.repetition_penalty,
        length_penalty=request.length_penalty,
    )
    result = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    return result
