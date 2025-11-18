import time
import uuid
from typing import Dict, List

import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from wordlist_generation.api.routers.models import ChatCompletionRequest
from wordlist_generation.api.dependencies import verify_token
from wordlist_generation.inference.vocab_constraints.tokens import get_stop_ids
from wordlist_generation.inference.generation import (
    tokenizer_encode_for_chat,
    strip_unused_model_inputs,
    move_inputs_to_correct_device,
    normalize_max_new_tokens,
    getgen_kwargs,
    truncate_inputs_to_max_length,
    is_model_parallel,
)
from wordlist_generation.inference.templates import (
    build_template_kwargs_for_model,
    is_cohere_reasoning_model,
)
from wordlist_generation.inference.vocab_constraints.prefix import build_regexp_prefix_fn

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request, auth_ok: bool = Depends(verify_token)):
    settings = request.app.state.settings
    ms = request.app.state.model_service
    tokenizer = ms.tokenizer
    model = ms.model

    # Rebuild messages: system first if provided
    system_prompt = ""
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
            break
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in req.messages:
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    template_kwargs = build_template_kwargs_for_model(settings.MODEL_NAME)

    # Build inputs
    if is_cohere_reasoning_model(settings.MODEL_NAME):
        inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = truncate_inputs_to_max_length(inputs, settings.MAX_INPUT_TOKENS)
        inputs = strip_unused_model_inputs(inputs)
        # For model-parallel keep on CPU
        if not is_model_parallel(model):
            inputs = move_inputs_to_correct_device(inputs, model)
        input_len = inputs["input_ids"].shape[1]
    else:
        prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
        if "glm-4.6" in settings.MODEL_NAME.lower():
            prompt_text += "<think></think>"
        inputs = tokenizer_encode_for_chat(
            tokenizer,
            prompt_text,
            settings.PAD_TO_MULTIPLE_OF,
            settings.MAX_INPUT_TOKENS,
        )
        inputs = strip_unused_model_inputs(inputs)
        if not is_model_parallel(model):
            inputs = move_inputs_to_correct_device(inputs, model)
        input_len = inputs["input_ids"].shape[1]

    # Optional constrained vocab
    prefix_fn = None
    if req.vocab_lang and req.vocab_n_words:
        prefix_fn = build_regexp_prefix_fn(
            tokenizer=tokenizer,
            lang=req.vocab_lang,
            n_words=req.vocab_n_words,
            wordlist_dir=settings.WORDLIST_DIR,
            allow_cohere_controls=ms.is_cohere_reasoning_model,
        )
        if prefix_fn is None:
            raise HTTPException(
                status_code=500,
                detail=f"Constrained vocabulary configuration failed for '{req.vocab_lang}'.",
            )

    max_new_tokens = normalize_max_new_tokens(req.max_tokens, settings.ALLOWED_MAX_NEW_TOKENS)
    stop_ids = get_stop_ids(tokenizer)
    gen_kwargs = getgen_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
        num_beams=req.num_beams,
        length_penalty=req.length_penalty if req.length_penalty is not None else 1.0,
        prefix_fn=prefix_fn,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )

    with ms.gpu_gate:
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

    gen_len = int(outputs[0].shape[0] - input_len)
    last_token = int(outputs[0][-1].item())
    finish_reason = (
        "stop"
        if stop_ids and (last_token in set(stop_ids))
        else ("length" if gen_len >= max_new_tokens else "stop")
    )
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": req.model or settings.MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": gen_len,
            "total_tokens": int(inputs["input_ids"].shape[1]) + gen_len,
        },
    }
