import time
import uuid
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, Depends, HTTPException, Request
from ..models.chat import ChatCompletionRequest
from ..core.auth import verify_token
from ..core.tokens import get_stop_ids
from ..core.generation import (
    tokenizer_encode_for_chat,
    strip_unused_model_inputs,
    move_inputs_to_correct_device,
    normalize_max_new_tokens,
    getgen_kwargs,
    truncate_inputs_to_max_length,
    build_template_kwargs_for_model,
    is_cohere_reasoning_model,
)
from ..core.prefix import build_regexp_prefix_fn

router = APIRouter(prefix="/v1", tags=["chat"])


@router.get("/models")
def list_models(request: Request, auth_ok: bool = Depends(verify_token)):
    settings = request.app.state.settings
    return {
        "object": "list",
        "data": [
            {
                "id": settings.MODEL_NAME,
                "object": "model",
                "owned_by": "owner",
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request, auth_ok: bool = Depends(verify_token)):
    settings = request.app.state.settings
    ms = request.app.state.model_service
    tokenizer = ms.tokenizer
    model = ms.model

    # Extract system prompt and rebuild messages (system first)
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

    # Build inputs: Cohere path tokenizes in template; others use text then tokenize
    if is_cohere_reasoning_model(settings.MODEL_NAME):
        inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = truncate_inputs_to_max_length(inputs, settings.MAX_INPUT_TOKENS)
        inputs = strip_unused_model_inputs(inputs)
        inputs = move_inputs_to_correct_device(inputs, model)
        input_len = inputs["input_ids"].shape[1]
    else:
        prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
        if "glm-4.6" in settings.MODEL_NAME.lower():
            prompt_text = prompt_text + "<think></think>"
        inputs = tokenizer_encode_for_chat(tokenizer, prompt_text, settings.PAD_TO_MULTIPLE_OF, settings.MAX_INPUT_TOKENS)
        inputs = strip_unused_model_inputs(inputs)
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
                detail=f"Constrained vocabulary configuration failed for language '{req.vocab_lang}'.",
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
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    gen_len = int(outputs[0].shape[0] - input_len)
    last_token = int(outputs[0][-1].item())
    finish_reason = "stop" if stop_ids and (last_token in set(stop_ids)) else ("length" if gen_len >= max_new_tokens else "stop")
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = gen_len
    created = int(time.time())

    resp = {
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp
