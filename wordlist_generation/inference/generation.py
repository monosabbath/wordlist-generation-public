from typing import Any, Dict, Optional
import torch


def move_inputs_to_correct_device(inputs: Dict[str, torch.Tensor], model):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {k: v.to(device) for k, v in inputs.items()}


def tokenizer_encode_for_chat(tokenizer, texts: Any, pad_to_multiple_of: int, max_input_tokens: int) -> Dict[str, torch.Tensor]:
    tok_kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "pad_to_multiple_of": pad_to_multiple_of,
        "max_length": max_input_tokens,
        "return_token_type_ids": False,
    }
    return tokenizer(texts, **tok_kwargs)


def strip_unused_model_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    inputs = dict(inputs)
    inputs.pop("token_type_ids", None)
    return inputs


def normalize_max_new_tokens(requested: Optional[int], allowed: Optional[tuple[int, ...]]) -> int:
    # Use the configured allowed set to determine the cap; fall back to 512 if not provided.
    if not allowed or len(allowed) == 0:
        cap = 512
        target = int(requested) if requested is not None else cap
        return min(target, cap)
    # allowed is sorted in Settings; use its max as cap
    cap = allowed[-1]
    target = int(requested) if requested is not None else cap
    target = min(target, cap)
    for a in allowed:
        if target <= a:
            return a
    return allowed[-1]


def truncate_inputs_to_max_length(inputs: Dict[str, torch.Tensor], max_length: int) -> Dict[str, torch.Tensor]:
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    if input_ids.dim() == 1:
        length = input_ids.size(0)
        if length > max_length:
            input_ids = input_ids[-max_length:]
            if attn is not None:
                attn = attn[-max_length:]
    elif input_ids.dim() == 2:
        bsz, seqlen = input_ids.size()
        if seqlen > max_length:
            input_ids = input_ids[:, -max_length:]
            if attn is not None:
                attn = attn[:, -max_length:]
    out = {"input_ids": input_ids}
    if attn is not None:
        out["attention_mask"] = attn
    for k, v in inputs.items():
        if k not in out:
            out[k] = v
    return out


def getgen_kwargs(
    tokenizer,
    max_new_tokens: int,
    stop_ids: Optional[list[int]],
    num_beams: Optional[int] = None,
    length_penalty: float = 1.0,
    prefix_fn=None,
):
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        # Consider lowering default beams to 1-4 for latency; 10 is very slow.
        num_beams=int(num_beams or 10),
        length_penalty=float(length_penalty),
    )
    if prefix_fn:
        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        else:
            gen_kwargs["pad_token_id"] = stop_ids[0] if isinstance(stop_ids, list) and stop_ids else None
    return gen_kwargs
