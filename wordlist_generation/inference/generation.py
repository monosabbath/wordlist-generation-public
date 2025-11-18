from typing import Any, Dict, Optional
import torch


def move_inputs_to_correct_device(inputs: Dict[str, torch.Tensor], model):
    """
    Safe device handling:

    - If the model is sharded across >1 devices (multi-GPU with hf_device_map), leave inputs on CPU.
      Accelerate's dispatch_model expects CPU (or neutral) inputs and will route them efficiently.
      Forcing them onto a single GPU can trigger implicit full-weight or buffer copies -> OOM.

    - If the model is on exactly one device, move inputs there.
    """
    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            unique_devices = {str(v) for v in device_map.values()}
            # Multi-device: keep on CPU
            if len(unique_devices) > 1:
                return inputs  # no move
            # Single-device sharded map case (rare but possible)
            sole = list(unique_devices)[0]
            if "cuda" in sole or "cpu" in sole:
                dev = torch.device(sole)
                return {k: v.to(dev) for k, v in inputs.items()}
        # Fallback: model has parameters on a single device
        device = next(model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        return inputs


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
    if not allowed or len(allowed) == 0:
        cap = 512
        target = int(requested) if requested is not None else cap
        return min(target, cap)
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
        if input_ids.size(0) > max_length:
            input_ids = input_ids[-max_length:]
            if attn is not None:
                attn = attn[-max_length:]
    elif input_ids.dim() == 2:
        if input_ids.size(1) > max_length:
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
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
):
    t = float(temperature if temperature is not None else 1.0)
    tp = float(top_p if top_p is not None else 1.0)
    tk = int(top_k if top_k is not None else 50)
    rp = float(repetition_penalty if repetition_penalty is not None else 1.0)

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=True,  # (Leave as-is; you asked not to change beam search config)
        num_beams=int(num_beams or 10),
        length_penalty=float(length_penalty),
        temperature=t,
        top_p=tp,
        top_k=tk,
        repetition_penalty=rp,
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
