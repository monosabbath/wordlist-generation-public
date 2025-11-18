from typing import Any, Dict, Optional
import os
import torch


def _local_rank_device() -> Optional[torch.device]:
    try:
        if torch.cuda.is_available():
            lr = os.getenv("LOCAL_RANK")
            if lr is not None and lr.isdigit():
                return torch.device(f"cuda:{int(lr)}")
            # fallback to current
            return torch.device(torch.cuda.current_device())
    except Exception:
        pass
    return None


def _model_primary_device(model) -> torch.device:
    """
    Best-effort way to pick the correct "entry" device for inputs.
    FSDP2 note: prefer LOCAL_RANK cuda device when available.
    - Prefer LOCAL_RANK CUDA device if present (FSDP2 multi-process).
    - Else input embedding weight device if available
    - Else first parameter device
    - Else CUDA:0 if available; else CPU
    """
    # Prefer local-rank device when running under torchrun (FSDP2)
    lr_dev = _local_rank_device()
    if lr_dev is not None:
        return lr_dev

    try:
        emb = getattr(model, "get_input_embeddings", None)
        if callable(emb):
            emb_mod = emb()
            if emb_mod is not None:
                p = next(emb_mod.parameters(), None)
                if p is not None:
                    return p.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_inputs_to_correct_device(inputs: Dict[str, torch.Tensor], model):
    device = _model_primary_device(model)
    out: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


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
    # Sampling parameters
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
):
    # Defaults for sampling params when not provided
    t = float(temperature if temperature is not None else 1.0)
    tp = float(top_p if top_p is not None else 1.0)
    tk = int(top_k if top_k is not None else 50)
    rp = float(repetition_penalty if repetition_penalty is not None else 1.0)

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        # Enable sampling together with beam search
        do_sample=True,
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
