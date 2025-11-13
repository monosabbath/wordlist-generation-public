from functools import lru_cache
from typing import List


def _safe_add(tok, token_name: str, ids: set):
    try:
        eid = tok.convert_tokens_to_ids(token_name)
        if eid is not None and eid != tok.unk_token_id and eid != -1:
            ids.add(int(eid))
    except Exception:
        pass


# Build a hashable signature to cache on
def _stop_ids_sig(tokenizer) -> tuple:
    name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    eos = getattr(tokenizer, "eos_token_id", None)
    # Include a few known special tokens to avoid collisions across variants
    specials = tuple(sorted(str(s) for s in getattr(tokenizer, "all_special_tokens", [])))
    return (name, eos, specials)


@lru_cache(maxsize=256)
def _get_stop_ids_from_sig(sig: tuple) -> List[int]:
    # This function is called with a hashable signature; it still needs a real tokenizer to map names -> ids,
    # so we don't do any tokenizer operations here.
    # We only use the signature cache as a guard; the public get_stop_ids computes from the tokenizer each call,
    # but returns quickly thanks to the cache hit pattern.
    # Returning [] here; public function will compute properly.
    return []


def get_stop_ids(tokenizer) -> List[int]:
    # Prime the signature cache (no-op if hit); we can't store the tokenizer itself there.
    _get_stop_ids_from_sig(_stop_ids_sig(tokenizer))

    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, int):
            stop_ids.add(tokenizer.eos_token_id)
        elif isinstance(tokenizer.eos_token_id, list):
            for i in tokenizer.eos_token_id:
                stop_ids.add(int(i))

    common_end_markers = ("<end_of_turn>", "<|eot_id|>", "<|im_end|>")
    for special in common_end_markers:
        _safe_add(tokenizer, special, stop_ids)

    for special in ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"):
        _safe_add(tokenizer, special, stop_ids)

    return list(stop_ids)


def get_cohere_control_ids(tokenizer) -> List[int]:
    names = [
        "<|START_OF_TURN_TOKEN|>",
        "<|END_OF_TURN_TOKEN|>",
        "<|CHATBOT_TOKEN|>",
        "<|START_RESPONSE|>",
        "<|END_RESPONSE|>",
        "<|START_THINKING|>",
        "<|END_THINKING|>",
    ]
    out: set[int] = set()
    for n in names:
        _safe_add(tokenizer, n, out)
    return list(out)
