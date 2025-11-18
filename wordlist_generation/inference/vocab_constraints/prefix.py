import os
import re
import functools
from typing import Any, Dict, Optional, Tuple

from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from wordlist_generation.inference.vocab_constraints.trie import (
    TrieNode,
    build_trie_with_ranks,
    trie_to_regex,
    normalize_word,
)
from wordlist_generation.inference.vocab_constraints.tokens import get_stop_ids, get_cohere_control_ids


def _safe_lang_name(lang: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", lang))


@functools.lru_cache(maxsize=16)
def get_or_build_trie(lang: str, wordlist_dir: str) -> Optional[Dict[str, Any]]:
    if not _safe_lang_name(lang):
        return None
    
    filename = os.path.join(wordlist_dir, f"{lang}.txt")
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, encoding="utf-8") as fin:
            words = [normalize_word(w) for w in fin if w.strip()]
    except Exception:
        return None
    if not words:
        return None
    trie = build_trie_with_ranks(words)
    return {"trie": trie}


def build_word_regex_for_n(lang: str, n_words: int, wordlist_dir: str) -> Optional[str]:
    data = get_or_build_trie(lang, wordlist_dir)
    if data is None:
        return None
    trie: TrieNode = data["trie"]
    if trie.min_rank > n_words:
        return None
    return trie_to_regex(trie, nlimit=n_words)


@functools.lru_cache(maxsize=64)
def _get_cached_prefix_fn(
    tokenizer_name_or_path: str,
    tokenizer_eos_id: str,
    lang: str,
    n_words: int,
    wordlist_dir: str,
    allow_cohere_controls: bool,
    stop_ids_tuple: Tuple[int, ...],
    cohere_ids_tuple: Tuple[int, ...]
):
    # Re-fetch trie/regex (cached)
    word_regex = build_word_regex_for_n(lang, n_words, wordlist_dir)
    if not word_regex:
        return None

    # allow words separated by punctuation/whitespace; flexible boundaries
    punct_regex = r'[.,!?¿¡…\s]+'
    flexible_grammar = fr'(?:{punct_regex})?(?:{word_regex}{punct_regex})*'

    parser = RegexParser(flexible_grammar)
    
    # NOTE: We cannot reconstruct the tokenizer here easily, so we rely on the caller 
    # providing a function that takes (tokenizer, parser).
    # However, build_transformers_prefix_allowed_tokens_fn NEEDS the tokenizer instance.
    # The cache key uses strings/tuples to be hashable.
    # We will return the parser and sets here, and the wrapper builds the fn.
    # Actually, we can't cache the *function* itself easily if it closes over the tokenizer object 
    # which is not hashable.
    # Instead, we will cache the PARSER and regex generation which is the expensive part.
    
    return parser


def build_regexp_prefix_fn(
    tokenizer,
    lang: str,
    n_words: int,
    wordlist_dir: str,
    allow_cohere_controls: bool = False,
):
    if get_or_build_trie(lang, wordlist_dir) is None:
        return None

    # Cache keys for hashability
    name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    eos = str(getattr(tokenizer, "eos_token_id", "None"))
    stop_ids = get_stop_ids(tokenizer)
    cohere_ids = get_cohere_control_ids(tokenizer) if allow_cohere_controls else []
    
    parser = _get_cached_prefix_fn(
        name,
        eos,
        lang,
        n_words,
        wordlist_dir,
        allow_cohere_controls,
        tuple(stop_ids),
        tuple(cohere_ids)
    )
    
    if parser is None:
        return None

    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    
    stop_ids_set = set(stop_ids)
    cohere_ids_set = set(cohere_ids)

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        return list(allowed | stop_ids_set | cohere_ids_set)

    return wrapped_prefix_fn
