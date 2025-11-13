import os
import re
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

# Process-local caches
TRIECACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}  # key: (wordlist_dir, lang)
_PREFIX_FN_CACHE: Dict[Tuple[str, str, int, str, bool], Any] = {}


def _safe_lang_name(lang: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", lang))


def get_or_build_trie(lang: str, wordlist_dir: str) -> Optional[Dict[str, Any]]:
    if not _safe_lang_name(lang):
        return None
    key = (wordlist_dir, lang)
    if key in TRIECACHE:
        return TRIECACHE[key]
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
    TRIECACHE[key] = {"trie": trie}
    return TRIECACHE[key]


def build_word_regex_for_n(lang: str, n_words: int, wordlist_dir: str) -> Optional[str]:
    data = get_or_build_trie(lang, wordlist_dir)
    if data is None:
        return None
    trie: TrieNode = data["trie"]
    if trie.min_rank > n_words:
        return None
    return trie_to_regex(trie, nlimit=n_words)


def _prefix_cache_key(tokenizer, lang: str, n_words: int, wordlist_dir: str, allow_cohere_controls: bool) -> Tuple[str, str, int, str, bool]:
    name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    # Include eos id to differentiate tokenizers with same name but different specials
    eos = str(getattr(tokenizer, "eos_token_id", "None"))
    return (name, eos, n_words, f"{wordlist_dir}:{lang}", allow_cohere_controls)


def build_regexp_prefix_fn(
    tokenizer,
    lang: str,
    n_words: int,
    wordlist_dir: str,
    allow_cohere_controls: bool = False,
):
    key = _prefix_cache_key(tokenizer, lang, n_words, wordlist_dir, allow_cohere_controls)
    if key in _PREFIX_FN_CACHE:
        return _PREFIX_FN_CACHE[key]

    if get_or_build_trie(lang, wordlist_dir) is None:
        return None
    word_regex = build_word_regex_for_n(lang, n_words, wordlist_dir)
    if not word_regex:
        return None

    # allow words separated by punctuation/whitespace; flexible boundaries
    punct_regex = r'[.,!?¿¡…\s]+'
    flexible_grammar = fr'(?:{punct_regex})?(?:{word_regex}{punct_regex})*'

    parser = RegexParser(flexible_grammar)
    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    stop_ids = set(get_stop_ids(tokenizer))
    cohere_ids = set(get_cohere_control_ids(tokenizer)) if allow_cohere_controls else set()

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        return list(allowed | stop_ids | cohere_ids)

    _PREFIX_FN_CACHE[key] = wrapped_prefix_fn
    return wrapped_prefix_fn
