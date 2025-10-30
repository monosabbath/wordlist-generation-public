import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from .tokens import get_stop_ids, get_cohere_control_ids


def normalizeword(w: str) -> str:
    return unicodedata.normalize("NFC", w.strip()).lower()


class _TrieNode:
    __slots__ = ("children", "end", "min_rank")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.end: bool = False
        self.min_rank: int = 10**12


def buildtrie_with_ranks(words: List[str]) -> _TrieNode:
    root = _TrieNode()
    for rank, w in enumerate(words, start=1):
        node = root
        node.min_rank = min(node.min_rank, rank)
        for ch in w:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
            node.min_rank = min(node.min_rank, rank)
        node.end = True
    return root


def escapefor_regex(ch: str) -> str:
    return re.escape(ch)


def trieto_regex(node: _TrieNode, nlimit: int) -> str:
    alts = []
    for ch, child in sorted(node.children.items()):
        if child.min_rank > nlimit:
            continue
        sub = trieto_regex(child, nlimit)
        alts.append(escapefor_regex(ch) + sub)
    if node.end and node.min_rank <= nlimit:
        alts.append("")
    if not alts:
        return ""
    if len(alts) == 1:
        return alts[0]
    return "(?:" + "|".join(alts) + ")"


TRIECACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}  # key: (wordlist_dir, lang)


def _safe_lang_name(lang: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", lang))


def getor_build_trie(lang: str, wordlist_dir: str) -> Optional[Dict[str, Any]]:
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
            words = [normalizeword(w) for w in fin if w.strip()]
    except Exception:
        return None
    if not words:
        return None
    trie = buildtrie_with_ranks(words)
    TRIECACHE[key] = {"trie": trie}
    return TRIECACHE[key]


def buildword_regex_for_n(lang: str, n_words: int, wordlist_dir: str) -> Optional[str]:
    data = getor_build_trie(lang, wordlist_dir)
    if data is None:
        return None
    trie: _TrieNode = data["trie"]
    if trie.min_rank > n_words:
        return None
    return trieto_regex(trie, nlimit=n_words)


@lru_cache(maxsize=128)
def build_regexp_prefix_fn(
    tokenizer,
    lang: str,
    n_words: int,
    wordlist_dir: str,
    allow_cohere_controls: bool = False,
):
    if getor_build_trie(lang, wordlist_dir) is None:
        return None
    word_regex = buildword_regex_for_n(lang, n_words, wordlist_dir)
    if not word_regex:
        return None

    punct_regex = r'[.,!?¿¡…\s]+'
    flexible_grammar = fr'(?:{punct_regex})?(?:{word_regex})(?:{punct_regex}(?:{word_regex}))*(?:{punct_regex})?'

    parser = RegexParser(flexible_grammar)
    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    stop_ids = set(get_stop_ids(tokenizer))
    cohere_ids = set(get_cohere_control_ids(tokenizer)) if allow_cohere_controls else set()

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        return list(allowed | stop_ids | cohere_ids)

    return wrapped_prefix_fn
