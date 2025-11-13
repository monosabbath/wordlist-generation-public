import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from .tokens import get_stop_ids, get_cohere_control_ids

# -----------------------------------------------------------------------------
# This module builds constrained-vocabulary prefix functions for LMFE.
#
# Key changes vs. previous version:
# - Uses a simplified grammar that avoids duplicating the large "word" alternation:
#     flexible_grammar = (?:punct)?(?:word punct)*
#   This:
#     * Allows optional leading punctuation/whitespace
#     * Requires any word to be followed by punctuation/whitespace (so outputs do not end in a word)
#     * Allows empty and punctuation-only outputs (as discussed and acceptable)
#     * Uses the "word" alternation exactly once to reduce RegexParser compile time
#
# - Adds bucketization: splits the word set into multiple smaller regexes/parsers and
#   unions their allowed tokens at runtime. This drastically cuts RegexParser build time
#   for thousands of words at the cost of a small per-step overhead.
#
# - EOS handling remains unchanged: we still union stop_ids so EOS is allowed at any time,
#   per the current setup and your preference.
#
# Tunables (env overrides):
#   LMFE_PUNCT_CLASS            - character class for separators (default: [.,;:!?¿¡…\\s]+)
#   LMFE_BUCKET_MAX_WORDS       - max words per compiled regex bucket (default: 200)
#   LMFE_USE_BUCKETS_MIN_WORDS  - if n_words >= this, use bucketization (default: 400)
# -----------------------------------------------------------------------------


def normalizeword(w: str) -> str:
    return unicodedata.normalize("NFC", w.strip()).lower()


class _TrieNode:
    __slots__ = ("children", "end", "min_rank")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.end: bool = False
        # min rank of any word passing through this node
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
    """
    Build a regex for all suffixes under 'node' up to rank limit.
    Includes the empty suffix "" if node.end and node.min_rank <= nlimit.
    """
    alts: List[str] = []
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


# Process-local cache keyed by hashable signature instead of tokenizer object
_PREFIX_FN_CACHE: Dict[Tuple[str, str, int, str, bool], Any] = {}


def _prefix_cache_key(tokenizer, lang: str, n_words: int, wordlist_dir: str, allow_cohere_controls: bool) -> Tuple[str, str, int, str, bool]:
    name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    # Include eos id to differentiate tokenizers with same name but different specials
    eos = str(getattr(tokenizer, "eos_token_id", "None"))
    return (name, eos, n_words, f"{wordlist_dir}:{lang}", allow_cohere_controls)


# -------------------------- Bucketization helpers ----------------------------

# Tunables via environment
_PUNCT_CLASS = os.getenv("LMFE_PUNCT_CLASS", r"[.,;:!?¿¡…\s]+")
_BUCKET_MAX_WORDS_DEFAULT = int(os.getenv("LMFE_BUCKET_MAX_WORDS", "200"))
_USE_BUCKETS_MIN_WORDS = int(os.getenv("LMFE_USE_BUCKETS_MIN_WORDS", "400"))


def _count_words_under(node: _TrieNode, nlimit: int, cache: Dict[int, int]) -> int:
    """
    Count how many words (ends) are below this node respecting the rank limit.
    Uses node.min_rank as a quick cutoff and caches by id(node).
    Note: Follows the same semantics as trieto_regex for including ends.
    """
    node_id = id(node)
    if node_id in cache:
        return cache[node_id]

    if node.min_rank > nlimit:
        cache[node_id] = 0
        return 0

    total = 1 if (node.end and node.min_rank <= nlimit) else 0
    for child in node.children.values():
        if child.min_rank <= nlimit:
            total += _count_words_under(child, nlimit, cache)
    cache[node_id] = total
    return total


class _Entry:
    """
    Represents a group of words under a prefix.
    mode = "full": includes the word equal to prefix (if exists) and all descendants
    mode = "end_only": includes only the word equal to prefix
    """
    __slots__ = ("prefix", "node", "mode")

    def __init__(self, prefix: str, node: _TrieNode, mode: str = "full"):
        self.prefix = prefix
        self.node = node
        self.mode = mode  # "full" | "end_only"


def _split_to_entries(node: _TrieNode, prefix: str, nlimit: int, max_words: int, counts: Dict[int, int]) -> List[_Entry]:
    """
    Split a subtree into entries such that each entry covers <= max_words words.
    If a subtree exceeds max_words, we split into:
      - an 'end_only' entry for the exact prefix (if that word is allowed)
      - recursively split entries for each child
    """
    total = _count_words_under(node, nlimit, counts)
    if total <= max_words:
        return [_Entry(prefix, node, "full")]

    entries: List[_Entry] = []
    if node.end and node.min_rank <= nlimit:
        entries.append(_Entry(prefix, node, "end_only"))

    for ch, child in sorted(node.children.items()):
        if child.min_rank > nlimit:
            continue
        entries.extend(_split_to_entries(child, prefix + ch, nlimit, max_words, counts))
    return entries


def _entries_for_top_n(trie: _TrieNode, nlimit: int, max_words: int) -> List[_Entry]:
    """
    Produce small entries for all words up to nlimit by splitting from the root.
    We do not create an entry at root; we start at first characters.
    """
    counts: Dict[int, int] = {}
    entries: List[_Entry] = []
    for ch, child in sorted(trie.children.items()):
        if child.min_rank > nlimit:
            continue
        entries.extend(_split_to_entries(child, ch, nlimit, max_words, counts))
    return entries


def _entry_word_count(entry: _Entry, nlimit: int, counts: Dict[int, int]) -> int:
    if entry.mode == "end_only":
        return 1
    return _count_words_under(entry.node, nlimit, counts)


def _pack_entries_into_buckets(entries: List[_Entry], nlimit: int, max_words: int) -> List[List[_Entry]]:
    """
    Greedy packing: assemble entries into buckets so that the total words per bucket
    stays <= max_words. This keeps each compiled regex relatively small.
    """
    buckets: List[List[_Entry]] = []
    counts: Dict[int, int] = {}

    current: List[_Entry] = []
    current_count = 0

    for e in entries:
        c = _entry_word_count(e, nlimit, counts)
        # If a single entry is already > max_words (pathological), still put it alone.
        if current and current_count + c > max_words:
            buckets.append(current)
            current = []
            current_count = 0
        current.append(e)
        current_count += c

    if current:
        buckets.append(current)
    return buckets


def _entry_to_regex(entry: _Entry, nlimit: int) -> str:
    """
    Convert a single entry to a regex fragment representing its words.
    - end_only: just the prefix
    - full: prefix + trieto_regex(node)
    """
    if entry.mode == "end_only":
        return escapefor_regex(entry.prefix)
    sub = trieto_regex(entry.node, nlimit)
    return escapefor_regex(entry.prefix) + sub


def _bucket_word_regex(bucket: List[_Entry], nlimit: int) -> str:
    """
    Build the alternation of all entries in a bucket.
    """
    alts: List[str] = []
    for e in bucket:
        frag = _entry_to_regex(e, nlimit)
        if frag == "":
            # Should not occur for non-root entries; skip defensively
            continue
        alts.append(frag)
    if not alts:
        return ""
    if len(alts) == 1:
        return alts[0]
    return "(?:" + "|".join(alts) + ")"


def _build_bucketed_prefix_fn(
    tokenizer,
    trie: _TrieNode,
    n_words: int,
    punct_regex: str,
    allow_cohere_controls: bool,
    bucket_max_words: int,
):
    """
    Build multiple small RegexParsers and union their allowed tokens each step.
    Grammar per bucket (single copy of the alternation):
        (?:punct)? (?:word punct)*
    """
    # 1) Create small entries
    entries = _entries_for_top_n(trie, n_words, bucket_max_words)
    if not entries:
        return None

    # 2) Pack entries into buckets capped by bucket_max_words words per compiled regex
    buckets = _pack_entries_into_buckets(entries, n_words, bucket_max_words)
    if not buckets:
        return None

    # 3) Compile one parser per bucket using the simplified grammar
    base_prefix_fns = []
    for bucket in buckets:
        word_regex = _bucket_word_regex(bucket, n_words)
        if not word_regex:
            continue
        # Only one copy of the big alternation; punctuation is a + class
        flexible_grammar = fr'(?:{punct_regex})?(?:{word_regex}{punct_regex})*'
        parser = RegexParser(flexible_grammar)
        base_prefix_fns.append(build_transformers_prefix_allowed_tokens_fn(tokenizer, parser))

    if not base_prefix_fns:
        return None

    stop_ids = set(get_stop_ids(tokenizer))
    cohere_ids = set(get_cohere_control_ids(tokenizer)) if allow_cohere_controls else set()

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed: set[int] = set()
        for fn in base_prefix_fns:
            allowed.update(fn(batch_id, input_ids))
        # Maintain existing behavior: unconditionally allow EOS/stop ids
        return list(allowed | stop_ids | cohere_ids)

    return wrapped_prefix_fn


# --------------------------- Public build function ---------------------------

def build_regexp_prefix_fn(
    tokenizer,
    lang: str,
    n_words: int,
    wordlist_dir: str,
    allow_cohere_controls: bool = False,
):
    """
    Build a prefix_allowed_tokens_fn that constrains outputs to:
      - Optional leading punctuation/whitespace
      - Zero or more (word + punctuation/whitespace) pairs
    So outputs never end in a bare word; they end in punctuation/whitespace or can be empty.
    EOS handling remains unchanged: we union stop_ids so EOS is allowed anytime.

    For performance:
      - If n_words >= LMFE_USE_BUCKETS_MIN_WORDS (default 400), use bucketization.
      - Otherwise, build a single-parser grammar (still without duplicating "word").
    """
    key = _prefix_cache_key(tokenizer, lang, n_words, wordlist_dir, allow_cohere_controls)
    if key in _PREFIX_FN_CACHE:
        return _PREFIX_FN_CACHE[key]

    data = getor_build_trie(lang, wordlist_dir)
    if data is None:
        return None
    trie: _TrieNode = data["trie"]
    if trie.min_rank > n_words:
        return None

    punct_regex = _PUNCT_CLASS
    bucket_max_words = _BUCKET_MAX_WORDS_DEFAULT

    # Use bucketization for larger n to avoid multi-minute RegexParser builds
    if n_words >= _USE_BUCKETS_MIN_WORDS:
        wrapped = _build_bucketed_prefix_fn(
            tokenizer=tokenizer,
            trie=trie,
            n_words=n_words,
            punct_regex=punct_regex,
            allow_cohere_controls=allow_cohere_controls,
            bucket_max_words=bucket_max_words,
        )
        if wrapped is not None:
            _PREFIX_FN_CACHE[key] = wrapped
            return wrapped
        # Fallback to single regex if bucketed failed (should be rare)

    # Single-parser path for small n_words
    word_regex = buildword_regex_for_n(lang, n_words, wordlist_dir)
    if not word_regex:
        return None

    # Simplified grammar with a single copy of the big alternation:
    #   (?:punct)? (?:word punct)*
    flexible_grammar = fr'(?:{punct_regex})?(?:{word_regex}{punct_regex})*'

    parser = RegexParser(flexible_grammar)
    base_prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    stop_ids = set(get_stop_ids(tokenizer))
    cohere_ids = set(get_cohere_control_ids(tokenizer)) if allow_cohere_controls else set()

    def wrapped_prefix_fn(batch_id, input_ids):
        allowed = set(base_prefix_fn(batch_id, input_ids))
        # Maintain existing behavior: unconditionally allow EOS/stop and optional cohere controls
        return list(allowed | stop_ids | cohere_ids)

    _PREFIX_FN_CACHE[key] = wrapped_prefix_fn
    return wrapped_prefix_fn
