import re
import unicodedata
from typing import Dict, List


def normalize_word(w: str) -> str:
    return unicodedata.normalize("NFC", w.strip()).lower()


class TrieNode:
    __slots__ = ("children", "end", "min_rank")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.end: bool = False
        self.min_rank: int = 10**12


def build_trie_with_ranks(words: List[str]) -> TrieNode:
    root = TrieNode()
    for rank, w in enumerate(words, start=1):
        node = root
        node.min_rank = min(node.min_rank, rank)
        for ch in w:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.min_rank = min(node.min_rank, rank)
        node.end = True
    return root


def trie_to_regex(node: TrieNode, nlimit: int) -> str:
    # Group children by their recursive subpattern
    grouped: Dict[str, List[str]] = {}
    for ch, child in sorted(node.children.items()):
        if child.min_rank > nlimit:
            continue
        sub = trie_to_regex(child, nlimit)
        grouped.setdefault(sub, []).append(ch)

    has_end = node.end and node.min_rank <= nlimit

    # Special-case: pure terminal set (all children have empty suffix)
    if grouped and set(grouped.keys()) == {""}:
        chars = "".join(re.escape(c) for c in grouped[""])
        if has_end:
            # Set plus optional termination
            return f"[{chars}]?"
        else:
            return f"[{chars}]"

    alts: List[str] = []

    # Build alternatives with suffix factoring
    for sub, chars in grouped.items():
        if len(chars) == 1:
            # Single char: escape and concatenate directly
            alts.append(re.escape(chars[0]) + sub)
        else:
            # Multiple chars share the same suffix; build a character class
            cls = "".join(re.escape(c) for c in chars)
            if sub == "":
                alts.append(f"[{cls}]")
            else:
                # Concatenation of the class and the suffix pattern
                alts.append(f"[{cls}]{sub}")

    # Handle end-of-word (empty alternative)
    if has_end:
        alts.append("")

    if not alts:
        return ""
    if len(alts) == 1:
        return alts[0]
    return "(?:" + "|".join(alts) + ")"
