class TokenTrieNode:
    """A node in the token trie."""

    def __init__(self):
        self.children = dict()
        self.is_end = False
        self.starts_with_space = False


class TokenTrie:
    """A token trie containing valid words."""

    def __init__(self):
        self.root = TokenTrieNode()
        self.max_depth = 0

    def insert(self, token_ids, raw_text):
        node = self.root

        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TokenTrieNode()
            node = node.children[token_id]
            node.starts_with_space = raw_text.startswith(" ")

        node.is_end = True
        self.max_depth = max(self.max_depth, len(token_ids))

    def get_next_tokens(self, prefix_ids, only_starts_with_space=False):
        node = self.root
        for token_id in prefix_ids:
            if token_id not in node.children:
                return []  # dead end
            node = node.children[token_id]
        if only_starts_with_space:
            return set(
                token_id
                for token_id, child in node.children.items()
                if child.starts_with_space
            )
        else:
            return set(node.children.keys())


class PrefixAllowedTokens:
    def __init__(self, words, prompt_len, tokenizer):
        self.trie = TokenTrie()
        self.words = words
        self.special_tokens = tokenizer.all_special_tokens + [
            "assistant", 
            "\n\n",
            "\n",
            ".",
            ",",
            " ",
            "?",
            "!",
            "¿",
            "¡",
            ":",
            ";",
            "-",
            "(",
            ")",
        ]
        self.prompt_len = prompt_len
        self.tokenizer = tokenizer
        for word in words + self.special_tokens:
            variants = [word, word.title(), " " + word, " " + word.title()]

            for variant in variants:
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                self.trie.insert(token_ids, variant)

    def __call__(self, batch_id, input_ids):
        """A function that returns the allowed tokens for the next generation."""

        full_input = input_ids.tolist()
        generated = full_input[self.prompt_len:]
        allowed = set()

        # Try all possible prefixes up to max token length
        for i in range(self.trie.max_depth):
            prefix = generated[-i:]
            allowed.update(self.trie.get_next_tokens(prefix))

        # Always allow starting a new word from scratch
        if len(generated) < 6:  # let it generate header unconstrained
            allowed.update(self.trie.get_next_tokens([]))
        else:
            allowed.update(self.trie.get_next_tokens([], only_starts_with_space=True))
            for token in self.special_tokens:
                allowed.add(self.tokenizer.encode(token, add_special_tokens=False)[0])

        return list(allowed) if allowed else [self.tokenizer.eos_token_id]

