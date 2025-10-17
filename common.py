from typing import Literal

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    vocab_lang: Literal["en", "es"]
    vocab_n_words: int
    max_new_tokens: int = 518
    num_beams: int = 10
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
