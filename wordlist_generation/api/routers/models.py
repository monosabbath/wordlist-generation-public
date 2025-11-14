from typing import Optional, List
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    vocab_lang: Optional[str] = None
    vocab_n_words: Optional[int] = None
    num_beams: Optional[int] = 5
    length_penalty: Optional[float] = 1.0

    # Sampling parameters
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0
