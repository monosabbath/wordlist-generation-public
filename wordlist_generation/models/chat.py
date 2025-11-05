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
    num_beams: Optional[int] = 10
    length_penalty: Optional[float] = 1.0
    # Request OpenAI-compatible SSE streaming when true
    stream: Optional[bool] = False
