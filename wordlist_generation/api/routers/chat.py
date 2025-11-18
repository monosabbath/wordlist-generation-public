import uuid
import time
from fastapi import APIRouter, Depends, Request

from wordlist_generation.schemas.chat import ChatCompletionRequest
from wordlist_generation.api.dependencies import verify_token, get_generation_service
from wordlist_generation.services.generation_service import GenerationService

router = APIRouter(prefix="/v1", tags=["chat"])


@router.get("/models")
def list_models(request: Request, auth_ok: bool = Depends(verify_token)):
    settings = request.app.state.settings
    return {
        "object": "list",
        "data": [
            {
                "id": settings.MODEL_NAME,
                "object": "model",
                "owned_by": "owner",
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(
    req: ChatCompletionRequest, 
    service: GenerationService = Depends(get_generation_service),
    auth_ok: bool = Depends(verify_token)
):
    result = service.generate(req)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": result["usage"],
    }
