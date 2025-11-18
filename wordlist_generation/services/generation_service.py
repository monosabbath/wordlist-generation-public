import torch
from typing import List, Optional, Dict, Any

from wordlist_generation.schemas.chat import ChatCompletionRequest
from wordlist_generation.services.model_service import ModelService
from wordlist_generation.inference.vocab_constraints.prefix import build_regexp_prefix_fn
from wordlist_generation.inference.vocab_constraints.tokens import get_stop_ids
from wordlist_generation.inference.templates import build_template_kwargs_for_model
from wordlist_generation.inference.generation import (
    truncate_inputs_to_max_length,
    strip_unused_model_inputs,
    move_inputs_to_correct_device,
    normalize_max_new_tokens,
    getgen_kwargs,
    tokenizer_encode_for_chat
)

class GenerationService:
    def __init__(self, model_service: ModelService, settings):
        self.ms = model_service
        self.settings = settings

    def generate(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Unified generation logic for both Chat and Batch interfaces.
        Returns the raw text content and usage stats.
        """
        tokenizer = self.ms.tokenizer
        model = self.ms.model

        # 1. Prepare Messages (System prompt handling)
        messages = self._prepare_messages(request)

        # 2. Prepare Inputs (Tokenization & Templates)
        template_kwargs = build_template_kwargs_for_model(self.settings.MODEL_NAME)
        
        # Specialized logic for Cohere vs Standard models
        if self.ms.is_cohere_reasoning_model:
            inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
            inputs = truncate_inputs_to_max_length(inputs, self.settings.MAX_INPUT_TOKENS)
            inputs = strip_unused_model_inputs(inputs)
            inputs = move_inputs_to_correct_device(inputs, model)
            input_len = inputs["input_ids"].shape[1]
        else:
            prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
            if "glm-4.6" in self.settings.MODEL_NAME.lower():
                prompt_text = prompt_text + "<think></think>"
            
            inputs = tokenizer_encode_for_chat(
                tokenizer, 
                prompt_text, 
                self.settings.PAD_TO_MULTIPLE_OF, 
                self.settings.MAX_INPUT_TOKENS
            )
            inputs = strip_unused_model_inputs(inputs)
            inputs = move_inputs_to_correct_device(inputs, model)
            input_len = inputs["input_ids"].shape[1]

        # 3. Prepare Constraints
        prefix_fn = None
        if request.vocab_lang and request.vocab_n_words:
            prefix_fn = build_regexp_prefix_fn(
                tokenizer=tokenizer,
                lang=request.vocab_lang,
                n_words=request.vocab_n_words,
                wordlist_dir=self.settings.WORDLIST_DIR,
                allow_cohere_controls=self.ms.is_cohere_reasoning_model,
            )

        # 4. Generation Config
        max_new_tokens = normalize_max_new_tokens(request.max_tokens, self.settings.ALLOWED_MAX_NEW_TOKENS)
        stop_ids = get_stop_ids(tokenizer)
        
        gen_kwargs = getgen_kwargs(
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            stop_ids=stop_ids,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty if request.length_penalty is not None else 1.0,
            prefix_fn=prefix_fn,
            # Sampling params
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )

        # 5. Run Inference (Thread-safe via GPU Gate)
        with self.ms.gpu_gate:
            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)

        # 6. Decode & Return
        gen_len = int(outputs[0].shape[0] - input_len)
        last_token = int(outputs[0][-1].item())
        finish_reason = "stop" if stop_ids and (last_token in set(stop_ids)) else ("length" if gen_len >= max_new_tokens else "stop")
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": int(input_len),
                "completion_tokens": gen_len,
                "total_tokens": int(input_len) + gen_len
            }
        }

    def _prepare_messages(self, req: ChatCompletionRequest) -> List[Dict[str, str]]:
        # Extract system prompt and rebuild messages (system first)
        system_prompt = ""
        for msg in req.messages:
            if msg.role == "system":
                system_prompt = msg.content
                break
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in req.messages:
            if msg.role != "system":
                messages.append({"role": msg.role, "content": msg.content})
        return messages

