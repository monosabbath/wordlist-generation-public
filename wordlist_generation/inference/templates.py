from typing import Any, Dict


def is_cohere_reasoning_model(model_name: str) -> bool:
    return model_name.strip().lower() == "coherelabs/command-a-reasoning-08-2025"


def build_template_kwargs_for_model(model_name: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"add_generation_prompt": True}
    if is_cohere_reasoning_model(model_name):
        kwargs.update({
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "reasoning": False,
        })
    else:
        kwargs["tokenize"] = False
    return kwargs
