import logging
from typing import Any, Dict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

logger = logging.getLogger("model-service")


class ModelService:
    def __init__(self, model, tokenizer, text_pipeline, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.text_pipeline = text_pipeline
        self.settings = settings

    @property
    def is_cohere_reasoning_model(self) -> bool:
        return self.settings.MODEL_NAME.strip().lower() == "coherelabs/command-a-reasoning-08-2025"

    @classmethod
    def from_settings(cls, s):
        # Torch backend knobs
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # Dtype selection
        dtype: Any = "auto"
        ts = s.TORCH_DTYPE.lower()
        if ts in ("bf16", "bfloat16", "torch.bfloat16"):
            dtype = torch.bfloat16
            logger.info("Using dtype: bfloat16")
        elif ts in ("fp16", "float16", "torch.float16"):
            dtype = torch.float16
            logger.info("Using dtype: float16")
        else:
            logger.info("Using dtype: auto")

        init_kwargs: Dict[str, Any] = {
            "trust_remote_code": s.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": True,
            "local_files_only": False,
            "device_map": s.DEVICE_MAP,
        }
        if dtype != "auto":
            init_kwargs["torch_dtype"] = dtype

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (trust_remote_code={s.TRUST_REMOTE_CODE}, "
            f"attn='{s.ATTN_IMPLEMENTATION}', device_map='{s.DEVICE_MAP}')"
        )

        # Inspect config to route Mistral3 correctly
        cfg = AutoConfig.from_pretrained(s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False)
        model_type = getattr(cfg, "model_type", "").lower()
        cfg_name = cfg.__class__.__name__

        # Load model
        if model_type == "mistral3" or "Mistral3Config" in cfg_name:
            try:
                from transformers import Mistral3ForConditionalGeneration  # type: ignore
                try:
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        s.MODEL_NAME,
                        attn_implementation=s.ATTN_IMPLEMENTATION,
                        **init_kwargs,
                    )
                except TypeError:
                    logger.warning("Model.from_pretrained() did not accept attn_implementation; retrying without it.")
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        s.MODEL_NAME,
                        **init_kwargs,
                    )
                    try:
                        model.set_attention_implementation(s.ATTN_IMPLEMENTATION)
                    except Exception as e2:
                        logger.warning(f"Could not set attention implementation: {e2}. Trying SDPA.")
                        try:
                            model.set_attention_implementation("sdpa")
                        except Exception as e3:
                            logger.warning(f"Could not set SDPA either: {e3}")
            except ImportError as e:
                raise RuntimeError(
                    "Your Transformers version does not expose Mistral3ForConditionalGeneration. "
                    "Please upgrade: pip install -U 'transformers>=4.46.2' 'mistral-common>=1.6.2'"
                ) from e
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    attn_implementation=s.ATTN_IMPLEMENTATION,
                    **init_kwargs,
                )
            except TypeError:
                logger.warning("Model.from_pretrained() did not accept attn_implementation; retrying without it.")
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    **init_kwargs,
                )
                try:
                    model.set_attention_implementation(s.ATTN_IMPLEMENTATION)
                except Exception as e:
                    logger.warning(f"Could not set attention implementation: {e}. Trying SDPA.")
                    try:
                        model.set_attention_implementation("sdpa")
                    except Exception as e2:
                        logger.warning(f"Could not set SDPA either: {e2}")

        # Load tokenizer with Mistral3 fallback
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                s.MODEL_NAME,
                trust_remote_code=s.TRUST_REMOTE_CODE,
                local_files_only=False,
            )
        except KeyError as e:
            # Example: KeyError: <class '...Mistral3Config'>
            if model_type == "mistral3" or "Mistral3Config" in str(e):
                try:
                    from transformers import Mistral3Tokenizer  # type: ignore
                except ImportError as e2:
                    raise RuntimeError(
                        "Transformers build does not expose Mistral3Tokenizer. "
                        "Upgrade with: pip install -U 'transformers>=4.46.2' 'tokenizers>=0.20.1'"
                    ) from e2
                tokenizer = Mistral3Tokenizer.from_pretrained(
                    s.MODEL_NAME,
                    trust_remote_code=s.TRUST_REMOTE_CODE,
                    local_files_only=False,
                )
            else:
                raise

        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=s.TRUST_REMOTE_CODE,
        )

        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipeline, settings=s)
