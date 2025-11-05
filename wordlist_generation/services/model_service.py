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

        # Inspect config first so we can route Mistral3 correctly
        cfg = AutoConfig.from_pretrained(s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False)
        model_type = getattr(cfg, "model_type", "").lower()
        cfg_name = cfg.__class__.__name__

        model = None
        if model_type == "mistral3" or "Mistral3Config" in cfg_name:
            # Use the explicit class for Mistral 3.x models
            try:
                # Requires transformers >= 4.46.x where this class is available
                from transformers import Mistral3ForConditionalGeneration  # type: ignore

                try:
                    model = Mistral3ForConditionalGeneration.from_pretrained(
                        s.MODEL_NAME,
                        attn_implementation=s.ATTN_IMPLEMENTATION,
                        **init_kwargs,
                    )
                except TypeError:
                    # Some builds don't accept attn_implementation in from_pretrained
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
                # Clear guidance to upgrade Transformers for Mistral3 support
                raise RuntimeError(
                    "Transformers build does not expose Mistral3ForConditionalGeneration. "
                    "Please upgrade: pip install -U 'transformers>=4.46.0' 'mistral-common>=1.6.2'"
                ) from e
        else:
            # Default path for decoder-only models that map to AutoModelForCausalLM
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

        if s.STATIC_KV_CACHE:
            try:
                model.generation_config.cache_implementation = "static"
                logger.info("Enabled static KV cache.")
            except Exception as e:
                logger.warning(f"Static KV cache not available: {e}")

        tokenizer = AutoTokenizer.from_pretrained(
            s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
        )
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
