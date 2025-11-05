import logging
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

        # Optional Triton autotuning logs for fused kernels
        if getattr(s, "MOE_FUSED_TRITON_AUTOTUNING", False):
            os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

        model = None
        tokenizer = None

        use_fused = bool(getattr(s, "MOE_FUSED_ENABLE", False)) and ("qwen3" in s.MODEL_NAME.lower()) and ("fused" in s.MODEL_NAME.lower())

        if use_fused:
            logger.info(f"Loading fused Qwen3 MoE model '{s.MODEL_NAME}' (device_map='{s.DEVICE_MAP}')")
            try:
                from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
            except Exception as e:
                raise RuntimeError(
                    "MOE_FUSED_ENABLE=true but qwen3_moe_fused is not importable. "
                    "Install it first, e.g.:\n"
                    "  pip install -U pip setuptools wheel\n"
                    "  sudo apt-get update && sudo apt-get install -y git  # ensure git is available\n"
                    "  pip install -r requirements-moe.txt\n"
                    "Or: pip install \"transformers-qwen3-moe-fused @ git+https://github.com/woct0rdho/transformers-qwen3-moe-fused@4caa1524b8f691d24da2fa26e99711b2ec77db44\""
                ) from e

            try:
                model = Qwen3MoeFusedForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    **init_kwargs,
                )
            except Exception as e:
                logger.error(f"Failed to load fused MoE model '{s.MODEL_NAME}': {e}. Falling back to AutoModel.")
                use_fused = False

        if not use_fused:
            logger.info(
                f"Loading model '{s.MODEL_NAME}' (trust_remote_code={s.TRUST_REMOTE_CODE}, "
                f"attn='{s.ATTN_IMPLEMENTATION}', device_map='{s.DEVICE_MAP}')"
            )
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
                    logger.warning(f"Could not set attention implementation: {e}. Falling back to SDPA if possible.")
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
