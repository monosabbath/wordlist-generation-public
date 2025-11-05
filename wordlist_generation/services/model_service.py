import logging
import os
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# NEW: import bnb config
from transformers import BitsAndBytesConfig

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
        # NOTE: PyTorch 2.9 deprecates old TF32 toggles; harmless warning.
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
        # IMPORTANT: when we pass a quantization_config for fused 4-bit, we should not also force torch_dtype here.
        # We'll add torch_dtype only for non-quantized loads.
        add_torch_dtype = dtype != "auto"

        # Optional Triton autotuning logs for fused kernels
        if getattr(s, "MOE_FUSED_TRITON_AUTOTUNING", False):
            os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

        model = None

        use_fused = bool(getattr(s, "MOE_FUSED_ENABLE", False)) and ("qwen3" in s.MODEL_NAME.lower()) and ("fused" in s.MODEL_NAME.lower())

        if use_fused:
            logger.info(f"Loading fused Qwen3 MoE model '{s.MODEL_NAME}' (device_map='{s.DEVICE_MAP}')")
            try:
                # 1) Patch the fused quantizer BEFORE importing fused modules
                if "bnb-4bit" in s.MODEL_NAME.lower():
                    from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer
                    patch_bnb_quantizer()
                    logger.info("Patched bitsandbytes quantizer for fused 4-bit.")

                # 2) Import fused class AFTER patching
                from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM

                # 3) Build a bnb config for 4-bit repos so the HF quantizer path is used
                quantization_config = None
                if "bnb-4bit" in s.MODEL_NAME.lower():
                    # Prefer bf16 compute on Ampere+/Ada/Hopper, else fp16
                    compute_dtype = torch.bfloat16 if dtype in (torch.bfloat16, "auto") else torch.float16
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )

                # 4) Don’t pass attn_implementation; fused class manages its own kernels.
                fused_kwargs = dict(init_kwargs)
                # Avoid torch_dtype here when quantized; compute dtype is set in quantization_config
                if quantization_config is None and add_torch_dtype:
                    fused_kwargs["torch_dtype"] = dtype

                if quantization_config is not None:
                    model = Qwen3MoeFusedForCausalLM.from_pretrained(
                        s.MODEL_NAME,
                        quantization_config=quantization_config,
                        **fused_kwargs,
                    )
                else:
                    model = Qwen3MoeFusedForCausalLM.from_pretrained(
                        s.MODEL_NAME,
                        **fused_kwargs,
                    )

            except Exception as e:
                # Don’t fall back to AutoModel for fused checkpoints — shapes/layouts are incompatible
                logger.error(f"Fused MoE load failed for '{s.MODEL_NAME}': {e}")
                raise

        else:
            logger.info(
                f"Loading model '{s.MODEL_NAME}' (trust_remote_code={s.TRUST_REMOTE_CODE}, "
                f"attn='{s.ATTN_IMPLEMENTATION}', device_map='{s.DEVICE_MAP}')"
            )
            nonfused_kwargs = dict(init_kwargs)
            if add_torch_dtype:
                nonfused_kwargs["torch_dtype"] = dtype
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    attn_implementation=s.ATTN_IMPLEMENTATION,
                    **nonfused_kwargs,
                )
            except TypeError:
                logger.warning("Model.from_pretrained() did not accept attn_implementation; retrying without it.")
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    **nonfused_kwargs,
                )
                try:
                    model.set_attention_implementation(s.ATTN_IMPLEMENTATION)
                except Exception as e:
                    logger.warning(f"Could not set attention implementation: {e}. Falling back to SDPA if possible.")
                    try:
                        model.set_attention_implementation("sdpa")
                    except Exception as e2:
                        logger.warning(f"Could not set SDPA either: {e2}")

        # Optional static KV cache
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
