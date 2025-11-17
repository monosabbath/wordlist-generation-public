import logging
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline

logger = logging.getLogger("model-service")


class GPUConcurrencyGate:
    def __init__(self, max_concurrency: int = 1):
        import threading
        self._sem = threading.Semaphore(max_concurrency)

    def __enter__(self):
        self._sem.acquire()

    def __exit__(self, exc_type, exc, tb):
        self._sem.release()


class ModelService:
    def __init__(self, model, tokenizer, text_pipeline, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.text_pipeline = text_pipeline
        self.settings = settings
        self.gpu_gate = GPUConcurrencyGate(max_concurrency=int(getattr(settings, "GENERATION_MAX_CONCURRENCY", 1)))

    @property
    def is_cohere_reasoning_model(self) -> bool:
        return self.settings.MODEL_NAME.strip().lower() == "coherelabs/command-a-reasoning-08-2025"

    @classmethod
    def from_settings(cls, s):
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

        base_init: Dict[str, Any] = {
            "trust_remote_code": s.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": True,
            "local_files_only": False,
        }
        if dtype != "auto":
            base_init["torch_dtype"] = dtype

        # Decide CPU-first load path
        cpu_first = bool(s.USE_GROUPED_GEMM and s.FUSE_ON_CPU_BEFORE_SHARD and not s.LOAD_FUSED_EXPERTS)
        device_map_to_use: Optional[str] = None if cpu_first else s.DEVICE_MAP

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (cpu_first={cpu_first}, device_map='{device_map_to_use}', "
            f"use_grouped_gemm={s.USE_GROUPED_GEMM}, load_fused_experts={s.LOAD_FUSED_EXPERTS})"
        )

        # Load config to toggle grouped_gemm if fused
        try:
            config = AutoConfig.from_pretrained(
                s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load config first: {e}")
            config = None

        if config and s.USE_GROUPED_GEMM and s.LOAD_FUSED_EXPERTS and hasattr(config, "use_grouped_gemm"):
            try:
                config.use_grouped_gemm = True
                logger.info("Set config.use_grouped_gemm=True (pre-fused checkpoint).")
            except Exception as e:
                logger.warning(f"Could not set use_grouped_gemm on config: {e}")

        # Model load
        init_kwargs = dict(base_init)
        if device_map_to_use is not None:
            init_kwargs["device_map"] = device_map_to_use

        try:
            if config is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    attn_implementation=s.ATTN_IMPLEMENTATION,
                    config=config,
                    **init_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    attn_implementation=s.ATTN_IMPLEMENTATION,
                    **init_kwargs,
                )
        except TypeError:
            logger.warning("attn_implementation not accepted; retrying without it.")
            if config is not None:
                model = AutoModelForCausalLM.from_pretrained(s.MODEL_NAME, config=config, **init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(s.MODEL_NAME, **init_kwargs)
            try:
                model.set_attention_implementation(s.ATTN_IMPLEMENTATION)
            except Exception as e:
                logger.warning(f"Could not set attention impl: {e}")
                try:
                    model.set_attention_implementation("sdpa")
                except Exception as e2:
                    logger.warning(f"Fallback SDPA also failed: {e2}")

        if s.STATIC_KV_CACHE:
            try:
                model.generation_config.cache_implementation = "static"
                logger.info("Enabled static KV cache.")
            except Exception as e:
                logger.warning(f"Static KV cache unavailable: {e}")

        tokenizer = AutoTokenizer.from_pretrained(
            s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
        )
        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Fuse experts if requested (unfused checkpoint path)
        if s.USE_GROUPED_GEMM and not s.LOAD_FUSED_EXPERTS:
            if hasattr(model, "fuse_experts"):
                try:
                    logger.info("Fusing experts on CPU prior to sharding...")
                    model.fuse_experts()
                    logger.info("Expert fusion complete.")
                except ImportError as e:
                    logger.error(
                        "Missing grouped_gemm. Install with: pip install git+https://github.com/fanshiqing/grouped_gemm@main"
                    )
                except Exception as e:
                    logger.error(f"Failed to fuse experts: {e}")
            else:
                logger.warning("Model has no fuse_experts(); verify Transformers branch installation.")

        # If we loaded on CPU first and a device_map was desired, dispatch now
        if cpu_first and s.DEVICE_MAP and s.DEVICE_MAP != "auto":
            # Explicit single-device move
            logger.info(f"Moving fused model to device '{s.DEVICE_MAP}'")
            model.to(s.DEVICE_MAP)
        elif cpu_first and s.DEVICE_MAP == "auto":
            # Use HF utility to infer device_map then load with weights already in RAM
            try:
                from transformers import infer_auto_device_map
                inferred = infer_auto_device_map(model, max_memory=None)  # optionally pass max_memory dict
                logger.info(f"Inferred device_map for fused model: {inferred}")
                model.tie_weights()
                # Dispatch manually
                for name, param in model.named_parameters():
                    # Simple heuristic: move parameter based on first submodule match
                    for dev_key, modules in inferred.items():
                        # modules is list of module names; if name starts with one
                        if any(name.startswith(m) for m in modules):
                            param.data = param.data.to(dev_key)
                            break
            except Exception as e:
                logger.warning(f"Automatic device sharding failed; model remains on CPU: {e}")

        model.eval()

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=s.TRUST_REMOTE_CODE,
        )

        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipeline, settings=s)
