import logging
from typing import Any, Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline

logger = logging.getLogger("model-service")


class GPUConcurrencyGate:
    """
    Simple per-process concurrency gate to serialize GPU-bound generation
    when desired (e.g., beam search).
    """
    def __init__(self, max_concurrency: int = 1):
        import threading
        self._sem = threading.Semaphore(max_concurrency)

    def __enter__(self):
        self._sem.acquire()

    def __exit__(self, exc_type, exc, tb):
        self._sem.release()


def _is_multi_gpu() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 1
    except Exception:
        return False


def _model_is_model_parallel(model) -> bool:
    """
    Detect model-parallel placement (multiple CUDA devices holding parameters).
    Works for HF device_map and Accelerate dispatch.
    """
    try:
        devs = {p.device for p in model.parameters()}
        cuda_devs = {d for d in devs if d.type == "cuda"}
        return len(cuda_devs) > 1
    except Exception:
        return False


class ModelService:
    def __init__(self, model, tokenizer, text_pipeline, settings):
        self.model = model
        self.tokenizer = tokenizer
        self.text_pipeline = text_pipeline  # None when model-parallel multi-GPU
        self.settings = settings
        self.gpu_gate = GPUConcurrencyGate(
            max_concurrency=int(getattr(settings, "GENERATION_MAX_CONCURRENCY", 1))
        )

    @property
    def is_cohere_reasoning_model(self) -> bool:
        return self.settings.MODEL_NAME.strip().lower() == "coherelabs/command-a-reasoning-08-2025"

    def is_model_parallel(self) -> bool:
        return _model_is_model_parallel(self.model)

    def supports_pipeline(self) -> bool:
        # Only use the pipeline when not model-parallel (single device)
        return (self.text_pipeline is not None) and (not self.is_model_parallel())

    @classmethod
    def from_settings(cls, s):
        # Performance knobs
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # Dtype selection
        dtype: Any = "auto"
        dtypestr = s.TORCH_DTYPE.lower()
        if dtypestr in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
            logger.info("Using dtype: bfloat16")
        elif dtypestr in ("fp16", "float16"):
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

        use_grouped_gemm = bool(getattr(s, "USE_GROUPED_GEMM", False))

        # Load config first
        config: Optional[Any] = None
        try:
            config = AutoConfig.from_pretrained(
                s.MODEL_NAME,
                trust_remote_code=s.TRUST_REMOTE_CODE,
                local_files_only=False,
            )
        except Exception as e:
            logger.warning(f"Config load failed: {e}")

        # Enable grouped_gemm kernels for fused checkpoints
        if config and use_grouped_gemm and hasattr(config, "use_grouped_gemm"):
            try:
                config.use_grouped_gemm = True
                logger.info("Set config.use_grouped_gemm=True (fused checkpoint expected).")
            except Exception as e:
                logger.warning(f"Could not set config.use_grouped_gemm: {e}")

        # Device map policy:
        # - grouped_gemm=true: CPU-first load (device_map=None), then manual Accelerate dispatch.
        # - grouped_gemm=false: from_pretrained with device_map=s.DEVICE_MAP (default 'auto').
        device_map_to_use = None if use_grouped_gemm else s.DEVICE_MAP

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (grouped_gemm={use_grouped_gemm}, "
            f"device_map='{device_map_to_use}', attn='{s.ATTN_IMPLEMENTATION}')"
        )

        init_kwargs = dict(base_init)
        if device_map_to_use is not None:
            init_kwargs["device_map"] = device_map_to_use

        # Load model
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
            logger.warning("Retrying load without attn_implementation (not supported).")
            if config is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    config=config,
                    **{k: v for k, v in init_kwargs.items() if k != "attn_implementation"},
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    **{k: v for k, v in init_kwargs.items() if k != "attn_implementation"},
                )
            try:
                model.set_attention_implementation(s.ATTN_IMPLEMENTATION)
            except Exception as e:
                logger.warning(f"Could not set attention implementation: {e}. Attempting SDPA fallback.")
                try:
                    model.set_attention_implementation("sdpa")
                except Exception as e2:
                    logger.warning(f"SDPA fallback failed: {e2}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            s.MODEL_NAME,
            trust_remote_code=s.TRUST_REMOTE_CODE,
            local_files_only=False,
        )
        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Optional static KV cache (mainly beneficial for repeated generate)
        if s.STATIC_KV_CACHE:
            try:
                model.generation_config.cache_implementation = "static"
                logger.info("Enabled static KV cache.")
            except Exception as e:
                logger.warning(f"Static KV cache not available: {e}")

        text_pipeline = None

        # Model-parallel (Accelerate) dispatch when grouped_gemm=true
        if use_grouped_gemm:
            if not _is_multi_gpu():
                logger.warning("Grouped GEMM requested but only a single GPU is visible; proceeding single-device.")
                if torch.cuda.is_available():
                    model.to("cuda:0")
            else:
                try:
                    from accelerate.utils import infer_auto_device_map
                    from accelerate import dispatch_model

                    device_map = infer_auto_device_map(model, include_embeddings=True)
                    logger.info(f"Inferred device_map: {device_map}")
                    model = dispatch_model(model, device_map=device_map)
                except Exception as e:
                    logger.error(f"Accelerate dispatch failed; falling back to device_map='auto' reload: {e}")
                    # Reload with HF device_map='auto' as fallback
                    try:
                        reload_kwargs = dict(base_init)
                        if dtype != "auto":
                            reload_kwargs["torch_dtype"] = dtype
                        model = AutoModelForCausalLM.from_pretrained(
                            s.MODEL_NAME,
                            trust_remote_code=s.TRUST_REMOTE_CODE,
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            **reload_kwargs,
                        )
                        logger.info("Reloaded model with device_map='auto'.")
                    except Exception as e2:
                        logger.error(f"Fallback reload failed: {e2}")
                        if torch.cuda.is_available():
                            model.to("cuda:0")
        else:
            # Non-grouped GEMM path: if device map places everything on one device we can safely build a pipeline.
            # For multi-GPU auto device_map, pipeline is often still okay—but to keep safety we only enable pipeline
            # when model is fully on a single device.
            single_cuda_dev = None
            try:
                devs = {p.device for p in model.parameters()}
                cuda_devs = {d for d in devs if d.type == "cuda"}
                if len(cuda_devs) == 1:
                    single_cuda_dev = list(cuda_devs)[0]
            except Exception:
                pass

            if single_cuda_dev is not None or not torch.cuda.is_available():
                text_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=s.TRUST_REMOTE_CODE,
                )
                logger.info("Built text-generation pipeline (single device).")
            else:
                logger.info("Model appears to be model-parallel; will skip pipeline and use direct generate().")

        model.eval()
        logger.info(f"Model parallel: {_model_is_model_parallel(model)}")
        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipeline, settings=s)
