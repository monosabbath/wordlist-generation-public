import logging
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
            # Critical: no device_map here if CPU_FIRST_LOAD
            "device_map": None if s.CPU_FIRST_LOAD else (s.DEVICE_MAP if s.DEVICE_MAP != "none" else None),
        }
        if dtype != "auto":
            base_init["torch_dtype"] = dtype

        use_grouped_gemm = bool(getattr(s, "USE_GROUPED_GEMM", False))
        load_fused_experts = bool(getattr(s, "LOAD_FUSED_EXPERTS", False))

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (cpu_first={s.CPU_FIRST_LOAD}, "
            f"device_map='{s.DEVICE_MAP}', use_grouped_gemm={use_grouped_gemm}, "
            f"prefused={load_fused_experts})"
        )

        # Load config first
        config: Optional[AutoConfig] = None
        try:
            config = AutoConfig.from_pretrained(
                s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load config first: {e}")

        if config and use_grouped_gemm and load_fused_experts and hasattr(config, "use_grouped_gemm"):
            try:
                config.use_grouped_gemm = True
                logger.info("Set config.use_grouped_gemm=True (prefused checkpoint).")
            except Exception as e:
                logger.warning(f"Could not set use_grouped_gemm on config: {e}")

        # Model load (always CPU if CPU_FIRST_LOAD)
        init_kwargs = dict(base_init)
        attn_impl = getattr(s, "ATTN_IMPLEMENTATION", None)

        def _load_model():
            try:
                if config is not None:
                    return AutoModelForCausalLM.from_pretrained(
                        s.MODEL_NAME,
                        attn_implementation=attn_impl,
                        config=config,
                        **{k: v for k, v in init_kwargs.items() if k != "attn_implementation"},
                    )
                else:
                    return AutoModelForCausalLM.from_pretrained(
                        s.MODEL_NAME,
                        attn_implementation=attn_impl,
                        **{k: v for k, v in init_kwargs.items() if k != "attn_implementation"},
                    )
            except TypeError:
                logger.warning("Retrying load without attn_implementation (not supported).")
                if config is not None:
                    return AutoModelForCausalLM.from_pretrained(
                        s.MODEL_NAME,
                        config=config,
                        **{k: v for k, v in init_kwargs.items() if k != "attn_implementation"},
                    )
                else:
                    return AutoModelForCausalLM.from_pretrained(
                        s.MODEL_NAME,
                        **{k: v for k, v in init_kwargs.items() if k != "attn_implementation"},
                    )

        model = _load_model()

        # If unfused and using grouped_gemm, fuse on CPU now (offline fuse alternative)
        if use_grouped_gemm and not load_fused_experts and hasattr(model, "fuse_experts"):
            try:
                logger.info("Fusing experts in-process on CPU...")
                model.fuse_experts()
                if hasattr(model.config, "use_grouped_gemm"):
                    model.config.use_grouped_gemm = True
                logger.info("Expert fusion complete.")
            except Exception as e:
                logger.error(f"Failed to fuse experts: {e}")

        # Static KV cache toggle
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

        # Dispatch to GPUs after CPU load
        if s.CPU_FIRST_LOAD:
            logger.info("CPU-first load complete. Preparing GPU dispatch...")
            if s.DEVICE_MAP == "auto":
                try:
                    from accelerate.utils import infer_auto_device_map
                    from accelerate import dispatch_model
                    device_map = infer_auto_device_map(
                        model,
                        max_memory=None,  # Could customize per-device memory
                        no_split_module_classes=None,
                    )
                    logger.info(f"Inferred device_map: {device_map}")
                    model = dispatch_model(model, device_map=device_map)
                except Exception as e:
                    logger.warning(f"Automatic device sharding failed; model remains on CPU: {e}")
            elif s.DEVICE_MAP and s.DEVICE_MAP != "none":
                # Single device or manual spec
                try:
                    logger.info(f"Moving model to device '{s.DEVICE_MAP}'")
                    model.to(s.DEVICE_MAP)
                except Exception as e:
                    logger.warning(f"Failed to move model to '{s.DEVICE_MAP}': {e}")
        else:
            logger.info("Device map applied during from_pretrained (not CPU-first).")

        model.eval()

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=s.TRUST_REMOTE_CODE,
        )

        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipeline, settings=s)
