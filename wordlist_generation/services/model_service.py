import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline

logger = logging.getLogger("model-service")


class GPUConcurrencyGate:
    """
    Simple process-local concurrency gate to avoid overlapping GPU-bound generation
    when running a single sharded model with device_map='auto'.
    """
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

        base_init: Dict[str, Any] = {
            "trust_remote_code": s.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": True,
            "local_files_only": False,
        }
        if dtype != "auto":
            base_init["torch_dtype"] = dtype

        # Read MoE/grouped_gemm flags with safe defaults
        use_grouped_gemm = bool(getattr(s, "USE_GROUPED_GEMM", False))
        load_fused_experts = bool(getattr(s, "LOAD_FUSED_EXPERTS", False))
        fuse_on_cpu_before_shard = bool(getattr(s, "FUSE_ON_CPU_BEFORE_SHARD", False))
        cpu_first = bool(use_grouped_gemm and fuse_on_cpu_before_shard and not load_fused_experts)

        device_map_to_use: Optional[str] = None if cpu_first else s.DEVICE_MAP

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (trust_remote_code={s.TRUST_REMOTE_CODE}, "
            f"attn='{s.ATTN_IMPLEMENTATION}', device_map='{device_map_to_use}', "
            f"use_grouped_gemm={use_grouped_gemm}, load_fused_experts={load_fused_experts}, "
            f"cpu_first={cpu_first})"
        )

        # Load config first to possibly set use_grouped_gemm for pre-fused checkpoints
        config = None
        try:
            config = AutoConfig.from_pretrained(
                s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load config first: {e}")

        if config and use_grouped_gemm and load_fused_experts and hasattr(config, "use_grouped_gemm"):
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
            logger.warning("Model.from_pretrained() did not accept attn_implementation; retrying without it.")
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
                logger.warning(f"Could not set attention implementation: {e}. Falling back to SDPA if possible.")
                try:
                    model.set_attention_implementation("sdpa")
                except Exception as e2:
                    logger.warning(f"Could not set SDPA either: {e2}")

        # Optional: static KV cache
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

        # Fuse experts on CPU if using unfused checkpoint
        if use_grouped_gemm and not load_fused_experts:
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
                logger.warning("Model has no fuse_experts(); verify Transformers feature branch is installed.")

        # If we loaded on CPU first and want multi-GPU, shard now
        if cpu_first:
            if s.DEVICE_MAP == "auto":
                try:
                    from accelerate.utils import infer_auto_device_map
                    from accelerate import dispatch_model
                    device_map = infer_auto_device_map(model)
                    logger.info(f"Inferred device_map for fused model: {device_map}")
                    model = dispatch_model(model, device_map=device_map)
                except Exception as e:
                    logger.warning(f"Automatic device sharding failed; model remains on CPU: {e}")
            elif s.DEVICE_MAP and s.DEVICE_MAP != "auto":
                try:
                    logger.info(f"Moving fused model to device '{s.DEVICE_MAP}'")
                    model.to(s.DEVICE_MAP)
                except Exception as e:
                    logger.warning(f"Failed to move fused model to '{s.DEVICE_MAP}': {e}")

        model.eval()

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=s.TRUST_REMOTE_CODE,
        )

        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipeline, settings=s)
