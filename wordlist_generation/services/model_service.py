import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger("model-service")


class GPUConcurrencyGate:
    """
    Simple process-local concurrency gate to avoid overlapping GPU-bound generation
    when running a single sharded model with device_map='auto'.
    """
    def __init__(self, max_concurrency: int = 1):
        import threading
        # A semaphore of size N allows up to N concurrent generations.
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
        # Serialize GPU work by default (can be >1 if you want limited parallelism)
        self.gpu_gate = GPUConcurrencyGate(max_concurrency=int(getattr(settings, "GENERATION_MAX_CONCURRENCY", 1)))

    @property
    def is_cohere_reasoning_model(self) -> bool:
        return self.settings.MODEL_NAME.strip().lower() == "coherelabs/command-a-reasoning-08-2025"

    @staticmethod
    def is_deepseek_v3_model(model_name: str) -> bool:
        """Check if the model is a Deepseek V3 MoE model."""
        normalized = model_name.strip().lower()
        return "deepseek-v3" in normalized or "deepseek_v3" in normalized

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

        # Deepseek V3 MoE grouped GEMM optimization
        if ModelService.is_deepseek_v3_model(s.MODEL_NAME) and s.USE_GROUPED_GEMM:
            logger.info("Deepseek V3 model detected with USE_GROUPED_GEMM=true")
            
            # Check if grouped_gemm is available
            try:
                import grouped_gemm  # noqa: F401
                logger.info("grouped_gemm module found")
            except ImportError:
                logger.error(
                    "grouped_gemm module not found. To use grouped GEMM optimization for Deepseek V3, "
                    "please install it: pip install grouped_gemm"
                )
                logger.warning("Continuing without grouped GEMM optimization")
            else:
                # Fuse experts if requested
                if s.FUSE_EXPERTS_ON_STARTUP:
                    try:
                        logger.info("Calling model.fuse_experts() to enable grouped GEMM kernel...")
                        model.fuse_experts()
                        logger.info("Successfully fused experts for Deepseek V3 MoE model")
                    except Exception as e:
                        logger.error(
                            f"Failed to fuse experts for Deepseek V3 model: {e}. "
                            "Falling back to standard MoE implementation."
                        )
                else:
                    logger.info("FUSE_EXPERTS_ON_STARTUP=false, skipping expert fusion")


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
