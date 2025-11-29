import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, BitsAndBytesConfig

# Try to import qwen3_moe_fused (bundled in this package)
try:
    from wordlist_generation.qwen3_moe_fused import Qwen3MoeFusedForCausalLM, patch_bnb_quantizer
    HAS_FUSED_MOE = True
except ImportError:
    HAS_FUSED_MOE = False

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

        # Determine if we should use Qwen3 Fused MoE
        use_fused_class = False
        if HAS_FUSED_MOE:
            try:
                config = AutoConfig.from_pretrained(s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE)
                if hasattr(config, "architectures") and config.architectures and "Qwen3MoeFused" in config.architectures[0]:
                    use_fused_class = True
                    logger.info("Detected Qwen3FusedMoe architecture.")
            except Exception as e:
                logger.warning(f"Could not inspect config for fused architecture: {e}")

        # Quantization config
        quantization_config = None
        if getattr(s, "LOAD_IN_4BIT", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype if dtype != "auto" else torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Enabled 4-bit quantization (load_in_4bit=True).")
            if use_fused_class:
                patch_bnb_quantizer()

        init_kwargs: Dict[str, Any] = {
            "trust_remote_code": s.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": True,
            "local_files_only": False,
            "device_map": s.DEVICE_MAP,
        }
        if dtype != "auto":
            init_kwargs["torch_dtype"] = dtype
        
        if quantization_config:
            init_kwargs["quantization_config"] = quantization_config

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (trust_remote_code={s.TRUST_REMOTE_CODE}, "
            f"attn='{s.ATTN_IMPLEMENTATION}', device_map='{s.DEVICE_MAP}')"
        )

        model = None
        if use_fused_class:
            logger.info("Using Qwen3MoeFusedForCausalLM for loading.")
            try:
                model = Qwen3MoeFusedForCausalLM.from_pretrained(
                    s.MODEL_NAME,
                    attn_implementation=s.ATTN_IMPLEMENTATION,
                    **init_kwargs,
                )
            except Exception as e:
                logger.error(f"Failed to load with Qwen3MoeFusedForCausalLM: {e}")
                # Fallback might fail if architecture is unknown to AutoModel, but we can try
        
        if model is None:
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
