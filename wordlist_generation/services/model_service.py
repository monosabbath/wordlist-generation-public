import logging
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
from transformers import infer_auto_device_map
from accelerate import dispatch_model

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


def _parse_max_memory_string(s: Optional[str]) -> Dict[str, str]:
    """
    Parse strings like "gpu0:70GiB,gpu1:70GiB,cpu:256GiB" into a dict accepted by HF.
    """
    out: Dict[str, str] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _build_max_memory(settings) -> Dict[str, str]:
    """
    Build a max_memory dict for infer_auto_device_map.
    Priority:
      1) Respect GPU_MAX_MEMORY/CPU_MAX_MEMORY if provided in .env.
      2) Otherwise, compute from current free mem per device (use ~90% of free).
    """
    user_mm = _parse_max_memory_string(getattr(settings, "GPU_MAX_MEMORY", None))
    if getattr(settings, "CPU_MAX_MEMORY", None):
        user_mm.update(_parse_max_memory_string(f"cpu:{settings.CPU_MAX_MEMORY}"))
    if user_mm:
        return user_mm

    max_memory: Dict[str, str] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                # use ~90% of free to leave headroom
                budget_gib = max(int((free_bytes / (1024**3)) * 0.9), 1)
                max_memory[f"cuda:{i}"] = f"{budget_gib}GiB"
            except Exception:
                # Fallback if mem_get_info not available
                max_memory[f"cuda:{i}"] = "80GiB"
    # Give CPU a generous cap to allow offloading if needed
    max_memory["cpu"] = getattr(settings, "CPU_MAX_MEMORY", "256GiB")
    return max_memory


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
        # Recommend expandable segments to reduce fragmentation during large moves
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments=True")

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

        cpu_first = bool(s.USE_GROUPED_GEMM and s.FUSE_ON_CPU_BEFORE_SHARD and not s.LOAD_FUSED_EXPERTS)
        device_map_to_use: Optional[str] = None if cpu_first else s.DEVICE_MAP

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (cpu_first={cpu_first}, device_map='{device_map_to_use}', "
            f"use_grouped_gemm={s.USE_GROUPED_GEMM}, load_fused_experts={s.LOAD_FUSED_EXPERTS})"
        )

        # Load config
        try:
            config = AutoConfig.from_pretrained(
                s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load config first: {e}")
            config = None

        # If checkpoint is pre-fused, set the flag on config
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

        # Static KV cache (optional)
        if s.STATIC_KV_CACHE:
            try:
                model.generation_config.cache_implementation = "static"
                logger.info("Enabled static KV cache.")
            except Exception as e:
                logger.warning(f"Static KV cache unavailable: {e}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
        )
        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Fuse experts on CPU if needed
        if s.USE_GROUPED_GEMM and not s.LOAD_FUSED_EXPERTS:
            if hasattr(model, "fuse_experts"):
                try:
                    # Ensure we are on CPU for fusion
                    cpu_device = torch.device("cpu")
                    model.to(cpu_device)
                    logger.info("Fusing experts on CPU prior to sharding...")
                    model.fuse_experts()
                    # After fusion, mark config if available
                    if hasattr(model.config, "use_grouped_gemm"):
                        model.config.use_grouped_gemm = True
                    logger.info("Expert fusion complete.")
                except ImportError:
                    logger.error(
                        "Missing grouped_gemm. Install with: pip install git+https://github.com/fanshiqing/grouped_gemm@main"
                    )
                except Exception as e:
                    logger.error(f"Failed to fuse experts: {e}")
            else:
                logger.warning("Model has no fuse_experts(); verify Transformers feature branch installation.")

        # If we loaded CPU-first and want auto sharding, infer device map and dispatch
        if cpu_first and s.DEVICE_MAP == "auto":
            try:
                os.makedirs(s.OFFLOAD_DIR, exist_ok=True)
                max_memory = _build_max_memory(s)
                logger.info(f"Inferring device_map with max_memory={max_memory} ...")
                # Determine no-split classes if present (optional)
                no_split = []
                try:
                    # Many HF models expose list of class names to avoid splitting
                    no_split = list(getattr(model, "_no_split_modules", []))
                except Exception:
                    pass

                inferred_device_map = infer_auto_device_map(
                    model,
                    dtype=dtype if dtype != "auto" else None,
                    max_memory=max_memory,
                    no_split_module_classes=no_split if no_split else None,
                )
                logger.info(f"Dispatching model with device_map across {len([k for k in inferred_device_map.values() if 'cuda' in k])} GPUs.")
                model = dispatch_model(model, device_map=inferred_device_map, offload_dir=s.OFFLOAD_DIR)
            except Exception as e:
                logger.warning(
                    "Automatic device sharding failed; model remains on CPU: "
                    + str(e)
                )

        model.eval()

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=s.TRUST_REMOTE_CODE,
        )

        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipeline, settings=s)
