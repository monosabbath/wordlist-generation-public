import logging
import os
from typing import Any, Iterable, List, Optional

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline

logger = logging.getLogger("model-service")


class GPUConcurrencyGate:
    """
    Process-local gate to avoid overlapping GPU-bound generation work.
    Still useful per-rank with FSDP2.
    """
    def __init__(self, max_concurrency: int = 1):
        import threading
        self._sem = threading.Semaphore(max_concurrency)

    def __enter__(self):
        self._sem.acquire()

    def __exit__(self, exc_type, exc, tb):
        self._sem.release()


def _get_local_device() -> torch.device:
    if torch.cuda.is_available():
        # Prefer torch.distributed LOCAL_RANK if set
        lr = os.getenv("LOCAL_RANK")
        if lr is not None and lr.isdigit():
            return torch.device(f"cuda:{int(lr)}")
        # Else fall back to current
        return torch.device(torch.cuda.current_device())
    return torch.device("cpu")


def _iter_named_modules(m) -> Iterable[tuple[str, torch.nn.Module]]:
    for name, mod in m.named_modules():
        yield name, mod


def _discover_transformer_layers(model) -> List[torch.nn.Module]:
    """
    Best-effort discovery of "block" layers to apply fully_shard to before the root.
    Tries common container paths, then falls back to a size-based heuristic.
    """
    candidates: List[torch.nn.Module] = []

    # Try common attributes: model.model.layers / model.transformer.layers / model.layers / layers
    try_paths = [
        "model.layers",
        "transformer.layers",
        "backbone.layers",
        "layers",
    ]
    for p in try_paths:
        try:
            obj = model
            for part in p.split("."):
                obj = getattr(obj, part)
            # If it's an iterable/modulelist of blocks, return those
            if isinstance(obj, (torch.nn.ModuleList, list, tuple)):
                cs = [m for m in obj if isinstance(m, torch.nn.Module)]
                if cs:
                    return cs
        except Exception:
            pass

    # Fallback: size-based heuristic for large child modules (transformer blocks)
    for name, mod in model.named_children():
        try:
            nparams = sum(p.numel() for p in mod.parameters())
            if nparams >= 5_000_000:  # heuristic threshold
                candidates.append(mod)
        except Exception:
            continue

    return candidates


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

        base_init: dict[str, Any] = {
            "trust_remote_code": s.TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": True,
            "local_files_only": False,
        }
        if dtype != "auto":
            base_init["torch_dtype"] = dtype

        use_grouped_gemm = bool(getattr(s, "USE_GROUPED_GEMM", False))

        # Load config
        config: Optional[Any] = None
        try:
            config = AutoConfig.from_pretrained(
                s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load config first: {e}")

        # For fused checkpoints, enable grouped_gemm kernels if config supports it
        if config and use_grouped_gemm and hasattr(config, "use_grouped_gemm"):
            try:
                config.use_grouped_gemm = True
                logger.info("Set config.use_grouped_gemm=True (expect fused checkpoint).")
            except Exception as e:
                logger.warning(f"Could not set use_grouped_gemm on config: {e}")

        # Device map policy:
        # - USE_GROUPED_GEMM: CPU-first load (device_map=None); we fully_shard (FSDP2) after load.
        # - Otherwise: respect DEVICE_MAP (defaults to 'auto') for standard flow.
        device_map_to_use = None if use_grouped_gemm else s.DEVICE_MAP

        logger.info(
            f"Loading model '{s.MODEL_NAME}' (trust_remote_code={s.TRUST_REMOTE_CODE}, "
            f"attn='{s.ATTN_IMPLEMENTATION}', device_map='{device_map_to_use}', "
            f"use_grouped_gemm={use_grouped_gemm})"
        )

        # Load model
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

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
        )
        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Optional: static KV cache
        if s.STATIC_KV_CACHE:
            try:
                model.generation_config.cache_implementation = "static"
                logger.info("Enabled static KV cache.")
            except Exception as e:
                logger.warning(f"Static KV cache not available: {e}")

        # FSDP2 fully_shard path when using grouped GEMM
        if use_grouped_gemm:
            try:
                # Require PyTorch with FSDP2
                from torch.distributed.fsdp import fully_shard  # FSDP2 API
                # Optional mixed precision for inference could be added with MixedPrecisionPolicy if desired.

                # Select device per rank
                device = _get_local_device()

                # Shard transformer blocks first, then shard the root
                layers = _discover_transformer_layers(model)
                if layers:
                    for i, layer in enumerate(layers):
                        fully_shard(layer)  # moves shards to device mesh (rank device)
                    logger.info(f"Applied fully_shard to {len(layers)} submodules.")
                else:
                    logger.warning("Could not auto-discover transformer layers; applying fully_shard to root only.")

                # Root wrap
                fully_shard(model)
                logger.info("Applied fully_shard to root model.")
            except ImportError as e:
                logger.error(
                    "FSDP2 (fully_shard) is not available in this PyTorch. "
                    "Upgrade to a version that provides torch.distributed.fsdp.fully_shard."
                )
                # Fallback: single-device
                if torch.cuda.is_available():
                    model.to("cuda:0")
            except Exception as e:
                logger.error(f"FSDP2 setup failed; falling back to single-device if possible: {e}")
                if torch.cuda.is_available():
                    model.to("cuda:0")

        model.eval()

        # Build generation pipeline.
        # For FSDP2, pass the per-rank device so pipeline places inputs correctly.
        if use_grouped_gemm:
            pipe_device = _get_local_device()
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=s.TRUST_REMOTE_CODE,
                device=pipe_device,
            )
        else:
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=s.TRUST_REMOTE_CODE,
            )

        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipeline, settings=s)
