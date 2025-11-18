import logging
import os
from typing import Any, List, Dict

import torch
import torch.distributed as dist
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
    def __init__(self, model, tokenizer, text_pipeline, settings, is_fsdp: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.text_pipeline = text_pipeline  # None when FSDP/grouped_gemm path
        self.settings = settings
        self.is_fsdp = is_fsdp
        self.gpu_gate = GPUConcurrencyGate(max_concurrency=int(getattr(settings, "GENERATION_MAX_CONCURRENCY", 1)))

    @property
    def is_cohere_reasoning_model(self) -> bool:
        return self.settings.MODEL_NAME.strip().lower() == "coherelabs/command-a-reasoning-08-2025"

    def _local_device(self) -> torch.device:
        if self.is_fsdp and dist.is_initialized():
            local_rank = int(os.getenv("LOCAL_RANK", dist.get_rank()))
            return torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def generate_prompts(
        self,
        prompts: List[str],
        generation_kwargs: Dict[str, Any],
        pad_to_multiple_of: int,
        max_length: int,
    ) -> List[str]:
        """
        Direct generate path for FSDP-wrapped or pipeline-unavailable models.
        All ranks must participate if FSDP is used (see NOTE below).
        """
        device = self._local_device()
        tok = self.tokenizer
        model = self.model

        # NOTE: For proper FSDP2 inference every rank should run this forward.
        # Current design only invokes generate on rank 0 (others idle) which can hang for FSDP2.
        # You must adapt orchestration later; this function assumes either:
        #   (a) FSDP1 still working with single-rank forward, or
        #   (b) You call it on all ranks in lock-step.
        out_texts: List[str] = []
        for batch_start in range(0, len(prompts), self.settings.BATCH_JOB_PIPELINE_SIZE):
            sub = prompts[batch_start: batch_start + self.settings.BATCH_JOB_PIPELINE_SIZE]
            enc = tok(
                sub,
                return_tensors="pt",
                padding=True,
                truncation=True,
                pad_to_multiple_of=pad_to_multiple_of,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                gen_out = model.generate(**enc, **generation_kwargs)
            for i, ids in enumerate(gen_out):
                prompt_len = enc["input_ids"][i].shape[0]
                # Use original tokenized prompt length for slicing
                # (If model.generate returns concatenated prompt+completion)
                completion = ids[enc["input_ids"][i].shape[0]:]
                out_texts.append(tok.decode(completion, skip_special_tokens=True))
        return out_texts

    @classmethod
    def from_settings(cls, s):
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

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

        config = None
        try:
            config = AutoConfig.from_pretrained(
                s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load config first: {e}")

        if config and use_grouped_gemm and hasattr(config, "use_grouped_gemm"):
            try:
                config.use_grouped_gemm = True
                logger.info("Set config.use_grouped_gemm=True (expect fused checkpoint).")
            except Exception as e:
                logger.warning(f"Could not set use_grouped_gemm: {e}")

        device_map_to_use = None if use_grouped_gemm else s.DEVICE_MAP
        logger.info(
            f"Loading model '{s.MODEL_NAME}' (device_map={device_map_to_use}, grouped_gemm={use_grouped_gemm})"
        )

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
            logger.warning("Retrying load without attn_implementation.")
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
                logger.warning(f"Could not set attention implementation: {e}")

        tokenizer = AutoTokenizer.from_pretrained(
            s.MODEL_NAME, trust_remote_code=s.TRUST_REMOTE_CODE, local_files_only=False
        )
        tokenizer.padding_side = s.TOKENIZER_PADDING_SIDE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # FSDP wrap (current FSDP v1 path). If migrating to fully_shard (FSDP2),
        # replace this block with fully_shard calls. We mark is_fsdp True to skip pipeline.
        is_fsdp = False
        if use_grouped_gemm:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                try:
                    local_rank = int(os.getenv("LOCAL_RANK", dist.get_rank()))
                    torch.cuda.set_device(local_rank)
                    device = torch.device(f"cuda:{local_rank}")
                    model.to(device)
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                    auto_wrap = size_based_auto_wrap_policy(min_num_params=5_000_000)
                    model = FSDP(model, auto_wrap_policy=auto_wrap, device_id=device)
                    is_fsdp = True
                    logger.info(f"Wrapped model with FSDP on rank {dist.get_rank()}.")
                except Exception as e:
                    logger.error(f"FSDP setup failed; using single GPU if available: {e}")
                    if torch.cuda.is_available():
                        model.to("cuda:0")
            else:
                if torch.cuda.is_available():
                    model.to("cuda:0")
                is_fsdp = True  # treat as special path even if single-device fallback

        model.eval()

        # Try pipeline only if not FSDP path
        text_pipe = None
        if not is_fsdp:
            try:
                text_pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=s.TRUST_REMOTE_CODE,
                )
            except Exception as e:
                logger.warning(f"Pipeline unavailable, will use direct generate: {e}")
                text_pipe = None
        else:
            logger.info("Skipping pipeline construction (FSDP path). Using direct generate.")

        return cls(model=model, tokenizer=tokenizer, text_pipeline=text_pipe, settings=s, is_fsdp=is_fsdp)
