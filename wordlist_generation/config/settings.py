import os
import tempfile
from typing import List, Tuple
from dotenv import load_dotenv

# Load .env early
load_dotenv()


def _parse_int_tuple(values: str) -> Tuple[int, ...]:
    try:
        return tuple(sorted({int(x.strip()) for x in values.split(",") if x.strip()}))
    except Exception:
        return (64, 128, 256, 512)


class Settings:
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    MODEL_NAME: str = os.getenv("MODEL_NAME", "zai-org/GLM-4.6")
    DEVICE_MAP: str = os.getenv("DEVICE_MAP", "auto")
    TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
    SECRET_TOKEN: str = os.getenv("SECRET_TOKEN", "changeme")
    ATTN_IMPLEMENTATION: str = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2").strip()
    TOKENIZER_PADDING_SIDE: str = os.getenv("TOKENIZER_PADDING_SIDE", "left").strip()
    PAD_TO_MULTIPLE_OF: int = int(os.getenv("PAD_TO_MULTIPLE_OF", "64"))
    MAX_INPUT_TOKENS: int = int(os.getenv("MAX_INPUT_TOKENS", "512"))
    ALLOWED_MAX_NEW_TOKENS: Tuple[int, ...] = _parse_int_tuple(
        os.getenv("ALLOWED_MAX_NEW_TOKENS", "64,128,256,512")
    )
    STATIC_KV_CACHE: bool = os.getenv("STATIC_KV_CACHE", "false").lower() == "true"

    GENERATION_MAX_CONCURRENCY: int = int(os.getenv("GENERATION_MAX_CONCURRENCY", "1"))

    # Grouped GEMM / MoE optimization switch:
    # - True: expect an offline fused checkpoint; CPU-first load; FSDP2 shards across GPUs if launched with torchrun.
    # - False: standard device_map path (e.g., 'auto'); no grouped_gemm or FSDP2.
    USE_GROUPED_GEMM: bool = os.getenv("USE_GROUPED_GEMM", "false").lower() == "true"

    PREBUILD_PREFIX: bool = os.getenv("PREBUILD_PREFIX", "true").lower() == "true"
    PREBUILD_WORD_COUNTS: Tuple[int, ...] = tuple(
        int(x) for x in os.getenv("PREBUILD_WORD_COUNTS", "3000").split(",")
    )
    PREBUILD_LANGS: List[str] = [x.strip() for x in os.getenv("PREBUILD_LANGS", "").split(",") if x.strip()]
    WORDLIST_DIR: str = os.getenv("WORDLIST_DIR", "wordlists")

    TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "auto")

    BATCH_JOB_TEMP_DIR: str = os.getenv("BATCH_JOB_TEMP_DIR", tempfile.gettempdir())
    BATCH_JOB_PIPELINE_SIZE: int = int(os.getenv("BATCH_JOB_PIPELINE_SIZE", "8"))
