import os
import time
import logging
from contextlib import asynccontextmanager

import torch
import torch.distributed as dist
from fastapi import FastAPI

from wordlist_generation.config.settings import Settings
from wordlist_generation.config.logging import configure_logging
from wordlist_generation.services.model_service import ModelService
from wordlist_generation.services.batch_processor import BatchProcessor
from wordlist_generation.api.routers import chat, batch
from wordlist_generation.inference.vocab_constraints.prefix import build_regexp_prefix_fn


def init_dist_if_needed(settings: Settings, backend: str = "nccl"):
    """
    Initialize torch.distributed if:
      - USE_GROUPED_GEMM is true, and
      - Distributed env vars (RANK/WORLD_SIZE) are present, and
      - Not already initialized.
    """
    if not getattr(settings, "USE_GROUPED_GEMM", False):
        return
    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        logger = logging.getLogger("dist")
        logger.info(
            f"Initialized process group: rank={dist.get_rank()} "
            f"world_size={dist.get_world_size()} local_rank={local_rank}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load settings and logging
    settings = Settings()
    configure_logging(settings)

    # Initialize distributed (no-op unless properly launched with torchrun)
    try:
        init_dist_if_needed(settings, backend="nccl")
    except Exception as e:
        logging.getLogger("dist").warning(f"Distributed init skipped/failed: {e}")

    # Initialize model service (loads model/tokenizer/pipeline; applies FSDP2 when enabled)
    model_service = ModelService.from_settings(settings)

    # Initialize batch processor service
    batch_processor = BatchProcessor(settings=settings, model_service=model_service)

    # Store on app.state
    app.state.settings = settings
    app.state.model_service = model_service
    app.state.batch_processor = batch_processor

    # Optional: prebuild regex prefix functions for constrained vocab
    if settings.PREBUILD_PREFIX and settings.PREBUILD_LANGS:
        logger = logging.getLogger("prebuild")
        logger.info("Prebuilding prefix functions...")
        for lang in settings.PREBUILD_LANGS:
            for n in settings.PREBUILD_WORD_COUNTS:
                try:
                    build_regexp_prefix_fn(
                        tokenizer=model_service.tokenizer,
                        lang=lang,
                        n_words=n,
                        wordlist_dir=settings.WORDLIST_DIR,
                        allow_cohere_controls=model_service.is_cohere_reasoning_model,
                    )
                except Exception as e:
                    logger.warning(f"Prebuild failed for {lang} n={n}: {e}")
        logger.info("Prebuild done.")

    yield
    # Optional: cleanup resources here if needed


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan, title="Wordlist-Constrained Generation")
    app.include_router(chat.router)
    app.include_router(batch.router)
    return app


app = create_app()


if __name__ == "__main__":
    # Integrated runner to support multi-GPU via FSDP2 inside the main app.
    # Launch with:
    #   torchrun --nproc_per_node=<GPUS> -m wordlist_generation.app.main --host 0.0.0.0 --port 8000
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", default="nccl")
    args = ap.parse_args()

    settings = Settings()
    configure_logging(settings)

    # Initialize distributed if needed
    try:
        init_dist_if_needed(settings, backend=args.backend)
    except Exception as e:
        logging.getLogger("dist").warning(f"Distributed init skipped/failed: {e}")

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Keep nonzero ranks alive to back FSDP; rank 0 hosts the API
        while True:
            time.sleep(60)
