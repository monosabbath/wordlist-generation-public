from contextlib import asynccontextmanager
from fastapi import FastAPI

from wordlist_generation.config.settings import Settings
from wordlist_generation.config.logging import configure_logging
from wordlist_generation.services.model_service import ModelService
from wordlist_generation.services.batch_processor import BatchProcessor
from wordlist_generation.api.routers import chat, batch
from wordlist_generation.inference.vocab_constraints.prefix import build_regexp_prefix_fn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load settings and logging
    settings = Settings()
    configure_logging(settings)

    # Initialize model service (loads model/tokenizer/pipeline)
    model_service = ModelService.from_settings(settings)

    # Initialize batch processor service
    batch_processor = BatchProcessor(settings=settings, model_service=model_service)

    # Store on app.state
    app.state.settings = settings
    app.state.model_service = model_service
    app.state.batch_processor = batch_processor

    # Optional: prebuild regex prefix functions for constrained vocab
    if settings.PREBUILD_PREFIX and settings.PREBUILD_LANGS:
        from logging import getLogger

        logger = getLogger("prebuild")
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
