from contextlib import asynccontextmanager
from fastapi import FastAPI
from .settings import Settings
from .logging_config import configure_logging
from .services.model_service import ModelService
from .routers import chat, batch
from .core.prefix import build_regexp_prefix_fn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load settings and logging
    settings = Settings()
    configure_logging(settings)

    # Initialize model service (loads model/tokenizer/pipeline)
    model_service = ModelService.from_settings(settings)

    # Store on app.state
    app.state.settings = settings
    app.state.model_service = model_service

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
