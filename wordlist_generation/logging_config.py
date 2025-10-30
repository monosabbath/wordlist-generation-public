import logging


def configure_logging(settings) -> None:
    level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
    logging.basicConfig(level=level)
    logging.getLogger("transformers").setLevel(logging.WARNING)
