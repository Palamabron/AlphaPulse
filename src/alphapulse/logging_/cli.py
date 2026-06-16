import sys

from loguru import logger

_CONFIGURED = False


def configure_cli_logging(*, level: str = "INFO") -> None:
    """Send structured progress logs to stderr for CLI scripts."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        enqueue=True,
    )
    _CONFIGURED = True
