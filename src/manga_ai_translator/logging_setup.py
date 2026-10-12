"""Logging to the console (rich) and to a rotating log file."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAME = "manga_ai_translator"
LOG_FILENAME = "mangat.log"
_MAX_LOG_BYTES = 2_000_000
_LOG_BACKUPS = 3


def setup_logging(level: str, logs_dir: Path | None) -> Path | None:
    """Configure the package logger and return the log file path, if file logging is active.

    The console shows messages at ``level``; the log file always records DEBUG so problems
    on another machine can be diagnosed from the file alone.
    """
    reset_logging()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console_handler = RichHandler(
        console=Console(stderr=True), show_path=False, markup=False, rich_tracebacks=True
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if logs_dir is None:
        return None
    log_file = logs_dir / LOG_FILENAME
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=_MAX_LOG_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("File logging disabled, cannot write to %s: %s", logs_dir, exc)
        return None
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)
    return log_file


def reset_logging() -> None:
    """Remove and close every handler installed by :func:`setup_logging`."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
