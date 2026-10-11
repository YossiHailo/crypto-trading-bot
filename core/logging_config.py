"""Console plus rotating file logging.

A ``RedactingFilter`` runs on both handlers, so registered secrets and
token-looking URL parameters are masked even inside exception tracebacks.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.redact import RedactingFilter


def setup_logging(level: str, log_dir: str) -> Path:
    """Configure the root logger. Returns the log file path."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "trading-bot.log"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level.upper())
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    rotating = RotatingFileHandler(
        log_path,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    rotating.setFormatter(formatter)
    redactor = RedactingFilter()
    console.addFilter(redactor)
    rotating.addFilter(redactor)
    root.addHandler(console)
    root.addHandler(rotating)

    # Library loggers can drown the decision log on a 24/7 process.
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return log_path
