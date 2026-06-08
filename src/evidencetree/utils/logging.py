"""Unified logger setup (console + optional file)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str = "evidencetree",
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a configured logger.

    Idempotent: calling twice with the same name does not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_evidencetree_configured", False):
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    logger._evidencetree_configured = True  # type: ignore[attr-defined]
    return logger
