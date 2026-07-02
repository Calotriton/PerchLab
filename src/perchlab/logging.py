"""Centralized structured logging for PerchLab.

Library code never calls :func:`print`; it logs through the ``"perchlab"`` logger
configured here. The console handler is human-friendly (via :mod:`rich`); an
optional file handler writes structured records (timestamp, level, module) for
reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAME = "perchlab"

_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s:%(module)s:%(lineno)d %(message)s"

# Shared console so progress bars and log records target the same stream.
console = Console(stderr=True)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Configure and return the root PerchLab logger.

    Idempotent: repeated calls replace existing handlers rather than stacking
    them, so calling this from both the CLI and tests is safe.

    Args:
        level: Logging level for the console handler (name or numeric).
        log_file: Optional path; when given, structured logs are also written
            there at DEBUG level regardless of the console level.

    Returns:
        The configured ``"perchlab"`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        markup=False,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the PerchLab logger.

    Args:
        name: Optional dotted suffix (e.g. ``"classify"``). When omitted the
            root PerchLab logger is returned.

    Returns:
        A :class:`logging.Logger` under the ``"perchlab"`` namespace.
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
