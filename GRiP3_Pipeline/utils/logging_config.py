"""Shared logging configuration for VL-GRiP3.

Call :func:`setup_logging` once from an entry point (``main.py`` /
``main_whisper.py`` / a training script). Library modules should only do
``logger = logging.getLogger(__name__)`` and never configure handlers
themselves, so the application stays in control of formatting and verbosity.
"""
from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging once.

    :param level: log level (int or name). Falls back to the ``VLGRIP3_LOGLEVEL``
        environment variable, then ``INFO``.
    """
    if level is None:
        level = os.environ.get("VLGRIP3_LOGLEVEL", "INFO")
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
