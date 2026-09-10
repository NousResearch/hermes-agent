# copied from setuptools.logging, omitting monkeypatching
from __future__ import annotations

import logging
import sys


def _not_warning(record: logging.LogRecord) -> bool:
    return record.levelno < logging.WARNING


def configure() -> None:
    """
    Configure logging to emit warning and above to stderr
    and everything else to stdout. This behavior is provided
    for compatibility with distutils.log but may change in
    the future.
    """
    err_handler = logging.StreamHandler()
    err_handler.setLevel(logging.WARNING)
    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.addFilter(_not_warning)
    handlers = err_handler, out_handler
    logging.basicConfig(
        format="{message}", style="{", handlers=handlers, level=logging.DEBUG
    )
