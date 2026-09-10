import inspect
import logging
import sys

from . import monkey

import distutils.log


def _not_warning(record):
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
        format="{message}", style='{', handlers=handlers, level=logging.DEBUG
    )
    if inspect.ismodule(distutils.dist.log):
        monkey.patch_func(set_threshold, distutils.log, 'set_threshold')
        # For some reason `distutils.log` module is getting cached in `distutils.dist`
        # and then loaded again when patched,
        # implying: id(distutils.log) != id(distutils.dist.log).
        # Make sure the same module object is used everywhere:
        distutils.dist.log = distutils.log


def set_threshold(level: int) -> int:
    logging.root.setLevel(level * 10)
    return set_threshold.unpatched(level)
