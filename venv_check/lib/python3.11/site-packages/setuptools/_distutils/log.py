"""
A simple log mechanism styled after PEP 282.

Retained for compatibility and should not be used.
"""

import logging
import warnings

from ._log import log as _global_log

DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
FATAL = logging.FATAL

log = _global_log.log
debug = _global_log.debug
info = _global_log.info
warn = _global_log.warning
error = _global_log.error
fatal = _global_log.fatal


def set_threshold(level):
    orig = _global_log.level
    _global_log.setLevel(level)
    return orig


def set_verbosity(v):
    if v <= 0:
        set_threshold(logging.WARN)
    elif v == 1:
        set_threshold(logging.INFO)
    elif v >= 2:
        set_threshold(logging.DEBUG)


class Log(logging.Logger):
    """distutils.log.Log is deprecated, please use an alternative from `logging`."""

    def __init__(self, threshold=WARN):
        warnings.warn(Log.__doc__)  # avoid DeprecationWarning to ensure warn is shown
        super().__init__(__name__, level=threshold)

    @property
    def threshold(self):
        return self.level

    @threshold.setter
    def threshold(self, level):
        self.setLevel(level)

    warn = logging.Logger.warning
