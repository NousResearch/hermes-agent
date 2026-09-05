"""Shared interpreter-shutdown predicate for background threads that can outlive
process teardown (cron delivery, concurrent tool submission, retry paths).

Once finalization starts, ``concurrent.futures`` refuses new work and asyncio's
default executor is gone, so any further scheduling only produces noise (stray
prints after the TUI exited, tracebacks in ``errors.log``, futile retries).
CPython raises ``cannot schedule new futures after interpreter shutdown``
(module-global flag) or ``... after shutdown`` (a pool whose ``shutdown()`` ran);
the short prefix catches both — safe here because every pool involved is a
module-global daemon or a ``with``-scoped local only finalization can shut down.
"""

from __future__ import annotations

import sys
from typing import Optional


def interpreter_shutting_down(exc: Optional[BaseException] = None) -> bool:
    """True when the interpreter is finalizing. ``exc`` lets a caller treat an
    already-raised scheduling error as a shutdown signal: the ``concurrent.futures``
    flag can be set a hair before ``sys.is_finalizing()`` flips."""
    if sys.is_finalizing():
        return True
    return exc is not None and "cannot schedule new futures" in str(exc).lower()
