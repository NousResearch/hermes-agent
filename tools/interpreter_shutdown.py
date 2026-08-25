"""Shared interpreter-shutdown detection.

Single home for the "is the Python interpreter finalizing?" predicate used
by every subsystem whose background threads can outlive process teardown
(cron delivery, concurrent tool submission, the conversation loop's retry
path, background review forks).

Once finalization starts, ``concurrent.futures`` refuses new work with
``RuntimeError: cannot schedule new futures after interpreter shutdown`` and
asyncio's default executor is gone — *any* further attempt to schedule work
(an API retry, a thread-pool submit, ``asyncio.run``) is doomed and only
produces noise: stray ``❌`` prints after the TUI exited, tracebacks in
``errors.log``, and futile retry loops that burn iterations against a dying
process (#55924, #58720, and the CLI-exit retry spam this module was
extracted for).

CPython emits two message variants depending on the failing site:

- ``cannot schedule new futures after interpreter shutdown`` — the
  module-global finalization flag (asyncio.run_coroutine_threadsafe, a
  torn-down default executor, ThreadPoolExecutor.submit during teardown).
- ``cannot schedule new futures after shutdown`` — a plain
  ``ThreadPoolExecutor`` whose ``shutdown()`` ran.

The common short prefix catches both. Matching the second variant is safe
for shutdown detection at every current call site: the pools involved are
either module-global daemons or ``with``-scoped locals that cannot be shut
down mid-use by anything except interpreter finalization.

Historically this predicate existed at three sites, each fixed
independently as its own incident — ``cron/scheduler.py`` (#55924/#58720),
``agent/tool_executor.py``, and nothing at all in the conversation loop's
outer retry handler (the CLI-exit spam). One predicate, all sites.
"""

from __future__ import annotations

import sys
from typing import Optional

_SHUTDOWN_SUBMIT_ERROR_PREFIX = "cannot schedule new futures"


def interpreter_shutting_down(exc: Optional[BaseException] = None) -> bool:
    """Return True when the Python interpreter is finalizing.

    ``exc`` lets a caller also treat an already-raised scheduling error as a
    shutdown signal: the ``concurrent.futures`` module-global flag can be set
    a hair before ``sys.is_finalizing()`` flips, so matching the error text
    is a safe fallback for that race.
    """
    if sys.is_finalizing():
        return True
    if exc is not None:
        return _SHUTDOWN_SUBMIT_ERROR_PREFIX in str(exc).lower()
    return False
