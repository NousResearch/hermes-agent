"""Shared daemon-thread ThreadPoolExecutor.

Stdlib ``ThreadPoolExecutor`` workers are non-daemon AND are registered in
``concurrent.futures.thread._threads_queues``, whose atexit hook
(``_python_exit``) joins every worker unconditionally — even after
``shutdown(wait=False)``.  A single wedged worker (tool blocked on network
I/O, hung provider daemon, stuck subagent) therefore blocks interpreter
exit forever.  This is the root cause of multi-minute CLI exits on long
sessions: every abandoned concurrent-tool batch leaves workers that the
exit hook insists on joining.

``DaemonThreadPoolExecutor`` spawns daemon workers and skips the
``_threads_queues`` registration, so:

  - ``_python_exit`` never joins them, and
  - the interpreter's non-daemon thread join at shutdown skips them.

Semantics are otherwise identical (initializer/initargs, work queue,
idle-thread reuse).  Use it for any pool whose work is best-effort or
independently interruptible and must never hold the process open:
concurrent tool execution, background memory sync, catalog fan-out,
subagent timeout wrappers.  Do NOT use it for work that must complete
before exit (durable writes) — those belong on foreground threads with
explicit bounded joins.

Python 3.14+ note
-----------------
CPython 3.14 changed ``ThreadPoolExecutor`` internals: ``_worker`` now
takes ``(executor_ref, ctx, work_queue)`` where ``ctx`` is a
``WorkerContext`` produced by ``self._create_worker_context()``, replacing
the old ``_worker(executor_ref, work_queue, initializer, initargs)``
signature and the ``self._initializer`` / ``self._initargs`` attributes
(they no longer exist on 3.14).  This module detects which layout the
active interpreter exposes and dispatches accordingly; the 3.8–3.13 path
is unchanged.
"""

from __future__ import annotations

import sys
import threading
import weakref
import inspect as _inspect

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker

__all__ = ["DaemonThreadPoolExecutor"]


# Python 3.14 refactored ThreadPoolExecutor: the per-executor initializer
# state moved from `self._initializer` / `self._initargs` into a WorkerContext
# produced by `self._create_worker_context()` (an instance attribute assigned
# in `__init__`), and the `_worker` target now takes `(ref, ctx, work_queue)`
# instead of the legacy `(ref, work_queue, initializer, initargs)`.
#
# Detect which layout the active interpreter exposes by inspecting `_worker`'s
# signature once at import time.  The check is robust against:
#   - instance-vs-class attribute pitfalls (3.11 sets `_initializer` only on
#     instances, so `hasattr(ThreadPoolExecutor, "_initializer")` is False);
#   - patched distributions that backport `_create_worker_context` without
#     switching the `_worker` signature;
#   - any future micro release that re-adds legacy attributes for compat.
# The `_worker` signature is the load-bearing contract.
try:
    _WORKER_PARAM_COUNT = len(_inspect.signature(_worker).parameters)
except (TypeError, ValueError):  # extremely defensive: built-in unavailable
    _WORKER_PARAM_COUNT = 4  # legacy 3.8–3.13 layout
_PY314_PLUS = _WORKER_PARAM_COUNT == 3


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def _adjust_thread_count(self) -> None:
        # Mirrors CPython's implementation with two changes:
        # daemon=True and no _threads_queues registration. Two code paths
        # cover Python 3.8–3.13 (legacy `_worker(ref, q, init, initargs)`)
        # and Python 3.14+ (`_worker(ref, ctx, q)` with WorkerContext).
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads >= self._max_workers:
            return

        thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)

        if _PY314_PLUS:
            # 3.14+: WorkerContext encapsulates initializer/initargs. The
            # ``weakref.ref`` callback wakes a parked worker; the ctx carries
            # per-executor init state. Worker is daemon so it won't block exit.
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._create_worker_context(),
                    self._work_queue,
                ),
                daemon=True,
            )
        else:
            # 3.8–3.13: initializer/initargs live on the executor instance.
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )

        t.start()
        self._threads.add(t)
