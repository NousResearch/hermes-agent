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
"""

from __future__ import annotations

import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker

__all__ = ["DaemonThreadPoolExecutor"]

# Python 3.14 replaced ThreadPoolExecutor._initializer/_initargs with a
# prepare_context() → (_create_worker_context, _resolve_work_item_task)
# pattern, and changed _worker's signature from
#   _worker(executor_ref, work_queue, initializer, initargs)   # 3.8–3.13
# to
#   _worker(executor_ref, ctx, work_queue)                     # 3.14+
# The override below must branch on the running interpreter version.
_PY314 = sys.version_info >= (3, 14)


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Belt-and-suspenders: on 3.14+ the stdlib __init__ no longer sets
        # _initializer/_initargs.  Ensure they exist so any external code
        # that reads them (or an older code path) doesn't AttributeError.
        if not hasattr(self, "_initializer"):
            self._initializer = None
        if not hasattr(self, "_initargs"):
            self._initargs = ()

    def _adjust_thread_count(self) -> None:
        # Mirrors CPython's implementation with two changes:
        # daemon=True and no _threads_queues registration.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            if _PY314:
                # Python 3.14+: _worker(executor_ref, ctx, work_queue)
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
                # Python 3.8–3.13: _worker(executor_ref, work_queue, initializer, initargs)
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
