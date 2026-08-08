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

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker

__all__ = ["DaemonThreadPoolExecutor"]


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def _adjust_thread_count(self) -> None:
        # Mirrors CPython's ThreadPoolExecutor._adjust_thread_count but:
        #  - daemon=True so workers don't block process exit
        #  - No _threads_queues registration (avoids non-daemon atexit joins)
        #
        # Python 3.14+ changed _worker's signature to take a context object
        # created via _create_worker_context(). Earlier versions used
        # _worker(executor_reference, work_queue, initializer, initargs).
        # We adapt to whichever signature the running interpreter uses.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (
                self._thread_name_prefix or self,
                num_threads,
            )
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=_daemon_worker_args(self, weakref_cb),
                daemon=True,
            )
            t.start()
            self._threads.add(t)


def _daemon_worker_args(executor, weakref_cb):
    """Build the args tuple for _worker, compatible across Python versions.

    Python 3.14+: _worker(executor_reference, ctx, work_queue)
    Python 3.8–3.13: _worker(executor_reference, work_queue, initializer, initargs)
    """
    if hasattr(executor, "_create_worker_context"):
        # Python 3.14+
        ctx = executor._create_worker_context()
        return (
            weakref.ref(executor, weakref_cb),
            ctx,
            executor._work_queue,
        )
    else:
        # Python 3.8–3.13
        initializer = getattr(executor, "_initializer", None)
        initargs = getattr(executor, "_initargs", ())
        return (
            weakref.ref(executor, weakref_cb),
            executor._work_queue,
            initializer,
            initargs,
        )
