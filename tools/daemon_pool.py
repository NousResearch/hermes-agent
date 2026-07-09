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
        # Mirrors CPython's implementation with two changes: daemon=True and no
        # _threads_queues registration.  CPython 3.14 changed the private
        # worker signature from (executor_ref, queue, initializer, initargs) to
        # (executor_ref, worker_context, queue); build the args at runtime so the
        # daemon pool works across both shapes.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            worker_args = self._worker_args(weakref_cb)
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=worker_args,
                daemon=True,
            )
            t.start()
            self._threads.add(t)

    def _worker_args(self, weakref_cb):
        """Return args for concurrent.futures.thread._worker.

        Python 3.14+ uses ``(executor_ref, worker_context, work_queue)`` while
        3.8–3.13 use ``(executor_ref, work_queue, initializer, initargs)``.
        Avoid version checks; the executor instance exposes the new context
        factory only on runtimes that need it.
        """
        executor_ref = weakref.ref(self, weakref_cb)
        create_context = getattr(self, "_create_worker_context", None)
        if callable(create_context):
            return (executor_ref, create_context(), self._work_queue)
        return (
            executor_ref,
            self._work_queue,
            getattr(self, "_initializer", None),
            getattr(self, "_initargs", ()),
        )
