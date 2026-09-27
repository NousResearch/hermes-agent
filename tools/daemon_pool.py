"""Shared daemon-thread ThreadPoolExecutor.

Stdlib workers are non-daemon AND registered in ``_threads_queues``, whose atexit
hook joins every worker even after ``shutdown(wait=False)`` — one wedged worker
(tool blocked on network I/O, hung provider, stuck subagent) blocks interpreter
exit forever. This variant spawns daemon workers and skips that registration.
Use it for best-effort/interruptible work that must never hold the process open;
NOT for work that must complete before exit (durable writes belong on foreground
threads with explicit bounded joins).
"""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker
from contextvars import copy_context

__all__ = ["DaemonThreadPoolExecutor"]


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def submit(self, fn, /, *args, **kwargs):
        """Keep each task in its caller's profile scope, even on a reused worker.

        Thread-start context cannot track later submissions from other profiles
        (#54937). The stdlib worker context manages initialization, not contextvars.
        """
        ctx = copy_context()

        def _run_with_context(*call_args, **call_kwargs):
            return ctx.run(fn, *call_args, **call_kwargs)
        return super().submit(_run_with_context, *args, **kwargs)

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
            executor_ref = weakref.ref(self, weakref_cb)
            if hasattr(self, "_create_worker_context"):
                # Python 3.14 replaced _initializer/_initargs with a factory
                # that supplies the worker's initializer context.
                worker_args = (
                    executor_ref,
                    self._create_worker_context(),
                    self._work_queue,
                )
            else:
                worker_args = (
                    executor_ref,
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                )
            t = threading.Thread(
                name=thread_name, target=_worker, daemon=True,
                args=worker_args,
            )
            t.start()
            self._threads.add(t)
