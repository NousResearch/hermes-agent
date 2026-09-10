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
        """Submit a callable, propagating the caller's contextvars. Stdlib only does
        this from 3.14; on 3.11-3.13 a bare worker starts with an EMPTY Context and
        drops profile secret scope / HERMES_HOME override — under the multiplexed
        gateway a credential read then fails closed with ``UnscopedSecretError``.
        Unconditional: on 3.14+ ``ctx.run`` re-applies the same context (no-op)."""
        ctx = copy_context()

        def _run_with_context(*call_args, **call_kwargs):
            return ctx.run(fn, *call_args, **call_kwargs)
        return super().submit(_run_with_context, *args, **kwargs)

    def _adjust_thread_count(self) -> None:
        # Mirrors CPython's implementation with two changes:
        # daemon=True and no _threads_queues registration.
        # Supports both CPython 3.8-3.13 (initializer/initargs) and
        # 3.14+ (WorkerContext via _create_worker_context) so the pool
        # works even when the entry-point is launched with a system
        # Python 3.14 (e.g. shebang resolving to /usr/bin/python).
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)
        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            # Python 3.14 removed _initializer/_initargs in favour of
            # WorkerContext (_create_worker_context/_resolve_work_item_task)
            # and changed _worker signature to (executor_ref, ctx, queue).
            if hasattr(self, "_create_worker_context"):  # pyright: ignore[reportAttributeAccessIssue]
                t = threading.Thread(
                    name=thread_name, target=_worker, daemon=True,
                    args=(weakref.ref(self, weakref_cb), self._create_worker_context(), self._work_queue),  # pyright: ignore[reportAttributeAccessIssue]
                )
            else:
                t = threading.Thread(
                    name=thread_name, target=_worker, daemon=True,
                    args=(weakref.ref(self, weakref_cb), self._work_queue, self._initializer, self._initargs),
                )
            t.start()
            self._threads.add(t)
