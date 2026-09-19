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
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)
        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            executor_ref = weakref.ref(self, weakref_cb)
            create_context = self._worker_context_factory()
            if create_context is not None:
                # 3.14+ worker ABI: (executor_ref, ctx, work_queue).
                worker_args = (
                    executor_ref,
                    create_context(),
                    self._work_queue,
                )
            else:
                # 3.8-3.13 worker ABI: (executor_ref, work_queue, initializer, initargs).
                # Missing both fields (patched/frozen build) degrades to "no
                # initializer" instead of raising AttributeError on every spawn.
                worker_args = (
                    executor_ref,
                    self._work_queue,
                    getattr(self, "_initializer", None),
                    getattr(self, "_initargs", ()),
                )
            # Carry the active profile into the review thread so MEMORY.md / skill review writes land in the
            # right profile (#54937).
            t = threading.Thread(
                name=thread_name, target=_worker, daemon=True,
                args=worker_args,
            )
            t.start()
            self._threads.add(t)

    def _worker_context_factory(self):
        """Factory for the worker's initializer context, or None for the 3.8-3.13 ABI.

        Probed by attribute rather than ``sys.version_info`` so patched/frozen
        interpreters that only shuffle these private names keep working. 3.14
        dropped ``_initializer``/``_initargs``; if a build also drops the
        instance attribute it still ships the official ``prepare_context``.
        """
        factory = getattr(self, "_create_worker_context", None)
        if factory is not None:
            return factory
        prepare = getattr(type(self), "prepare_context", None)
        if prepare is None:
            return None
        try:
            return prepare(None, ())[0]
        except Exception:
            return None
