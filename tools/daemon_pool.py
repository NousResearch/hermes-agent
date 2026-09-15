"""Shared daemon-thread ThreadPoolExecutor.

Stdlib workers are non-daemon AND registered in ``_threads_queues``, whose atexit
hook joins every worker even after ``shutdown(wait=False)`` — one wedged worker
(tool blocked on network I/O, hung provider, stuck subagent) blocks interpreter
exit forever. This variant spawns daemon workers and skips that registration.
Use it for best-effort/interruptible work that must never hold the process open;
NOT for work that must complete before exit (durable writes belong on foreground
threads with explicit bounded joins).

Compatibility note (Python 3.13+/3.14): stdlib 3.14 creates ``_initializer``/
``_initargs`` lazily inside its own ``_adjust_thread_count`` (which this class
overrides — so on 3.14 they never come into existence) and changed ``_worker()``
to 3 positional args with a ``WorkerContext``. Capture both attrs in
``__init__`` and build worker args for whichever signature the running stdlib
uses; reading ``self._initializer`` unconditionally in the override crashes the
first ``submit()`` on 3.14 with AttributeError, and the 4-arg call crashes the
worker thread with TypeError.
"""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker
from contextvars import copy_context

try:
    # Python 3.14: initializer/initargs travel via WorkerContext
    from concurrent.futures.thread import WorkerContext as _StdlibWorkerContext
except ImportError:  # pragma: no cover — Python <= 3.13
    _StdlibWorkerContext = None

__all__ = ["DaemonThreadPoolExecutor"]


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def __init__(self, max_workers=None, thread_name_prefix=None,
                 initializer=None, initargs=()):
        super().__init__(max_workers=max_workers,
                         thread_name_prefix=thread_name_prefix)
        # Python 3.14 lazily creates _initializer/_initargs in the stdlib's own
        # _adjust_thread_count; this class overrides that method, so capture
        # them here to keep them available regardless of stdlib version.
        self._initializer = initializer
        self._initargs = initargs

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
            # Python 3.14 expects _worker(executor_ref, ctx, work_queue).
            # Older stdlib expected _worker(executor_ref, work_queue, initializer, initargs).
            if _StdlibWorkerContext is not None:
                worker_args = (
                    weakref.ref(self, weakref_cb),
                    _StdlibWorkerContext(self._initializer, self._initargs),
                    self._work_queue,
                )
            else:
                worker_args = (
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                )
            # Carry the active profile into the review thread so MEMORY.md / skill review writes land in the
            # right profile (#54937).
            t = threading.Thread(
                name=thread_name, target=_worker, daemon=True,
                args=worker_args,
            )
            t.start()
            self._threads.add(t)
