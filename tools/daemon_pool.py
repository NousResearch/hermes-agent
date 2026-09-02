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
idle-thread reuse), plus context propagation: ``submit`` snapshots the
submitting context with ``copy_context()`` and runs each work item inside
it.  Stdlib ``ThreadPoolExecutor`` only does this from Python 3.14; on the
3.11-3.13 runtimes Hermes ships, a bare pool worker starts with an EMPTY
Context and silently drops contextvar-based state (profile secret scope,
HERMES_HOME override) — under the multiplexed gateway a credential read in
such a worker fails closed with ``UnscopedSecretError``.  Propagating by
default makes every consumer safe even when it forgets
``propagate_context_to_thread``.  Use it for any pool whose work is
best-effort or independently interruptible and must never hold the process open:
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
from contextvars import copy_context

__all__ = ["DaemonThreadPoolExecutor"]


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def submit(self, fn, /, *args, **kwargs):
        """Submit a callable, propagating the caller's contextvars.

        Python 3.14's ``ThreadPoolExecutor`` snapshots the submitting
        context with ``copy_context()`` and runs each work item inside it;
        3.11-3.13 (the runtimes Hermes ships) do not, so a pool worker
        starts with an empty Context and loses the multiplexed profile
        secret scope / HERMES_HOME override.  Do it here unconditionally so
        the daemon pool behaves identically on every runtime; on 3.14+ the
        inner ``ctx.run`` re-applies the same immutable context and is a
        no-op.
        """
        ctx = copy_context()

        def _run_with_context(*call_args, **call_kwargs):
            return ctx.run(fn, *call_args, **call_kwargs)

        return super().submit(_run_with_context, *args, **kwargs)

    def _adjust_thread_count(self) -> None:
        # Mirrors CPython's implementation (3.8–3.13) with two changes:
        # daemon=True and no _threads_queues registration.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
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
