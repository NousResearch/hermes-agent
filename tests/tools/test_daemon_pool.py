"""Tests for tools.daemon_pool.DaemonThreadPoolExecutor.

The daemon pool exists so abandoned workers (interrupted/timed-out tool
batches, wedged memory-provider syncs) can never block interpreter exit:
stdlib ThreadPoolExecutor workers are non-daemon AND registered in
concurrent.futures.thread._threads_queues, whose atexit hook joins every
worker unconditionally — even after shutdown(wait=False).
"""

import inspect
import subprocess
import sys
import threading
import time

from concurrent.futures.thread import _threads_queues

import tools.daemon_pool as daemon_pool
from tools.daemon_pool import DaemonThreadPoolExecutor


def test_workers_are_daemon_threads():
    pool = DaemonThreadPoolExecutor(max_workers=2)
    try:
        info = pool.submit(
            lambda: (threading.current_thread().daemon, threading.current_thread())
        ).result(timeout=10)
        is_daemon, worker = info
        assert is_daemon is True
        # Not registered with concurrent.futures' atexit join hook.
        assert worker not in _threads_queues
    finally:
        pool.shutdown(wait=True)


def test_idle_worker_reuse():
    pool = DaemonThreadPoolExecutor(max_workers=4)
    try:
        tid1 = pool.submit(threading.get_ident).result(timeout=10)
        time.sleep(0.05)  # let the worker park on the idle semaphore
        tid2 = pool.submit(threading.get_ident).result(timeout=10)
        assert tid1 == tid2
    finally:
        pool.shutdown(wait=True)


def test_wedged_worker_does_not_block_interpreter_exit():
    """A worker stuck in a long sleep must not hold the process open.

    With stdlib ThreadPoolExecutor this subprocess hangs until the sleep
    finishes (the atexit hook joins the worker); with the daemon pool it
    exits as soon as the main thread returns.
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tools.daemon_pool import DaemonThreadPoolExecutor\n"
        "import time\n"
        "pool = DaemonThreadPoolExecutor(max_workers=1)\n"
        "pool.submit(time.sleep, 120)\n"
        "time.sleep(0.3)\n"
        "pool.shutdown(wait=False)\n"
        "print('main-done', flush=True)\n"
    ) % (str(_repo_root()),)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "main-done" in proc.stdout


def test_submit_propagates_caller_contextvars():
    """Pool workers inherit contextvars set in the submitting context.

    Stdlib ThreadPoolExecutor snapshots the caller's context with
    ``copy_context()``; some bundled CPython runtime builds strip that, so
    the daemon pool restores it explicitly.  Without the fix this returns
    the default because the worker runs in a bare context.
    """
    from contextvars import ContextVar

    var = ContextVar("daemon_pool_test_var", default="unset")

    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        token = var.set("hello")
        try:
            seen = pool.submit(var.get).result(timeout=10)
        finally:
            var.reset(token)
        assert seen == "hello"
    finally:
        pool.shutdown(wait=True)


def _capture_worker_args(monkeypatch, pool):
    """Swap the stdlib worker for one that records its args and resolves one item.

    The stdlib ``_worker`` signature differs between interpreters, so the fake
    accepts anything and completes the work item's future directly — the test
    then runs on 3.11 and 3.14 alike and asserts only on the arg shape chosen.
    """
    seen = []

    def fake_worker(*args):
        seen.append(args)
        pool._work_queue.get().future.set_result("done")

    monkeypatch.setattr(daemon_pool, "_worker", fake_worker)
    return seen


def test_worker_gets_context_when_executor_builds_worker_contexts(monkeypatch):
    """3.14+ shape (#58596, #111813): the executor exposes ``_create_worker_context``
    and no ``_initializer``/``_initargs``; the worker must receive
    ``(executor_ref, ctx, work_queue)`` — reading the legacy fields raised
    ``AttributeError`` on every pool spawn."""
    pool = DaemonThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(pool, "_create_worker_context", lambda: "worker-context", raising=False)
    monkeypatch.delattr(pool, "_initializer", raising=False)
    monkeypatch.delattr(pool, "_initargs", raising=False)
    seen = _capture_worker_args(monkeypatch, pool)
    try:
        assert pool.submit(lambda: None).result(timeout=10) == "done"
    finally:
        pool.shutdown(wait=True)
    ((executor_ref, ctx, work_queue),) = seen
    assert executor_ref() is pool
    assert ctx == "worker-context"
    assert work_queue is pool._work_queue


def test_worker_gets_initializer_when_executor_stores_initializer_fields(monkeypatch):
    """3.11–3.13 shape: no ``_create_worker_context``; the worker must receive
    ``(executor_ref, work_queue, initializer, initargs)``."""

    def init(*_):
        return None

    pool = DaemonThreadPoolExecutor(max_workers=1)
    monkeypatch.delattr(pool, "_create_worker_context", raising=False)
    monkeypatch.setattr(pool, "_initializer", init, raising=False)
    monkeypatch.setattr(pool, "_initargs", (1, 2), raising=False)
    seen = _capture_worker_args(monkeypatch, pool)
    try:
        assert pool.submit(lambda: None).result(timeout=10) == "done"
    finally:
        pool.shutdown(wait=True)
    ((executor_ref, work_queue, initializer, initargs),) = seen
    assert executor_ref() is pool
    assert work_queue is pool._work_queue
    assert (initializer, initargs) == (init, (1, 2))


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]


class _WorkerContext314:
    """Stand-in for 3.14's ``concurrent.futures.thread.WorkerContext``."""

    def __init__(self, initializer, initargs, events):
        self.initializer = initializer
        self.initargs = initargs
        self.events = events

    def initialize(self):
        self.events.append("initialize")
        if self.initializer is not None:
            self.initializer(*self.initargs)

    def finalize(self):
        self.events.append("finalize")

    def run(self, task):
        fn, args, kwargs = task
        return fn(*args, **kwargs)


def _worker_314(_executor_reference, worker_context, work_queue):
    """3.14's worker ABI: ``(executor_reference, ctx, work_queue)``."""
    worker_context.initialize()
    try:
        while True:
            item = work_queue.get()
            if item is None:
                return
            if inspect.signature(item.run).parameters:
                item.run(worker_context)  # 3.14 ``_WorkItem.run(ctx)``
            else:
                item.run()  # 3.8–3.13
    finally:
        worker_context.finalize()


def test_python314_worker_context_runs_the_task_and_still_spawns_daemon_threads(monkeypatch):
    """3.14 hand-off, end to end (#107121): the executor's worker context reaches a
    worker that initializes with it, runs real work items, and stays a daemon —
    reading the removed ``_initializer`` raised ``AttributeError`` on every spawn."""
    events = []
    pool = DaemonThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(daemon_pool, "_worker", _worker_314)
    monkeypatch.setattr(
        pool,
        "_create_worker_context",
        lambda: _WorkerContext314(None, (), events),
        raising=False,
    )
    monkeypatch.delattr(pool, "_initializer", raising=False)
    monkeypatch.delattr(pool, "_initargs", raising=False)
    try:
        is_daemon, worker, value = pool.submit(
            lambda: (threading.current_thread().daemon, threading.current_thread(), "done")
        ).result(timeout=10)
    finally:
        pool.shutdown(wait=True)
    assert value == "done"
    assert is_daemon is True
    assert worker not in _threads_queues
    assert events == ["initialize", "finalize"]


def test_worker_uses_official_prepare_context_when_instance_factory_is_missing(monkeypatch):
    """3.14 builds that only expose the classmethod still hand the worker a context."""
    pool = DaemonThreadPoolExecutor(max_workers=1)
    monkeypatch.delattr(pool, "_create_worker_context", raising=False)
    monkeypatch.delattr(pool, "_initializer", raising=False)
    monkeypatch.delattr(pool, "_initargs", raising=False)
    monkeypatch.setattr(
        type(pool),
        "prepare_context",
        lambda initializer, initargs: (
            lambda: "official-context",
            lambda fn, args, kwargs: (fn, args, kwargs),
        ),
        raising=False,
    )
    seen = _capture_worker_args(monkeypatch, pool)
    try:
        assert pool.submit(lambda: None).result(timeout=10) == "done"
    finally:
        pool.shutdown(wait=True)
    ((executor_ref, ctx, work_queue),) = seen
    assert executor_ref() is pool
    assert ctx == "official-context"
    assert work_queue is pool._work_queue


def test_worker_runs_without_initializer_when_no_worker_api_is_readable(monkeypatch):
    """Neither ABI readable: spawn a plain worker instead of raising
    ``AttributeError: 'DaemonThreadPoolExecutor' object has no attribute '_initializer'``."""
    pool = DaemonThreadPoolExecutor(max_workers=1)
    monkeypatch.delattr(pool, "_create_worker_context", raising=False)
    monkeypatch.delattr(pool, "_initializer", raising=False)
    monkeypatch.delattr(pool, "_initargs", raising=False)
    try:
        is_daemon, worker, value = pool.submit(
            lambda: (threading.current_thread().daemon, threading.current_thread(), "done")
        ).result(timeout=10)
    finally:
        pool.shutdown(wait=True)
    assert value == "done"
    assert is_daemon is True
    assert worker not in _threads_queues


def test_legacy_initializer_still_runs_inside_the_worker():
    """3.11–3.13 ABI unchanged: the initializer is passed through and runs on the
    worker thread, so the probe never silently drops it."""
    initialized_in = []
    pool = DaemonThreadPoolExecutor(
        max_workers=1, initializer=lambda: initialized_in.append(threading.get_ident())
    )
    try:
        task_ident = pool.submit(threading.get_ident).result(timeout=10)
    finally:
        pool.shutdown(wait=True)
    assert initialized_in == [task_ident]
