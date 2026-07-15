"""Tests for tools.daemon_pool.DaemonThreadPoolExecutor.

The daemon pool exists so abandoned workers (interrupted/timed-out tool
batches, wedged memory-provider syncs) can never block interpreter exit:
stdlib ThreadPoolExecutor workers are non-daemon AND registered in
concurrent.futures.thread._threads_queues, whose atexit hook joins every
worker unconditionally — even after shutdown(wait=False).
"""

import subprocess
import sys
import threading
import time

from concurrent.futures.thread import _threads_queues

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


def test_results_and_initializer_work_like_stdlib():
    seen = []

    def _init(tag):
        seen.append(tag)

    pool = DaemonThreadPoolExecutor(max_workers=1, initializer=_init, initargs=("t",))
    try:
        assert pool.submit(lambda: 41 + 1).result(timeout=10) == 42
        assert seen == ["t"]
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


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]


# ── Python 3.14 _worker signature compatibility ─────────────────────────────
#
# CPython 3.14 changed ThreadPoolExecutor._worker from
# (executor_ref, work_queue, initializer, initargs) to
# (executor_ref, ctx, work_queue), folding initializer/initargs into a
# worker-context object built by _create_worker_context(). The subclass
# feature-detects that method; both branches are pinned here directly so the
# suite covers them regardless of which interpreter it runs on.


def test_worker_context_probe_matches_interpreter():
    """The 3.14 branch keys off _create_worker_context — pin that the probe
    tracks the running interpreter, so a future stdlib reshape shows up here
    instead of as silently-dying worker threads.

    On 3.14 it is an INSTANCE attribute (unpacked in __init__ from
    prepare_context()), so probe a constructed executor, exactly like the
    hasattr(self, ...) guard in _worker_args does.
    """
    from concurrent.futures import ThreadPoolExecutor

    expected = sys.version_info >= (3, 14)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        assert hasattr(pool, "_create_worker_context") is expected
    finally:
        pool.shutdown(wait=False)


class _OldShapeExecutor:
    """Stand-in with the pre-3.14 attribute surface (no _create_worker_context)."""

    def __init__(self):
        self._work_queue = object()
        self._initializer = lambda: None
        self._initargs = ("tag",)


class _NewShapeExecutor:
    """Stand-in with the 3.14+ attribute surface."""

    _ctx = object()

    def __init__(self):
        self._work_queue = object()

    def _create_worker_context(self):
        return self._ctx


def test_worker_args_pre_314_shape():
    """Without _create_worker_context the historical 4-tuple is built."""
    fake = _OldShapeExecutor()

    args = DaemonThreadPoolExecutor._worker_args(fake, lambda *_: None)

    assert len(args) == 4
    executor_ref, work_queue, initializer, initargs = args
    assert executor_ref() is fake
    assert work_queue is fake._work_queue
    assert initializer is fake._initializer
    assert initargs == ("tag",)


def test_worker_args_314_shape():
    """With _create_worker_context the 3.14 3-tuple (ref, ctx, queue) is built."""
    fake = _NewShapeExecutor()

    args = DaemonThreadPoolExecutor._worker_args(fake, lambda *_: None)

    assert len(args) == 3
    executor_ref, ctx, work_queue = args
    assert executor_ref() is fake
    assert ctx is _NewShapeExecutor._ctx
    assert work_queue is fake._work_queue


def test_live_pool_spawns_working_threads_on_this_interpreter():
    """End-to-end guard for whichever branch is live: on 3.14 the pre-fix
    4-tuple made every worker thread die at spawn (TypeError in _worker), so
    submitted futures never completed."""
    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        assert pool.submit(lambda: "alive").result(timeout=10) == "alive"
        assert len(pool._threads) == 1
        assert all(t.is_alive() for t in pool._threads)
    finally:
        pool.shutdown(wait=True)
