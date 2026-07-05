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

from concurrent.futures.thread import _threads_queues, _worker

from tools.daemon_pool import DaemonThreadPoolExecutor, _WORKER_USES_CTX


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


# ---------------------------------------------------------------------------
# Regression tests for Python 3.14 _worker signature change (#58596)
# ---------------------------------------------------------------------------

def test_worker_uses_ctx_matches_signature():
    """_WORKER_USES_CTX must reflect the actual _worker parameter count."""
    params = len(inspect.signature(_worker).parameters)
    if params == 3:
        assert _WORKER_USES_CTX is True
    elif params == 4:
        assert _WORKER_USES_CTX is False
    else:
        raise AssertionError(
            f"Unexpected _worker signature: {params} params (expected 3 or 4)"
        )


def test_submit_result_py314_path():
    """Submit+result works regardless of which _worker branch is taken."""
    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        assert pool.submit(lambda: "ok-" + "py314").result(timeout=10) == "ok-py314"
    finally:
        pool.shutdown(wait=True)


def test_initializer_works_across_worker_signatures():
    """initializer/initargs must be honoured on both Python paths."""
    seen: list = []

    def _init(tag: str) -> None:
        seen.append(tag)

    pool = DaemonThreadPoolExecutor(
        max_workers=1, initializer=_init, initargs=("tag-58596",)
    )
    try:
        pool.submit(lambda: None).result(timeout=10)
        assert seen == ["tag-58596"]
    finally:
        pool.shutdown(wait=True)


def test_pool_reuse_across_worker_signatures():
    """Multiple submits reuse the same worker (idle-reuse path)."""
    pool = DaemonThreadPoolExecutor(max_workers=3)
    try:
        tid1 = pool.submit(threading.get_ident).result(timeout=10)
        time.sleep(0.05)
        tid2 = pool.submit(threading.get_ident).result(timeout=10)
        assert tid1 == tid2
    finally:
        pool.shutdown(wait=True)
