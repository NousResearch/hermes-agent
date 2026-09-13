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


def test_initializer_runs_on_each_worker_before_tasks():
    local = threading.local()
    ready = threading.Barrier(2, timeout=10)
    initialized = []
    lock = threading.Lock()
    marker = object()

    def initialize(value):
        local.value = value
        with lock:
            initialized.append(threading.current_thread())

    def task():
        ready.wait()
        return local.value, threading.current_thread()

    with DaemonThreadPoolExecutor(max_workers=2, initializer=initialize, initargs=(marker,)) as pool:
        futures = [pool.submit(task) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]

    workers = {worker for _, worker in results}
    assert len(workers) == 2
    assert set(initialized) == workers
    assert len(initialized) == len(workers)
    assert all(value is marker for value, _ in results)
    assert all(worker.daemon and worker not in _threads_queues for worker in workers)


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
    """A reused worker must not mix profile scopes or leak task mutations."""
    from contextvars import ContextVar

    var = ContextVar("daemon_pool_test_var", default="unset")

    def read_and_mutate():
        seen = var.get()
        var.set("worker mutation")
        return seen, threading.current_thread()

    with DaemonThreadPoolExecutor(max_workers=1) as pool:
        futures = []
        for profile in ("first", "second"):
            token = var.set(profile)
            try:
                futures.append(pool.submit(read_and_mutate))
            finally:
                var.reset(token)
        results = [future.result(timeout=10) for future in futures]
        assert [seen for seen, _ in results] == ["first", "second"]
        assert results[0][1] is results[1][1]
        assert pool.submit(var.get).result(timeout=10) == "unset"
        assert var.get() == "unset"


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]
