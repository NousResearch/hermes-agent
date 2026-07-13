"""Tests for tools.daemon_pool.DaemonThreadPoolExecutor.

The daemon pool exists so abandoned workers (interrupted/timed-out tool
batches, wedged memory-provider syncs) can never block interpreter exit:
stdlib ThreadPoolExecutor workers are non-daemon AND registered in
concurrent.futures.thread._threads_queues, whose atexit hook joins every
worker unconditionally — even after shutdown(wait=False).

Additional Python 3.14+ coverage: the stdlib ``ThreadPoolExecutor`` changed
its internal ``_worker`` signature and dropped ``self._initializer`` /
``self._initargs`` in favor of a WorkerContext (issue #63769). The daemon
pool must keep working on both layouts.
"""

import subprocess
import sys
import threading
import time

from concurrent.futures.thread import _threads_queues

from tools.daemon_pool import DaemonThreadPoolExecutor, _PY314_PLUS


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


def test_python_314_worker_context_signature_does_not_crash():
    """Regression test for issue #63769.

    On Python 3.14+, stdlib ``ThreadPoolExecutor`` drops ``self._initializer``
    / ``self._initargs`` and switches ``_worker`` to a ``(ref, ctx, q)``
    signature with a ``WorkerContext``. The daemon pool must:

      1. submit without raising ``AttributeError: ... _initializer``.
      2. still produce daemon workers that are not registered with
         ``concurrent.futures.thread._threads_queues``.
      3. still honor ``initializer`` / ``initargs`` semantics via the new
         WorkerContext path.
    """
    if not _supports_worker_context():
        # On 3.11–3.13 the legacy path is exercised by the other tests; this
        # assertion is a 3.14+ guard, so skip rather than duplicate work.
        import pytest

        pytest.skip("WorkerContext path only exists on Python 3.14+")

    init_seen = []

    def _init(tag):
        init_seen.append(tag)

    pool = DaemonThreadPoolExecutor(max_workers=1, initializer=_init, initargs=("ctx",))
    try:
        # Submit must not raise AttributeError: '_initializer'
        result = pool.submit(lambda: 21 * 2).result(timeout=10)
        assert result == 42
        # Initializer ran exactly once per worker spawn.
        assert init_seen == ["ctx"]

        # Worker is still daemon and unregistered with the atexit join hook.
        worker_id = pool.submit(threading.get_ident).result(timeout=10)
        worker = next(
            (t for t in pool._threads if t.ident == worker_id),
            None,
        )
        assert worker is not None
        assert worker.daemon is True
        assert worker not in _threads_queues
    finally:
        pool.shutdown(wait=True)


def test_python_314_submit_many_tasks_exercises_reused_workers():
    """Submit several sequential tasks on a single-worker pool.

    On 3.14+ this exercises both _adjust_thread_count branches: the first
    submit spawns the worker, subsequent submits reuse it via the idle
    semaphore. This guards against any silent ctx/state mismatch where a
    second task would fail to run because the worker context was tied to
    the first submit only.
    """
    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        results = [pool.submit(lambda i=i: i * i).result(timeout=10) for i in range(5)]
        assert results == [0, 1, 4, 9, 16]
    finally:
        pool.shutdown(wait=True)


def _supports_worker_context() -> bool:
    """True if the active Python ships the 3.14+ ThreadPoolExecutor layout.

    Reuses ``tools.daemon_pool._PY314_PLUS`` so the test gate and the
    dispatch logic stay in lockstep — a future Python release that changes
    ``_worker``'s signature again only needs to be updated in one place.
    """
    return _PY314_PLUS


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]
