"""Invariant tests for DaemonThreadPoolExecutor.

Regression context: the class overrides ``_adjust_thread_count``, so on
Python 3.14 the stdlib's lazy creation of ``_initializer``/``_initargs``
(inside its own ``_adjust_thread_count``) never happens; the first
``submit()`` must not raise AttributeError, and the worker thread must
receive initializer args in whichever ``_worker()`` signature the running
stdlib uses (WorkerContext on 3.14, 4 positional args on 3.8-3.13).
"""

import contextvars
import threading

import pytest

from tools.daemon_pool import DaemonThreadPoolExecutor


def test_submit_runs_and_returns_result():
    pool = DaemonThreadPoolExecutor()
    try:
        assert pool.submit(lambda: 42).result(timeout=10) == 42
    finally:
        pool.shutdown(wait=False)


def test_initializer_attrs_captured_at_construction():
    # On 3.14 this attribute does not exist until a worker spawns via the
    # stdlib path; the subclass override must capture it in __init__ so the
    # override never reads a missing attribute.
    def _init():
        pass

    pool = DaemonThreadPoolExecutor(initializer=_init, initargs=(1, 2))
    assert pool._initializer is _init
    assert pool._initargs == (1, 2)


def test_initializer_executes_on_worker_thread():
    ran = threading.Event()
    pool = DaemonThreadPoolExecutor(initializer=ran.set)
    try:
        assert pool.submit(lambda: 1).result(timeout=10) == 1
        # First worker spawn runs the initializer regardless of stdlib version.
        assert ran.wait(timeout=10)
    finally:
        pool.shutdown(wait=False)


def test_submit_propagates_contextvars():
    var = contextvars.ContextVar("daemon_pool_test_var")
    var.set("caller-value")
    pool = DaemonThreadPoolExecutor()
    try:
        assert pool.submit(var.get).result(timeout=10) == "caller-value"
    finally:
        pool.shutdown(wait=False)


def test_workers_are_daemons():
    pool = DaemonThreadPoolExecutor(max_workers=2)
    try:
        pool.submit(int).result(timeout=10)  # force one worker spawn
        (worker,) = pool._threads
        assert worker.daemon is True
    finally:
        pool.shutdown(wait=False)
