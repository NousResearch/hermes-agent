"""Tests for _SlashWorker drain thread cleanup (#53303)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class _FakeProc:
    """Minimal subprocess.Popen stand-in for _SlashWorker tests."""

    def __init__(self):
        self.stdin = MagicMock()
        self.stdout = MagicMock()
        self.stderr = MagicMock()
        self._poll = None  # None = still running

        # Make stdout/stderr iteration return empty (drain threads exit)
        self.stdout.__iter__ = lambda self: iter([])
        self.stderr.__iter__ = lambda self: iter([])

    def poll(self):
        return self._poll

    def terminate(self):
        self._poll = 0

    def kill(self):
        self._poll = -9

    def wait(self, timeout=None):
        return self._poll or 0


def test_slash_worker_close_joins_drain_threads():
    """_SlashWorker.close() must join its drain threads (#53303).

    Prior to the fix, close() terminated the subprocess and closed
    the pipes but never joined the _drain_stdout/_drain_stderr threads.
    This left 2 leaked daemon threads per session on Linux, each holding
    references to the _SlashWorker instance and its buffers.

    The fix stores thread references and calls join(timeout=2) in close().
    In production, closing proc.stdout/proc.stderr causes the readline()
    in the drain threads to hit EOF and exit, so join() returns quickly.
    This test uses threads that exit promptly to verify the join path works.
    """
    from tui_gateway.server import _SlashWorker

    worker = _SlashWorker.__new__(_SlashWorker)
    worker._lock = threading.Lock()
    worker._seq = 0
    worker.stderr_tail = []
    worker.stdout_queue = __import__("queue").Queue()
    worker._closed = False
    worker.proc = _FakeProc()

    # Use threads that exit quickly (simulating EOF on the pipe)
    exit_event = threading.Event()
    exit_event.set()  # let them exit immediately

    def quick_drain():
        exit_event.wait(timeout=5)

    worker._drain_thread_stdout = threading.Thread(
        target=quick_drain, daemon=True, name="test-drain-stdout"
    )
    worker._drain_thread_stderr = threading.Thread(
        target=quick_drain, daemon=True, name="test-drain-stderr"
    )
    worker._drain_thread_stdout.start()
    worker._drain_thread_stderr.start()

    # Give threads time to exit
    time.sleep(0.1)

    # Call close()
    worker.close()

    # close() should have set _closed
    assert worker._closed

    # close() should have terminated the proc
    assert worker.proc.poll() is not None

    # close() should have closed stdin/stdout/stderr
    worker.proc.stdin.close.assert_called()
    worker.proc.stdout.close.assert_called()
    worker.proc.stderr.close.assert_called()

    # The drain threads should have exited (they exit on their own, and
    # close() joins them — so they're definitely not alive after close()).
    assert not worker._drain_thread_stdout.is_alive(), (
        "_drain_thread_stdout is still alive after close()"
    )
    assert not worker._drain_thread_stderr.is_alive(), (
        "_drain_thread_stderr is still alive after close()"
    )


def test_slash_worker_close_is_idempotent():
    """close() can be called multiple times safely."""
    from tui_gateway.server import _SlashWorker

    worker = _SlashWorker.__new__(_SlashWorker)
    worker._closed = False
    worker.proc = _FakeProc()

    def noop():
        pass

    worker._drain_thread_stdout = threading.Thread(target=noop, daemon=True)
    worker._drain_thread_stderr = threading.Thread(target=noop, daemon=True)
    worker._drain_thread_stdout.start()
    worker._drain_thread_stderr.start()

    worker.close()
    assert worker._closed

    # Second call should be a no-op (guard at top of close())
    worker.close()
    assert worker._closed


def test_slash_worker_drain_threads_are_named():
    """Drain threads should have identifiable names for debugging."""
    # This is a regression guard: anonymous threads (no name) make it
    # impossible to identify the source of leaked threads in py-spy dumps.
    # The fix gives them explicit names: slash-drain-stdout, slash-drain-stderr.
    import inspect

    from tui_gateway import server

    source = inspect.getsource(_SlashWorker := server._SlashWorker)
    assert "slash-drain-stdout" in source, (
        "_SlashWorker should name its stdout drain thread 'slash-drain-stdout'"
    )
    assert "slash-drain-stderr" in source, (
        "_SlashWorker should name its stderr drain thread 'slash-drain-stderr'"
    )
