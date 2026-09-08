"""Tests for the Windows non-blocking stdout drain (``_drain_fd_windows``) and its
helpers. These mirror the POSIX ``_drain_fd_select`` behaviour: the drain must stop
promptly when *stop* is set, or ~300ms after the child exits with the pipe idle, so a
backgrounded grandchild holding the pipe write end can no longer hang the tool
(see issue #105865). The Windows-only ``ctypes``/``msvcrt`` calls are isolated behind
``_peek_pipe_available`` / ``_windows_pipe_handle`` so the drain logic is unit-testable
on POSIX without a live Windows pipe.
"""

import codecs
import os
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.environments import base_output
from tools.environments.base_output import (
    _BoundedOutputCollector,
    _drain_fd_windows,
    _drain_stdout,
    _peek_pipe_available,
    _windows_pipe_handle,
)


def _make_decoder():
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


def _idle_proc():
    """A proc that has already exited (poll() returns 0)."""
    proc = MagicMock()
    proc.poll.return_value = 0
    return proc


def _running_proc():
    """A proc that is still running (poll() returns None)."""
    proc = MagicMock()
    proc.poll.return_value = None
    return proc


class TestDrainFdWindowsControlFlow:
    """The drain must honour *stop* and exit shortly after the child dies — never block."""

    def test_stops_immediately_when_stop_event_set(self):
        proc = _running_proc()
        stop = threading.Event()
        stop.set()
        with patch.object(base_output, "_windows_pipe_handle", return_value=42), \
             patch.object(base_output, "_peek_pipe_available", return_value=0):
            _drain_fd_windows(proc, 3, _BoundedOutputCollector(max_chars=1_000_000), _make_decoder(), stop)
        # Must not have polled the child's lifetime — it would still be "running" and the
        # only exit is via stop, so reaching here without hanging is the assertion.
        assert proc.poll.call_count <= 1  # at most one peek before stop checked

    def test_exits_after_process_exit_with_idle_pipe(self):
        proc = _idle_proc()
        with patch.object(base_output, "_windows_pipe_handle", return_value=42), \
             patch.object(base_output, "_peek_pipe_available", return_value=0):
            _drain_fd_windows(proc, 3, _BoundedOutputCollector(max_chars=1_000_000), _make_decoder(), None)
        # ~3 idle cycles after exit before returning.
        assert proc.poll.call_count >= 3

    def test_reads_available_bytes_and_appends_output(self):
        proc = _idle_proc()
        output = _BoundedOutputCollector(max_chars=1_000_000)
        # One cycle with data (peek=5 -> os.read=b"hello"), then the pipe goes idle so
        # the proc.poll() branch is exercised and the drain exits after ~3 idle cycles.
        peek_calls = {"n": 0}

        def peek_side(handle):
            peek_calls["n"] += 1
            return 5 if peek_calls["n"] == 1 else 0

        with patch.object(base_output, "_windows_pipe_handle", return_value=42), \
             patch.object(base_output, "_peek_pipe_available", side_effect=peek_side), \
             patch("os.read", return_value=b"hello"):
            _drain_fd_windows(proc, 3, output, _make_decoder(), None)
        assert "hello" in output.render()

    def test_returns_on_true_eof(self):
        proc = _running_proc()
        with patch.object(base_output, "_windows_pipe_handle", return_value=42), \
             patch.object(base_output, "_peek_pipe_available", return_value=4), \
             patch("os.read", return_value=b""):
            _drain_fd_windows(proc, 3, _BoundedOutputCollector(max_chars=1_000_000), _make_decoder(), None)
        # Reaching here without hanging is the assertion (EOF exits the loop).

    def test_returns_when_pipe_broken(self):
        proc = _running_proc()
        with patch.object(base_output, "_windows_pipe_handle", return_value=42), \
             patch.object(base_output, "_peek_pipe_available", return_value=-1):
            _drain_fd_windows(proc, 3, _BoundedOutputCollector(max_chars=1_000_000), _make_decoder(), None)
        assert proc.poll.call_count == 0  # broken pipe exits before checking poll

    def test_returns_when_handle_unavailable(self):
        proc = _running_proc()
        with patch.object(base_output, "_windows_pipe_handle", return_value=None):
            _drain_fd_windows(proc, 3, _BoundedOutputCollector(max_chars=1_000_000), _make_decoder(), None)
        assert proc.poll.call_count == 0  # no handle -> immediate return

    def test_stop_checked_each_cycle_even_with_data(self):
        """If data is perpetually available, stop must still be honoured (no infinite loop)."""
        proc = _running_proc()
        stop = threading.Event()
        stop.set()
        with patch.object(base_output, "_windows_pipe_handle", return_value=42), \
             patch.object(base_output, "_peek_pipe_available", return_value=99), \
             patch("os.read", return_value=b"x"):
            _drain_fd_windows(proc, 3, _BoundedOutputCollector(max_chars=1_000_000), _make_decoder(), stop)
        # stop is checked first each cycle -> returns without reading forever.


class TestDrainStdoutDispatchOnWindows:
    """``_drain_stdout`` must route to ``_drain_fd_windows`` (not the blocking os.read loop)
    on Windows, and still fall back to iterating non-fd streams."""

    def test_dispatches_to_drain_fd_windows_on_nt(self):
        stream = MagicMock()
        stream.fileno.return_value = 5
        proc = MagicMock()
        proc.stdout = stream  # real int fileno -> reaches the os.name branch
        with patch("os.name", "nt"), \
             patch.object(base_output, "_drain_fd_windows") as mock_win, \
             patch.object(base_output, "_drain_fd_select") as mock_select:
            _drain_stdout(proc, _BoundedOutputCollector(max_chars=1_000_000), stop=None)
        mock_win.assert_called_once()
        mock_select.assert_not_called()

    def test_dispatches_to_drain_fd_select_on_posix(self):
        stream = MagicMock()
        stream.fileno.return_value = 5
        proc = MagicMock()
        proc.stdout = stream  # real int fileno -> reaches the os.name branch
        with patch("os.name", "posix"), \
             patch.object(base_output, "_drain_fd_windows") as mock_win, \
             patch.object(base_output, "_drain_fd_select") as mock_select:
            _drain_stdout(proc, _BoundedOutputCollector(max_chars=1_000_000), stop=None)
        mock_select.assert_called_once()
        mock_win.assert_not_called()

    def test_none_stream_returns_early(self):
        proc = MagicMock()
        proc.stdout = None
        with patch.object(base_output, "_drain_fd_windows") as mock_win, \
             patch.object(base_output, "_drain_fd_select") as mock_select:
            _drain_stdout(proc, _BoundedOutputCollector(max_chars=1_000_000), stop=None)
        mock_win.assert_not_called()
        mock_select.assert_not_called()


class TestWindowsPipeHandle:
    """``_windows_pipe_handle`` lazily imports msvcrt (Windows-only) and maps fd -> handle."""

    def test_returns_handle_from_get_osfhandle(self):
        fake_msvcrt = MagicMock()
        fake_msvcrt.get_osfhandle.return_value = 4242
        with patch.dict(sys.modules, {"msvcrt": fake_msvcrt}):
            assert _windows_pipe_handle(3) == 4242

    def test_returns_none_on_oserror(self):
        fake_msvcrt = MagicMock()
        fake_msvcrt.get_osfhandle.side_effect = OSError("bad fd")
        with patch.dict(sys.modules, {"msvcrt": fake_msvcrt}):
            assert _windows_pipe_handle(3) is None


class TestPeekPipeAvailable:
    """``_peek_pipe_available`` wraps PeekNamedPipe and returns bytes available (-1 on error)."""

    def test_returns_available_bytes(self):
        fake_kernel32 = MagicMock()
        fake_kernel32.PeekNamedPipe.return_value = 1  # BOOL TRUE

        # PeekNamedPipe writes into the c_ulong passed via byref. Simulate by capturing
        # the byref object's underlying c_ulong and setting its value.
        def fake_peek(handle, buf, size, written, total_avail, left):
            # total_avail is ctypes.byref(c_ulong) -> a CArgObject wrapping the c_ulong.
            # Walk the caller's frame to find the original c_ulong and set it.
            import ctypes
            # The caller created `avail = ctypes.c_ulong(0)` then passed ctypes.byref(avail).
            # Easiest robust approach: set via the c_ulong object reachable through the
            # CArgObject's _obj (cpython implementation detail, but stable for tests).
            obj = getattr(total_avail, "_obj", None) or getattr(total_avail, "_obj", None)
            if obj is not None:
                obj.value = 7
            return 1

        fake_kernel32.PeekNamedPipe.side_effect = fake_peek
        fake_windll = MagicMock()
        fake_windll.kernel32 = fake_kernel32
        with patch("ctypes.windll", fake_windll, create=True):
            assert _peek_pipe_available(4242) == 7

    def test_returns_neg1_when_peeknamedpipe_fails(self):
        fake_kernel32 = MagicMock()
        fake_kernel32.PeekNamedPipe.return_value = 0  # BOOL FALSE -> failure
        fake_windll = MagicMock()
        fake_windll.kernel32 = fake_kernel32
        with patch("ctypes.windll", fake_windll, create=True):
            assert _peek_pipe_available(4242) == -1
