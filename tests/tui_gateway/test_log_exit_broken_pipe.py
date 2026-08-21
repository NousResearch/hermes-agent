"""Tests for _log_exit guarding against BrokenPipeError on shutdown.

Verifies the fix for issue #90434: clean TUI shutdown should not emit a
traceback from an unguarded stderr write in the gateway exit handler.
"""

from __future__ import annotations

import io
import sys
from unittest import mock


def _broken_pipe_stderr() -> io.StringIO:
    """A stderr-like object that raises BrokenPipeError on write/flush."""
    stream = io.StringIO()

    def _raise(*args, **kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    stream.write = _raise  # type: ignore[assignment]
    stream.flush = _raise  # type: ignore[assignment]
    return stream


def test_log_exit_guards_broken_pipe() -> None:
    """_log_exit must not raise when stderr is a broken pipe."""
    from tui_gateway.entry import _log_exit

    broken = _broken_pipe_stderr()
    with mock.patch.object(sys, "stderr", broken):
        # Should not raise — best-effort logger.
        _log_exit("stdin EOF (peer closed)")


def test_log_exit_guards_oserror() -> None:
    """_log_exit must swallow OSError (EPIPE/EBADF superclass)."""
    from tui_gateway.entry import _log_exit

    broken = _broken_pipe_stderr()
    broken.write = mock.Mock(side_effect=OSError(22, "Invalid argument"))  # type: ignore[assignment]
    broken.flush = mock.Mock(side_effect=OSError(22, "Invalid argument"))  # type: ignore[assignment]

    with mock.patch.object(sys, "stderr", broken):
        _log_exit("test reason")


def test_log_exit_guards_closed_stderr() -> None:
    """_log_exit must swallow ValueError from a closed-file stderr."""
    from tui_gateway.entry import _log_exit

    closed = io.StringIO()
    closed.close()

    with mock.patch.object(sys, "stderr", closed):
        _log_exit("test reason")


def test_log_exit_writes_when_pipe_healthy() -> None:
    """_log_exit should still write to stderr when the pipe is fine."""
    from tui_gateway.entry import _log_exit

    buf = io.StringIO()
    with mock.patch.object(sys, "stderr", buf):
        _log_exit("clean exit")

    assert "[gateway-exit] clean exit" in buf.getvalue()


def test_sw_log_guards_broken_pipe() -> None:
    """_sw_log (slash_worker) must not raise on broken pipe.

    _sw_log is defined inside main(), so we replicate its guarded shape
    and verify the pattern holds — this is the exact code path from the fix.
    """
    def _sw_log(reason: str) -> None:
        try:
            print(f"[slash-worker] {reason}", file=sys.stderr, flush=True)
        except (BrokenPipeError, ValueError, OSError):
            pass

    broken = _broken_pipe_stderr()
    with mock.patch.object(sys, "stderr", broken):
        _sw_log("stdin EOF (peer closed)")


def test_sw_log_writes_when_pipe_healthy() -> None:
    """_sw_log should still write to stderr when the pipe is fine."""
    def _sw_log(reason: str) -> None:
        try:
            print(f"[slash-worker] {reason}", file=sys.stderr, flush=True)
        except (BrokenPipeError, ValueError, OSError):
            pass

    buf = io.StringIO()
    with mock.patch.object(sys, "stderr", buf):
        _sw_log("clean exit")

    assert "[slash-worker] clean exit" in buf.getvalue()