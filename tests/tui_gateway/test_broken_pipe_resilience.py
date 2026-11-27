"""Tests for the TUI gateway's broken-pipe resilience (#68614).

When the TUI parent closes its stdout/stderr pipe (session recovery, UI
recycling), the gateway child's diagnostic prints to stderr raise
``BrokenPipeError``. In the signal handler and exit logger those prints sit
outside any guard, so the exception propagates and kills the child — which
the TUI parent immediately respawns, producing a visible crash loop.

These tests assert the invariant: ``_log_signal`` and ``_log_exit`` must not
raise when stderr is a broken pipe. They exercise the real functions with a
real closed-pipe stderr, not a mock.

Note: the pipe setup runs inside the test body, not a fixture — pytest's
default capture re-installs ``sys.stderr`` after fixture setup, which would
silently undo the patch before the test body runs.
"""

from __future__ import annotations

import os
import signal
import sys
import threading

import pytest


def _broken_stderr(monkeypatch):
    """Point sys.stderr at a pipe whose read end is closed (EPIPE on write)."""
    r, w = os.pipe()
    os.close(r)  # read end gone → writes raise BrokenPipeError
    monkeypatch.setattr(sys, "stderr", os.fdopen(w, "w"))


def test_log_exit_survives_broken_stderr(monkeypatch):
    """_log_exit must not raise when stderr is a broken pipe."""
    from tui_gateway import entry

    _broken_stderr(monkeypatch)
    entry._log_exit("test-reason")  # must not raise


def test_log_signal_survives_broken_stderr(monkeypatch):
    """_log_signal must not raise when stderr is a broken pipe."""
    from tui_gateway import entry

    _broken_stderr(monkeypatch)
    # Keep the handler from actually exiting or arming the grace timer.
    monkeypatch.setattr(entry, "_shutdown_grace_seconds", lambda: 0.0)
    real_timer = threading.Timer
    monkeypatch.setattr(threading, "Timer", lambda *a, **k: real_timer(0.0, lambda: None))
    monkeypatch.setattr(entry.sys, "exit", lambda code: None)

    entry._log_signal(signal.SIGTERM, None)  # must not raise


def test_record_crash_survives_broken_stderr(monkeypatch):
    """The panic hook must not raise when stderr is a broken pipe.

    This is the worst place for a broken pipe to kill the child: the crash
    reporter itself would die before the user sees anything.
    """
    from tui_gateway import server

    _broken_stderr(monkeypatch)
    server._record_crash("test", ValueError, ValueError("boom"), None)  # must not raise
