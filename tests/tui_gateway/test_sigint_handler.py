"""Tests for the TUI gateway two-stage SIGINT handler (#53362).

Verifies the review-remediation semantics:

1. Stage-1 interrupt clears pending prompts PER SESSION (never globally) so
   one Ctrl+C cannot dismiss clarify/sudo/secret prompts on unrelated live
   sessions.
2. The grace-window failsafe is CONDITIONAL — it only hard-exits while at
   least one session is genuinely still running after the interrupt.  Once
   every interrupted session has drained, the failsafe is disarmed so a
   healthy recoverable session is never hard-killed.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types


def _load_entry() -> types.ModuleType:
    """Import tui_gateway/entry with signal registration stubbed out.

    entry.py registers a real SIGINT handler at import time, which would
    clobber pytest's own handler in this process.  We intercept
    ``signal.signal`` so the module loads but the global handler isn't
    installed.
    """
    import signal

    real_signal = signal.signal
    try:
        signal.signal = lambda *a, **k: None
        sys.modules.pop("tui_gateway.entry", None)
        mod = importlib.import_module("tui_gateway.entry")
        return mod
    finally:
        signal.signal = real_signal


# ---------------------------------------------------------------------------
# _sessions_still_running
# ---------------------------------------------------------------------------


def test_sessions_still_running_true_when_any_running(monkeypatch):
    mod = _load_entry()
    fake_sessions = {
        "a": {"running": False, "agent": object()},
        "b": {"running": True, "agent": object()},
    }
    monkeypatch.setattr(
        sys.modules["tui_gateway.server"], "_sessions", fake_sessions,
    )
    assert mod._sessions_still_running() is True


def test_sessions_still_running_false_when_all_drained(monkeypatch):
    mod = _load_entry()
    fake_sessions = {
        "a": {"running": False, "agent": object()},
        "b": {"running": False, "agent": object()},
    }
    monkeypatch.setattr(
        sys.modules["tui_gateway.server"], "_sessions", fake_sessions,
    )
    assert mod._sessions_still_running() is False


def test_sessions_still_running_true_when_empty_guard_fails(monkeypatch):
    # If we cannot inspect sessions we fail closed (arm the failsafe).
    mod = _load_entry()
    monkeypatch.setattr(
        sys.modules["tui_gateway.server"], "_sessions", None,
    )
    assert mod._sessions_still_running() is True


# ---------------------------------------------------------------------------
# _handle_sigint — per-session pending clear + conditional failsafe
# ---------------------------------------------------------------------------


def test_stage1_clears_pending_per_session(monkeypatch):
    mod = _load_entry()
    monkeypatch.setattr(mod, "_sigint_stage", 0, raising=False)

    interrupted = []
    cleared = []
    agent = type("Agent", (), {"interrupt": lambda self: interrupted.append(1)})()
    fake_sessions = {
        "a": {"running": True, "agent": agent},
        "b": {"running": False, "agent": object()},
    }
    monkeypatch.setattr(
        sys.modules["tui_gateway.server"], "_sessions", fake_sessions,
    )

    def _fake_clear(sid):
        cleared.append(sid)

    monkeypatch.setattr(
        sys.modules["tui_gateway.server"], "_clear_pending", _fake_clear,
    )
    # Don't arm the real timer — record whether it would be started.
    timer_starts = []

    class _FakeTimer:
        def __init__(self, delay, fn):
            timer_starts.append((delay, fn))
            self.delay = delay
            self.fn = fn

        def start(self):
            pass

    monkeypatch.setattr(threading, "Timer", _FakeTimer)

    mod._handle_sigint(2, None)

    assert interrupted == [1]  # only the running session was interrupted
    # Pending cleared per session, scoped to session id — never globally.
    assert cleared == ["a", "b"]
    # Failsafe armed (a session was still running).
    assert timer_starts != []


def test_stage1_disarms_failsafe_when_all_drained(monkeypatch):
    mod = _load_entry()
    monkeypatch.setattr(mod, "_sigint_stage", 0, raising=False)

    fake_sessions = {
        "a": {"running": False, "agent": object()},
    }
    monkeypatch.setattr(
        sys.modules["tui_gateway.server"], "_sessions", fake_sessions,
    )

    def _fake_clear(sid):
        pass

    monkeypatch.setattr(
        sys.modules["tui_gateway.server"], "_clear_pending", _fake_clear,
    )
    timer_starts = []

    class _FakeTimer:
        def __init__(self, delay, fn):
            timer_starts.append((delay, fn))

        def start(self):
            pass

    monkeypatch.setattr(threading, "Timer", _FakeTimer)

    mod._handle_sigint(2, None)

    # No session running → failsafe NOT armed; a healthy gateway survives.
    assert timer_starts == []


def test_stage2_hard_exits(monkeypatch):
    mod = _load_entry()
    monkeypatch.setattr(mod, "_sigint_stage", 1)
    exits = []
    monkeypatch.setattr(mod.os, "_exit", lambda code: exits.append(code))
    mod._handle_sigint(2, None)
    assert exits == [0]
