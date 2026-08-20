"""Diagnosability of empty window discovery + CLI-fallback fail-fast.

Live-QA findings (Aug 2026, locked KDE desktop):
1. capture() returned a bare ``capture mode=ax 0x0`` with zero hint while
   the desktop was locked — undiagnosable for the model AND the user.
2. `_call_tool_via_cli` retried 4x with ~3.5s of sleeps on "daemon is not
   running", a permanent condition for that invocation.
"""
from typing import Any, Dict

import pytest

from tools.computer_use import cua_backend as cb


# ── _empty_discovery_reason ─────────────────────────────────────────────


def test_locked_session_reason_names_the_lock(monkeypatch):
    monkeypatch.setattr(cb, "_linux_session_locked", lambda: True)
    reason = cb._empty_discovery_reason()
    assert "LOCKED" in reason
    assert "unlock" in reason.lower()


def test_no_display_reason(monkeypatch):
    monkeypatch.setattr(cb, "_linux_session_locked", lambda: False)
    monkeypatch.setattr(cb.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    reason = cb._empty_discovery_reason()
    assert "DISPLAY" in reason


def test_unknown_reason_points_at_doctor(monkeypatch):
    monkeypatch.setattr(cb, "_linux_session_locked", lambda: None)
    monkeypatch.setenv("DISPLAY", ":0")
    reason = cb._empty_discovery_reason()
    assert "doctor" in reason


def test_locked_probe_fails_safe(monkeypatch):
    def _boom(*a, **kw):
        raise OSError("no loginctl")
    monkeypatch.setattr(cb.subprocess, "run", _boom)
    assert cb._linux_session_locked() in (None, False) if cb.sys.platform != "linux" else cb._linux_session_locked() is None


def test_empty_capture_carries_reason(monkeypatch):
    """capture() with zero discovered windows must explain itself."""
    backend = object.__new__(cb.CuaDriverBackend)
    backend._active_pid = None
    backend._active_window_id = None
    backend._last_app = None
    backend._last_target = None
    backend._snapshot_tokens = {}
    monkeypatch.setattr(backend, "_load_windows", lambda: [], raising=False)
    monkeypatch.setattr(cb, "_empty_discovery_reason",
                        lambda: "the desktop session is LOCKED (test)")
    cap = backend.capture(mode="ax")
    assert cap.width == 0 and cap.height == 0
    assert "LOCKED" in cap.window_title


# ── CLI fallback fail-fast on dead daemon ───────────────────────────────


class _Proc:
    def __init__(self, stdout="", stderr=""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0


def _make_session():
    session = object.__new__(cb._CuaDriverSession)
    session._embedded_daemon = None
    return session


def test_cli_fallback_fails_fast_on_daemon_not_running(monkeypatch):
    session = _make_session()
    calls = {"n": 0}
    sleeps = []

    def _fake_run(cmd, **kw):
        calls["n"] += 1
        return _Proc(stdout="Cua Driver daemon is not running on /x.sock.\nStart it first with: cua-driver serve")

    monkeypatch.setattr(cb, "resolve_cua_driver_cmd", lambda override=None: "cua-driver")
    import subprocess as _sp
    import time as _time
    monkeypatch.setattr(_sp, "run", _fake_run)
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="daemon is not running"):
        session._call_tool_via_cli("list_windows", {}, 10.0)

    assert calls["n"] == 1, f"expected fail-fast, got {calls['n']} attempts"
    assert sleeps == [], f"expected no backoff sleeps, got {sleeps}"


def test_cli_fallback_still_retries_transient_empty(monkeypatch):
    """Genuinely empty output (EAGAIN congestion) keeps the retry loop."""
    session = _make_session()
    calls = {"n": 0}

    def _fake_run(cmd, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            return _Proc(stdout="")
        return _Proc(stdout='{"windows": []}')

    monkeypatch.setattr(cb, "resolve_cua_driver_cmd", lambda override=None: "cua-driver")
    import subprocess as _sp
    import time as _time
    monkeypatch.setattr(_sp, "run", _fake_run)
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    out = session._call_tool_via_cli("list_windows", {}, 10.0)
    assert calls["n"] == 3
    assert out.get("structuredContent") == {"windows": []}
