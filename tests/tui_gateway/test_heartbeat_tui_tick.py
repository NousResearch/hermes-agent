"""Tests for the TUI/Desktop heartbeat driver (issue #102056).

The slash worker that parses ``/heartbeat`` starts a watchdog thread in its
own process, where the queued prompt is never consumed. The real driver lives
in ``tui_gateway/server._maybe_fire_tui_heartbeat_tick``, which polls the
session-owner process and re-enters the live session via ``_run_prompt_submit``.
"""

import time
from types import SimpleNamespace

import pytest

import tui_gateway.server as server
from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState


def _due_manager(session_id):
    """A HeartbeatManager whose state is already due, bypassing SessionDB."""
    mgr = HeartbeatManager.__new__(HeartbeatManager)
    mgr.session_id = session_id
    # created long ago so is_due() is immediately true.
    mgr._state = HeartbeatState(
        prompt="report backend health",
        interval_seconds=60,
        status="active",
        created_at=time.time() - 3600,
    )
    return mgr


def _idle_session():
    return {
        "agent": SimpleNamespace(),
        "session_key": "hb-tui-key",
        "history": [],
        "history_lock": server.threading.Lock(),
        "running": False,
    }


def test_tui_heartbeat_fires_when_idle_and_due(monkeypatch):
    submitted = []

    def _submit(_rid, sid, session, text):
        submitted.append(text)

    monkeypatch.setattr(server, "_run_prompt_submit", _submit)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.heartbeat.HeartbeatManager", lambda session_id: _due_manager(session_id)
    )

    session = _idle_session()
    server._maybe_fire_tui_heartbeat_tick("sid-hb", session)

    assert len(submitted) == 1
    assert "report backend health" in submitted[0]
    assert session["running"] is True


def test_tui_heartbeat_skips_when_busy(monkeypatch):
    submitted = []

    def _submit(_rid, sid, session, text):
        submitted.append(text)

    monkeypatch.setattr(server, "_run_prompt_submit", _submit)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.heartbeat.HeartbeatManager", lambda session_id: _due_manager(session_id)
    )

    session = _idle_session()
    session["running"] = True  # an in-flight turn owns the session
    server._maybe_fire_tui_heartbeat_tick("sid-hb", session)

    assert submitted == []
    # The tick stays due; the driver never claims the session.
    assert session["running"] is True


def test_tui_heartbeat_skips_when_not_due(monkeypatch):
    submitted = []

    def _submit(_rid, sid, session, text):
        submitted.append(text)

    monkeypatch.setattr(server, "_run_prompt_submit", _submit)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    mgr = HeartbeatManager.__new__(HeartbeatManager)
    mgr.session_id = "hb-tui-key"
    mgr._state = HeartbeatState(
        prompt="not yet",
        interval_seconds=60,
        status="active",
        created_at=time.time(),  # just armed — not due
    )
    monkeypatch.setattr("hermes_cli.heartbeat.HeartbeatManager", lambda session_id: mgr)

    session = _idle_session()
    server._maybe_fire_tui_heartbeat_tick("sid-hb", session)

    assert submitted == []
    assert session["running"] is False