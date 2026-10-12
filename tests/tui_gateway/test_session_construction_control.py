"""A live session-control token fences cold construction and late attachment."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from tui_gateway import server

    monkeypatch.setattr(server, "_hermes_home", home)
    with server._sessions_lock:
        saved_sessions = dict(server._sessions)
        server._sessions.clear()
    try:
        yield server
    finally:
        with server._sessions_lock:
            server._sessions.clear()
            server._sessions.update(saved_sessions)


@pytest.fixture()
def held_control_session(server):
    sid = "session-construction-control"
    token = object()
    current = {
        "session_key": "session-construction-control-key",
        "agent": None,
        "agent_ready": threading.Event(),
        "agent_build_lock": threading.Lock(),
        "agent_build_started": False,
        "history_lock": threading.Lock(),
        "history": [],
        "running": False,
        "lazy": True,
        "_session_control_inflight": token,
    }
    with server._sessions_lock:
        server._sessions[sid] = current
    return sid, current, token


def test_control_token_blocks_cold_agent_build(server, held_control_session, monkeypatch):
    sid, current, token = held_control_session
    monkeypatch.setattr(server, "_child_run_active", lambda *args: False)
    # An unfenced build on the base must exit without loading an agent or profile.
    monkeypatch.setattr(server, "_await_resume_history", lambda *args: False)
    monkeypatch.setattr(server, "_finish_agent_build", lambda *args, **kwargs: None)

    try:
        server._start_agent_build(sid, current)

        assert current["agent_build_started"] is False, (
            "a held session-control token must keep a cold build unstarted"
        )
        assert current["lazy"] is True
        assert current.get("_agent_build_thread") is None
        assert not current["agent_ready"].is_set()
        assert current["_session_control_inflight"] is token
    finally:
        build_thread = current.get("_agent_build_thread")
        if build_thread is not None:
            build_thread.join(timeout=5)
            assert not build_thread.is_alive(), "the base's rejected build must be cleaned up"


def test_control_token_blocks_late_agent_attachment(server, held_control_session, monkeypatch):
    sid, current, token = held_control_session
    monkeypatch.setattr(server, "_register_session_cwd", lambda *args: None)
    monkeypatch.setattr(server, "_session_todo_state", lambda *args: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("test-model", "test-provider"))
    agent = SimpleNamespace()

    attached = server._attach_built_agent(sid, current, agent)

    assert attached is False, "a held session-control token must reject late attachment"
    assert current["agent"] is None
    assert current["_session_control_inflight"] is token
