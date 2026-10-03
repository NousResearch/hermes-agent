"""Other-installs notice on the TUI: interactive stdio session only, once, setting-aware."""

from __future__ import annotations

import json

import pytest

from tui_gateway import server
from tui_gateway.session_installs_notice import NOTICE_KEY, maybe_show

NOTICE = ("2 other Hermes installs found on this machine. Run `hermes installs` to see them, "
          "or `hermes installs dismiss` to hide this.")


@pytest.fixture
def frames(monkeypatch):
    sent = []
    monkeypatch.setattr(server, "write_json", lambda frame: sent.append(frame) or True)
    return sent


@pytest.fixture
def other_installs(monkeypatch):
    monkeypatch.setattr("hermes_cli.installs.launch_notice", lambda *a, **k: NOTICE)


def _show_payloads(frames):
    return [f["params"]["payload"] for f in frames
            if f.get("params", {}).get("type") == "notification.show"]


def _tui_session(**extra):
    return {"transport": server._stdio_transport, **extra}


def _suppress_config(tmp_path, monkeypatch, suppress):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    cfg = {"display": {"suppress_warning_notifications": suppress}} if suppress else {}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")


def test_interactive_tui_session_shows_the_notice_once(frames, other_installs, tmp_path, monkeypatch):
    _suppress_config(tmp_path, monkeypatch, suppress=False)
    session = _tui_session()
    maybe_show("s1", session)
    maybe_show("s1", session)

    payloads = _show_payloads(frames)
    assert len(payloads) == 1
    assert payloads[0]["text"] == NOTICE
    assert payloads[0]["level"] == "warn"
    assert payloads[0]["key"] == NOTICE_KEY


def test_no_other_installs_sends_nothing(frames, tmp_path, monkeypatch):
    _suppress_config(tmp_path, monkeypatch, suppress=False)
    monkeypatch.setattr("hermes_cli.installs.launch_notice", lambda *a, **k: None)
    maybe_show("s1", _tui_session())
    assert _show_payloads(frames) == []


def test_suppressed_warning_notifications_send_nothing(frames, other_installs, tmp_path, monkeypatch):
    _suppress_config(tmp_path, monkeypatch, suppress=True)
    maybe_show("s1", _tui_session())
    assert _show_payloads(frames) == []


def test_non_stdio_transport_sends_nothing(frames, other_installs, tmp_path, monkeypatch):
    _suppress_config(tmp_path, monkeypatch, suppress=False)
    maybe_show("s1", {"transport": object()})
    assert _show_payloads(frames) == []


@pytest.mark.parametrize("flag", ["room_plumbing", "pending_hidden"])
def test_hidden_and_messaging_sessions_send_nothing(frames, other_installs, tmp_path, monkeypatch, flag):
    _suppress_config(tmp_path, monkeypatch, suppress=False)
    maybe_show("s1", _tui_session(**{flag: True}))
    assert _show_payloads(frames) == []


def test_launch_notice_failure_is_swallowed(frames, tmp_path, monkeypatch):
    _suppress_config(tmp_path, monkeypatch, suppress=False)

    def boom(*a, **k):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr("hermes_cli.installs.launch_notice", boom)
    maybe_show("s1", _tui_session())
    assert _show_payloads(frames) == []


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False


def _build_env(monkeypatch, tmp_path, events, *, resume_ok=True):
    import types

    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_await_resume_history", lambda sid, current: resume_ok)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: None)
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_bind_build_profile_scopes", lambda home: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(server, "_make_agent", lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(server, "_attach_built_agent", lambda sid, current, agent: True)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a, **k: False)
    monkeypatch.setattr(server, "_announce_built_agent", lambda *a, **k: events.append("announced"))
    monkeypatch.setattr(server, "_finish_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_show_installs_notice", lambda sid, session: events.append("notice"))


@pytest.mark.parametrize(("resume_ok", "expected"), [(True, ["announced", "notice"]), (False, [])])
def test_notice_follows_the_built_agent_not_the_create_response(monkeypatch, tmp_path, resume_ok, expected):
    import threading

    events: list[str] = []
    _build_env(monkeypatch, tmp_path, events, resume_ok=resume_ok)
    session = {"agent": None, "agent_error": None, "session_key": "k", "agent_ready": threading.Event(),
               "cwd": str(tmp_path), "profile_home": None}
    server._sessions["build-sid"] = session
    try:
        server._start_agent_build("build-sid", session)
    finally:
        server._sessions.pop("build-sid", None)

    assert events == expected
