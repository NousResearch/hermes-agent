"""Agent construction is not a session reset (regression for #156607)."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import server


@pytest.mark.parametrize("build_path", ["eager", "deferred"])
def test_same_session_agent_build_does_not_emit_reset(monkeypatch, build_path):
    """Both resume build paths keep plugin-owned resources on the stored id."""
    sid, key = "resume-runtime", "stored-conversation"
    hook = Mock(return_value=[])
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a: threading.Event())
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a: False)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_hydrate_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_session_todo_state", lambda *a: None)
    monkeypatch.setattr(server, "_session_info", lambda *a: {})
    monkeypatch.setattr(server, "_emit", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: None)
    monkeypatch.setattr(server, "_load_cfg", dict)
    monkeypatch.setattr(server, "_probe_config_health", lambda *a: None)
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: None)
    agent = SimpleNamespace(session_id=key)
    monkeypatch.setattr(server, "_make_agent", lambda *a, **kw: agent)
    try:
        if build_path == "eager":
            server._init_session(sid, key, agent, [], source="desktop")
        else:
            record = server._deferred_session_record(
                key, cols=80, cwd="", history=[], lease=None, source="desktop")
            server._sessions[sid] = record
            server._start_agent_build(sid, record)
            assert record["agent_ready"].wait(timeout=10), "resume build did not settle"
            assert record.get("agent_error") is None
        session = server._sessions[sid]
        assert session["agent"].session_id == key
        # Rebuilding services for this same conversation is not a rotation either.
        server._start_session_services(sid, key, session)
        assert session["_notif_stop"] is not None
        hook.assert_not_called()
    finally:
        server._sessions.pop(sid, None)
