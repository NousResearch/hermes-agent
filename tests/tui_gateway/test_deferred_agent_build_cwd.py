"""Regression coverage for logical cwd propagation in deferred TUI/Desktop builds."""

import threading
import uuid
from types import SimpleNamespace

from tui_gateway import server


def test_deferred_agent_build_threads_session_cwd(monkeypatch, tmp_path):
    """A cold Desktop build must not fall back to the serve process cwd."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = {}
    built = threading.Event()
    ready = threading.Event()
    sid = f"cwd-build-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"cwd-key-{uuid.uuid4().hex[:8]}",
        "cwd": str(workspace),
    }

    def fake_set_session_context(key, cwd=None):
        captured["context_key"] = key
        captured["context_cwd"] = cwd
        return []

    def fake_make_agent(*args, **kwargs):
        captured["agent_cwd"] = kwargs.get("cwd_override")
        built.set()
        return SimpleNamespace(model="test", session_id=session["session_key"])

    monkeypatch.setattr(server, "_set_session_context", fake_set_session_context)
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(
        "tui_gateway.entry.ensure_mcp_discovery_started", lambda: None
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=15), "agent build thread never called _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)
        from tools.approval import unregister_gateway_notify

        unregister_gateway_notify(session["session_key"])

    assert captured["context_cwd"] == str(workspace)
    assert captured["agent_cwd"] == str(workspace)
