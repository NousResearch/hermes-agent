"""TUI gateway transport errors must not echo credentials."""

from __future__ import annotations

import threading


SECRET = "sk-tuisecretvalue1234567890"


def test_err_redacts_secret_assignments():
    from tui_gateway.server import _err

    result = _err("r1", -32000, f"provider failed: access_token={SECRET}")

    msg = result["error"]["message"]
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == "r1"
    assert result["error"]["code"] == -32000
    assert SECRET not in msg
    assert "access_token=***" in msg


def test_err_fails_closed_when_redactor_fails(monkeypatch):
    from tui_gateway import server

    def broken_redactor(*_args, **_kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", broken_redactor)

    result = server._err("r2", -32000, f"provider failed: api_key={SECRET}")

    assert result["error"]["message"] == "Gateway error"
    assert SECRET not in result["error"]["message"]


def test_start_agent_build_redacts_init_error(monkeypatch):
    from tui_gateway import server

    emitted = []

    def fake_make_agent(*_args, **_kwargs):
        raise RuntimeError(f"init failed: refresh_token={SECRET}")

    monkeypatch.setattr(server, "_set_session_context", lambda target: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted.append(args))

    sid = "redact-build-sid"
    session = {
        "agent": None,
        "agent_ready": threading.Event(),
        "session_key": "k1",
        "profile_home": None,
        "model_override": None,
        "create_reasoning_override": None,
        "create_service_tier_override": None,
    }
    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert session["agent_ready"].wait(timeout=3), "agent build did not finish"
    finally:
        server._sessions.clear()

    assert SECRET not in session["agent_error"]
    assert "refresh_token=***" in session["agent_error"]

    error_events = [args for args in emitted if args and args[0] == "error"]
    assert error_events
    payload = error_events[-1][2]
    assert SECRET not in payload["message"]
    assert "refresh_token=***" in payload["message"]
