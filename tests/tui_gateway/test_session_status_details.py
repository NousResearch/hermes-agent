"""Structured session status matches the live agent and the compatibility text output."""

import threading
import types

from tui_gateway import server


def test_session_status_reads_live_gateway_agent(monkeypatch):
    agent = types.SimpleNamespace(
        model="live-model",
        provider="live-provider",
        session_total_tokens=1234,
    )
    monkeypatch.setattr(server, "_sessions", {"sid": {
        "agent": agent, "running": True, "session_key": "session-key",
        "history": [], "history_lock": threading.Lock(),
    }})

    class _DB:
        def get_session(self, key):
            assert key == "session-key"
            return {
                "title": "Live TUI",
                "started_at": 1_700_000_000,
                "updated_at": 1_700_000_060,
            }

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.status", "params": {"session_id": "sid"}}
        )
    finally:
        server._sessions.pop("sid", None)

    out = resp["result"]["output"]
    for value in ("session-key", "Live TUI", "live-model", "live-provider", "1,234"):
        assert value in out
    details = resp["result"]["details"]
    assert details["session_id"] == "session-key"
    assert details["project"] == ""
    assert details["title"] == "Live TUI"
    assert details["model"] == "live-model"
    assert details["provider"] == "live-provider"
    assert details["tokens"] == 1234
    assert details["agent_running"] is True
    assert details["created"]
    assert details["last_activity"]
