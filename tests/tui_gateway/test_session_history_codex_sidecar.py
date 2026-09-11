"""RPC history must retain visible Responses-API assistant sidecars (#68321)."""

from __future__ import annotations

import threading

from hermes_state import SessionDB
import tui_gateway.server as server


def test_session_history_preserves_codex_message_items(tmp_path):
    message_items = [
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Persisted answer"}],
        }
    ]
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("stored-session", source="desktop")
    db.append_message(
        "stored-session",
        "assistant",
        "",
        codex_message_items=message_items,
        reasoning="Thinking before the tool",
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
    )
    previous_db = server._db
    setattr(server, "_db", db)
    server._sessions["runtime-session"] = {
        "session_key": "stored-session",
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "agent": None,
    }

    try:
        response = server.handle_request(
            {"id": "1", "method": "session.history", "params": {"session_id": "runtime-session"}}
        )
    finally:
        server._sessions.pop("runtime-session", None)
        setattr(server, "_db", previous_db)
        db.close()

    assert isinstance(response, dict)
    assert "error" not in response
    assert response["result"]["count"] == 1
    messages = response["result"]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["text"] == ""
    assert messages[0]["codex_message_items"] == message_items
