"""Agent reactions address durable session ids, not routing keys (#139102)."""

import json

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB
from tools import react_to_message_tool as reactions


@pytest.mark.parametrize("routing_key", ["agent:main:local:dm:chat-a", "chat-b"])
def test_reaction_uses_durable_session_with_routing_key(tmp_path, monkeypatch, routing_key):
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("chat-a", "local")
        db.append_message("chat-a", "user", "Earlier message")
        earlier = db.latest_message_row_id("chat-a", role="user")
        db.append_message("chat-a", "user", "Latest message")
        latest = db.latest_message_row_id("chat-a", role="user")
        db.create_session("chat-b", "local")
        db.append_message("chat-b", "user", "Other conversation")
        foreign = db.latest_message_row_id("chat-b", role="user")
    emitted = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", path)  # conftest pins this per test
    monkeypatch.setattr(reactions.desktop_ui, "emit", lambda *args: emitted.append(args))
    tokens = set_session_vars(session_key=routing_key, session_id="chat-a")
    try:
        def react(**args):
            return json.loads(reactions.registry.dispatch("react_to_message", args))

        result = react(emoji="👍")
        assert result.get("success"), result
        assert result["row_id"] == latest
        assert react(emoji="❤️", messages_back=1)["row_id"] == earlier
        assert react(emoji="😂", message_row_id=earlier)["success"]
        assert "error" in react(emoji="👍", message_row_id=foreign)
        assert "error" in react(emoji="👍", messages_back=99)
        assert react(emoji="", message_row_id=earlier)["success"]
    finally:
        clear_session_vars(tokens)
    with SessionDB(db_path=path) as reopened:
        assert reopened.get_message_reactions("chat-a", latest)[0]["emoji"] == "👍"
        assert reopened.get_message_reactions("chat-a", earlier) == []
        assert reopened.get_message_reactions("chat-b", foreign) == []
    assert [payload["row_id"] for _, payload in emitted] == [latest, earlier, earlier, earlier]
    assert all(name == "message.reaction" for name, _ in emitted)
    assert emitted[0][1]["reactions"][0]["emoji"] == "👍"
    assert emitted[-1][1]["reactions"] == []


def test_reaction_keeps_legacy_key_fallback_and_empty_session(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("legacy", "local")
        db.append_message("legacy", "user", "Legacy message")
        row = db.latest_message_row_id("legacy", role="user")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", path)  # conftest pins this per test
    emitted = []
    monkeypatch.setattr(reactions.desktop_ui, "emit", lambda *args: emitted.append(args))
    tokens = set_session_vars(session_key="legacy", session_id="missing")
    try:
        result = json.loads(reactions.registry.dispatch("react_to_message", {"emoji": "👍"}))
        assert "No user message" in result["error"]
        assert emitted == []
    finally:
        clear_session_vars(tokens)
    with SessionDB(db_path=path) as db:
        assert db.get_message_reactions("legacy", row) == []
    tokens = set_session_vars(session_key="legacy", session_id="")
    try:
        assert json.loads(reactions.registry.dispatch("react_to_message", {"emoji": "👍"}))["success"]
    finally:
        clear_session_vars(tokens)
    result = json.loads(reactions.registry.dispatch("react_to_message", {"emoji": "👍"}))
    assert "No active session" in result["error"]
