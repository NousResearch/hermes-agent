"""Agent reactions address durable session ids, not routing keys (#139102)."""

import json

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tools import react_to_message_tool as reactions


def _reactions(store, session_id, row_id):
    """The reactions stored on one row of the session's transcript (none for a row it does not hold)."""
    row = next((r for r in transcript_rows(store, session_id) if r["id"] == row_id), None)
    return (row["display_metadata"] or {}).get("reactions", []) if row else []


@pytest.mark.parametrize("routing_key", ["agent:main:local:dm:chat-a", "chat-b"])
def test_reaction_uses_durable_session_with_routing_key(tmp_path, monkeypatch, routing_key):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, "chat-a", "local")
        seed_messages(db, "chat-a", {"role": "user", "content": "Earlier message"})
        earlier = db.messages.latest_row_id("chat-a", role="user")
        seed_messages(db, "chat-a", {"role": "user", "content": "Latest message"})
        latest = db.messages.latest_row_id("chat-a", role="user")
        seed_session(db, "chat-b", "local")
        seed_messages(db, "chat-b", {"role": "user", "content": "Other conversation"})
        foreign = db.messages.latest_row_id("chat-b", role="user")
    emitted = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
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
    with open_persistence(profile_scope(home=tmp_path)) as reopened:
        assert _reactions(reopened, "chat-a", latest)[0]["emoji"] == "👍"
        assert _reactions(reopened, "chat-a", earlier) == []
        assert _reactions(reopened, "chat-b", foreign) == []
    assert [payload["row_id"] for _, payload in emitted] == [latest, earlier, earlier, earlier]
    assert all(name == "message.reaction" for name, _ in emitted)
    assert emitted[0][1]["reactions"][0]["emoji"] == "👍"
    assert emitted[-1][1]["reactions"] == []


def test_reaction_keeps_legacy_key_fallback_and_empty_session(tmp_path, monkeypatch):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, "legacy", "local")
        seed_messages(db, "legacy", {"role": "user", "content": "Legacy message"})
        row = db.messages.latest_row_id("legacy", role="user")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(reactions.desktop_ui, "emit", lambda *args: emitted.append(args))
    tokens = set_session_vars(session_key="legacy", session_id="missing")
    try:
        result = json.loads(reactions.registry.dispatch("react_to_message", {"emoji": "👍"}))
        assert "No user message" in result["error"]
        assert emitted == []
    finally:
        clear_session_vars(tokens)
    with open_persistence(profile_scope(home=tmp_path)) as db:
        assert _reactions(db, "legacy", row) == []
    tokens = set_session_vars(session_key="legacy", session_id="")
    try:
        assert json.loads(reactions.registry.dispatch("react_to_message", {"emoji": "👍"}))["success"]
    finally:
        clear_session_vars(tokens)
    result = json.loads(reactions.registry.dispatch("react_to_message", {"emoji": "👍"}))
    assert "No active session" in result["error"]
