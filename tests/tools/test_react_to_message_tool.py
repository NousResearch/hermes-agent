"""Ownership tests for desktop message reactions."""

from unittest.mock import MagicMock

from tools import react_to_message_tool as reactions


def test_reaction_database_closes_when_write_fails(monkeypatch):
    persistence = MagicMock()
    persistence.messages.latest_row_id.return_value = 42
    persistence.messages.set_reaction.side_effect = RuntimeError("write failed")
    monkeypatch.setattr(reactions, "_open_session_db", lambda: persistence)
    monkeypatch.setattr(
        reactions,
        "get_session_env",
        lambda _name, _default="": "session-1",
    )

    result = reactions.react_to_message_tool("👍")

    assert "write failed" in result
    persistence.close.assert_called_once()
