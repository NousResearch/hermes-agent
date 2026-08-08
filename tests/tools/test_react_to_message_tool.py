"""Resource-lifecycle tests for the desktop reaction tool."""

import json
from unittest.mock import MagicMock

from tools import react_to_message_tool as reaction_tool


def _install_db(monkeypatch, db):
    monkeypatch.setattr(reaction_tool, "get_session_env", lambda *_args: "session-1")
    monkeypatch.setattr(reaction_tool, "_open_session_db", lambda: db)
    monkeypatch.setattr(reaction_tool.desktop_ui, "emit", MagicMock())


def test_reaction_closes_session_db_after_success(monkeypatch):
    db = MagicMock()
    db.latest_message_row_id.return_value = 42
    db.set_message_reaction.return_value = [{"emoji": "👍", "author": "agent"}]
    _install_db(monkeypatch, db)

    result = json.loads(reaction_tool.react_to_message_tool("👍"))

    assert result["success"] is True
    assert result["row_id"] == 42
    db.close.assert_called_once_with()


def test_reaction_closes_session_db_when_target_is_missing(monkeypatch):
    db = MagicMock()
    db.latest_message_row_id.return_value = None
    _install_db(monkeypatch, db)

    result = json.loads(reaction_tool.react_to_message_tool("👍"))

    assert "No user message" in result["error"]
    db.close.assert_called_once_with()


def test_reaction_closes_session_db_after_write_failure(monkeypatch):
    db = MagicMock()
    db.latest_message_row_id.return_value = 42
    db.set_message_reaction.side_effect = RuntimeError("write failed")
    _install_db(monkeypatch, db)

    result = json.loads(reaction_tool.react_to_message_tool("👍"))

    assert "write failed" in result["error"]
    db.close.assert_called_once_with()
