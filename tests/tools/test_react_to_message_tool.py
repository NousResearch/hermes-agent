"""Ownership tests for desktop message reactions."""

import json

from unittest.mock import MagicMock

from hermes_state import SessionDB
from tools import react_to_message_tool as reactions


def test_reaction_database_closes_when_write_fails(monkeypatch):
    db = MagicMock()
    db.latest_message_row_id.return_value = 42
    db.set_message_reaction.side_effect = RuntimeError("write failed")
    monkeypatch.setattr(reactions, "_open_session_db", lambda: db)
    monkeypatch.setattr(
        reactions,
        "get_session_env",
        lambda _name, _default="": "session-1",
    )

    result = reactions.react_to_message_tool("👍")

    assert "write failed" in result
    db.close.assert_called_once()


def test_explicit_reaction_accepts_compression_ancestor_and_exact_session(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    parent = db.create_session("parent", "test")
    parent_row = db.append_message(parent, "user", "before compression")
    db.end_session(parent, "compression")
    tip = db.create_session("tip", "test", parent_session_id=parent)
    tip_row = db.append_message(tip, "user", "after compression")
    db.close()

    monkeypatch.setattr(reactions, "_open_session_db", lambda: SessionDB(db_path=db_path))
    monkeypatch.setattr(reactions, "get_session_env", lambda _name, _default="": tip)

    ancestor_result = json.loads(reactions.react_to_message_tool("❤️", message_row_id=parent_row))
    exact_result = json.loads(reactions.react_to_message_tool("👍", message_row_id=tip_row))

    assert ancestor_result["success"] is True
    assert ancestor_result["row_id"] == parent_row
    assert exact_result["success"] is True
    assert exact_result["row_id"] == tip_row


def test_explicit_reaction_rejects_unrelated_and_branch_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    parent = db.create_session("parent", "test")
    db.end_session(parent, "compression")
    tip = db.create_session("tip", "test", parent_session_id=parent)
    unrelated = db.create_session("unrelated", "test")
    unrelated_row = db.append_message(unrelated, "user", "other conversation")
    branch = db.create_session(
        "branch", "test", parent_session_id=parent, model_config={"_branched_from": parent}
    )
    branch_row = db.append_message(branch, "user", "explicit branch")
    db.close()

    monkeypatch.setattr(reactions, "_open_session_db", lambda: SessionDB(db_path=db_path))
    monkeypatch.setattr(reactions, "get_session_env", lambda _name, _default="": tip)

    unrelated_result = json.loads(
        reactions.react_to_message_tool("❤️", message_row_id=unrelated_row)
    )
    branch_result = json.loads(reactions.react_to_message_tool("❤️", message_row_id=branch_row))

    assert unrelated_result["error"] == f"Message {unrelated_row} is not part of this conversation."
    assert branch_result["error"] == f"Message {branch_row} is not part of this conversation."
