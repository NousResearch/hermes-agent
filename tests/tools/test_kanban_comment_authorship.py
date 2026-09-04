"""Comment authorship provenance: task_comments.session_id.

On a claimed card, the ``author`` profile alone cannot tell WHICH agent
context wrote a comment: a bot-mode self-DM fork (incident class above) ran a
second full agent context of the same profile, and its comment was
indistinguishable from the claimant's. The fix stamps the calling session's
id at insert time (from HERMES_SESSION_ID in the agent tool layer), recorded
once, immutable — never the card's claimant identity, never caller-supplied.
"""
from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def comment_board(monkeypatch, tmp_path):
    """Isolated board DB + clean session env; returns a created task id."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="authored", assignee="forge")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    return tid


def _comment_row(task_id):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        return conn.execute(
            "SELECT author, body, session_id FROM task_comments "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,)).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool layer stamps the calling session at call time
# ---------------------------------------------------------------------------

def test_kanban_comment_stamps_calling_session(comment_board, monkeypatch):
    tid = comment_board
    fork_session = "20260830_144815_forksession"
    monkeypatch.setenv("HERMES_SESSION_ID", fork_session)
    from tools import kanban_tools as kt

    out = kt._handle_comment({"task_id": tid, "body": "from the fork"})
    assert json.loads(out).get("ok") is True, out
    row = _comment_row(tid)
    assert row["author"] == "forge"          # profile identity (unchanged)
    assert row["session_id"] == fork_session  # provenance: the real writer


def test_fork_session_differs_from_claimant_session(comment_board, monkeypatch):
    """Two agent contexts of one profile leave distinguishable trails."""
    tid = comment_board
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_SESSION_ID", "claimant_session_1")
    kt._handle_comment({"task_id": tid, "body": "claimant here"})
    monkeypatch.setenv("HERMES_SESSION_ID", "fork_session_2")
    kt._handle_comment({"task_id": tid, "body": "fork here"})

    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        rows = conn.execute(
            "SELECT session_id FROM task_comments WHERE task_id = ? "
            "ORDER BY id ASC", (tid,)).fetchall()
    finally:
        conn.close()
    assert [r["session_id"] for r in rows] == [
        "claimant_session_1", "fork_session_2"]


def test_missing_session_env_records_null(comment_board, monkeypatch):
    tid = comment_board
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from tools import kanban_tools as kt

    out = kt._handle_comment({"task_id": tid, "body": "no session env"})
    assert json.loads(out).get("ok") is True, out
    assert _comment_row(tid)["session_id"] is None


# ---------------------------------------------------------------------------
# DB layer: provenance is write-once, defaults to NULL, never caller-authored
# ---------------------------------------------------------------------------

def test_add_comment_session_id_defaults_and_strips(comment_board):
    from hermes_cli import kanban_db as kb

    tid = comment_board
    conn = kb.connect()
    try:
        kb.add_comment(conn, tid, author="forge", body="no provenance")
        kb.add_comment(conn, tid, author="forge", body="blank provenance",
                       session_id="   ")  # whitespace-only → NULL
        kb.add_comment(conn, tid, author="forge", body="padded",
                       session_id="  sess7  ")
        legacy = conn.execute(
            "SELECT session_id FROM task_comments WHERE task_id = ? "
            "ORDER BY id ASC", (tid,)).fetchall()
    finally:
        conn.close()
    assert legacy[0]["session_id"] is None
    assert legacy[1]["session_id"] is None
    assert legacy[2]["session_id"] == "sess7"


def test_comment_dataclass_carries_session_id(comment_board):
    from hermes_cli import kanban_db as kb

    tid = comment_board
    conn = kb.connect()
    try:
        kb.add_comment(conn, tid, author="forge", body="hi",
                       session_id="sessX")
        comments = kb.list_comments(conn, tid)
    finally:
        conn.close()
    assert comments[-1].session_id == "sessX"


def test_add_comment_rejects_bad_input(comment_board):
    """Pre-existing validation intact (author/body required)."""
    from hermes_cli import kanban_db as kb

    tid = comment_board
    conn = kb.connect()
    try:
        with pytest.raises(ValueError):
            kb.add_comment(conn, tid, author="", body="x")
        with pytest.raises(ValueError):
            kb.add_comment(conn, tid, author="forge", body="   ")
    finally:
        conn.close()
