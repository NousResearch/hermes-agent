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
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="authored", assignee="forge")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    return tid


def _comment_row(task_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
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
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
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
    from hermes_cli import kanban_db_connect as kbc

    tid = comment_board
    conn = kbc.connect()
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
    from hermes_cli import kanban_db_connect as kbc

    tid = comment_board
    conn = kbc.connect()
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
    from hermes_cli import kanban_db_connect as kbc

    tid = comment_board
    conn = kbc.connect()
    try:
        with pytest.raises(ValueError):
            kb.add_comment(conn, tid, author="", body="x")
        with pytest.raises(ValueError):
            kb.add_comment(conn, tid, author="forge", body="   ")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Blocker-3 regression: stamp the TASK-LOCAL session identity, not the
# process-global env (gateway made HERMES_SESSION_* ContextVars for exactly
# this reason — concurrent tasks in one process clobber os.environ)
# ---------------------------------------------------------------------------

def test_comment_stamps_task_local_session_id(comment_board, monkeypatch):
    """With the gateway's task-local binding active, get_session_env wins
    over os.environ: the comment records the session of the task context
    that issued it, not whatever the process env happens to hold."""
    tid = comment_board
    from gateway import session_context as sc

    monkeypatch.setenv("HERMES_SESSION_ID", "stale_process_env_session")
    with sc.scoped_current_session_id("task_local_session_A"):
        from tools import kanban_tools as kt
        out = kt._handle_comment({"task_id": tid, "body": "from context A"})
    assert json.loads(out).get("ok") is True, out
    assert _comment_row(tid)["session_id"] == "task_local_session_A"


def test_two_task_local_contexts_one_process_leave_distinct_trails(
        comment_board):
    """Two task-local session contexts sharing ONE process stamp their own
    ids — the process-global env could only ever record the last writer."""
    tid = comment_board
    from gateway import session_context as sc
    from tools import kanban_tools as kt

    with sc.scoped_current_session_id("session_of_task_1"):
        kt._handle_comment({"task_id": tid, "body": "context 1"})
    with sc.scoped_current_session_id("session_of_task_2"):
        kt._handle_comment({"task_id": tid, "body": "context 2"})

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        rows = conn.execute(
            "SELECT body, session_id FROM task_comments WHERE task_id = ? "
            "ORDER BY id ASC", (tid,)).fetchall()
    finally:
        conn.close()
    assert [(r["body"], r["session_id"]) for r in rows] == [
        ("context 1", "session_of_task_1"),
        ("context 2", "session_of_task_2"),
    ]


def test_worker_context_rendering_marks_off_run_session(comment_board):
    """The #98750 stale-comment symptom: worker_context must distinguish
    provenance, not render 'comment from worker X' bare. session-bearing
    comments render their (raw) session id for current-vs-ended ownership
    checks; legacy NULL rows render unchanged."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = comment_board
    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="forge", body="with provenance",
                       session_id="ended_run_session")
        kb.add_comment(conn, tid, author="forge", body="legacy row")
        ctx = kb.build_worker_context(conn, tid)
    finally:
        conn.close()
    assert "ended_run_session" in ctx          # provenance surfaced
    assert "legacy row" in ctx                 # NULL session: unchanged form
    assert "comment from worker `forge`" in ctx  # author framing intact
