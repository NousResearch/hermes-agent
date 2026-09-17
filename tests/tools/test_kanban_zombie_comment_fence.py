"""Kanban worker comment fencing at the tool layer.

Companion to tests/hermes_cli/test_kanban_stale_claim_fencing.py (#99283):
the DB-level fence is exercised there; this file proves the real worker path
carries it. A dispatcher-spawned worker has ``HERMES_KANBAN_RUN_ID`` pinned in
its env; after its claim is reclaimed and a successor re-claims, the zombie's
``kanban_comment`` (still holding the dead run id) must come back as a tool
error, and the thread the successor reads stays clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools import kanban_tools as kt


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_zombie_worker_comment_rejected_through_tool(kanban_home, monkeypatch):
    """Worker claims, env pins its run id, claim is reclaimed + re-claimed by
    a successor — the zombie's kanban_comment is a structured tool error and
    nothing lands on the thread."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="zombie comment", assignee="w")
        zombie = kb.claim_task(conn, tid, claimer="10.9.9.9:9:1")
        assert zombie is not None
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(zombie.current_run_id))

        # Reclaim the zombie's claim, then a successor claims.
        conn.execute("UPDATE tasks SET claim_expires = 1 WHERE id = ?", (tid,))
        conn.commit()
        assert kb.release_stale_claims(conn, signal_fn=None) == 1
        successor = kb.claim_task(conn, tid, claimer=kb._claimer_id())
        assert successor is not None
        assert successor.current_run_id != zombie.current_run_id

    out = json.loads(kt._handle_comment({"task_id": tid, "body": "zombie says hi"}))
    assert out.get("error"), "zombie worker comment landed on successor's thread"

    with kbc.connect() as conn:
        assert kb.list_comments(conn, tid) == []
        # The successor's own comment (live run id) still lands.
        monkeypatch.setenv(
            "HERMES_KANBAN_RUN_ID", str(kb.get_task(conn, tid).current_run_id))
        out = json.loads(kt._handle_comment(
            {"task_id": tid, "body": "successor note"}))
        assert out.get("ok") is True
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert "successor note" in comments[0].body
