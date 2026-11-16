"""A cron run fired in-process from a Kanban worker inherits ``HERMES_KANBAN_*`` but does not
own the worker's card (#149512, #103883): naming that card explicitly must not pass the
ownership gate, borrow the worker's run id, or extend the worker's claim."""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def worker_board(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        worker_tid = kb.create_task(conn, title="worker card", assignee="test-worker")
        kb.claim_task(conn, worker_tid)
        run_id = kb._current_run_id(conn, worker_tid)
        other_tid = kb.create_task(conn, title="other card")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", worker_tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock-abc")
    return worker_tid, other_tid


def _status(tid):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return kb.get_task(conn, tid).status
    finally:
        conn.close()


def _comments(tid):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return kb.list_comments(conn, tid)
    finally:
        conn.close()


@pytest.mark.parametrize("tool, extra", [
    ("_handle_complete", {"summary": "cron says done"}),
    ("_handle_block", {"reason": "cron says blocked"}),
    ("_handle_heartbeat", {"note": "cron beat"}),
    ("_handle_comment", {"body": "cron note"}),
])
def test_cron_run_cannot_mutate_inherited_worker_card(worker_board, tool, extra):
    from agent.delegation_context import non_dispatcher_owned_context
    from tools import kanban_tools as kt

    worker_tid, _ = worker_board
    with non_dispatcher_owned_context():
        out = json.loads(getattr(kt, tool)({"task_id": worker_tid, **extra}))
    assert not out.get("ok"), out
    assert "does not own it" in json.dumps(out)
    assert _status(worker_tid) == "running"
    assert _comments(worker_tid) == []


def test_cron_run_does_not_borrow_worker_identity(worker_board, monkeypatch):
    from agent.delegation_context import non_dispatcher_owned_context
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    worker_tid, _ = worker_board
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
    assert kt._worker_run_id(worker_tid) is not None
    with non_dispatcher_owned_context():
        assert kt._worker_run_id(worker_tid) is None
        assert kt.heartbeat_current_worker_from_env() is False
    conn = kbc.connect()
    try:
        row = conn.execute("SELECT last_heartbeat_at FROM tasks WHERE id = ?", (worker_tid,)).fetchone()
    finally:
        conn.close()
    assert row[0] is None


def test_cron_run_may_still_touch_other_cards(worker_board):
    from agent.delegation_context import non_dispatcher_owned_context
    from tools import kanban_tools as kt

    _, other_tid = worker_board
    with non_dispatcher_owned_context():
        out = json.loads(kt._handle_comment({"task_id": other_tid, "body": "finding"}))
    assert out.get("ok") is True, out
    assert len(_comments(other_tid)) == 1
    with non_dispatcher_owned_context():
        out = json.loads(kt._handle_block({"task_id": other_tid, "reason": "cron found a blocker"}))
    assert out.get("ok") is True, out
    assert _status(other_tid) == "blocked"


def test_dispatcher_worker_still_completes_own_card(worker_board):
    from tools import kanban_tools as kt

    worker_tid, _ = worker_board
    out = json.loads(kt._handle_complete({"task_id": worker_tid, "summary": "done"}))
    assert out.get("ok") is True, out
    assert _status(worker_tid) == "done"
