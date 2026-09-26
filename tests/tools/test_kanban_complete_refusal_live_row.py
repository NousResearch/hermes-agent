"""A refusal answers from the LIVE row — never from a durable crash string.

``tools/kanban_tools._handle_complete`` used to fall back to ``tasks.last_failure_error`` whenever
``complete_task`` returned ``False``. That column is durable and describes a run that is OVER, so
the incident's call was answered with a two-day-old crash string and sent the worker chasing a stale
run while the live row said something else entirely (#123811). The reason now comes from
``kanban_db.live_row_refusal``: the card's own state, the run that owns it, and whether the caller's
run was superseded or closed by the infrastructure. Crash text survives only as labelled history.
"""

import json
import time

import pytest


@pytest.fixture
def superseded_worker(monkeypatch, tmp_path):
    """A running card owned by a run NEWER than the caller's, with a planted crash string."""
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
    planted = "crashed 2 days ago: this text must not be presented as the live reason"
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="live-row refusal", assignee="test-worker")
        assert kb.claim_task(conn, tid) is not None
        superseded = kb._current_run_id(conn, tid)
        assert superseded is not None
        # A newer attempt takes the card underneath the caller.
        with kb.write_txn(conn):
            cur = conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at) VALUES (?, 'running', ?)",
                (tid, int(time.time())),
            )
            conn.execute(
                "UPDATE tasks SET current_run_id = ?, last_failure_error = ? WHERE id = ?",
                (cur.lastrowid, planted, tid),
            )
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(superseded))
    return tid, superseded, planted


def test_the_refusal_names_the_live_owner_and_quotes_the_crash_only_as_history(superseded_worker):
    from tools import kanban_tools as kt

    tid, superseded, planted = superseded_worker
    out = json.loads(kt._handle_complete({"task_id": tid, "summary": "done, honestly"}))
    err = out.get("error") or ""
    assert err, out
    assert "status=running" in err, err
    assert f"your run {superseded} is SUPERSEDED" in err, err
    assert f"history only — NOT the current reason: {planted!r}" in err, err
    assert planted not in err.split("history only")[0], err


def test_the_refusal_does_not_mutate_the_card_a_newer_run_owns(superseded_worker):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    tid, superseded, _planted = superseded_worker
    kt._handle_complete({"task_id": tid, "summary": "done, honestly"})
    conn = kbc.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.current_run_id != superseded
    finally:
        conn.close()
