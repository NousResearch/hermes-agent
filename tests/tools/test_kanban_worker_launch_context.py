"""Fail-closed kanban worker launch validation (issue #77825).

A kanban worker is a heredoc intent that must not silently enter the agent loop
without a live dispatcher-pinned task on the resolved board. The pre-model check
catches six launch contexts and refuses to start the agent when the worker would
otherwise race the canonical owner:

  - no worker intent (regular chat session)                       -> OK to start
  - dispatcher env + valid live task on the resolved board        -> OK to start
  - explicit "work kanban task <id>" prompt with no env           -> refuse
  - dispatcher env pointing at a wrong/missing board DB           -> refuse
  - dispatcher env whose task row is no longer running            -> refuse
  - dispatcher env whose run id / claim lock / claim expiry       -> refuse
    mismatch (stale claim, already-claimed by another worker)

The validation lives in tools/kanban_tools.py because the worker registration
already lives there; cli_single_query.py calls it before the agent starts so a
refused launch is a single, actionable stderr line and a nonzero exit, never a
partial model execution.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def launch_env(monkeypatch, tmp_path):
    """Isolated ``HERMES_HOME`` with one running claim. Returns the task id,
    its run id, and the board DB path. The conftest autouse already redirects
    ``HERMES_HOME``; we just pin the boards root for the DB."""
    boards_root = tmp_path / "boards"
    boards_root.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_BOARDS_ROOT", str(boards_root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="worker-launch", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
        lock = conn.execute(
            "SELECT claim_lock FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["claim_lock"] or ""
    finally:
        conn.close()

    db_path = kb.kanban_db_path()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    return {"tid": tid, "run_id": run_id, "claim_lock": lock, "db_path": db_path}


def _scrub_worker_env(monkeypatch) -> None:
    for k in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(k, raising=False)


def _row_factory_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db_path.absolute().as_uri()}?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# 1. Valid canonical launch -> proceed
# ---------------------------------------------------------------------------


def test_valid_canonical_launch_returns_true(monkeypatch, launch_env):
    """A worker launched with the full set of dispatcher-pinned coordinates,
    all of them matching the live run on the resolved board, is allowed to
    start. The bare happy path that registers and tests against every
    acceptance criterion would mask a regression on the read path itself."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", launch_env["tid"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(launch_env["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", launch_env["claim_lock"])
    monkeypatch.setenv("HERMES_KANBAN_DB", str(launch_env["db_path"]))

    from tools.kanban_tools import validate_worker_launch_context
    assert validate_worker_launch_context(query=None) is True, (
        f"valid canonical launch refused; env HERMES_KANBAN_TASK="
        f"{launch_env['tid']}, HERMES_KANBAN_RUN_ID={launch_env['run_id']}, "
        f"claim_lock={launch_env['claim_lock']!r}"
    )


# ---------------------------------------------------------------------------
# 2. Worker-intent query with no env -> refuse
# ---------------------------------------------------------------------------


def test_work_kanban_task_prompt_without_env_returns_false(monkeypatch, tmp_path):
    """A helper launched with the human query ``work kanban task t_<id>``
    but without dispatcher-provided environment. Without DB / run id / claim
    lock, the worker would race the canonical owner. Fail closed."""
    _scrub_worker_env(monkeypatch)

    from tools.kanban_tools import validate_worker_launch_context
    # Synthetic task id matches the dispatcher's t_<hex> shape so the regex
    # recognises it as worker intent.
    fake_task_id = "t_" + "0123456789abcdef"
    query = f"work kanban task {fake_task_id}"
    assert validate_worker_launch_context(query=query) is False, (
        "prompt-only worker intent must be refused when env is incomplete; "
        "the agent would otherwise run unclaimed against an unknown task"
    )


# ---------------------------------------------------------------------------
# 3. Wrong DB -> refuse
# ---------------------------------------------------------------------------


def test_kanban_db_path_does_not_exist_returns_false(monkeypatch, launch_env):
    """``HERMES_KANBAN_DB`` pointing at a non-existent board database."""
    _scrub_worker_env(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", launch_env["tid"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(launch_env["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", launch_env["claim_lock"])
    # Point at a board DB that does not exist on disk.
    monkeypatch.setenv(
        "HERMES_KANBAN_DB",
        str(launch_env["db_path"].parent / "does-not-exist" / "kanban.db"),
    )

    from tools.kanban_tools import validate_worker_launch_context
    assert validate_worker_launch_context(query=None) is False, (
        "missing board DB must refuse; the agent would otherwise run with "
        "no board, no claim, and no run it can prove"
    )


def test_kanban_db_path_points_at_wrong_board_returns_false(monkeypatch, launch_env):
    """``HERMES_KANBAN_DB`` points at a DIFFERENT board's DB where the
    task id does not exist. Same symptom, different fail-closed condition."""
    _scrub_worker_env(monkeypatch)
    # Create a SECOND board with its own DB; the worker is pinned to that
    # DB instead of the one that owns the task.
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="other")
    other_db = kb.kanban_db_path(board="other")

    monkeypatch.setenv("HERMES_KANBAN_TASK", launch_env["tid"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(launch_env["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", launch_env["claim_lock"])
    monkeypatch.setenv("HERMES_KANBAN_DB", str(other_db))

    from tools.kanban_tools import validate_worker_launch_context
    assert validate_worker_launch_context(query=None) is False, (
        f"wrong-board DB {other_db} must refuse; task {launch_env['tid']} "
        f"lives on a different board so this worker cannot prove ownership"
    )


# ---------------------------------------------------------------------------
# 4. Stale run id / claim lock / expiry -> refuse
# ---------------------------------------------------------------------------


def test_run_id_mismatch_returns_false(monkeypatch, launch_env):
    """The dispatcher-pinned run id no longer matches the live run."""
    _scrub_worker_env(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", launch_env["tid"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(launch_env["run_id"] + 9999))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", launch_env["claim_lock"])

    from tools.kanban_tools import validate_worker_launch_context
    assert validate_worker_launch_context(query=None) is False, (
        f"HERMES_KANBAN_RUN_ID={launch_env['run_id'] + 9999} does not match "
        f"task's live current_run_id={launch_env['run_id']}; this is a stale "
        f"worker that would corrupt the live run on completion"
    )


def test_claim_lock_mismatch_returns_false(monkeypatch, launch_env):
    """The dispatcher-pinned claim lock is no longer the live lock."""
    _scrub_worker_env(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", launch_env["tid"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(launch_env["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "wrong-lock")

    from tools.kanban_tools import validate_worker_launch_context
    assert validate_worker_launch_context(query=None) is False, (
        "claim_lock mismatch means the live claim belongs to another "
        "worker (or was reset); this worker cannot prove ownership"
    )


def test_expired_claim_returns_false(monkeypatch, launch_env):
    """The task's claim_expires is in the past; a successful pre-model
    launch would resurrect an expired claim and race the canonical owner."""
    _scrub_worker_env(monkeypatch)
    # Push claim_expires into the past.
    conn = _row_factory_conn(launch_env["db_path"])
    try:
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 3600, launch_env["tid"]),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", launch_env["tid"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(launch_env["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", launch_env["claim_lock"])

    from tools.kanban_tools import validate_worker_launch_context
    assert validate_worker_launch_context(query=None) is False, (
        "expired claim means the dispatcher has already rotated to a fresh "
        "claim; this worker must not start"
    )


# ---------------------------------------------------------------------------
# 5. Task already in a non-running state -> refuse
# ---------------------------------------------------------------------------


def test_task_not_running_returns_false(monkeypatch, launch_env):
    """The task row exists but its status is no longer ``running``
    (e.g., already completed or blocked). Launching an unclaimed worker
    against a terminal task is the canonical untracked-editor race."""
    _scrub_worker_env(monkeypatch)
    conn = _row_factory_conn(launch_env["db_path"])
    try:
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (launch_env["tid"],))
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", launch_env["tid"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(launch_env["run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", launch_env["claim_lock"])

    from tools.kanban_tools import validate_worker_launch_context
    assert validate_worker_launch_context(query=None) is False, (
        "task status is no longer running; the worker would either run on a "
        "terminal card or silently no-op the dispatcher"
    )


# ---------------------------------------------------------------------------
# 6. No worker intent at all -> proceed
# ---------------------------------------------------------------------------


def test_no_worker_intent_returns_true(monkeypatch, tmp_path):
    """A normal ``hermes chat`` session (no env, no worker prompt) returns
    True so the agent starts. The validation must not gate regular chat."""
    _scrub_worker_env(monkeypatch)

    from tools.kanban_tools import validate_worker_launch_context
    assert validate_worker_launch_context(query=None) is True
    assert validate_worker_launch_context(query="hello world") is True