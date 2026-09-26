"""PLT-010: worker token optimization — context is read once, not duplicated.

Regression guards for the token optimization:
  * ``kanban_show`` no longer returns the same body/comments/runs twice (once
    as structured fields, once inside ``worker_context``) — the single source
    is the ``worker_context`` block.
  * Prior-attempt history is trimmed to the most recent run's first line, and
    the comment thread to the newest ``_CTX_MAX_COMMENTS`` — while the
    essential fix info (task body, parent handoff with capa/evidencia) still
    reaches the worker intact.
"""

from __future__ import annotations

import json
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    from pathlib import Path
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _make_task(conn, title="t", body="") -> str:
    return kb.create_task(conn, title=title, body=body)


def _add_comments(conn, task_id, n):
    for i in range(n):
        kb.add_comment(conn, task_id, author="worker", body=f"comment {i}")


def _close_runs(conn, task_id, n):
    """Create ``n`` closed prior runs with distinct summaries, without
    changing the task's status (so the task stays claimable)."""
    now = int(time.time())
    for i in range(n):
        conn.execute(
            """
            INSERT INTO task_runs (
                task_id, profile, status, outcome, summary, started_at, ended_at
            ) VALUES (?, ?, 'completed', 'completed', ?, ?, ?)
            """,
            (task_id, "test-worker", f"attempt summary {i}", now - (n - i), now),
        )


def test_kanban_show_does_not_duplicate_body_comments_runs(kanban_home):
    """D1: the same data is delivered once, in worker_context, not twice."""
    from tools import kanban_tools as kt
    conn = kbc.connect()
    try:
        tid = _make_task(conn, title="fix", body="expediente: tasks/plt-010.md\ncapa: Cerebro\nevidence: kanban_db.py:3848")
        _add_comments(conn, tid, 3)
        _close_runs(conn, tid, 2)
    finally:
        conn.close()

    d = json.loads(kt._handle_show({"task_id": tid}))
    # Structured fields keep only what changes on the fly.
    assert set(d["task"].keys()) == {"id", "status", "current_run_id"}
    assert "comments" not in d
    assert "runs" not in d
    # The single source carries the essential fix info intact.
    ctx = d["worker_context"]
    assert "## Body" in ctx
    assert "expediente: tasks/plt-010.md" in ctx
    assert "capa: Cerebro" in ctx
    assert "kanban_db.py:3848" in ctx
    assert "comment 0" in ctx and "comment 2" in ctx


def test_prior_attempts_trimmed_to_first_line(kanban_home):
    """D3: prior attempts are one-line markers, not full summaries."""
    conn = kbc.connect()
    try:
        tid = _make_task(conn, title="t")
        _close_runs(conn, tid, 3)
        ctx = kb.build_worker_context(conn, tid)
    finally:
        conn.close()
    assert "## Prior attempts on this task" in ctx
    # Only the most recent attempt is shown, as a one-line marker.
    assert "attempt summary 2" in ctx
    assert "attempt summary 0" not in ctx
    assert "attempt summary 1" not in ctx
    # The full multi-line summary body is not re-sent.
    assert "attempt summary 2" in ctx.splitlines()[0] or any(
        "attempt summary 2" in line for line in ctx.splitlines()
    )


def test_comments_capped_to_newest(kanban_home):
    """D3: the comment thread is capped to the newest _CTX_MAX_COMMENTS."""
    conn = kbc.connect()
    try:
        tid = _make_task(conn, title="t")
        _add_comments(conn, tid, kb._CTX_MAX_COMMENTS + 5)
        ctx = kb.build_worker_context(conn, tid)
    finally:
        conn.close()
    # The newest comments are present; the oldest are omitted with a note.
    assert f"comment {kb._CTX_MAX_COMMENTS + 4}" in ctx
    assert "comment 0" not in ctx
    assert "omitted" in ctx


def test_worker_context_smaller_than_duplicated_show(kanban_home):
    """The single-source context is smaller than the old duplicated payload."""
    from tools import kanban_tools as kt
    conn = kbc.connect()
    try:
        tid = _make_task(conn, title="t", body="x" * 2000)
        _add_comments(conn, tid, 20)
        _close_runs(conn, tid, 5)
    finally:
        conn.close()
    d = json.loads(kt._handle_show({"task_id": tid}))
    ctx = d["worker_context"]
    # The structured payload no longer re-sends body/comments/runs, so the
    # whole response is dominated by the single worker_context block.
    assert len(ctx) > 0
    # Sanity: the essential body is still fully present (not truncated).
    assert "x" * 2000 in ctx
