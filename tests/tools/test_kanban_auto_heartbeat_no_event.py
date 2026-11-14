"""The automatic worker heartbeat must not append a ``task_events`` row (#149385).

``heartbeat_current_worker_from_env`` fires about once a minute per running worker.
The dispatcher watchdog only reads ``tasks.last_heartbeat_at``, and the tick carries
no durable note, so every one of those events is write amplification: on a busy board
heartbeats reached 46-70% of ``task_events`` and drowned ``kanban tail``/``watch`` and
the dashboard's event stream. The explicit ``kanban_heartbeat`` tool still records its
event — that one is a model action with a note worth keeping.

The auto-path fixture is rebuilt here (not imported from ``tests/tools/test_kanban_tools.py``)
because importing another module's fixture fails the health gate as ``F811``; it is the
minimal board + claim + worker env those tests use.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def running_worker(monkeypatch, tmp_path):
    """A started board with one running task claimed by this process, env-scoped to it."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path

    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="auto-heartbeat-noise", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def _events(tid: str, kind: str = "heartbeat") -> list:
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = ?",
                (tid, kind),
            ).fetchall()
        ]
    finally:
        conn.close()


def _task_heartbeat_at(tid: str):
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        return conn.execute(
            "SELECT last_heartbeat_at FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["last_heartbeat_at"]
    finally:
        conn.close()


def test_auto_heartbeat_advances_liveness_without_appending_an_event(running_worker, monkeypatch):
    """The periodic tick still proves liveness; it just stops growing the event log."""
    from tools import kanban_tools as kt

    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
    before = int(time.time())

    assert kt.heartbeat_current_worker_from_env() is True

    assert _task_heartbeat_at(running_worker) >= before
    assert _events(running_worker) == []


def test_explicit_heartbeat_tool_still_records_its_event(running_worker):
    """The model-invoked tool is a durable action: it keeps its note and its row."""
    import json

    from tools import kanban_tools as kt

    assert json.loads(kt._handle_heartbeat({"note": "still alive"})).get("ok") is True

    events = _events(running_worker)
    assert len(events) == 1
    assert json.loads(events[0]["payload"]) == {"note": "still alive"}


def test_heartbeat_worker_keeps_recording_by_default(running_worker):
    """``record_event`` is opt-out: every other caller keeps today's behaviour."""
    from hermes_cli import kanban_db_dispatch as kbd

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        assert kbd.heartbeat_worker(conn, running_worker, note="manual") is True
    finally:
        conn.close()
    assert len(_events(running_worker)) == 1

    conn = kbc.connect()
    try:
        assert kbd.heartbeat_worker(
            conn, running_worker, expected_run_id=kb._current_run_id(conn, running_worker),
            record_event=False,
        ) is True
    finally:
        conn.close()
    assert len(_events(running_worker)) == 1
