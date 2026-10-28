"""The runtime-activity → board-heartbeat bridge must not wait out a write-locked board.

``_touch_activity`` calls ``heartbeat_current_worker_from_env`` inline on the turn thread
(stream monitor heartbeat, chunk path). With the board's default busy timeout a locked board
held that thread for minutes, so the stream stale watchdog never fired and the turn-liveness
watchdog aborted the turn instead (#145886). A busy board now skips the beat quickly.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
import tools.kanban_tools as kt


@pytest.fixture
def running_task(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_BOARD", "HERMES_KANBAN_CLAIM_LOCK", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="busy board")
        run = kb.claim_task(conn, tid)
        assert run is not None
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.id))
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
    return tid


def test_auto_heartbeat_skips_write_locked_board_instead_of_blocking(running_task, monkeypatch):
    tid = running_task
    holder = kbc.connect()
    try:
        before = kb.get_task(holder, tid)
        holder.execute("BEGIN IMMEDIATE")
        result: list[bool] = []
        beat = threading.Thread(target=lambda: result.append(kt.heartbeat_current_worker_from_env()),
                                daemon=True)
        started = time.monotonic()
        beat.start()
        # Default busy timeout is 120 s per boundary attempt; the bridge must give up in seconds.
        beat.join(timeout=10.0)
        finished = not beat.is_alive()
        elapsed = time.monotonic() - started
        holder.execute("ROLLBACK")
        beat.join()
        assert finished, f"auto-heartbeat blocked {elapsed:.1f}s on a write-locked board"
        assert result == [False]

        after = kb.get_task(holder, tid)
        assert after.claim_expires == before.claim_expires
        assert after.last_heartbeat_at == before.last_heartbeat_at
    finally:
        holder.close()

    # The board is free again: the next window heartbeats normally.
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
    assert kt.heartbeat_current_worker_from_env() is True
