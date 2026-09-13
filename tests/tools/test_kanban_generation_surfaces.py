"""Worker heartbeats must preserve the attempt boundary through real tool dispatch."""

from __future__ import annotations

import hermes_cli.kanban_claims as _owner_kanban_claims

import json

import pytest


@pytest.mark.parametrize("automatic", [False, True])
def test_heartbeat_rejects_old_run_and_accepts_current_run(tmp_path, monkeypatch, automatic):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt
    from tools.registry import registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="heartbeat generation", assignee="worker")
        assert _owner_kanban_claims.claim_task(conn, tid)
        old_run = kb.get_task(conn, tid).current_run_id
        # Model a completed attempt followed by a retry owned by the same
        # dispatcher; the stale worker retains its original environment.
        conn.execute("UPDATE task_runs SET ended_at = 1 WHERE id = ?", (old_run,))
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, "
            "current_run_id = NULL WHERE id = ?", (tid,))
        conn.commit()
        assert _owner_kanban_claims.claim_task(conn, tid)
        current_run = kb.get_task(conn, tid).current_run_id
        conn.execute("UPDATE tasks SET claim_expires = 1 WHERE id = ?", (tid,))
        conn.commit()
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", kb.get_task(conn, tid).claim_lock)

        def heartbeat(run_id):
            monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
            if automatic:
                monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
                assert kt.heartbeat_current_worker_from_env()
            else:
                return json.loads(registry.dispatch("kanban_heartbeat", {"task_id": tid}))

        before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())
        heartbeat(old_run)
        after = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())
        assert after == before
        result = heartbeat(current_run)
        if not automatic:
            assert result["ok"]
        current = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert current["current_run_id"] == current_run
        assert current["claim_expires"] > before["claim_expires"]
        assert current["last_heartbeat_at"] is not None
        assert current["worker_registered_at"] is not None
