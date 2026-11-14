"""The legacy roster must not hydrate every historical session on each poll (#151727)."""
import json
import sqlite3
import time

from hermes_state import SessionDB
import tui_gateway.server as srv
from tui_gateway import profile_roster_cache


def test_roster_reads_recent_sessions_with_a_bounded_sql_budget(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    now = time.time()
    try:
        db._conn.executemany(
            "INSERT INTO sessions(id,source,started_at,last_activity_at) VALUES(?, 'cron', ?, ?)",
            [(f"history-{i}", now - 20000 + i, now - 20000 + i) for i in range(10000)],
        )
        db.create_session("chat", "cli")
        db.append_message("chat", "user", "recent chat")
        db.create_session("worker", "kanban")
        db.append_message("worker", "user", "recent work")
        db._conn.commit()
    finally:
        db.close()

    # Count actual SQLite VM work, not elapsed time on a shared CI host. Keep the
    # test's budget active even when the query installs its own deadline callback.
    connections = []

    class BudgetConnection(sqlite3.Connection):
        steps = 0

        def set_progress_handler(self, callback, n=1000):
            def progress():
                self.steps += 1000
                return self.steps > 500000 or (callback is not None and callback())
            super().set_progress_handler(progress, 1000)

    connect = sqlite3.connect

    def budget_connect(*args, **kwargs):
        conn = connect(*args, **dict(kwargs, factory=BudgetConnection))
        conn.set_progress_handler(None)
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", budget_connect)
    profile_roster_cache.invalidate(tmp_path)
    row = {}
    srv._profile_session_fields(row, tmp_path)
    assert row["last_session"] is not None
    assert row["last_session"]["id"] == "chat"
    assert row["last_session"]["preview"] == "recent chat"
    assert row["worker_session"]["id"] == "worker"
    assert connections and all(conn.steps <= 500000 for conn in connections)


def test_roster_keeps_compression_tip_and_excludes_retired_local_segments(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    now = time.time()
    try:
        db.create_session("root", "cli")
        db.append_message("root", "user", "before compression")
        db.end_session("root", "compression")
        db.create_session("tip", "cli", parent_session_id="root")
        db.append_message("tip", "user", "current text")
        db.create_session("retired", "local")
        db.create_session("reset", "local", parent_session_id="retired",
                          model_config={"_reset_from": "retired"})
        db._conn.execute("UPDATE sessions SET end_reason='session_reset', last_activity_at=? WHERE id='retired'",
                         (now + 10,))
        db._conn.execute("UPDATE sessions SET last_activity_at=? WHERE id='reset'", (now - 10,))
        db._conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)", (
            "gateway.local_policy.v1:retired",
            json.dumps({"session_id": "retired", "lineage": ["retired", "reset"]}),
        ))
        db._conn.commit()
    finally:
        db.close()
    profile_roster_cache.invalidate(tmp_path)
    row = {}
    srv._profile_session_fields(row, tmp_path)
    assert row["last_session"]["id"] == "tip"
    assert row["last_session"]["preview"] == "current text"
    assert row["worker_session"] is None
