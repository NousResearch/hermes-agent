"""
tests/tools/test_shadow_clone_sqlite_persistence.py

P1 — Shadow clone SQLite persistence tests.

Tests the four new SessionDB methods:
  • insert_shadow_clone_task   — idempotent INSERT
  • update_shadow_clone_task   — status + result write
  • gc_shadow_clone_tasks      — 24 h GC deletes only completed/failed/timeout
  • recover_inflight_shadow_clone_tasks — TTL classification on startup

Plus smoke tests for the dispatch / completion hooks in async_delegation.dispatch()
and the gateway startup recovery code path.

All tests use a fresh in-memory (or tmp-file) SessionDB so they are fully isolated
and never touch the live state.db.
"""
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated SessionDB backed by a tmp file
# ---------------------------------------------------------------------------

@pytest.fixture()
def sdb(tmp_path):
    """Return a SessionDB backed by a temporary SQLite file."""
    from hermes_state import SessionDB
    db_path = tmp_path / "test_state.db"
    return SessionDB(db_path=db_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _insert_running(sdb, delegation_id: str, session_key: str = "sk1",
                    dispatched_at: Optional[float] = None):
    sdb.insert_shadow_clone_task(
        delegation_id=delegation_id,
        session_key=session_key,
        goal="test goal",
        dispatched_at=dispatched_at or time.time(),
    )


def _row(sdb, delegation_id: str) -> Optional[Dict[str, Any]]:
    """Read a row directly from the DB (bypasses SessionDB public API)."""
    cur = sdb._conn.execute(
        "SELECT * FROM shadow_clone_tasks WHERE delegation_id = ?",
        (delegation_id,),
    )
    r = cur.fetchone()
    if r is None:
        return None
    keys = [d[0] for d in cur.description]
    return dict(zip(keys, r))


# ---------------------------------------------------------------------------
# 1. Schema: table exists on first open
# ---------------------------------------------------------------------------

def test_schema_table_exists(sdb):
    """shadow_clone_tasks table must be created on SessionDB init."""
    cur = sdb._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_clone_tasks'"
    )
    assert cur.fetchone() is not None, "shadow_clone_tasks table not found"


def test_schema_columns(sdb):
    """All required columns must be present."""
    cur = sdb._conn.execute("PRAGMA table_info(shadow_clone_tasks)")
    cols = {row[1] for row in cur.fetchall()}
    expected = {
        "delegation_id", "session_key", "kanban_ticket_id", "goal",
        "status", "dispatched_at", "completed_at", "result_json", "routing_meta",
    }
    assert expected.issubset(cols), f"Missing columns: {expected - cols}"


# ---------------------------------------------------------------------------
# 2. insert_shadow_clone_task
# ---------------------------------------------------------------------------

def test_insert_creates_running_row(sdb):
    """Insert must create a row with status='running'."""
    _insert_running(sdb, "d1")
    r = _row(sdb, "d1")
    assert r is not None
    assert r["status"] == "running"
    assert r["session_key"] == "sk1"
    assert r["completed_at"] is None


def test_insert_idempotent(sdb):
    """Second INSERT OR IGNORE on same delegation_id must not raise and not change the row."""
    _insert_running(sdb, "d2")
    first = _row(sdb, "d2")
    _insert_running(sdb, "d2")  # second call — should be silently ignored
    second = _row(sdb, "d2")
    assert first == second


def test_insert_stores_kanban_ticket_id(sdb):
    sdb.insert_shadow_clone_task(
        delegation_id="d3",
        session_key="sk_x",
        kanban_ticket_id="t_abc123",
        goal="some work",
    )
    r = _row(sdb, "d3")
    assert r["kanban_ticket_id"] == "t_abc123"


def test_insert_stores_routing_meta_as_json(sdb):
    meta = {"platform": "telegram", "chat_id": "999"}
    sdb.insert_shadow_clone_task(
        delegation_id="d4", session_key="sk_m", routing_meta=meta
    )
    r = _row(sdb, "d4")
    assert r["routing_meta"] is not None
    assert json.loads(r["routing_meta"]) == meta


def test_insert_truncates_goal(sdb):
    """Goals longer than 500 chars should be stored truncated."""
    long_goal = "X" * 600
    sdb.insert_shadow_clone_task(delegation_id="d5", session_key="sk1", goal=long_goal)
    r = _row(sdb, "d5")
    assert len(r["goal"]) <= 500


# ---------------------------------------------------------------------------
# 3. update_shadow_clone_task
# ---------------------------------------------------------------------------

def test_update_completed(sdb):
    """Update to 'completed' must set status, completed_at, and result_json."""
    _insert_running(sdb, "u1")
    result = {"summary": "done", "status": "completed"}
    sdb.update_shadow_clone_task("u1", status="completed", result=result)
    r = _row(sdb, "u1")
    assert r["status"] == "completed"
    assert r["completed_at"] is not None
    assert json.loads(r["result_json"])["summary"] == "done"


def test_update_failed(sdb):
    _insert_running(sdb, "u2")
    sdb.update_shadow_clone_task("u2", status="failed", result={"error": "boom"})
    r = _row(sdb, "u2")
    assert r["status"] == "failed"


def test_update_result_truncated_to_8k(sdb):
    """result_json must be truncated at 8000 bytes to avoid oversized rows."""
    _insert_running(sdb, "u3")
    big_result = {"data": "Y" * 10_000}
    sdb.update_shadow_clone_task("u3", status="completed", result=big_result)
    r = _row(sdb, "u3")
    assert r["result_json"] is not None
    assert len(r["result_json"]) <= 8000


def test_update_does_not_raise_on_missing_row(sdb):
    """Updating a non-existent delegation_id must not raise."""
    sdb.update_shadow_clone_task("no_such_id", status="failed")


# ---------------------------------------------------------------------------
# 4. gc_shadow_clone_tasks
# ---------------------------------------------------------------------------

def test_gc_deletes_old_completed_rows(sdb):
    old_ts = time.time() - 25 * 3600  # 25 h ago — beyond 24 h retention
    _insert_running(sdb, "g1", dispatched_at=old_ts)
    sdb.update_shadow_clone_task("g1", status="completed", completed_at=old_ts + 60)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 1
    assert _row(sdb, "g1") is None


def test_gc_deletes_old_failed_rows(sdb):
    old_ts = time.time() - 25 * 3600
    _insert_running(sdb, "g2", dispatched_at=old_ts)
    sdb.update_shadow_clone_task("g2", status="failed", completed_at=old_ts + 60)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 1
    assert _row(sdb, "g2") is None


def test_gc_deletes_old_timeout_rows(sdb):
    old_ts = time.time() - 25 * 3600
    _insert_running(sdb, "g3", dispatched_at=old_ts)
    # manually set timeout
    sdb._conn.execute(
        "UPDATE shadow_clone_tasks SET status='timeout', completed_at=? WHERE delegation_id=?",
        (old_ts + 60, "g3"),
    )
    sdb._conn.commit()
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 1


def test_gc_keeps_fresh_completed_rows(sdb):
    """Rows completed less than 24 h ago must NOT be deleted."""
    recent_ts = time.time() - 1 * 3600  # 1 h ago
    _insert_running(sdb, "g4")
    sdb.update_shadow_clone_task("g4", status="completed", completed_at=recent_ts)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 0
    assert _row(sdb, "g4") is not None


def test_gc_never_deletes_running_rows(sdb):
    """Running rows must not be deleted regardless of age."""
    old_ts = time.time() - 48 * 3600
    _insert_running(sdb, "g5", dispatched_at=old_ts)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 0
    assert _row(sdb, "g5") is not None


def test_gc_mixed_batch(sdb):
    now = time.time()
    old = now - 25 * 3600

    _insert_running(sdb, "m1", dispatched_at=old)
    sdb.update_shadow_clone_task("m1", status="completed", completed_at=old + 60)  # old → delete

    _insert_running(sdb, "m2", dispatched_at=now - 3600)
    sdb.update_shadow_clone_task("m2", status="completed", completed_at=now - 3600)  # fresh → keep

    _insert_running(sdb, "m3", dispatched_at=now)  # still running → keep

    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 1
    assert _row(sdb, "m1") is None
    assert _row(sdb, "m2") is not None
    assert _row(sdb, "m3") is not None


# ---------------------------------------------------------------------------
# 5. recover_inflight_shadow_clone_tasks
# ---------------------------------------------------------------------------

def test_recover_empty_db(sdb):
    fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    assert fresh == []
    assert stale == []


def test_recover_classifies_fresh(sdb):
    """A row dispatched 1 h ago (< TTL 2 h) must appear in fresh."""
    ts = time.time() - 3600
    _insert_running(sdb, "r1", dispatched_at=ts)
    fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    assert len(fresh) == 1
    assert fresh[0]["delegation_id"] == "r1"
    assert stale == []


def test_recover_classifies_stale(sdb):
    """A row dispatched 3 h ago (> TTL 2 h) must appear in stale and be marked timeout."""
    ts = time.time() - 3 * 3600
    _insert_running(sdb, "r2", dispatched_at=ts)
    fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    assert "r2" in stale
    assert fresh == []
    r = _row(sdb, "r2")
    assert r["status"] == "timeout"
    assert r["completed_at"] is not None


def test_recover_mixed(sdb):
    now = time.time()
    _insert_running(sdb, "x1", dispatched_at=now - 1 * 3600)  # fresh
    _insert_running(sdb, "x2", dispatched_at=now - 3 * 3600)  # stale
    _insert_running(sdb, "x3", dispatched_at=now - 0.5 * 3600)  # fresh

    fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    fresh_ids = {f["delegation_id"] for f in fresh}
    assert fresh_ids == {"x1", "x3"}
    assert "x2" in stale


def test_recover_skips_completed_rows(sdb):
    """Completed rows must NOT appear in fresh or stale — only 'running' is scanned."""
    _insert_running(sdb, "y1")
    sdb.update_shadow_clone_task("y1", status="completed")
    fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    assert fresh == []
    assert stale == []


def test_recover_returns_routing_meta(sdb):
    meta = {"platform": "discord", "chat_id": "777"}
    sdb.insert_shadow_clone_task(
        delegation_id="z1", session_key="sk_d",
        dispatched_at=time.time() - 60,
        routing_meta=meta,
    )
    fresh, _ = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    assert len(fresh) == 1
    assert fresh[0]["routing_meta"] == meta


def test_recover_idempotent(sdb):
    """Calling recover twice should mark stale only once (no re-marking already-timeout rows)."""
    ts = time.time() - 3 * 3600
    _insert_running(sdb, "idem", dispatched_at=ts)
    _, stale1 = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    _, stale2 = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    assert "idem" in stale1
    assert stale2 == []  # second call sees 'timeout' row — not 'running'


# ---------------------------------------------------------------------------
# 6. Dispatch hook: insert_shadow_clone_task called at dispatch time
# ---------------------------------------------------------------------------

def test_dispatch_inserts_row(tmp_path):
    """async_delegation.dispatch() must call insert_shadow_clone_task for shadow clones."""
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "test_dispatch.db")

    calls = []

    original_insert = db.insert_shadow_clone_task

    def _capturing_insert(*args, **kwargs):
        calls.append((args, kwargs))
        original_insert(*args, **kwargs)

    db.insert_shadow_clone_task = _capturing_insert

    import tools.async_delegation as ad
    import queue

    q = queue.Queue()

    def _runner():
        return {"status": "completed", "summary": "ok"}

    task_info = {
        "shadow_clone": True,
        "goal": "hello world",
        "kanban_ticket_id": "t_testkt",
        "routing_meta": {"platform": "telegram"},
        "context": "",
        "toolsets": None,
        "role": "leaf",
        "model": "test",
        "provider": "test",
    }

    # Patch hermes_state.SessionDB so the lazy import inside dispatch() returns our db
    with patch("hermes_state.SessionDB", return_value=db):
        ret = ad.dispatch(
            runner_fn=_runner,
            task_info=task_info,
            completion_queue=q,
            session_key="test_session",
        )

    assert ret["status"] in ("dispatched", "queued")
    assert len(calls) == 1
    _, kw = calls[0]
    assert kw["delegation_id"] == ret["delegation_id"]
    assert kw["goal"] == "hello world"


# ---------------------------------------------------------------------------
# 7. Completion hook: update_shadow_clone_task called on runner finish
# ---------------------------------------------------------------------------

def test_completion_hook_updates_row(tmp_path):
    """The background runner must call update_shadow_clone_task after finishing."""
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "test_completion.db")

    update_calls = []

    original_update = db.update_shadow_clone_task

    def _capturing_update(*args, **kwargs):
        update_calls.append((args, kwargs))
        original_update(*args, **kwargs)

    db.update_shadow_clone_task = _capturing_update

    import tools.async_delegation as ad
    import queue
    import threading

    q = queue.Queue()
    done = threading.Event()

    def _runner():
        done.wait(timeout=5)
        return {"status": "completed", "summary": "all good"}

    task_info = {
        "shadow_clone": True,
        "goal": "completion test",
        "kanban_ticket_id": None,
        "routing_meta": {},
        "context": "",
        "toolsets": None,
        "role": "leaf",
        "model": "test",
        "provider": "test",
    }

    with patch("hermes_state.SessionDB", return_value=db):
        ret = ad.dispatch(
            runner_fn=_runner,
            task_info=task_info,
            completion_queue=q,
            session_key="sk_completion",
        )
        did = ret["delegation_id"]

        done.set()  # let the runner finish
        # Wait for the completion event
        evt = q.get(timeout=5)

    assert evt["status"] in ("completed", "error")
    # update must have been called at least once with our delegation_id
    assert any(
        (args and args[0] == did) or kw.get("delegation_id") == did
        for args, kw in update_calls
    )


# ---------------------------------------------------------------------------
# 8. Gateway startup recovery smoke test
# ---------------------------------------------------------------------------

def test_gateway_startup_recovery_smoke(tmp_path, monkeypatch):
    """Startup recovery code must not raise and must return two lists."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "startup_recovery.db")
    now = time.time()
    _insert_running(db, "s_fresh", dispatched_at=now - 3600)   # < 2 h TTL
    _insert_running(db, "s_stale", dispatched_at=now - 5 * 3600)  # > 2 h TTL

    fresh, stale = db.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
    assert len(fresh) == 1
    assert fresh[0]["delegation_id"] == "s_fresh"
    assert "s_stale" in stale

    # Verify the stale row is now 'timeout'
    r = _row(db, "s_stale")
    assert r["status"] == "timeout"


# ---------------------------------------------------------------------------
# 9. GC covers all runner-written terminal statuses (regression test for
#    the GC enum mismatch bug — 'error', 'cancelled', 'timed_out' rows were
#    previously omitted from the DELETE WHERE clause and leaked permanently)
# ---------------------------------------------------------------------------

def test_gc_deletes_old_error_rows(sdb):
    """Rows with runner-written status='error' must be GC'd after retention."""
    old_ts = time.time() - 25 * 3600
    _insert_running(sdb, "e1", dispatched_at=old_ts)
    sdb.update_shadow_clone_task("e1", status="error", completed_at=old_ts + 60)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 1
    assert _row(sdb, "e1") is None


def test_gc_deletes_old_cancelled_rows(sdb):
    """Rows with runner-written status='cancelled' must be GC'd after retention."""
    old_ts = time.time() - 25 * 3600
    _insert_running(sdb, "e2", dispatched_at=old_ts)
    sdb.update_shadow_clone_task("e2", status="cancelled", completed_at=old_ts + 60)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 1
    assert _row(sdb, "e2") is None


def test_gc_deletes_old_timed_out_rows(sdb):
    """Rows with runner-written status='timed_out' must be GC'd after retention."""
    old_ts = time.time() - 25 * 3600
    _insert_running(sdb, "e3", dispatched_at=old_ts)
    sdb.update_shadow_clone_task("e3", status="timed_out", completed_at=old_ts + 60)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 1
    assert _row(sdb, "e3") is None


def test_gc_all_terminal_statuses_in_one_batch(sdb):
    """GC must delete ALL six terminal statuses in a single pass, keeping only running rows."""
    old_ts = time.time() - 25 * 3600
    terminal_statuses = ["completed", "failed", "timeout", "error", "cancelled", "timed_out"]
    for i, status in enumerate(terminal_statuses):
        did = f"all_{i}"
        _insert_running(sdb, did, dispatched_at=old_ts)
        # 'timeout' is set by recovery, others by update_shadow_clone_task
        if status == "timeout":
            sdb._conn.execute(
                "UPDATE shadow_clone_tasks SET status='timeout', completed_at=? WHERE delegation_id=?",
                (old_ts + 60, did),
            )
            sdb._conn.commit()
        else:
            sdb.update_shadow_clone_task(did, status=status, completed_at=old_ts + 60)

    # one still-running row must survive
    _insert_running(sdb, "all_running")

    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == len(terminal_statuses)
    for i in range(len(terminal_statuses)):
        assert _row(sdb, f"all_{i}") is None
    assert _row(sdb, "all_running") is not None
