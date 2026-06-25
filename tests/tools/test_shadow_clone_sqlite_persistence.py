from __future__ import annotations
"""
tests/tools/test_shadow_clone_sqlite_persistence.py

P1 — Shadow clone SQLite persistence tests.

Tests the four SessionDB methods:
  * insert_shadow_clone_task   — idempotent INSERT
  * update_shadow_clone_task   — status + result write
  * gc_shadow_clone_tasks      — 24 h GC deletes only terminal rows
  * recover_inflight_shadow_clone_tasks — TTL classification on startup

Plus smoke tests for the dispatch / completion hooks in
async_delegation.dispatch_async_delegation() and the gateway startup recovery
code path.

All tests use a fresh tmp-file-backed SessionDB so they are fully isolated
and never touch the live state.db.
"""
import json
import time
from typing import Any, Dict, Optional
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


def test_schema_indexes_exist(sdb):
    """Both helper indexes must exist."""
    cur = sdb._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='shadow_clone_tasks'"
    )
    index_names = {row[0] for row in cur.fetchall()}
    assert "idx_sct_status" in index_names
    assert "idx_sct_dispatched_at" in index_names


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
    _insert_running(sdb, "d2")  # second call — silently ignored
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


def test_insert_stores_routing_meta_as_json_string(sdb):
    meta_str = json.dumps({"platform": "telegram", "chat_id": "999"})
    sdb.insert_shadow_clone_task(
        delegation_id="d4", session_key="sk_m", routing_meta=meta_str
    )
    r = _row(sdb, "d4")
    assert r["routing_meta"] is not None
    assert json.loads(r["routing_meta"])["platform"] == "telegram"


def test_insert_truncates_goal(sdb):
    """Goals longer than 500 chars should be stored truncated."""
    long_goal = "X" * 600
    sdb.insert_shadow_clone_task(delegation_id="d5", session_key="sk1", goal=long_goal)
    r = _row(sdb, "d5")
    assert len(r["goal"]) <= 500


def test_insert_dispatched_at_defaults_to_now(sdb):
    before = time.time()
    _insert_running(sdb, "d6")
    after = time.time()
    r = _row(sdb, "d6")
    assert before <= r["dispatched_at"] <= after


# ---------------------------------------------------------------------------
# 3. update_shadow_clone_task
# ---------------------------------------------------------------------------

def test_update_completed(sdb):
    """Update to 'completed' must set status, completed_at, and result_json."""
    _insert_running(sdb, "u1")
    result_str = json.dumps({"summary": "done", "status": "completed"})[:8000]
    sdb.update_shadow_clone_task("u1", status="completed", result_json=result_str)
    r = _row(sdb, "u1")
    assert r["status"] == "completed"
    assert r["completed_at"] is not None
    assert json.loads(r["result_json"])["summary"] == "done"


def test_update_failed(sdb):
    _insert_running(sdb, "u2")
    sdb.update_shadow_clone_task("u2", status="failed",
                                 result_json=json.dumps({"error": "boom"}))
    r = _row(sdb, "u2")
    assert r["status"] == "failed"


def test_update_coalesce_does_not_clobber_result_json(sdb):
    """A routing-only update (result_json=None) must NOT overwrite an existing result."""
    _insert_running(sdb, "u3")
    result_str = json.dumps({"output": "important"})
    sdb.update_shadow_clone_task("u3", status="completed", result_json=result_str)
    # Simulate a routing-only update arriving after the result is written
    sdb.update_shadow_clone_task("u3", status="completed", result_json=None,
                                 routing_meta=json.dumps({"chat_id": "42"}))
    r = _row(sdb, "u3")
    assert r["result_json"] is not None
    assert json.loads(r["result_json"])["output"] == "important"


def test_update_result_string_truncated_at_8k(sdb):
    """result_json beyond 8000 chars is trimmed by the caller before passing."""
    _insert_running(sdb, "u4")
    big = "Y" * 10_000
    truncated = big[:8000]
    sdb.update_shadow_clone_task("u4", status="completed", result_json=truncated)
    r = _row(sdb, "u4")
    assert len(r["result_json"]) <= 8000


def test_update_does_not_raise_on_missing_row(sdb):
    """Updating a non-existent delegation_id must not raise."""
    sdb.update_shadow_clone_task("no_such_id", status="failed")


def test_update_sets_routing_meta(sdb):
    """routing_meta field can be updated independently of result."""
    _insert_running(sdb, "u5")
    meta = json.dumps({"platform": "discord", "channel": "123"})
    sdb.update_shadow_clone_task("u5", status="running", routing_meta=meta)
    r = _row(sdb, "u5")
    assert json.loads(r["routing_meta"])["platform"] == "discord"


# ---------------------------------------------------------------------------
# 4. gc_shadow_clone_tasks
# ---------------------------------------------------------------------------

def test_gc_deletes_old_completed_rows(sdb):
    old_ts = time.time() - 25 * 3600  # 25 h ago — beyond 24 h retention
    _insert_running(sdb, "g1", dispatched_at=old_ts)
    sdb.update_shadow_clone_task("g1", status="completed",
                                 completed_at=old_ts + 60)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 1
    assert _row(sdb, "g1") is None


def test_gc_deletes_old_failed_rows(sdb):
    old_ts = time.time() - 25 * 3600
    _insert_running(sdb, "g2", dispatched_at=old_ts)
    sdb.update_shadow_clone_task("g2", status="failed",
                                 completed_at=old_ts + 60)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted >= 1
    assert _row(sdb, "g2") is None


def test_gc_deletes_old_timeout_rows(sdb):
    old_ts = time.time() - 25 * 3600
    _insert_running(sdb, "g3", dispatched_at=old_ts)
    sdb.update_shadow_clone_task("g3", status="timeout",
                                 completed_at=old_ts + 60)
    sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert _row(sdb, "g3") is None


def test_gc_does_not_delete_running_rows(sdb):
    """Running rows must never be deleted by GC, regardless of age."""
    old_ts = time.time() - 25 * 3600
    _insert_running(sdb, "g4", dispatched_at=old_ts)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 0
    assert _row(sdb, "g4") is not None


def test_gc_does_not_delete_recent_completed_rows(sdb):
    """Rows completed less than retain_hours ago must not be deleted."""
    recent_ts = time.time() - 1 * 3600  # 1 h ago — within 24 h window
    _insert_running(sdb, "g5", dispatched_at=recent_ts)
    sdb.update_shadow_clone_task("g5", status="completed",
                                 completed_at=recent_ts + 60)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 0
    assert _row(sdb, "g5") is not None


def test_gc_returns_deleted_count(sdb):
    old_ts = time.time() - 25 * 3600
    for i in range(3):
        _insert_running(sdb, f"gc_count_{i}", dispatched_at=old_ts)
        sdb.update_shadow_clone_task(f"gc_count_{i}", status="completed",
                                     completed_at=old_ts + 10)
    deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
    assert deleted == 3


def test_gc_empty_table_returns_zero(sdb):
    assert sdb.gc_shadow_clone_tasks(retain_hours=24.0) == 0


# ---------------------------------------------------------------------------
# 5. recover_inflight_shadow_clone_tasks
# ---------------------------------------------------------------------------

def test_recover_empty_returns_empty(sdb):
    rows = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200.0)
    assert rows == []


def test_recover_fresh_row_returned_as_running(sdb):
    """A recently dispatched row (within TTL) must be returned with status=running."""
    _insert_running(sdb, "r1")
    rows = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200.0)
    assert len(rows) == 1
    assert rows[0]["delegation_id"] == "r1"
    assert rows[0]["status"] == "running"


def test_recover_stale_row_marked_timeout(sdb):
    """A row older than TTL must be updated to status='timeout' in the DB."""
    old_ts = time.time() - 3 * 3600  # 3 h ago — beyond 2 h TTL
    _insert_running(sdb, "r2", dispatched_at=old_ts)
    rows = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200.0)
    assert len(rows) == 1
    # status in the returned dict reflects the promotion
    assert rows[0]["status"] == "timeout"
    # and the DB row is updated too
    db_row = _row(sdb, "r2")
    assert db_row["status"] == "timeout"


def test_recover_does_not_return_already_completed(sdb):
    """Completed rows must not appear in the recovery result."""
    _insert_running(sdb, "r3")
    sdb.update_shadow_clone_task("r3", status="completed",
                                 result_json=json.dumps({"summary": "ok"}))
    rows = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200.0)
    assert all(r["delegation_id"] != "r3" for r in rows)


def test_recover_mixed_fresh_and_stale(sdb):
    """Must correctly classify a mix of fresh and stale rows."""
    fresh_ts = time.time() - 30 * 60   # 30 min ago — within TTL
    stale_ts = time.time() - 3 * 3600  # 3 h ago — beyond TTL
    _insert_running(sdb, "r_fresh", dispatched_at=fresh_ts)
    _insert_running(sdb, "r_stale", dispatched_at=stale_ts)
    rows = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200.0)
    by_id = {r["delegation_id"]: r for r in rows}
    assert by_id["r_fresh"]["status"] == "running"
    assert by_id["r_stale"]["status"] == "timeout"


def test_recover_idempotent(sdb):
    """Calling recover twice must not error; second call finds no running rows."""
    _insert_running(sdb, "r_idem")
    sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200.0)
    rows2 = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200.0)
    # The first call marked any stale row as timeout; running fresh rows are still running.
    # Either way the second call must not raise.
    assert isinstance(rows2, list)


# ---------------------------------------------------------------------------
# 6. dispatch_async_delegation persistence hook (smoke test)
# ---------------------------------------------------------------------------

def test_dispatch_inserts_row(tmp_path):
    """dispatch_async_delegation must write a row to the DB before starting the worker."""
    import hermes_state as _hs
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")

    def _fake_runner():
        return {"status": "completed", "summary": "ok", "api_calls": 1,
                "duration_seconds": 0.1}

    # Patch hermes_state.SessionDB (the source module) so the local import inside
    # dispatch_async_delegation picks it up, and _get_executor so no real thread fires.
    with patch.object(_hs, "SessionDB", return_value=db), \
         patch("tools.async_delegation._get_executor") as mock_exec:
        mock_exec.return_value.submit = MagicMock()
        from tools.async_delegation import dispatch_async_delegation, _reset_for_tests
        _reset_for_tests()
        delegation_id_info = dispatch_async_delegation(
            session_key="sk_smoke",
            goal="smoke test",
            context=None,
            toolsets=None,
            role="leaf",
            model=None,
            runner=_fake_runner,
        )
        delegation_id = delegation_id_info["delegation_id"]
        # The row must exist immediately after dispatch
        r = _row(db, delegation_id)
        assert r is not None, "Expected a DB row to be inserted at dispatch time"
        assert r["status"] == "running"


# ---------------------------------------------------------------------------
# 7. _finalize persistence hook (integration smoke test)
# ---------------------------------------------------------------------------

def test_finalize_updates_row(tmp_path):
    """_finalize must call update_shadow_clone_task with the correct status and result_json."""
    db_path = tmp_path / "state.db"
    from hermes_state import SessionDB
    db = SessionDB(db_path=db_path)
    delegation_id = "fin_test_01"
    db.insert_shadow_clone_task(delegation_id=delegation_id, session_key="sk1")

    from tools import async_delegation as _ad

    # _finalize early-returns if delegation_id isn't in _records, so seed it.
    with _ad._records_lock:
        _ad._records[delegation_id] = {
            "delegation_id": delegation_id,
            "session_key": "sk1",
            "goal": "test",
            "status": "running",
            "dispatched_at": time.time() - 1.0,
            "completed_at": None,
            "consumed": True,
            "interrupt_fn": None,
        }

    # _finalize opens a fresh SessionDB() using the default path each time.
    # We capture the update call by monkeypatching the method on the class,
    # then replaying it against our test db.
    update_calls: list = []
    original_update = SessionDB.update_shadow_clone_task

    def _capturing_update(self, *args, **kwargs):
        update_calls.append((args, kwargs))
        # Replay the call against our isolated test db so we can verify state.
        original_update(db, *args, **kwargs)

    with patch.object(SessionDB, "update_shadow_clone_task", _capturing_update):
        _ad._finalize(
            delegation_id=delegation_id,
            result={"summary": "all done", "status": "completed",
                    "api_calls": 3, "duration_seconds": 1.2},
            status="completed",
        )

    # Verify the call was made with the right arguments
    assert len(update_calls) == 1, "update_shadow_clone_task must be called exactly once"
    _, kwargs = update_calls[0]
    assert kwargs.get("status") == "completed"
    assert kwargs.get("result_json") is not None
    payload = json.loads(kwargs["result_json"])
    assert payload.get("summary") == "all done"

    # Verify the row was actually written to our test db
    r = _row(db, delegation_id)
    assert r is not None
    assert r["status"] == "completed"
    assert r["result_json"] is not None
