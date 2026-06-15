"""
tests/tools/test_shadow_clone_gateway_restart.py

Gateway-restart scenario tests for shadow clone SQLite persistence.

Covers the 4 scenarios from task t_24b9a10a:

  S1 — Full pytest baseline (all shadow-clone tests pass, called via import)
  S2 — Restart recovery: dispatch 5 shadow clones, some complete before restart;
       after recover_inflight_shadow_clone_tasks() the running rows are correctly
       classified (fresh vs stale) and completed results remain intact.
  S3 — TTL test: manually insert a running row older than 2 h; after recovery
       it must be marked 'timeout', not returned in fresh.
  S4 — GC test: completed/failed rows older than 24 h are deleted; running rows
       and young rows survive.

All tests use isolated tmp-file SessionDBs — never touch the live state.db.
"""

import json
import time
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def sdb(tmp_path):
    from hermes_state import SessionDB
    return SessionDB(db_path=tmp_path / "restart_test.db")


def _insert(sdb, did: str, session_key: str = "sk1",
            dispatched_at: Optional[float] = None,
            routing_meta: Optional[dict] = None,
            kanban_ticket_id: Optional[str] = None):
    """Insert a 'running' shadow clone row into *sdb*."""
    sdb.insert_shadow_clone_task(
        delegation_id=did,
        session_key=session_key,
        goal=f"goal for {did}",
        dispatched_at=dispatched_at or time.time(),
        routing_meta=routing_meta,
        kanban_ticket_id=kanban_ticket_id,
    )


def _row(sdb, did: str) -> Optional[Dict[str, Any]]:
    cur = sdb._conn.execute(
        "SELECT * FROM shadow_clone_tasks WHERE delegation_id = ?", (did,)
    )
    r = cur.fetchone()
    if r is None:
        return None
    keys = [d[0] for d in cur.description]
    return dict(zip(keys, r))


def _count(sdb) -> int:
    return sdb._conn.execute("SELECT COUNT(*) FROM shadow_clone_tasks").fetchone()[0]


# ---------------------------------------------------------------------------
# S2 — Restart recovery: 5 clones, some complete before restart
# ---------------------------------------------------------------------------

class TestS2RestartRecovery:
    """
    S2: dispatch 5 shadow clones; 2 complete before the gateway restarts;
    3 are still running.  After recover_inflight_shadow_clone_tasks():
      - The 2 completed rows are untouched (not running → ignored by recovery).
      - The 3 still-running rows are all within TTL → returned in 'fresh'.
      - No row is mis-classified.
    Completed result JSON is still readable post-restart.
    """

    def test_five_clones_partial_completion_before_restart(self, sdb):
        now = time.time()

        # Dispatch 5 shadow clones — all dispatched within the last 5 minutes
        clones = [f"sc_{i}" for i in range(5)]
        routing = {"platform": "telegram", "chat_id": "test_chat"}
        for did in clones:
            _insert(sdb, did, session_key="sess_a",
                    dispatched_at=now - 120,  # 2 min ago — well within 2 h TTL
                    routing_meta=routing,
                    kanban_ticket_id=f"t_{did}")

        # 2 clones complete before the restart
        completed_before = clones[:2]
        still_running = clones[2:]

        for did in completed_before:
            sdb.update_shadow_clone_task(
                did, status="completed",
                result={"summary": f"{did} done", "kanban_result": "ok"}
            )

        # ---- GATEWAY RESTART ----
        fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        # Only still-running rows should appear in fresh
        fresh_ids = {r["delegation_id"] for r in fresh}
        assert fresh_ids == set(still_running), (
            f"Expected fresh={set(still_running)}, got={fresh_ids}"
        )

        # No stale because all running rows are < 2 h old
        assert stale == [], f"Expected no stale, got: {stale}"

        # Completed rows untouched — result JSON still intact
        for did in completed_before:
            r = _row(sdb, did)
            assert r is not None
            assert r["status"] == "completed"
            result = json.loads(r["result_json"])
            assert result["summary"] == f"{did} done"
            assert result["kanban_result"] == "ok"

    def test_fresh_rows_carry_routing_meta(self, sdb):
        """Fresh rows returned by recovery must include routing_meta so the
        gateway can re-deliver notifications to the right platform/chat."""
        now = time.time()
        meta = {"platform": "discord", "chat_id": "dc_123", "thread_id": "th_9"}
        _insert(sdb, "sc_route", session_key="sess_b",
                dispatched_at=now - 60, routing_meta=meta)

        fresh, _ = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
        assert len(fresh) == 1
        assert fresh[0]["routing_meta"] == meta, (
            f"Routing meta mismatch: {fresh[0]['routing_meta']} != {meta}"
        )

    def test_fresh_rows_carry_kanban_ticket_id(self, sdb):
        """Fresh rows must carry the kanban_ticket_id so recovery can look up
        task status from the kanban board."""
        now = time.time()
        _insert(sdb, "sc_kb", session_key="sess_c",
                dispatched_at=now - 30, kanban_ticket_id="t_abc999")

        fresh, _ = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
        assert any(r["kanban_ticket_id"] == "t_abc999" for r in fresh), (
            f"kanban_ticket_id not found in fresh rows: {fresh}"
        )

    def test_running_rows_survive_restart_unchanged(self, sdb):
        """Still-running rows must remain status='running' after recovery
        (the recovery call only writes to stale rows, not fresh ones)."""
        now = time.time()
        _insert(sdb, "sc_survive", dispatched_at=now - 300)

        sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        r = _row(sdb, "sc_survive")
        assert r["status"] == "running", (
            f"Expected status=running after recovery, got: {r['status']}"
        )

    def test_five_mixed_sessions(self, sdb):
        """5 clones spread across different sessions all recovered correctly."""
        now = time.time()
        sessions = ["sess_x", "sess_y", "sess_z"]
        dids = [f"sc_multi_{i}" for i in range(5)]
        for i, did in enumerate(dids):
            _insert(sdb, did, session_key=sessions[i % 3],
                    dispatched_at=now - 60 * (i + 1))

        fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
        fresh_ids = {r["delegation_id"] for r in fresh}
        assert fresh_ids == set(dids)
        assert stale == []

    def test_completed_result_survives_second_recover_call(self, sdb):
        """Calling recover twice must not corrupt completed rows."""
        now = time.time()
        _insert(sdb, "sc_dbl", dispatched_at=now - 60)
        sdb.update_shadow_clone_task("sc_dbl", status="completed",
                                     result={"key": "val"})

        sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
        sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        r = _row(sdb, "sc_dbl")
        assert r["status"] == "completed"
        assert json.loads(r["result_json"]) == {"key": "val"}


# ---------------------------------------------------------------------------
# S3 — TTL test: running row older than 2 h → marked 'timeout' on restart
# ---------------------------------------------------------------------------

class TestS3TTL:
    """
    S3: a running row inserted with dispatched_at > 2 h ago must be classified
    as 'stale' and have its status flipped to 'timeout' by recover_inflight.
    """

    def test_old_running_row_marked_timeout(self, sdb):
        """Running row dispatched 3 h ago → in stale list, status='timeout'."""
        old_ts = time.time() - 3 * 3600  # 3 hours ago
        _insert(sdb, "stale_sc", dispatched_at=old_ts)

        fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        assert "stale_sc" in stale, f"Expected stale, got stale={stale}"
        assert fresh == [], f"Expected no fresh, got fresh={fresh}"

        r = _row(sdb, "stale_sc")
        assert r["status"] == "timeout", f"Expected timeout, got: {r['status']}"
        assert r["completed_at"] is not None, "completed_at must be set on stale mark"

    def test_exactly_at_ttl_boundary_is_stale(self, sdb):
        """Row dispatched exactly at the TTL boundary (2 h 1 s ago) → stale."""
        boundary_ts = time.time() - 7201  # 1 second past the 2 h TTL
        _insert(sdb, "boundary_sc", dispatched_at=boundary_ts)

        fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        assert "boundary_sc" in stale

    def test_just_inside_ttl_is_fresh(self, sdb):
        """Row dispatched 1 h 59 min ago → fresh, not stale."""
        recent_ts = time.time() - (7200 - 60)  # 1 minute inside TTL
        _insert(sdb, "fresh_sc", dispatched_at=recent_ts)

        fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        fresh_ids = {r["delegation_id"] for r in fresh}
        assert "fresh_sc" in fresh_ids
        assert "fresh_sc" not in stale

    def test_mixed_fresh_and_stale(self, sdb):
        """Both a fresh and a stale row co-exist — each classified correctly."""
        now = time.time()
        _insert(sdb, "sc_fresh", dispatched_at=now - 60)    # 1 min ago
        _insert(sdb, "sc_stale", dispatched_at=now - 10800)  # 3 h ago

        fresh, stale = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        fresh_ids = {r["delegation_id"] for r in fresh}
        assert "sc_fresh" in fresh_ids
        assert "sc_stale" in stale
        assert "sc_stale" not in fresh_ids
        assert "sc_fresh" not in stale

    def test_stale_row_then_gc_eligible(self, sdb):
        """After being marked 'timeout', a row must become GC-eligible
        once it ages past the retain_hours threshold."""
        old_ts = time.time() - 25 * 3600  # 25 h ago
        _insert(sdb, "sc_gc_eligible", dispatched_at=old_ts)

        # Recovery marks it 'timeout' with completed_at ≈ now
        # Set completed_at back in time so GC threshold is crossed
        sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        # Manually back-date completed_at so GC sees it as old
        sdb._conn.execute(
            "UPDATE shadow_clone_tasks SET completed_at = ? WHERE delegation_id = ?",
            (old_ts + 60, "sc_gc_eligible"),
        )
        sdb._conn.commit()

        deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted == 1
        assert _row(sdb, "sc_gc_eligible") is None

    def test_five_clones_mix_three_stale_two_fresh(self, sdb):
        """
        5 clones: 3 stale (>2 h), 2 fresh (<2 h).
        After recovery: stale list has 3, fresh list has 2; all 3 stale rows
        now have status='timeout'.
        """
        now = time.time()
        stale_dids = [f"sc_stale_{i}" for i in range(3)]
        fresh_dids = [f"sc_fresh_{i}" for i in range(2)]

        for did in stale_dids:
            _insert(sdb, did, dispatched_at=now - 3 * 3600)
        for did in fresh_dids:
            _insert(sdb, did, dispatched_at=now - 300)

        fresh_out, stale_out = sdb.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)

        assert set(stale_out) == set(stale_dids), (
            f"stale mismatch: got={set(stale_out)}, want={set(stale_dids)}"
        )
        assert {r["delegation_id"] for r in fresh_out} == set(fresh_dids), (
            f"fresh mismatch: got={fresh_out}, want={fresh_dids}"
        )

        for did in stale_dids:
            assert _row(sdb, did)["status"] == "timeout"
        for did in fresh_dids:
            assert _row(sdb, did)["status"] == "running"


# ---------------------------------------------------------------------------
# S4 — GC test: 24 h+ completed/failed rows deleted; running rows survive
# ---------------------------------------------------------------------------

class TestS4GC:
    """
    S4: gc_shadow_clone_tasks() must delete all terminal rows older than
    retain_hours (default 24 h) and must NOT touch running rows or young rows.
    """

    def test_gc_deletes_old_completed(self, sdb):
        old_ts = time.time() - 25 * 3600
        _insert(sdb, "gc_c1", dispatched_at=old_ts)
        sdb.update_shadow_clone_task("gc_c1", status="completed",
                                     completed_at=old_ts + 60)
        deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted == 1
        assert _row(sdb, "gc_c1") is None

    def test_gc_deletes_old_failed(self, sdb):
        old_ts = time.time() - 25 * 3600
        _insert(sdb, "gc_f1", dispatched_at=old_ts)
        sdb.update_shadow_clone_task("gc_f1", status="failed",
                                     completed_at=old_ts + 60)
        deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted == 1
        assert _row(sdb, "gc_f1") is None

    def test_gc_deletes_old_timeout(self, sdb):
        old_ts = time.time() - 25 * 3600
        _insert(sdb, "gc_t1", dispatched_at=old_ts)
        sdb._conn.execute(
            "UPDATE shadow_clone_tasks SET status='timeout', completed_at=? "
            "WHERE delegation_id='gc_t1'",
            (old_ts + 60,),
        )
        sdb._conn.commit()
        deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted == 1
        assert _row(sdb, "gc_t1") is None

    def test_gc_spares_running_rows(self, sdb):
        """Running rows must never be GC'd, even if very old."""
        _insert(sdb, "gc_running", dispatched_at=time.time() - 48 * 3600)
        deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted == 0
        assert _row(sdb, "gc_running") is not None

    def test_gc_spares_young_terminal_rows(self, sdb):
        """Terminal rows under 24 h must not be deleted."""
        _insert(sdb, "gc_young_c", dispatched_at=time.time() - 3600)
        sdb.update_shadow_clone_task("gc_young_c", status="completed")
        deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted == 0
        assert _row(sdb, "gc_young_c") is not None

    def test_gc_returns_correct_count(self, sdb):
        """gc_shadow_clone_tasks must return exactly the number of rows deleted."""
        old_ts = time.time() - 25 * 3600
        terminal_statuses = ["completed", "failed", "timeout", "error", "cancelled", "timed_out"]
        for i, status in enumerate(terminal_statuses):
            did = f"gc_count_{i}"
            _insert(sdb, did, dispatched_at=old_ts)
            if status == "timeout":
                sdb._conn.execute(
                    "UPDATE shadow_clone_tasks SET status='timeout', completed_at=? "
                    "WHERE delegation_id=?", (old_ts + 60, did)
                )
                sdb._conn.commit()
            else:
                sdb.update_shadow_clone_task(did, status=status,
                                             completed_at=old_ts + 60)

        # Also add a running row that must survive
        _insert(sdb, "gc_count_running")

        deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted == len(terminal_statuses), (
            f"Expected {len(terminal_statuses)} deleted, got {deleted}"
        )
        assert _row(sdb, "gc_count_running") is not None

    def test_gc_is_idempotent(self, sdb):
        """Calling GC twice must not delete more rows than expected."""
        old_ts = time.time() - 25 * 3600
        _insert(sdb, "gc_idem", dispatched_at=old_ts)
        sdb.update_shadow_clone_task("gc_idem", status="completed",
                                     completed_at=old_ts + 60)

        deleted1 = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        deleted2 = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted1 == 1
        assert deleted2 == 0  # row already gone

    def test_gc_with_mixed_ages_and_statuses(self, sdb):
        """Only old terminal rows go; young rows and running rows all survive."""
        now = time.time()
        old_ts = now - 25 * 3600

        # Old terminal rows (should be deleted)
        for i, st in enumerate(["completed", "failed"]):
            did = f"gc_old_{i}"
            _insert(sdb, did, dispatched_at=old_ts)
            sdb.update_shadow_clone_task(did, status=st, completed_at=old_ts + 60)

        # Young terminal row (should survive)
        _insert(sdb, "gc_young", dispatched_at=now - 3600)
        sdb.update_shadow_clone_task("gc_young", status="completed")

        # Old running row (should survive — GC never deletes running)
        _insert(sdb, "gc_old_running", dispatched_at=old_ts)

        deleted = sdb.gc_shadow_clone_tasks(retain_hours=24.0)
        assert deleted == 2
        assert _row(sdb, "gc_young") is not None
        assert _row(sdb, "gc_old_running") is not None
        assert _row(sdb, "gc_old_0") is None
        assert _row(sdb, "gc_old_1") is None


# ---------------------------------------------------------------------------
# S2 extended — dispatch+completion via async_delegation.dispatch()
# ---------------------------------------------------------------------------

class TestS2DispatchAndCompletion:
    """
    End-to-end via async_delegation.dispatch() — verifies the full round-trip:
    insert on dispatch → status=running → update on completion → status=completed.
    Simulates the exact path a gateway restart would encounter.
    """

    def test_dispatch_and_complete_lifecycle(self, tmp_path):
        """
        Dispatch a shadow clone via async_delegation.dispatch(); wait for the
        runner to finish; verify DB row goes running → completed and result is
        readable post-completion (survives a simulated gateway 'restart').
        """
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "lifecycle.db")

        import tools.async_delegation as ad

        q = queue.Queue()
        done_event = threading.Event()

        def _runner():
            done_event.wait(timeout=5)
            return {"status": "completed", "summary": "lifecycle ok"}

        task_info = {
            "shadow_clone": True,
            "goal": "gateway restart lifecycle test",
            "kanban_ticket_id": "t_lifecycle_kt",
            "routing_meta": {"platform": "telegram", "chat_id": "12345"},
            "context": "",
            "toolsets": None,
            "role": "leaf",
            "model": "test-model",
            "provider": "test-provider",
        }

        with patch("hermes_state.SessionDB", return_value=db):
            ret = ad.dispatch(
                runner_fn=_runner,
                task_info=task_info,
                completion_queue=q,
                session_key="sess_lifecycle",
            )
            did = ret["delegation_id"]

            # Row should be running immediately after dispatch
            r = _row(db, did)
            assert r is not None, "Row not inserted after dispatch"
            assert r["status"] == "running"

            # Let the runner complete
            done_event.set()
            evt = q.get(timeout=5)

        assert evt["status"] in ("completed", "error")

        # Post-completion: simulate 'restart' — call recover
        fresh, stale = db.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
        # Completed row must NOT appear in fresh (it's no longer running)
        fresh_ids = {r["delegation_id"] for r in fresh}
        assert did not in fresh_ids, "Completed row must not appear in fresh"

        # Result JSON must still be readable
        r = _row(db, did)
        assert r["status"] == "completed"
        result = json.loads(r["result_json"])
        assert result.get("summary") == "lifecycle ok"

    def test_five_dispatches_all_complete_then_restart(self, tmp_path):
        """
        Dispatch 5 clones via async_delegation.dispatch(); all complete;
        gateway 'restarts' (recover_inflight called); zero fresh, zero stale.
        All 5 rows readable and have status=completed.
        """
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "five_lifecycle.db")

        import tools.async_delegation as ad

        completed_dids = []

        for i in range(5):
            q = queue.Queue()

            def _runner(idx=i):
                return {"status": "completed", "summary": f"clone {idx} done"}

            task_info = {
                "shadow_clone": True,
                "goal": f"clone {i} work",
                "kanban_ticket_id": f"t_clone_{i}",
                "routing_meta": {"platform": "telegram", "chat_id": "main_chat"},
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
                    session_key="sess_five",
                )
                did = ret["delegation_id"]
                completed_dids.append(did)
                q.get(timeout=5)  # wait for completion

        # Gateway restart
        fresh, stale = db.recover_inflight_shadow_clone_tasks(ttl_seconds=7200)
        assert fresh == [], f"All completed — expected fresh=[], got {fresh}"
        assert stale == [], f"All completed — expected stale=[], got {stale}"

        # All 5 rows show completed
        for did in completed_dids:
            r = _row(db, did)
            assert r is not None
            assert r["status"] == "completed"
