"""Tests for async (background) delegation — tools/async_delegation.py.

Covers the dispatch handle, non-blocking behavior, completion-event delivery
onto the shared process_registry.completion_queue, the rich re-injection block
formatting, capacity rejection, and crash handling.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry, format_process_notification


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    # Give just-released workers a beat to finalize BEFORE draining, so their
    # completion events land now instead of leaking into the next test's
    # queue (worker threads push events asynchronously; a drain that races an
    # in-flight _finalize misses it).
    deadline = time.monotonic() + 2.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _drain_one(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


def _drain_for(delegation_id, timeout=5.0):
    """Drain until the event for *delegation_id* appears (discarding others).

    Completion events are pushed asynchronously by worker threads, so a
    straggler from a PREVIOUS test can land after that test's teardown drain
    and leak into the current test's queue. Matching on delegation_id makes
    the assertion immune to that cross-test leak.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("delegation_id") == delegation_id:
                return evt
            continue
        time.sleep(0.02)
    return None


def _install_goal_owner(goal_id: str, session_id: str = "owner") -> None:
    with ad._connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO state_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (
                f"goal:{session_id}",
                json.dumps({"goal_id": goal_id, "status": "active"}),
            ),
        )


def test_dispatch_returns_immediately_without_blocking():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "done", "api_calls": 1,
                "duration_seconds": 0.1, "model": "m"}

    t0 = time.monotonic()
    res = ad.dispatch_async_delegation(
        goal="g", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=3,
    )
    elapsed = time.monotonic() - t0

    assert res["status"] == "dispatched"
    assert res["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: dispatch returned while the runner is still
    # gated (active), so it cannot have waited on the gate. The active_count
    # check is the environment-independent proof; the generous wall-clock
    # bound is a loose sanity backstop, not the primary assertion (a loaded
    # CI runner can be slow but never anywhere near the runner's 5s gate).
    assert ad.active_count() == 1
    assert elapsed < 4.0, f"dispatch blocked {elapsed:.2f}s (gate is 5s)"
    gate.set()


def test_async_executor_workers_are_daemon_threads():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "done"}

    res = ad.dispatch_async_delegation(
        goal="daemon check", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1,
    )
    assert res["status"] == "dispatched"

    deadline = time.monotonic() + 2
    worker = None
    while time.monotonic() < deadline:
        worker = next(
            (t for t in threading.enumerate() if t.name.startswith("async-delegate")),
            None,
        )
        if worker is not None:
            break
        time.sleep(0.02)
    assert worker is not None
    assert worker.daemon is True
    gate.set()
    assert _drain_one() is not None


def test_completion_event_lands_on_shared_queue_with_session_key():
    def runner():
        return {"status": "completed", "summary": "the result",
                "api_calls": 3, "duration_seconds": 2.0, "model": "test-model"}

    res = ad.dispatch_async_delegation(
        goal="compute X", context="some context", toolsets=["web", "file"],
        role="leaf", model="test-model", session_key="agent:main:cli:dm:local",
        parent_session_id="20260703_parent_sid",
        runner=runner, max_async_children=3,
    )
    assert res["status"] == "dispatched"

    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["summary"] == "the result"
    assert evt["session_key"] == "agent:main:cli:dm:local"
    assert evt["parent_session_id"] == "20260703_parent_sid"
    assert evt["delegation_id"] == res["delegation_id"]


def test_rich_reinjection_block_is_self_contained():
    def runner():
        return {"status": "completed", "summary": "The answer is 42.",
                "api_calls": 7, "duration_seconds": 3.5, "model": "test-model"}

    ad.dispatch_async_delegation(
        goal="Compute the meaning of life",
        context="User is a philosopher. Respond tersely.",
        toolsets=["web"], role="leaf", model="test-model",
        session_key="", runner=runner, max_async_children=3,
    )
    evt = _drain_one()
    assert evt is not None
    text = format_process_notification(evt)
    assert text is not None
    for needle in [
        "ASYNC DELEGATION COMPLETE",
        "Compute the meaning of life",
        "User is a philosopher",
        "Toolsets: web",
        "The answer is 42.",
        "Status: completed",
        "API calls: 7",
    ]:
        assert needle in text, f"missing {needle!r}"


def test_dispatch_rejected_at_capacity():
    ev = threading.Event()

    def blocker():
        ev.wait(timeout=60)
        return {"status": "completed", "summary": "x"}

    for i in range(2):
        r = ad.dispatch_async_delegation(
            goal=f"task{i}", context=None, toolsets=None, role="leaf",
            model="m", session_key="", runner=blocker, max_async_children=2,
        )
        assert r["status"] == "dispatched"

    r3 = ad.dispatch_async_delegation(
        goal="task3", context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=blocker, max_async_children=2,
    )
    assert r3["status"] == "rejected"
    assert "capacity reached" in r3["error"]
    ev.set()


def test_interrupt_all_signals_running_children():
    ev = threading.Event()
    interrupted = {"count": 0}
    # No short internal timeout: the blocker holds until interrupt_fn fires.
    # The old ev.wait(timeout=5) made this test a change-detector for CI
    # worker load — on a CPU-starved runner the 5s expired before
    # interrupt_all() ran, the record finalized, and interrupt_all() found
    # nothing running (n == 0). The pytest-level timeout is the real
    # runaway guard.

    def blocker():
        ev.wait(timeout=60)
        return {"status": "interrupted", "summary": None,
                "error": "cancelled"}

    def interrupt_fn():
        interrupted["count"] += 1
        ev.set()

    r = ad.dispatch_async_delegation(
        goal="long task", context=None, toolsets=None, role="leaf",
        model="m", session_key="", runner=blocker,
        interrupt_fn=interrupt_fn, max_async_children=3,
    )
    n = ad.interrupt_all(reason="test")
    assert n == 1
    assert interrupted["count"] == 1
    # child still emits a completion event after interrupt. Match on THIS
    # delegation's id — straggler 'completed' events from a previous test's
    # workers can finalize after that test's teardown drain and leak into
    # this queue (observed on loaded CI workers).
    evt = _drain_for(r["delegation_id"])
    assert evt is not None
    assert evt["status"] == "interrupted"


def test_completed_records_pruned_to_cap():
    # Run more than the retention cap quickly; ensure list doesn't grow forever.
    for i in range(ad._MAX_RETAINED_COMPLETED + 10):
        ad.dispatch_async_delegation(
            goal=f"t{i}", context=None, toolsets=None, role="leaf", model="m",
            session_key="", runner=lambda: {"status": "completed", "summary": "ok"},
            max_async_children=ad._MAX_RETAINED_COMPLETED + 20,
        )
    # let workers finish
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and ad.active_count() > 0:
        time.sleep(0.05)
    assert len(ad.list_async_delegations()) <= ad._MAX_RETAINED_COMPLETED


def test_completion_is_persisted_and_delivery_can_be_acknowledged(tmp_path, monkeypatch):
    """A finished child remains pending on disk until its queue consumer acks it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_goal_owner("goal-123")
    dispatched = ad.dispatch_async_delegation(
        goal="durable", context="ctx", toolsets=["terminal"], role="leaf",
        model="m", session_key="owner", parent_session_id="parent",
        goal_id="goal-123", requires_goal_join=True,
        goal_owner_session_id="owner",
        parent_delegation_id="deleg-parent",
        runner=lambda: {
            "status": "completed",
            "summary": "survived",
            "tokens": {"input": 9, "output": 5},
            "cost_usd": 0.02,
        },
    )
    assert dispatched["goal_owner"] == {
        "goal_id": "goal-123",
        "requires_goal_join": True,
    }
    completion = _drain_one()
    assert completion is not None
    assert completion["goal_id"] == "goal-123"
    assert completion["requires_goal_join"] is True
    assert completion["parent_delegation_id"] == "deleg-parent"
    assert completion["goal_owner_session_id"] == "owner"
    assert completion["tokens"] == {"input": 9, "output": 5}
    assert completion["cost_usd"] == 0.02

    restored = queue.Queue()
    assert ad.restore_undelivered_completions(restored) == 1
    row = ad.get_durable_delegation(dispatched["delegation_id"])
    assert row is not None
    assert row["origin_session"] == "owner"
    assert row["state"] == "completed"
    assert row["result"]["summary"] == "survived"
    assert row["goal_id"] == "goal-123"
    assert row["requires_goal_join"] is True
    assert row["parent_delegation_id"] == "deleg-parent"
    assert row["goal_owner_session_id"] == "owner"
    assert row["reconciliation_state"] == "pending"
    assert row["delivery_state"] == "pending"
    # Queue publication/restoration is not a destination delivery attempt.
    assert row["delivery_attempts"] == 0

    assert ad.mark_completion_delivered(dispatched["delegation_id"])
    semantic = queue.Queue()
    assert ad.restore_undelivered_completions(semantic) == 1
    assert semantic.get_nowait()["semantic_recovery"] is True
    assert ad.get_durable_delegation(dispatched["delegation_id"])["delivery_state"] == "delivered"


def test_completion_persistence_recovers_from_one_transient_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    real_persist_once = ad._persist_completion_once
    attempts = 0

    def fail_once(event, result):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        assert not ad._DB_LOCK.locked()
        real_persist_once(event, result)

    monkeypatch.setattr(ad, "_persist_completion_once", fail_once)
    dispatched = ad.dispatch_async_delegation(
        goal="retry terminal write",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        delegation_id="deleg_transient_completion_lock",
        runner=lambda: {"status": "completed", "summary": "persisted"},
    )

    event = _drain_for(dispatched["delegation_id"])
    assert event is not None
    assert attempts == 2
    durable = ad.get_durable_delegation(dispatched["delegation_id"])
    assert durable is not None and durable["state"] == "completed"
    assert ad.active_count() == 0


def test_completion_persistence_continues_after_immediate_retries_exhaust(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ad, "_COMPLETION_PERSIST_IMMEDIATE_ATTEMPTS", 2)
    monkeypatch.setattr(ad, "_COMPLETION_PERSIST_RETRY_BASE_SECONDS", 0.01)
    real_persist_once = ad._persist_completion_once
    allow_persistence = threading.Event()
    immediate_exhausted = threading.Event()
    attempts = 0

    def locked_until_released(event, result):
        nonlocal attempts
        attempts += 1
        if attempts >= 2:
            immediate_exhausted.set()
        if not allow_persistence.is_set():
            raise sqlite3.OperationalError("database is locked")
        real_persist_once(event, result)

    monkeypatch.setattr(ad, "_persist_completion_once", locked_until_released)
    dispatched = ad.dispatch_async_delegation(
        goal="preserve terminal result",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        delegation_id="deleg_background_completion_retry",
        runner=lambda: {"status": "completed", "summary": "eventual result"},
    )
    assert dispatched["status"] == "dispatched"
    assert immediate_exhausted.wait(timeout=5)
    assert process_registry.completion_queue.empty()
    running = ad.get_durable_delegation(dispatched["delegation_id"])
    assert running is not None and running["state"] == "running"
    assert ad.list_async_delegations()[0]["status"] == "finalizing"

    blocked = ad.dispatch_async_delegation(
        goal="must not outgrow retry capacity",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        delegation_id="deleg_blocked_while_finalizing",
        runner=lambda: {"status": "completed", "summary": "should not run"},
        max_async_children=1,
    )
    assert blocked["status"] == "rejected"
    assert "capacity reached" in blocked["error"]

    allow_persistence.set()
    event = _drain_for(dispatched["delegation_id"])
    assert event is not None and event["summary"] == "eventual result"
    durable = ad.get_durable_delegation(dispatched["delegation_id"])
    assert durable is not None and durable["state"] == "completed"
    assert ad.active_count() == 0


def test_completion_remains_durable_when_registry_import_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "tools.process_registry", None)
    dispatched = ad.dispatch_async_delegation(
        goal="persist without queue import",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        delegation_id="deleg_registry_import_failure",
        runner=lambda: {"status": "completed", "summary": "durable result"},
    )

    deadline = time.monotonic() + 5.0
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)

    durable = ad.get_durable_delegation(dispatched["delegation_id"])
    assert durable is not None and durable["state"] == "completed"
    assert durable["result"]["summary"] == "durable result"
    assert ad.active_count() == 0


def test_durable_schema_migrates_goal_ownership_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    initial = ad._connect()
    initial.close()

    with sqlite3.connect(ad._db_path()) as conn:
        conn.execute("DROP INDEX idx_async_delegations_goal_reconciliation")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN goal_id")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN requires_goal_join")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN parent_delegation_id")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN goal_owner_session_id")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN reconciliation_state")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN reconciliation_claim")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN reconciliation_claimed_at")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN reconciliation_attempts")
        conn.execute("ALTER TABLE async_delegations DROP COLUMN reconciled_at")

    with ad._connect() as conn:
        columns = {
            row[1]: row for row in conn.execute("PRAGMA table_info(async_delegations)")
        }
    assert columns["goal_id"][3] == 1  # NOT NULL
    assert columns["goal_id"][4] == "''"  # default preserves legacy rows
    assert columns["requires_goal_join"][4] == "0"
    assert columns["goal_owner_session_id"][4] == "''"
    assert columns["reconciliation_state"][4] == "'not_required'"
    assert "reconciliation_claim" in columns


def test_concurrent_processes_can_migrate_goal_ownership_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    initial = ad._connect()
    initial.close()
    with sqlite3.connect(ad._db_path()) as conn:
        conn.execute("DROP INDEX idx_async_delegations_goal_reconciliation")
        for column in (
            "goal_id",
            "requires_goal_join",
            "parent_delegation_id",
            "goal_owner_session_id",
            "reconciliation_state",
            "reconciliation_claim",
            "reconciliation_claimed_at",
            "reconciliation_attempts",
            "reconciled_at",
        ):
            conn.execute(f"ALTER TABLE async_delegations DROP COLUMN {column}")

    start_file = tmp_path / "start-migration"
    script = """
import os
import sys
import time
from tools import async_delegation as ad

while not os.path.exists(sys.argv[1]):
    time.sleep(0.01)
with ad._connect():
    pass
"""
    env = dict(os.environ)
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(start_file)],
            cwd=os.getcwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    start_file.touch()
    failures = []
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=45)
        if worker.returncode:
            failures.append((worker.returncode, stdout, stderr))

    assert failures == []
    with ad._connect() as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")
        }
    assert "goal_id" in columns
    assert "reconciled_at" in columns


def test_duplicate_delegation_id_preserves_original_durable_row(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    base = {
        "delegation_id": "deleg_fixed",
        "goal": "original",
        "session_key": "owner",
        "status": "running",
        "dispatched_at": time.time(),
    }
    ad._persist_dispatch(base)

    with pytest.raises(sqlite3.IntegrityError):
        ad._persist_dispatch({**base, "goal": "replacement"})

    with sqlite3.connect(ad._db_path()) as conn:
        task_json = conn.execute(
            "SELECT task_json FROM async_delegations WHERE delegation_id='deleg_fixed'"
        ).fetchone()[0]
    assert json.loads(task_json)["goal"] == "original"


def test_duplicate_live_delegation_id_does_not_replace_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    gate = threading.Event()

    def original_runner():
        gate.wait(timeout=5)
        return {"status": "completed", "summary": "original finished"}

    first = ad.dispatch_async_delegation(
        goal="original live task",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        delegation_id="deleg_live_fixed",
        runner=original_runner,
    )
    second = ad.dispatch_async_delegation(
        goal="replacement task",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        delegation_id="deleg_live_fixed",
        runner=lambda: {"status": "completed", "summary": "replacement ran"},
    )
    assert first["status"] == "dispatched"
    assert second["status"] == "rejected"
    gate.set()
    event = _drain_for("deleg_live_fixed")
    assert event is not None
    assert event["summary"] == "original finished"


def test_dispatch_persistence_failure_releases_in_memory_slot(monkeypatch):
    def fail_persist(_record):
        raise OSError("disk")

    monkeypatch.setattr(ad, "_persist_dispatch", fail_persist)

    result = ad.dispatch_async_delegation(
        goal="must persist",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        runner=lambda: {"status": "completed"},
    )

    assert result["status"] == "rejected"
    assert ad.active_count() == 0
    assert ad.list_async_delegations() == []

    batch_result = ad.dispatch_async_delegation_batch(
        goals=["also persist"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        runner=lambda: {"results": []},
    )
    assert batch_result["status"] == "rejected"
    assert ad.active_count() == 0
    assert ad.list_async_delegations() == []


def test_goal_snapshot_reports_mixed_batch_child_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_goal_owner("goal-mixed")
    dispatched = ad.dispatch_async_delegation_batch(
        goals=["succeeds", "fails"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        goal_id="goal-mixed",
        requires_goal_join=True,
        goal_owner_session_id="owner",
        runner=lambda: {
            "status": "completed",
            "results": [
                {"status": "completed", "summary": "ok"},
                {"status": "error", "error": "child exploded"},
            ],
        },
    )
    assert _drain_for(dispatched["delegation_id"]) is not None

    snapshot = ad.get_goal_work_snapshot("goal-mixed")
    assert snapshot["completed_count"] == 1
    assert snapshot["failed_count"] == 1
    assert snapshot["last_error"] == "child exploded"


def test_in_memory_pruning_keeps_finalizing_records(monkeypatch):
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 1)
    with ad._records_lock:
        ad._records.update({
            "deleg_finalizing": {
                "status": "finalizing",
                "dispatched_at": 1,
            },
            "deleg_old": {
                "status": "completed",
                "completed_at": 2,
            },
            "deleg_new": {
                "status": "completed",
                "completed_at": 3,
            },
        })
        ad._prune_completed_locked()

    assert "deleg_finalizing" in ad._records
    assert "deleg_old" not in ad._records
    assert "deleg_new" in ad._records


def test_reconciliation_claim_is_fenced_idempotent_and_stale_recoverable(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_goal_owner("goal-claim")
    dispatched = ad.dispatch_async_delegation(
        goal="claimable",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        goal_id="goal-claim",
        requires_goal_join=True,
        goal_owner_session_id="owner",
        runner=lambda: {"status": "completed", "summary": "ready"},
    )
    completion = _drain_one()
    assert completion is not None
    assert completion["delegation_id"] == dispatched["delegation_id"]

    first = ad.claim_goal_reconciliation("goal-claim", "consumer-a")
    assert first is not None
    assert first["attempt"] == 1
    assert ad.claim_goal_reconciliation("goal-claim", "consumer-b") is None
    assert ad.release_goal_reconciliation("wrong-token") is False
    assert ad.release_goal_reconciliation(first["claim_id"]) is True
    assert ad.release_goal_reconciliation(first["claim_id"]) is True

    second = ad.claim_goal_reconciliation("goal-claim", "consumer-b")
    assert second is not None
    assert second["attempt"] == 2
    with sqlite3.connect(ad._db_path()) as conn:
        conn.execute(
            "UPDATE async_delegations SET reconciliation_claimed_at=? WHERE delegation_id=?",
            (time.time() - 999, dispatched["delegation_id"]),
        )
        conn.execute(
            "UPDATE async_delegations SET delivery_state='delivered' WHERE delegation_id=?",
            (dispatched["delegation_id"],),
        )
    restored = queue.Queue()
    assert ad.recover_stale_goal_reconciliation_claims(restored) == 1
    retry_event = restored.get_nowait()
    assert retry_event["semantic_recovery"] is True
    assert retry_event["goal_owner_session_id"] == "owner"
    third = ad.claim_goal_reconciliation("goal-claim", "consumer-c")
    assert third is not None
    assert third["attempt"] == 3
    assert ad.complete_goal_reconciliation(second["claim_id"]) is False
    assert ad.complete_goal_reconciliation(third["claim_id"]) is True
    assert ad.complete_goal_reconciliation(third["claim_id"]) is True
    assert ad.get_goal_work_snapshot("goal-claim")["reconciled_count"] == 1


def test_pruning_protects_required_unreconciled_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_goal_owner("goal-retain")
    dispatched = ad.dispatch_async_delegation(
        goal="retain evidence",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        goal_id="goal-retain",
        requires_goal_join=True,
        goal_owner_session_id="owner",
        runner=lambda: {"status": "completed", "summary": "evidence"},
    )
    assert _drain_one() is not None
    delegation_id = dispatched["delegation_id"]
    assert ad.mark_completion_delivered(delegation_id)
    with sqlite3.connect(ad._db_path()) as conn:
        conn.execute(
            "UPDATE async_delegations SET delivered_at=? WHERE delegation_id=?",
            (time.time() - 10_000, delegation_id),
        )
    monkeypatch.setattr(ad, "_DURABLE_RETENTION_SECONDS", 1)

    ad._prune_durable_records()
    assert ad.get_durable_delegation(delegation_id) is not None

    assert ad.abandon_goal_work("goal-retain", "test cleanup") == 1
    with sqlite3.connect(ad._db_path()) as conn:
        conn.execute(
            "UPDATE async_delegations SET updated_at=? WHERE delegation_id=?",
            (time.time() - 10_000, delegation_id),
        )
    ad._prune_durable_records()
    assert ad.get_durable_delegation(delegation_id) is None


def test_reconciliation_claim_tolerates_malformed_result_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_goal_owner("goal-malformed")
    dispatched = ad.dispatch_async_delegation(
        goal="malformed evidence",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        goal_id="goal-malformed",
        requires_goal_join=True,
        goal_owner_session_id="owner",
        runner=lambda: {"status": "completed", "summary": "valid first"},
    )
    assert _drain_one() is not None
    with sqlite3.connect(ad._db_path()) as conn:
        conn.execute(
            "UPDATE async_delegations SET result_json=? WHERE delegation_id=?",
            ("{not-json", dispatched["delegation_id"]),
        )

    claim = ad.claim_goal_reconciliation("goal-malformed", "test")
    assert claim is not None
    assert claim["delegations"][0]["result"] == {}


def test_pre_turn_release_does_not_consume_retry_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_goal_owner("goal-pre-turn")
    ad.dispatch_async_delegation(
        goal="enqueue later",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        goal_id="goal-pre-turn",
        requires_goal_join=True,
        goal_owner_session_id="owner",
        runner=lambda: {"status": "completed", "summary": "ready"},
    )
    assert _drain_one() is not None
    first = ad.claim_goal_reconciliation("goal-pre-turn", "test")
    assert first is not None and first["attempt"] == 1
    assert ad.release_goal_reconciliation(
        first["claim_id"],
        decrement_attempt=True,
    )
    second = ad.claim_goal_reconciliation("goal-pre-turn", "test")
    assert second is not None and second["attempt"] == 1


def test_concurrent_reconciliation_claims_have_one_winner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_goal_owner("goal-contention")
    ad.dispatch_async_delegation(
        goal="claim once",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="owner",
        goal_id="goal-contention",
        requires_goal_join=True,
        goal_owner_session_id="owner",
        runner=lambda: {"status": "completed", "summary": "ready"},
    )
    assert _drain_one() is not None

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(
            executor.map(
                lambda index: ad.claim_goal_reconciliation(
                    "goal-contention", f"consumer-{index}"
                ),
                range(8),
            )
        )
    assert len([claim for claim in claims if claim is not None]) == 1


def test_reconciliation_claim_batches_never_hide_claimed_rows(tmp_path, monkeypatch):
    from hermes_cli.goals import complete_goal_reconciliation_turn
    from tools.process_registry import format_goal_reconciliation_claim
    from tools.process_registry import process_registry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_goal_owner("goal-bounded-batch")
    for index in range(21):
        ad.dispatch_async_delegation(
            goal=f"work {index}",
            context=None,
            toolsets=None,
            role="leaf",
            model=None,
            session_key="batch-session",
            parent_session_id="batch-session",
            goal_id="goal-bounded-batch",
            requires_goal_join=True,
            goal_owner_session_id="owner",
            delegation_id=f"deleg_bounded_{index:02d}",
            runner=lambda i=index: {
                "status": "completed",
                "summary": f"bounded result {i} " + ("x" * 20_000),
            },
        )
        _drain_one()

    claimed_ids = []
    batch_sizes = []
    while True:
        claim = ad.claim_goal_reconciliation("goal-bounded-batch", "test")
        if claim is None:
            break
        batch_ids = list(claim["delegation_ids"])
        batch_sizes.append(len(batch_ids))
        prompt = format_goal_reconciliation_claim(claim, goal="bounded goal")
        assert len(prompt) < 60_000
        for delegation_id in batch_ids:
            assert f"async_delegations:{delegation_id}" in prompt
        claimed_ids.extend(batch_ids)
        assert complete_goal_reconciliation_turn(claim["claim_id"])
        if len(claimed_ids) < 21:
            followup = process_registry.completion_queue.get(timeout=1)
            assert followup["semantic_recovery"] is True

    assert batch_sizes == [10, 10, 1]
    assert claimed_ids == [f"deleg_bounded_{index:02d}" for index in range(21)]


def test_real_process_restart_restores_owned_completion_once(tmp_path):
    """Real-import E2E: a fresh interpreter restores a prior process's result."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": repo}
    producer = r'''
import time
from tools import async_delegation as ad
r = ad.dispatch_async_delegation(
    goal="restart", context=None, toolsets=None, role="leaf", model="m",
    session_key="owner-session", parent_session_id="durable-parent",
    runner=lambda: {"status": "completed", "summary": "after restart"},
)
deadline = time.time() + 5
while ad.active_count() and time.time() < deadline:
    time.sleep(.01)
print(r["delegation_id"])
'''
    first = subprocess.run(
        [sys.executable, "-c", producer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    delegation_id = first.stdout.strip().splitlines()[-1]

    consumer = r'''
import json
from tools.process_registry import process_registry
evt = process_registry.completion_queue.get_nowait()
print(json.dumps(evt, sort_keys=True))
'''
    second = subprocess.run(
        [sys.executable, "-c", consumer], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    evt = json.loads(second.stdout.strip().splitlines()[-1])
    assert evt["delegation_id"] == delegation_id
    assert evt["session_key"] == "owner-session"
    assert evt["parent_session_id"] == "durable-parent"
    assert evt["summary"] == "after restart"

    acker = f'''
from tools import async_delegation as ad
assert ad.mark_completion_delivered({delegation_id!r})
'''
    subprocess.run(
        [sys.executable, "-c", acker], cwd=repo, env=env,
        text=True, capture_output=True, timeout=15, check=True,
    )
    probe = subprocess.run(
        [sys.executable, "-c", "from tools.process_registry import process_registry; print(process_registry.completion_queue.qsize())"],
        cwd=repo, env=env, text=True, capture_output=True, timeout=15, check=True,
    )
    assert probe.stdout.strip().splitlines()[-1] == "0"


def test_submit_failure_removes_durable_running_record(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _BrokenExecutor:
        def submit(self, *_args, **_kwargs):
            raise RuntimeError("submit failed")

    monkeypatch.setattr(ad, "_get_executor", lambda _max_workers: _BrokenExecutor())
    result = ad.dispatch_async_delegation(
        goal="never ran", context=None, toolsets=None, role="leaf", model="m",
        session_key="owner", runner=lambda: {},
    )

    assert result["status"] == "rejected"
    with ad._DB_LOCK, ad._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM async_delegations").fetchone()[0] == 0


def test_pending_retention_prunes_delivered_before_undelivered(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 2)
    for index, delivery_state in enumerate(("pending", "delivered", "pending")):
        delegation_id = f"deleg_{index}"
        record = {
            "delegation_id": delegation_id,
            "session_key": "owner",
            "origin_ui_session_id": "",
            "parent_session_id": None,
            "dispatched_at": float(index + 1),
        }
        ad._persist_dispatch(record)
        ad._persist_completion(
            {
                "delegation_id": delegation_id,
                "status": "completed",
                "completed_at": float(index + 1),
            },
            {"status": "completed", "summary": delegation_id},
        )
        if delivery_state == "delivered":
            ad.mark_completion_delivered(delegation_id)

    ad._prune_durable_records()

    assert ad.get_durable_delegation("deleg_0") is not None
    assert ad.get_durable_delegation("deleg_1") is None
    assert ad.get_durable_delegation("deleg_2") is not None


def test_recover_marks_abandoned_running_record_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = {
        "delegation_id": "deleg_abandoned",
        "session_key": "owner",
        "origin_ui_session_id": "",
        "parent_session_id": None,
        "dispatched_at": 1.0,
    }
    ad._persist_dispatch(record)
    with ad._DB_LOCK, ad._connect() as conn:
        conn.execute(
            "UPDATE async_delegations SET owner_pid=?, owner_started_at=NULL WHERE delegation_id=?",
            (99999999, "deleg_abandoned"),
        )

    assert ad.recover_abandoned_delegations() == 1
    durable = ad.get_durable_delegation("deleg_abandoned")
    assert durable["state"] == "unknown"
    assert durable["delivery_state"] == "pending"
    restored = queue.Queue()
    assert ad.restore_undelivered_completions(restored) == 1
    assert restored.get_nowait()["status"] == "unknown"


def test_durable_delivery_claim_is_exclusive_and_retryable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = {
        "delegation_id": "deleg_claim", "session_key": "owner",
        "origin_ui_session_id": "", "parent_session_id": None,
        "dispatched_at": 1.0,
    }
    ad._persist_dispatch(record)
    ad._persist_completion(
        {"delegation_id": "deleg_claim", "status": "completed", "completed_at": 2.0},
        {"status": "completed", "summary": "done"},
    )

    assert ad.claim_completion_delivery("deleg_claim", "consumer-a")
    assert not ad.claim_completion_delivery("deleg_claim", "consumer-b")
    assert ad.release_completion_delivery("deleg_claim", "consumer-a")
    assert ad.claim_completion_delivery("deleg_claim", "consumer-b")
    assert ad.complete_completion_delivery("deleg_claim", "consumer-b")
    assert not ad.claim_completion_delivery("deleg_claim", "consumer-c")
    assert ad.get_durable_delegation("deleg_claim")["delivery_state"] == "delivered"


# ---------------------------------------------------------------------------
# Integration: delegate_task(background=True) routing
# ---------------------------------------------------------------------------

def test_delegate_task_background_routes_async_and_does_not_block(monkeypatch):
    """delegate_task(background=True) returns a handle without running the
    child synchronously, and the child completes on the background thread.
    A single task is dispatched as a one-item background batch unit."""
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)  # a sync impl would hang delegate_task here
        return {
            "task_index": 0, "status": "completed", "summary": f"done: {goal}",
            "api_calls": 1, "duration_seconds": 0.1, "model": "m",
            "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    # monkeypatch (not `with`) so patches outlive delegate_task's return and
    # remain active while the background worker runs.
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", slow_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    out = dt.delegate_task(
        goal="the real task", context="ctx",
        background=True, parent_agent=parent,
    )

    import json
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert parsed["mode"] == "background"
    assert parsed["delegation_id"].startswith("deleg_")
    # Non-blocking invariant: delegate_task returned while the child is STILL
    # blocked on the closed gate, so no completion event exists yet.
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 1  # one background batch unit, not finished

    gate.set()
    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    # Single task rides the batch path → carries a 1-item results list.
    assert evt.get("is_batch") is True
    assert len(evt["results"]) == 1
    assert evt["results"][0]["summary"] == "done: the real task"
    text = format_process_notification(evt)
    assert text is not None
    assert "the real task" in text


def test_delegate_task_runs_synchronously_when_goal_ownership_is_unavailable(monkeypatch):
    from unittest.mock import MagicMock

    import hermes_cli.goals as goals_module
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    creds = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }
    monkeypatch.setattr(dt, "_build_child_agent", lambda **_kw: fake_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *_a, **_k: creds)
    monkeypatch.setattr(
        dt,
        "_run_single_child",
        lambda *_a, **_k: {
            "task_index": 0,
            "status": "completed",
            "summary": "joined in the parent turn",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        },
    )
    monkeypatch.setattr(
        goals_module,
        "capture_goal_delegation_owner",
        lambda _session_id: (_ for _ in ()).throw(RuntimeError("database locked")),
    )
    dispatch = MagicMock(side_effect=AssertionError("must not detach"))
    monkeypatch.setattr(ad, "dispatch_async_delegation_batch", dispatch)

    result = json.loads(
        dt.delegate_task(goal="safe fallback", background=True, parent_agent=parent)
    )

    assert result["results"][0]["summary"] == "joined in the parent turn"
    assert "ran synchronously" in result["note"]
    dispatch.assert_not_called()


def test_delegate_task_background_uses_live_tui_agent_session_id(monkeypatch):
    """TUI async delegation must route to the live/compressed agent id.

    Regression: delegate_task captured the stale approval/session context key
    after compression rotated parent_agent.session_id. The resulting completion
    was orphaned and could be consumed by an unrelated desktop session poller.
    """
    import json
    from unittest.mock import MagicMock
    import tools.delegate_tool as dt
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.approval import reset_current_session_key, set_current_session_key

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "post-compress-tip"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(
        dt,
        "_run_single_child",
        lambda *a, **k: {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        },
    )

    approval_token = set_current_session_key("pre-compress-parent")
    session_tokens = set_session_vars(
        source="tui",
        session_key="pre-compress-parent",
        ui_session_id="origin-tab",
    )
    try:
        out = dt.delegate_task(goal="bg task", background=True, parent_agent=parent)
        assert json.loads(out)["status"] == "dispatched"
        evt = _drain_one()
    finally:
        reset_current_session_key(approval_token)
        clear_session_vars(session_tokens)

    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt["session_key"] == "post-compress-tip"
    assert evt["origin_ui_session_id"] == "origin-tab"


def test_delegate_task_background_batch_runs_as_one_unit(monkeypatch):
    """A multi-item batch with background=True dispatches the WHOLE fan-out as
    ONE background unit (one handle, one async slot). The children run in
    parallel and join; the consolidated results come back as a single
    completion event when ALL of them finish."""
    import json
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt
    from hermes_cli.goals import GoalManager

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"

    gate = threading.Event()

    def _blocking_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)
        return {
            "task_index": task_index, "status": "completed",
            "summary": f"done: {goal}", "api_calls": 1,
            "duration_seconds": 0.1, "model": "m", "exit_reason": "completed",
        }

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }

    # Use monkeypatch (not a `with` block) so the patches stay active while the
    # background worker thread runs _execute_and_aggregate AFTER delegate_task
    # has already returned.
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", _blocking_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    standing_goal = GoalManager("sess").set("Finish the parent objective")
    out = dt.delegate_task(
        tasks=[{"goal": "a"}, {"goal": "b"}, {"goal": "c"}],
        background=True,
        parent_agent=parent,
    )

    parsed = json.loads(out)
    assert parsed["status"] == "dispatched"
    assert parsed["mode"] == "background"
    assert parsed["count"] == 3
    assert parsed["delegation_id"].startswith("deleg_")
    assert parsed["goals"] == ["a", "b", "c"]
    assert parsed["goal_owner"] == {
        "goal_id": standing_goal.goal_id,
        "requires_goal_join": True,
        "parent_delegation_id": "",
    }
    durable = ad.get_durable_delegation(parsed["delegation_id"])
    assert durable is not None
    assert durable["requires_goal_join"] is True
    assert durable["reconciliation_state"] == "pending"
    assert fake_child._root_goal_id == standing_goal.goal_id
    assert fake_child._root_goal_owner_session_id == "sess"
    assert fake_child._delegation_id == parsed["delegation_id"]
    # ONE background unit for the whole fan-out (not three), and the call
    # returned while all children are still blocked → chat not blocked.
    assert process_registry.completion_queue.empty()
    assert ad.active_count() == 1

    # Release the children; the whole batch joins and emits ONE event.
    gate.set()
    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    assert evt.get("is_batch") is True
    assert evt["goal_id"] == standing_goal.goal_id
    assert evt["requires_goal_join"] is True
    assert len(evt["results"]) == 3
    summaries = sorted(r["summary"] for r in evt["results"])
    assert summaries == ["done: a", "done: b", "done: c"]
    # The consolidated notification names all three tasks in one block.
    text = format_process_notification(evt)
    assert text is not None
    assert "TASK 1/3" in text and "TASK 2/3" in text and "TASK 3/3" in text
    assert "done: a" in text and "done: b" in text and "done: c" in text
    # No more events — it's a single combined completion, not N of them.
    assert _drain_one() is None


def test_nested_async_dispatch_inherits_root_goal_and_parent_delegation(monkeypatch):
    import json
    from unittest.mock import MagicMock

    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 1
    parent.session_id = "nested-session"
    parent._root_goal_id = "goal-root"
    parent._root_goal_owner_session_id = "root-owner-session"
    parent._delegation_id = "deleg-parent"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    child = MagicMock()
    captured = {}

    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)
    monkeypatch.setattr(dt, "_build_child_agent", lambda **_kw: child)
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda *_a, **_k: {
            "model": "m",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )

    def _dispatch(**kwargs):
        captured.update(kwargs)
        return {"status": "dispatched", "delegation_id": kwargs["delegation_id"]}

    monkeypatch.setattr(ad, "dispatch_async_delegation_batch", _dispatch)
    payload = json.loads(
        dt.delegate_task(
            tasks=[{"goal": "nested work"}],
            background=True,
            parent_agent=parent,
        )
    )

    assert payload["status"] == "dispatched"
    assert captured["goal_id"] == "goal-root"
    assert captured["requires_goal_join"] is True
    assert captured["parent_delegation_id"] == "deleg-parent"
    assert captured["goal_owner_session_id"] == "root-owner-session"
    assert child._root_goal_id == "goal-root"
    assert child._root_goal_owner_session_id == "root-owner-session"
    assert child._delegation_id == payload["delegation_id"]


def test_model_dispatch_forces_background():
    """The MODEL-facing dispatch path forces background=True for any top-level
    delegation (single task OR batch), and keeps it off for an orchestrator
    subagent (depth > 0). Direct delegate_task() callers are unaffected (they
    keep the synchronous default)."""
    import tools.delegate_tool as dt
    from unittest.mock import MagicMock

    top = MagicMock()
    top._delegate_depth = 0
    sub = MagicMock()
    sub._delegate_depth = 1

    # Registry-fallback helper: top-level always background, regardless of
    # single vs batch; subagent never.
    assert dt._model_background_value({"goal": "x"}, top) is True
    assert dt._model_background_value(
        {"tasks": [{"goal": "a"}, {"goal": "b"}]}, top
    ) is True
    assert dt._model_background_value({"tasks": [{"goal": "a"}]}, top) is True
    assert dt._model_background_value({"goal": "x"}, sub) is False
    assert dt._model_background_value(
        {"tasks": [{"goal": "a"}, {"goal": "b"}]}, sub
    ) is False


def test_run_agent_dispatch_forces_background():
    """run_agent._dispatch_delegate_task — the live model path — forces
    background on for any top-level delegation (single OR batch) and off for a
    subagent."""
    from unittest.mock import patch
    import run_agent

    class _FakeAgent:
        _delegate_depth = 0

    captured = {}

    def _fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", _fake_delegate):
        agent = _FakeAgent()
        run_agent.AIAgent._dispatch_delegate_task(agent, {"goal": "x"})
        assert captured["background"] is True

        run_agent.AIAgent._dispatch_delegate_task(
            agent, {"tasks": [{"goal": "a"}, {"goal": "b"}]}
        )
        assert captured["background"] is True

        sub = _FakeAgent()
        sub._delegate_depth = 1
        run_agent.AIAgent._dispatch_delegate_task(sub, {"goal": "x"})
        assert captured["background"] is False


def test_dispatch_never_forwards_model_toolsets():
    """The model has no toolsets argument — subagents always inherit the
    parent's toolsets. Even if a model smuggles a `toolsets` key into the
    tool-call args, the live dispatch path must NOT forward it to
    delegate_task (which no longer accepts it) and must not crash."""
    from unittest.mock import patch
    import run_agent

    class _FakeAgent:
        _delegate_depth = 0

    captured = {}

    def _fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", _fake_delegate):
        run_agent.AIAgent._dispatch_delegate_task(
            _FakeAgent(), {"goal": "x", "toolsets": ["web", "terminal"]}
        )
    assert "toolsets" not in captured


def test_delegate_task_background_detaches_child_from_parent(monkeypatch):
    """A background child must NOT remain in parent._active_children —
    otherwise parent-turn interrupts / cache evicts / session close would
    kill the detached subagent mid-run."""
    from unittest.mock import MagicMock, patch
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    gate = threading.Event()

    def slow_child(task_index, goal, child=None, parent_agent=None, **kw):
        gate.wait(timeout=60)
        return {"task_index": 0, "status": "completed", "summary": "ok"}

    def build_and_register(**kw):
        # Mirror what the real _build_child_agent does: register the child
        # for interrupt propagation.
        parent._active_children.append(fake_child)
        return fake_child

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    with patch.object(dt, "_build_child_agent", side_effect=build_and_register), \
         patch.object(dt, "_run_single_child", side_effect=slow_child), \
         patch.object(dt, "_resolve_delegation_credentials", return_value=creds):
        out = dt.delegate_task(goal="bg task", background=True, parent_agent=parent)

    import json
    assert json.loads(out)["status"] == "dispatched"
    # Child detached immediately at dispatch, while it is still running.
    assert fake_child not in parent._active_children
    gate.set()
    assert _drain_one() is not None


def test_concurrent_dispatch_respects_capacity():
    """Two threads racing dispatch with cap=1 must yield exactly one accept
    (capacity check and record insert are atomic under the records lock)."""
    gate = threading.Event()

    def blocker():
        gate.wait(timeout=60)
        return {"status": "completed", "summary": "x"}

    results = []
    barrier = threading.Barrier(2)

    def racer():
        barrier.wait(timeout=5)
        results.append(
            ad.dispatch_async_delegation(
                goal="race", context=None, toolsets=None, role="leaf",
                model="m", session_key="", runner=blocker,
                max_async_children=1,
            )
        )

    threads = [threading.Thread(target=racer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["dispatched", "rejected"]
    gate.set()


# ---------------------------------------------------------------------------
# Gateway routing: session_key -> platform/chat_id, rich formatting, injection
# ---------------------------------------------------------------------------

def _make_async_evt(**over):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_x1",
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "context": "repo /tmp/p",
        "toolsets": ["terminal"],
        "role": "leaf",
        "model": "m",
        "status": "completed",
        "summary": "Found the bug in test_foo",
        "api_calls": 4,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }
    evt.update(over)
    return evt


def test_gateway_enriches_routing_from_session_key():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt()
    runner._enrich_async_delegation_routing(evt)
    assert evt["platform"] == "telegram"
    assert evt["chat_id"] == "12345"
    assert evt["thread_id"] == "678"


def test_gateway_formatter_renders_async_block():
    from gateway.run import _format_gateway_process_notification

    txt = _format_gateway_process_notification(_make_async_evt())
    assert txt is not None
    assert "ASYNC DELEGATION COMPLETE" in txt
    assert "Found the bug in test_foo" in txt
    assert "Investigate flaky test" in txt


def test_gateway_watch_drain_requeues_async_without_looping():
    from gateway.run import _drain_gateway_watch_events

    q = queue.Queue()
    async_evt = _make_async_evt()
    watch_evt = {
        "type": "watch_match",
        "session_id": "proc_1",
        "command": "pytest",
        "pattern": "READY",
        "output": "READY",
    }
    q.put(async_evt)
    q.put(watch_evt)

    watch_events = _drain_gateway_watch_events(q)

    assert watch_events == [watch_evt]
    assert q.qsize() == 1
    assert q.get_nowait() == async_evt


def test_gateway_builds_routable_source_from_enriched_event():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt()
    runner._enrich_async_delegation_routing(evt)
    src = runner._build_process_event_source(evt)
    assert src is not None
    assert src.platform.value == "telegram"
    assert src.chat_id == "12345"


def test_gateway_cli_origin_event_left_unrouted():
    """An empty session_key (CLI origin) is left without routing fields."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    evt = _make_async_evt(session_key="")
    runner._enrich_async_delegation_routing(evt)
    assert "platform" not in evt


