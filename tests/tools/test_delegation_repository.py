"""Behavior contracts for normalized delegation persistence."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from tools.delegation_repository import DelegationRepository

def _released_schema(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE async_delegations (
            delegation_id TEXT PRIMARY KEY,
            origin_session TEXT NOT NULL,
            origin_ui_session_id TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT,
            state TEXT NOT NULL,
            dispatched_at REAL NOT NULL,
            completed_at REAL,
            updated_at REAL NOT NULL,
            event_json TEXT,
            result_json TEXT,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            delivered_at REAL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            task_json TEXT,
            delivery_claim TEXT,
            delivery_claimed_at REAL,
            root_subagent_ids_json TEXT NOT NULL DEFAULT '[]',
            children_json TEXT NOT NULL DEFAULT '{}',
            interrupt_requests_json TEXT NOT NULL DEFAULT '{}',
            interrupt_reason TEXT,
            abandon_reason TEXT
        )"""
    )
    conn.commit()
    conn.close()

def _open_snapshot(path: str, delegation_id: str, start, output) -> None:
    start.wait()
    try:
        snap = DelegationRepository(Path(path)).snapshot(delegation_id)
        output.put(("ok", snap["lifecycle_version"] if snap else None))
    except BaseException as exc:  # pragma: no cover - asserted in parent
        output.put(("error", f"{type(exc).__name__}: {exc}"))

def _reserve_attempt(path: str, logical_id: str, worker_id: str, start, output) -> None:
    start.wait()
    try:
        outcome = DelegationRepository(Path(path)).reserve_resumed_attempt(
            logical_id, physical_worker_id=worker_id
        )
        output.put(outcome)
    except BaseException as exc:  # pragma: no cover - asserted in parent
        output.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

@pytest.fixture
def repo(tmp_path: Path) -> DelegationRepository:
    return DelegationRepository(tmp_path / "state.db")

def _record(delegation_id="deleg-1", roots=None, dispatched_at=10.0):
    roots = ["sa-root"] if roots is None else roots
    return {
        "delegation_id": delegation_id,
        "session_key": "owner",
        "origin_ui_session_id": "ui-owner",
        "parent_session_id": "parent-owner",
        "dispatched_at": dispatched_at,
        "goal": "do work",
        "context": "context",
        "role": "leaf",
        "model": "test",
        "root_subagent_ids": roots,
    }

def test_initial_dispatch_persists_supplied_attempt_mapping_atomically(repo):
    record = _record("deleg-protected", ["sa-a", "sa-b"])
    record["attempt_ids_by_logical_id"] = {
        "sa-a": "attempt-physical-a",
        "sa-b": "attempt-physical-b",
    }

    created = repo.register_initial_dispatch(record)

    assert created["status"] == "registered"
    assert [item["attempt_id"] for item in created["attempts"]] == [
        "attempt-physical-a",
        "attempt-physical-b",
    ]
    with sqlite3.connect(repo.db_path) as conn:
        rows = conn.execute(
            "SELECT attempt_id, logical_id, physical_worker_id "
            "FROM delegation_attempts ORDER BY logical_id"
        ).fetchall()
    assert rows == [
        ("attempt-physical-a", "sa-a", "attempt-physical-a"),
        ("attempt-physical-b", "sa-b", "attempt-physical-b"),
    ]


def test_initial_dispatch_persists_authority_inside_immutable_logical_spec(repo):
    authority = {
        "version": 1,
        "profile": {"name": "isolated", "hash": "hash", "snapshot": {}},
        "state": {"revoked": False, "cleaned": False},
    }
    record = _record("deleg-authority", ["sa-root"])
    record["authority_by_logical_id"] = {"sa-root": authority}

    created = repo.register_initial_dispatch(record)
    repo.transition_attempt(
        created["attempts"][0]["attempt_id"],
        {"starting"},
        "completed",
        metadata={"status_observation": "mutable"},
    )

    with sqlite3.connect(repo.db_path) as conn:
        raw_spec = conn.execute(
            "SELECT spec_json FROM delegation_logical_subagents WHERE logical_id=?",
            ("sa-root",),
        ).fetchone()[0]
    assert json.loads(raw_spec)["authority"] == authority



def test_authority_revocation_tombstone_preserves_immutable_snapshot(repo):
    authority = {
        "version": 1,
        "authority_hash": "immutable-hash",
        "profile": {"name": "isolated"},
        "state": {"revoked": False, "cleaned": False},
    }
    record = _record("deleg-authority-tombstone", ["sa-authority"])
    record["attempt_ids_by_logical_id"] = {
        "sa-authority": "attempt-authority"
    }
    record["authority_by_logical_id"] = {"sa-authority": authority}
    repo.register_initial_dispatch(record)

    outcome = repo.tombstone_logical_authority(
        "sa-authority", revoked=True, cleaned=True
    )

    assert outcome == {"status": "tombstoned"}
    with sqlite3.connect(repo.db_path) as conn:
        spec = json.loads(
            conn.execute(
                "SELECT spec_json FROM delegation_logical_subagents WHERE logical_id = ?",
                ("sa-authority",),
            ).fetchone()[0]
        )
    assert spec["authority"] == {
        **authority,
        "state": {"revoked": True, "cleaned": True},
    }


def test_tombstoned_authority_denies_late_steer_and_interrupt(repo):
    authority = {
        "version": 1,
        "authority_hash": "immutable-hash",
        "profile": {"name": "isolated"},
        "state": {"revoked": False, "cleaned": False},
    }
    record = _record("deleg-authority-denial", ["sa-authority-denial"])
    record["attempt_ids_by_logical_id"] = {
        "sa-authority-denial": "attempt-authority-denial"
    }
    record["authority_by_logical_id"] = {
        "sa-authority-denial": authority
    }
    repo.register_initial_dispatch(record)
    repo.tombstone_logical_authority("sa-authority-denial", revoked=True)

    assert repo.enqueue_steer(
        "deleg-authority-denial", "sa-authority-denial", "owner", "late"
    ) == {"status": "authority_revoked"}
    assert repo.request_interrupt("attempt-authority-denial", "late") == {
        "status": "authority_revoked"
    }


def test_fresh_resumed_attempt_preserves_immutable_logical_authority(repo):
    authority = {
        "version": 1,
        "authority_hash": "immutable-hash",
        "lineage": {
            "scope_id": "scope-original",
            "attempt_id": "attempt-original",
            "parent_attempt_id": None,
        },
        "state": {"revoked": False, "cleaned": False},
    }
    record = _record("deleg-authority-resume", ["sa-authority-resume"])
    record["attempt_ids_by_logical_id"] = {
        "sa-authority-resume": "attempt-original"
    }
    record["authority_by_logical_id"] = {
        "sa-authority-resume": authority
    }
    repo.register_initial_dispatch(record)
    repo.transition_attempt("attempt-original", {"starting"}, "completed")

    resumed = repo.reserve_resumed_attempt("sa-authority-resume")

    with sqlite3.connect(repo.db_path) as conn:
        stored = json.loads(
            conn.execute(
                "SELECT spec_json FROM delegation_logical_subagents WHERE logical_id=?",
                ("sa-authority-resume",),
            ).fetchone()[0]
        )["authority"]
        physical_worker_id = conn.execute(
            "SELECT physical_worker_id FROM delegation_attempts WHERE attempt_id=?",
            (resumed["attempt_id"],),
        ).fetchone()[0]
    assert resumed["status"] == "reserved"
    assert resumed["attempt_id"] != "attempt-original"
    assert physical_worker_id == resumed["attempt_id"]
    assert stored == authority


def test_retry_replacement_and_resume_allocate_fresh_physical_attempt_ids(repo):
    record = _record("deleg-retry", ["sa-a"])
    record["attempt_ids_by_logical_id"] = {"sa-a": "attempt-initial"}
    initial = repo.register_initial_dispatch(record)
    assert initial["status"] == "registered"
    assert repo.transition_attempt(
        "attempt-initial", {"starting"}, "completed"
    )["status"] == "updated"

    resumed = repo.reserve_resumed_attempt("sa-a")
    assert resumed["status"] == "reserved"
    assert repo.transition_attempt(
        resumed["attempt_id"], {"starting"}, "completed"
    )["status"] == "updated"
    replacement = repo.reserve_resumed_attempt("sa-a")
    assert replacement["status"] == "reserved"

    ids = {"attempt-initial", resumed["attempt_id"], replacement["attempt_id"]}
    assert len(ids) == 3
    conn = repo._connect()
    try:
        rows = conn.execute(
            "SELECT attempt_id,physical_worker_id FROM delegation_attempts "
            "WHERE logical_id=? ORDER BY attempt_number",
            ("sa-a",),
        ).fetchall()
    finally:
        conn.close()
    assert {
        str(row["attempt_id"]): str(row["physical_worker_id"]) for row in rows
    } == {attempt_id: attempt_id for attempt_id in ids}


def test_initial_single_and_batch_shape_preserves_root_order(repo):
    single = repo.register_initial_dispatch(_record())
    batch = repo.register_initial_dispatch(
        _record("deleg-batch", ["sa-z", "sa-a"], dispatched_at=20.0)
    )

    assert single["status"] == "registered"
    assert single["run_id"]
    assert [(a["logical_id"], a["attempt_number"]) for a in single["attempts"]] == [
        ("sa-root", 1)
    ]
    assert [a["logical_id"] for a in batch["attempts"]] == ["sa-z", "sa-a"]
    snap = repo.snapshot("deleg-batch")
    assert snap["lifecycle_version"] == 2
    assert snap["root_subagent_ids"] == ["sa-z", "sa-a"]
    assert list(snap["children"]) == ["sa-z", "sa-a"]
    assert snap["state"] == "running"
    assert snap["children"]["sa-z"]["status"] == "starting"

def test_batch_result_mapping_is_ordinal_not_uuid_order(repo):
    created = repo.register_initial_dispatch(_record(roots=["sa-z", "sa-a"]))
    result = {
        "results": [
            {"status": "completed", "summary": "first"},
            {"status": "error", "error": "second"},
        ]
    }
    event = {"status": "completed", "completed_at": 30.0, "results": result["results"]}

    assert repo.complete_run(created["run_id"], event, result)["status"] == "completed"
    children = repo.snapshot("deleg-1")["children"]
    assert children["sa-z"]["status"] == "completed"
    assert children["sa-z"]["summary"] == "first"
    assert children["sa-a"]["status"] == "error"
    assert children["sa-a"]["error"] == "second"

def test_concurrent_resume_has_one_winner(repo):
    initial = repo.register_initial_dispatch(_record())
    attempt_id = initial["attempts"][0]["attempt_id"]
    assert repo.transition_attempt(
        attempt_id, {"starting"}, "completed", completed_at=11.0
    )["status"] == "updated"
    barrier = threading.Barrier(9)
    outcomes = []

    def reserve(index):
        barrier.wait()
        outcomes.append(
            repo.reserve_resumed_attempt("sa-root", physical_worker_id=f"worker-{index}")
        )

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert [outcome["status"] for outcome in outcomes].count("reserved") == 1
    assert [outcome["status"] for outcome in outcomes].count("already_running") == 7

def test_stale_attempt_and_interrupt_cannot_poison_new_attempt(repo):
    initial = repo.register_initial_dispatch(_record())
    old_id = initial["attempts"][0]["attempt_id"]
    repo.transition_attempt(old_id, {"starting"}, "completed", completed_at=11.0)
    current = repo.reserve_resumed_attempt("sa-root", physical_worker_id="worker-new")

    assert repo.transition_attempt(old_id, {"completed"}, "error")["status"] == "stale"
    assert repo.request_interrupt(old_id, "too late")["status"] == "already_terminal"
    requested = repo.request_interrupt(current["attempt_id"], "stop")
    assert requested["status"] == "interrupt_requested"
    assert repo.request_interrupt(current["attempt_id"], "again")["status"] == "already_requested"
    assert repo.take_interrupt(current["attempt_id"]) == {"status": "taken", "reason": "stop"}
    assert repo.take_interrupt(current["attempt_id"])["status"] == "not_pending"
    assert repo.snapshot("deleg-1")["children"]["sa-root"]["status"] == "interrupt_requested"

def test_exact_run_completion_is_independent_of_newer_reserved_run(repo):
    initial = repo.register_initial_dispatch(_record())
    old_attempt = initial["attempts"][0]["attempt_id"]
    repo.transition_attempt(old_attempt, {"starting"}, "completed")
    resumed = repo.reserve_resumed_attempt("sa-root")
    initial_event = {"status": "error", "summary": "initial", "completed_at": 19.0}
    assert repo.complete_run(initial["run_id"], initial_event, initial_event)["status"] == "completed"
    assert repo.pending_events() == [{"run_id": initial["run_id"], "event": initial_event}]
    current = repo.snapshot("deleg-1")["children"]["sa-root"]
    assert current["attempt_id"] == resumed["attempt_id"]
    assert current["status"] == "starting"
    assert repo.inspect_delivery("deleg-1", resumed["run_id"])["completed_at"] is None

    duplicate = {"status": "error", "summary": "duplicate", "completed_at": 20.0}
    assert repo.complete_run(initial["run_id"], duplicate, duplicate)["status"] == "stale"
    assert len(repo.pending_events()) == 1

    fresh_event = {"status": "completed", "summary": "fresh", "completed_at": 20.0}
    assert repo.complete_run(resumed["run_id"], fresh_event, fresh_event)["status"] == "completed"
    assert len(repo.pending_events()) == 2
    snap = repo.snapshot("deleg-1")
    assert snap["event"]["summary"] == "fresh"
    assert snap["result"]["summary"] == "fresh"

def test_owner_recovery_is_exact_and_allows_later_reserve(repo):
    initial = repo.register_initial_dispatch(
        _record(), owner_pid=999999, owner_started_at=1
    )
    attempt_id = initial["attempts"][0]["attempt_id"]

    recovered = repo.recover_orphaned_attempts(lambda _pid, _started: False)
    assert recovered["attempts"] == [{"attempt_id": attempt_id, "run_id": initial["run_id"], "delegation_id": "deleg-1"}]
    assert repo.snapshot("deleg-1")["children"]["sa-root"]["status"] == "unknown"
    resumed = repo.reserve_resumed_attempt("sa-root", owner_pid=123, owner_started_at=2)
    assert resumed["status"] == "reserved"
    assert resumed["attempt_number"] == 2

def test_initial_registration_logical_conflict_rolls_back_every_row(repo):
    repo.register_initial_dispatch(_record("deleg-existing", ["sa-collision"]))
    rejected = _record("deleg-rejected", ["sa-collision"])
    rejected.update(run_id="run-rejected", attempt_ids=["attempt-rejected"])
    outcome = repo.register_initial_dispatch(rejected)
    assert outcome["status"] == "conflict"
    assert repo.register_initial_dispatch(_record("deleg-existing", ["sa-other"]))["status"] == "already_exists"
    with sqlite3.connect(repo.db_path) as conn:
        counts = conn.execute("""SELECT
            (SELECT COUNT(*) FROM async_delegations WHERE delegation_id='deleg-rejected'),
            (SELECT COUNT(*) FROM delegation_runs WHERE run_id='run-rejected'),
            (SELECT COUNT(*) FROM delegation_logical_subagents WHERE delegation_id='deleg-rejected'),
            (SELECT COUNT(*) FROM delegation_attempts WHERE attempt_id='attempt-rejected')""").fetchone()
        assert counts == (0, 0, 0, 0)

def test_legacy_migration_is_idempotent_and_preserves_disposition(tmp_path):
    path = tmp_path / "state.db"
    _released_schema(str(path))
    children = {
        "sa-z": {"subagent_id": "sa-z", "status": "success", "summary": "z"},
        "sa-a": {"subagent_id": "sa-a", "status": "failed", "error": "a"},
        "sa-child": {
            "subagent_id": "sa-child",
            "parent_id": "sa-z",
            "status": "running",
            "goal": "nested",
        },
    }
    conn = sqlite3.connect(path)
    conn.execute(
        """INSERT INTO async_delegations
           (delegation_id, origin_session, state, dispatched_at, completed_at,
            updated_at, event_json, result_json, delivery_state,
            delivery_attempts, delivery_claim, delivery_claimed_at,
            delivered_at, owner_pid, owner_started_at, task_json,
            root_subagent_ids_json, children_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            "deleg-old", "owner", "completed", 1.0, 2.0, 2.0,
            json.dumps({"status": "completed", "summary": "event"}),
            json.dumps({"status": "completed", "summary": "result"}),
            "pending", 3, "claim-old", 1.5, None, 4321, 99,
            json.dumps({"goal": "legacy"}),
            json.dumps(["sa-z", "sa-a"]), json.dumps(children),
        ),
    )
    conn.commit()
    conn.close()

    repository = DelegationRepository(path)
    first = repository.snapshot("deleg-old")
    second = repository.snapshot("deleg-old")
    assert first == second
    assert first["lifecycle_version"] == 2
    assert first["root_subagent_ids"] == ["sa-z", "sa-a"]
    assert first["children"]["sa-child"]["parent_id"] == "sa-z"
    assert first["children"]["sa-child"]["status"] == "running"
    assert first["children"]["sa-z"]["status"] == "completed"
    assert first["children"]["sa-a"]["status"] == "error"
    assert first["delivery_state"] == "delivering"
    assert first["delivery_claim"] == "claim-old"
    assert first["delivery_attempts"] == 3
    assert first["owner_pid"] == 4321

    with sqlite3.connect(path) as check:
        assert check.execute(
            "SELECT COUNT(*) FROM delegation_runs WHERE delegation_id='deleg-old'"
        ).fetchone()[0] == 1
        assert check.execute(
            "SELECT lifecycle_version FROM async_delegations WHERE delegation_id='deleg-old'"
        ).fetchone()[0] == 2

def test_released_schema_migrates_under_concurrent_process_open(tmp_path):
    path = tmp_path / "state.db"
    _released_schema(str(path))
    conn = sqlite3.connect(path)
    conn.execute(
        """INSERT INTO async_delegations
           (delegation_id, origin_session, state, dispatched_at, updated_at,
            root_subagent_ids_json, children_json)
           VALUES ('deleg-mp','owner','running',1,1,'["sa-root"]',
                   '{"sa-root":{"status":"running"}}')"""
    )
    conn.commit()
    conn.close()

    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    output = ctx.Queue()
    workers = [ctx.Process(target=_open_snapshot, args=(str(path), "deleg-mp", start, output)) for _ in range(6)]
    for worker in workers:
        worker.start()
    start.set()
    outcomes = [output.get(timeout=20) for _ in workers]
    for worker in workers:
        worker.join(20)
        assert worker.exitcode == 0

    assert outcomes == [("ok", 2)] * 6

def test_run_delivery_is_exact_and_omitted_run_fails_ambiguous(repo):
    initial = repo.register_initial_dispatch(_record())
    repo.complete_run(initial["run_id"], {"status": "completed"}, {"summary": "initial"})
    repo.transition_attempt(initial["attempts"][0]["attempt_id"], {"completed"}, "completed")
    resumed = repo.reserve_resumed_attempt("sa-root")
    repo.complete_run(resumed["run_id"], {"status": "completed"}, {"summary": "resumed"})

    first = repo.claim_run_delivery("deleg-1", initial["run_id"], "claim-1")
    second = repo.claim_run_delivery("deleg-1", resumed["run_id"], "claim-2")
    assert first["status"] == second["status"] == "claimed"
    assert repo.commit_run_delivery(initial["run_id"], "claim-1")["status"] == "delivered"
    assert repo.release_run_delivery(resumed["run_id"], "claim-2")["status"] == "released"
    assert repo.claim_run_delivery("deleg-1", None, "claim-3") == {
        "status": "ambiguous_run"
    }

def test_retention_protects_active_attempts_and_nonterminal_delivery(repo):
    repo.register_initial_dispatch(_record("deleg-active", ["sa-active"], 1.0))
    protected = {}
    for index, state in enumerate(("pending", "held_by_wait", "delivering"), 2):
        delegation_id = f"deleg-{state}"
        created = repo.register_initial_dispatch(
            _record(delegation_id, [f"sa-{state}"], float(index))
        )
        repo.complete_run(created["run_id"], {"status": "completed"}, {"summary": state})
        if state == "held_by_wait":
            repo.hold_for_wait(delegation_id, "owner", "hold")
        elif state == "delivering":
            repo.claim_run_delivery(delegation_id, created["run_id"], "claim")
        protected[delegation_id] = state
    for index, state in enumerate(("delivered", "consumed", "suppressed"), 5):
        delegation_id = f"deleg-{state}"
        created = repo.register_initial_dispatch(
            _record(delegation_id, [f"sa-{state}"], float(index))
        )
        repo.complete_run(created["run_id"], {"status": "completed"}, {"summary": state})
        if state == "delivered":
            repo.acknowledge_pending(delegation_id)
        elif state == "consumed":
            repo.hold_for_wait(delegation_id, "owner", "hold")
            repo.consume_wait_hold(delegation_id, "owner", "hold")
        else:
            repo.suppress_delivery(delegation_id, "owner")
    outcome = repo.prune(cutoff=time.time() + 100.0, max_terminal=0)
    assert outcome["deleted"] == 3
    assert repo.snapshot("deleg-active") is not None
    for delegation_id, state in protected.items():
        assert repo.snapshot(delegation_id)["delivery_state"] == state
    for state in ("delivered", "consumed", "suppressed"):
        assert repo.snapshot(f"deleg-{state}") is None

def test_authorized_snapshot_derives_compatibility_fields(repo):
    created = repo.register_initial_dispatch(_record())
    attempt_id = created["attempts"][0]["attempt_id"]
    repo.transition_attempt(
        attempt_id,
        {"starting"},
        "running",
        metadata={"goal": "updated", "tool_count": 2, "last_tool": "terminal"},
    )
    own = repo.snapshot("deleg-1", session_key="owner")

    assert own["worker_status"] == own["state"] == "running"
    assert own["delivery_disposition"] == own["delivery_state"] == "pending"
    assert own["children"]["sa-root"]["goal"] == "updated"
    assert own["children"]["sa-root"]["tool_count"] == 2
    assert own["owner_pid"] is None
    assert repo.snapshot("deleg-1", session_key="foreign") is None

def test_list_snapshots_hydrates_one_hundred_records_with_one_connection(repo):
    for index in range(100):
        repo.register_initial_dispatch(_record(f"deleg-{index}", [f"sa-{index}"]))
    opened, connect = 0, repo._connect
    def counted_connect():
        nonlocal opened
        opened += 1; return connect()
    repo._connect = counted_connect
    assert len(repo.list_snapshots(limit=100)) == 100
    assert opened == 1

@pytest.mark.parametrize(("legacy_state", "expected"), [
    ("starting", "starting"), ("running", "running"), ("finalizing", "finalizing"),
    ("interrupt_requested", "interrupt_requested"), ("completed", "completed"),
    ("success", "completed"), ("error", "error"), ("failed", "error"),
    ("budget_exhausted", "error"), ("interrupted", "interrupted"),
    ("cancelled", "interrupted"), ("timeout", "timeout"), ("mystery", "unknown"),
])
@pytest.mark.parametrize("delivery", [
    "pending", "held_by_wait", "delivering", "delivered", "consumed", "suppressed"
])
def test_released_migration_matrix_and_frozen_legacy_fields(tmp_path, legacy_state, expected, delivery):
    path = tmp_path / "state.db"
    _released_schema(str(path))
    claim = "legacy-claim" if delivery in {"held_by_wait", "delivering"} else None
    with sqlite3.connect(path) as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id,origin_session,state,dispatched_at,updated_at,
                completed_at,event_json,result_json,delivery_state,delivery_claim,
                root_subagent_ids_json,children_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "deleg-matrix", "owner", legacy_state, 1.0, 2.0,
                None if legacy_state in {"starting", "running", "finalizing", "interrupt_requested"} else 2.0,
                json.dumps({"status": legacy_state}), json.dumps({"status": legacy_state}),
                delivery, claim, '["sa-root"]',
                json.dumps({"sa-root": {"status": legacy_state}}),
            ),
        )
    repository = DelegationRepository(path)
    before = repository.snapshot("deleg-matrix")
    assert before["children"]["sa-root"]["status"] == expected
    assert before["delivery_state"] == delivery
    with sqlite3.connect(path) as conn:
        conn.execute(
            """UPDATE async_delegations SET state='failed',delivery_state='suppressed',
               children_json='{}',event_json='{}' WHERE delegation_id='deleg-matrix'"""
        )
    after = repository.snapshot("deleg-matrix")
    assert after["children"] == before["children"]
    assert after["delivery_state"] == delivery
    assert after["event"] == before["event"]

@pytest.mark.parametrize("repository_first", [True, False])
def test_repository_and_session_db_bootstrap_in_either_order(tmp_path, repository_first):
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    repository = DelegationRepository(path)
    if repository_first:
        assert repository.snapshot("missing") is None
    database = SessionDB(db_path=path)
    database.close()
    if not repository_first:
        assert repository.snapshot("missing") is None
    assert repository.register_initial_dispatch(_record())["status"] == "registered"
    assert repository.snapshot("deleg-1")["lifecycle_version"] == 2

def test_preallocated_ids_survive_registration_archive_completion_and_event(repo):
    record = _record()
    record.update(run_id="run-preallocated", attempt_ids=["attempt-preallocated"])
    created = repo.register_initial_dispatch(record)
    assert created["run_id"] == "run-preallocated"
    assert created["attempts"][0]["attempt_id"] == "attempt-preallocated"
    registered = repo.register_subagent(
        {"subagent_id": "sa-root", "status": "running", "tool_count": 1}
    )
    assert registered["attempt_id"] == "attempt-preallocated"
    repo.transition_attempt(
        "attempt-preallocated", {"running"}, "completed", metadata={"summary": "archived"}
    )
    event = {"status": "completed", "run_id": "run-preallocated", "completed_at": 5.0}
    assert repo.complete_run("run-preallocated", event, {"summary": "done"})["status"] == "completed"
    snapshot = repo.snapshot("deleg-1")
    assert snapshot["children"]["sa-root"]["attempt_id"] == "attempt-preallocated"
    assert snapshot["children"]["sa-root"]["summary"] == "archived"
    assert repo.pending_events()[0] == {"run_id": "run-preallocated", "event": event}

def test_repeated_multiprocess_resume_has_one_winner(tmp_path):
    path = tmp_path / "state.db"
    repository = DelegationRepository(path)
    created = repository.register_initial_dispatch(_record())
    repository.transition_attempt(created["attempts"][0]["attempt_id"], {"starting"}, "completed")
    ctx = multiprocessing.get_context("spawn")
    for round_number in range(3):
        start, output = ctx.Event(), ctx.Queue()
        workers = [
            ctx.Process(
                target=_reserve_attempt,
                args=(str(path), "sa-root", f"worker-{round_number}-{index}", start, output),
            )
            for index in range(6)
        ]
        for worker in workers:
            worker.start()
        start.set()
        outcomes = [output.get(timeout=20) for _ in workers]
        for worker in workers:
            worker.join(20)
            assert worker.exitcode == 0
        winners = [outcome for outcome in outcomes if outcome["status"] == "reserved"]
        assert len(winners) == 1
        assert [outcome["status"] for outcome in outcomes].count("already_running") == 5
        repository.transition_attempt(winners[0]["attempt_id"], {"starting"}, "completed")
