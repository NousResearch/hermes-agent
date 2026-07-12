"""Durable restart recovery for gateway background delegations."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield tmp_path
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _spec(*, goal="continue the report", generation=0):
    return {
        "profile": "default",
        "source": {
            "kind": "single",
            "tasks": [{
                "goal": goal,
                "context": "write /tmp/report.md",
                "role": "leaf",
                "inherit_context": False,
            }],
            "shared_context": None,
        },
        "execution": {
            "model": "test-model",
            "provider": "test-provider",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
            "acp_command": None,
            "acp_args": [],
            "reasoning_config": None,
            "fallback_chain": None,
            "service_tier": None,
            "provider_preferences": None,
            "toolsets": ["file"],
            "max_iterations": 50,
            "parent_depth": 0,
            "workspace_hint": "/tmp",
            "credential_ref": {"provider": "test-provider", "custom_provider": None},
        },
        "route": {
            "session_key": "agent:main:telegram:dm:123",
            "parent_session_id": "parent-1",
            "origin_ui_session_id": "",
            "platform": "telegram",
            "chat_type": "dm",
            "chat_id": "123",
            "thread_id": None,
            "user_id": "u1",
            "user_name": "Ace",
            "profile": "default",
        },
    }


def _dispatch(gate=None):
    gate = gate or threading.Event()

    def runner():
        gate.wait(timeout=5)
        return {"status": "completed", "summary": "done"}

    result = ad.dispatch_async_delegation_batch(
        goals=["continue the report"],
        context="write /tmp/report.md",
        toolsets=["file"],
        role="leaf",
        model="test-model",
        session_key="agent:main:telegram:dm:123",
        parent_session_id="parent-1",
        runner=runner,
        max_async_children=3,
        durable_spec=_spec(),
        current_boot_id="100:1.0",
    )
    return result, gate


def _load():
    return json.loads(ad._registry_path().read_text(encoding="utf-8"))


def _write_record(record):
    state = {"schema_version": 1, "updated_at": time.time(), "records": {record["delegation_id"]: record}}
    ad._write_registry_for_tests(state)


def _running_record(*, delegation_id="deleg_rc", owner="100:1.0", generation=0,
                    redispatch_count=0, submitted_at=1.0):
    now = time.time()
    spec = _spec()
    return {
        "delegation_id": delegation_id,
        "state": "running",
        "created_at": now,
        "updated_at": now,
        "profile": "default",
        "source": spec["source"],
        "execution": spec["execution"],
        "route": spec["route"],
        "attempt": {
            "attempt_id": f"{delegation_id}:g{generation}:old",
            "generation": generation,
            "redispatch_count": redispatch_count,
            "owner_boot_id": owner,
            "started_at": now,
            "submitted_at": submitted_at,
            "last_interrupted_at": None,
            "last_error": None,
        },
        "terminal": None,
        "outbox": [],
    }


def test_dispatch_persists_generation_zero_before_success():
    result, gate = _dispatch()
    assert result["status"] == "dispatched"
    record = _load()["records"][result["delegation_id"]]
    assert record["state"] == "running"
    assert record["attempt"]["generation"] == 0
    assert record["attempt"]["redispatch_count"] == 0
    assert record["attempt"]["owner_boot_id"] == "100:1.0"
    assert "submitted_at" in record["attempt"]
    assert record["source"]["tasks"][0]["goal"] == "continue the report"
    assert record["route"]["parent_session_id"] == "parent-1"
    assert "api_key" not in json.dumps(record)
    gate.set()


def test_registry_permissions_are_owner_only():
    result, gate = _dispatch()
    assert result["status"] == "dispatched"
    assert ad._registry_path().stat().st_mode & 0o777 == 0o600
    assert ad._registry_path().parent.stat().st_mode & 0o077 == 0
    gate.set()


def test_stale_attempt_finalize_cannot_overwrite_newer_attempt():
    result, gate = _dispatch()
    first = _load()["records"][result["delegation_id"]]
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    with ad._locked_registry() as registry:
        current = registry["records"][result["delegation_id"]]
        current["attempt"]["attempt_id"] = "new-attempt"
        current["attempt"]["generation"] = 1
    ad._finalize(result["delegation_id"], {"status": "completed", "summary": "stale"}, "completed",
                 attempt_id=first["attempt"]["attempt_id"], registry_path=ad._registry_path())
    assert _load()["records"][result["delegation_id"]]["state"] == "running"
    assert process_registry.completion_queue.empty()
    gate.set()


def test_recovery_claims_dead_boot_once_and_submits_existing_executor(monkeypatch):
    record = _running_record()
    _write_record(record)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    calls = []
    gate = threading.Event()
    started = threading.Event()

    def factory(claimed, continuation):
        calls.append((claimed["attempt"]["generation"], continuation))
        def run():
            started.set()
            return gate.wait(timeout=5) or {"status": "completed", "summary": "resumed"}
        return run

    summary = ad.recover_async_delegations(
        current_boot_id="200:2.0", runner_factory=factory, max_async_children=1
    )
    assert started.wait(timeout=5)
    current = _load()["records"][record["delegation_id"]]
    assert summary["claimed"] == 1
    assert current["attempt"]["generation"] == 1
    assert current["attempt"]["redispatch_count"] == 1
    assert current["attempt"]["submitted_at"] is not None
    assert calls and "CONTINUE" in calls[0][1]
    assert "/tmp/report.md" not in calls[0][1]
    gate.set()


def test_concurrent_recovery_scans_claim_exactly_once(monkeypatch):
    _write_record(_running_record())
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    barrier = threading.Barrier(2)
    calls = []
    gate = threading.Event()

    def scan():
        barrier.wait(timeout=5)
        return ad.recover_async_delegations(
            current_boot_id="200:2.0",
            runner_factory=lambda record, continuation: (
                calls.append(record["attempt"]["attempt_id"]) or
                (lambda: (gate.wait(timeout=5) or {"status": "completed"}))
            ),
            max_async_children=1,
        )

    results = []
    threads = [threading.Thread(target=lambda: results.append(scan())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert sum(item["claimed"] for item in results) == 1
    assert len(calls) == 1
    gate.set()


def test_rc1_claimed_but_never_submitted_does_not_count_attempt(monkeypatch):
    """RC-1: dead claim without submission telemetry is reconciled, then replaced."""
    record = _running_record(generation=1, redispatch_count=1, submitted_at=None)
    _write_record(record)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    gate = threading.Event()
    started = threading.Event()

    def resumed_runner():
        started.set()
        return gate.wait(timeout=5) or {"status": "completed"}

    result = ad.recover_async_delegations(
        current_boot_id="300:3.0",
        runner_factory=lambda record, continuation: resumed_runner,
        max_async_children=1,
    )
    assert started.wait(timeout=5)
    current = _load()["records"][record["delegation_id"]]
    assert result["claimed"] == 1
    assert current["attempt"]["generation"] == 2
    assert current["attempt"]["redispatch_count"] == 1
    assert current["attempt"]["submitted_at"] is not None
    gate.set()


def test_real_replacement_attempts_exhaust_after_two(monkeypatch):
    record = _running_record(generation=2, redispatch_count=2, submitted_at=1.0)
    _write_record(record)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    called = []
    result = ad.recover_async_delegations(
        current_boot_id="400:4.0",
        runner_factory=lambda record, continuation: called.append(record),
    )
    current = _load()["records"][record["delegation_id"]]
    assert result["exhausted"] == 1
    assert current["state"] == "failed"
    assert called == []
    assert len([event for event in current["outbox"] if event["type"] == "async_delegation"]) == 1


def test_rc2_pending_restart_event_replays_and_superseded_event_drops(monkeypatch):
    """RC-2: restart events replay like terminal events and stale generations drop."""
    record = _running_record(generation=1, redispatch_count=1, submitted_at=1.0)
    old = {
        "event_id": f"{record['delegation_id']}:restart:g1",
        "type": "async_delegation_restarted",
        "state": "pending",
        "queued_boot_id": "100:1.0",
        "created_at": time.time(),
        "delivered_at": None,
        "drop_reason": None,
        "payload": {"type": "async_delegation_restarted", "delegation_id": record["delegation_id"],
                    "attempt_generation": 1, "session_key": record["route"]["session_key"]},
    }
    record["outbox"].append(old)
    _write_record(record)

    queued = ad.enqueue_pending_outbox(current_boot_id="150:1.5")
    assert queued == 1
    assert process_registry.completion_queue.get_nowait()["type"] == "async_delegation_restarted"

    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    gate = threading.Event()
    ad.recover_async_delegations(
        current_boot_id="200:2.0",
        runner_factory=lambda record, continuation: (
            lambda: (gate.wait(timeout=5) or {"status": "completed"})
        ),
        max_async_children=1,
    )
    current = _load()["records"][record["delegation_id"]]
    stale = next(event for event in current["outbox"] if event["event_id"] == old["event_id"])
    assert stale["state"] == "dropped"
    assert stale["drop_reason"] == "superseded"
    assert any(event["event_id"].endswith(":restart:g2") for event in current["outbox"])
    gate.set()


def test_terminal_outbox_replays_and_acknowledges():
    record = _running_record()
    event = {
        "event_id": f"{record['delegation_id']}:terminal:g0",
        "type": "async_delegation",
        "state": "pending",
        "queued_boot_id": None,
        "created_at": time.time(),
        "delivered_at": None,
        "drop_reason": None,
        "payload": {"type": "async_delegation", "event_id": f"{record['delegation_id']}:terminal:g0",
                    "delegation_id": record["delegation_id"], "session_key": record["route"]["session_key"]},
    }
    record["state"] = "done"
    record["outbox"].append(event)
    _write_record(record)
    assert ad.enqueue_pending_outbox(current_boot_id="200:2.0") == 1
    assert ad.acknowledge_outbox_event(event["event_id"], outcome="delivered") is True
    current = _load()["records"][record["delegation_id"]]
    assert current["outbox"][0]["state"] == "delivered"
    assert ad.enqueue_pending_outbox(current_boot_id="300:3.0") == 0


def test_session_cancel_persists_before_interrupt_signal():
    result, gate = _dispatch()
    observed = []
    with ad._records_lock:
        ad._records[result["delegation_id"]]["interrupt_fn"] = lambda: observed.append(
            _load()["records"][result["delegation_id"]]["state"]
        )
    assert ad.interrupt_for_session(parent_session_id="parent-1") == 1
    assert observed == ["cancelled"]
    gate.set()


def test_clean_shutdown_recoverable_is_best_effort_and_bounded(monkeypatch):
    result, gate = _dispatch()
    observed = []
    with ad._records_lock:
        ad._records[result["delegation_id"]]["interrupt_fn"] = lambda: observed.append(True)
    monkeypatch.setattr(ad, "_mark_recoverable", lambda *args, **kwargs: False)
    started = time.monotonic()
    assert ad.interrupt_all(reason="shutdown", recoverable=True, lock_timeout=0.01) == 1
    assert time.monotonic() - started < 1.0
    assert observed == [True]
    gate.set()


def test_rc3_failed_recoverable_marker_cannot_terminalize_restart_intent(monkeypatch):
    """RC-3: lock contention leaves running intent for dead-owner recovery."""
    result, gate = _dispatch()
    with ad._records_lock:
        ad._records[result["delegation_id"]]["interrupt_fn"] = lambda: None
    monkeypatch.setattr(ad, "_mark_recoverable", lambda *args, **kwargs: False)
    assert ad.interrupt_all(reason="shutdown", recoverable=True, lock_timeout=0.01) == 1
    ad._finalize_batch(
        result["delegation_id"],
        {"results": [], "error": "interrupted"},
        "interrupted",
    )
    assert _load()["records"][result["delegation_id"]]["state"] == "running"
    gate.set()


def test_record_cap_rejects_without_deleting_live_records_and_counts_degradation(monkeypatch):
    monkeypatch.setattr(ad, "_MAX_REGISTRY_RECORDS", 2)
    monkeypatch.setattr(ad._store, "MAX_REGISTRY_RECORDS", 2)
    records = {
        "a": _running_record(delegation_id="a"),
        "b": _running_record(delegation_id="b"),
    }
    ad._write_registry_for_tests({"schema_version": 1, "updated_at": time.time(), "records": records})
    result, gate = _dispatch()
    assert result["status"] == "rejected"
    assert result["reason"] == "registry_cap"
    assert set(_load()["records"]) == {"a", "b"}
    assert ad.observability_counters()["sync_fallback_registry_cap"] == 1
    gate.set()


def test_resume_disabled_does_not_mutate_or_dispatch(monkeypatch):
    record = _running_record()
    _write_record(record)
    before = ad._registry_path().read_bytes()
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    result = ad.recover_async_delegations(
        current_boot_id="200:2.0", runner_factory=lambda *args: pytest.fail("must not run"),
        resume_enabled=False,
    )
    assert result["claimed"] == 0
    assert ad._registry_path().read_bytes() == before


def test_dispatch_intent_is_durable_before_executor_submit(monkeypatch):
    queued = []

    class InspectingExecutor:
        def submit(self, fn):
            record = next(iter(_load()["records"].values()))
            assert record["attempt"]["submitted_at"] is None
            queued.append(fn)
            return object()

    monkeypatch.setattr(ad, "_get_executor", lambda workers: InspectingExecutor())
    result, _ = _dispatch()
    assert result["status"] == "dispatched"
    assert _load()["records"][result["delegation_id"]]["attempt"]["submitted_at"] is None
    queued[0]()
    current = _load()["records"][result["delegation_id"]]
    assert current["attempt"]["submitted_at"] is not None


def test_submission_telemetry_failure_cannot_trigger_inline_duplicate(monkeypatch):
    queued = []
    ran = []

    class QueuingExecutor:
        def submit(self, fn):
            queued.append(fn)
            return object()

    monkeypatch.setattr(ad, "_get_executor", lambda workers: QueuingExecutor())
    monkeypatch.setattr(
        ad._store,
        "mark_submitted",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("lock busy")),
    )
    result = ad.dispatch_async_delegation_batch(
        goals=["continue the report"],
        context=None,
        toolsets=None,
        role="leaf",
        model="test-model",
        session_key="agent:main:telegram:dm:123",
        parent_session_id="parent-1",
        runner=lambda: (ran.append(True) or {"status": "completed"}),
        durable_spec=_spec(),
        current_boot_id="100:1.0",
    )

    assert result["status"] == "dispatched"
    queued[0]()
    assert ran == []
    assert _load()["records"][result["delegation_id"]]["state"] == "failed"


def test_executor_submit_failure_terminalizes_attempt_before_fallback(monkeypatch):
    class FailingExecutor:
        def submit(self, fn):
            raise RuntimeError("executor unavailable")

    monkeypatch.setattr(ad, "_get_executor", lambda workers: FailingExecutor())
    result = ad.dispatch_async_delegation_batch(
        goals=["continue the report"],
        context=None,
        toolsets=None,
        role="leaf",
        model="test-model",
        session_key="agent:main:telegram:dm:123",
        parent_session_id="parent-1",
        runner=lambda: pytest.fail("must not run"),
        durable_spec=_spec(),
        current_boot_id="100:1.0",
    )

    record = _load()["records"][result["delegation_id"]]
    assert result["status"] == "rejected"
    assert result["reason"] == "submission_failed"
    assert record["state"] == "failed"
    assert result["fallback_event_id"] == record["outbox"][0]["event_id"]


def test_registry_is_profile_scoped(tmp_path):
    first = tmp_path / "profiles" / "first"
    second = tmp_path / "profiles" / "second"
    record = _running_record(delegation_id="only-first")
    ad._store.write_for_tests(
        {"schema_version": 1, "updated_at": time.time(), "records": {"only-first": record}},
        profile_home=first,
    )
    assert set(ad._store.read_registry(first)["records"]) == {"only-first"}
    assert ad._store.read_registry(second)["records"] == {}


def test_registry_integrity_mismatch_fails_closed():
    record = _running_record()
    _write_record(record)
    raw = json.loads(ad._registry_path().read_text(encoding="utf-8"))
    raw["records"][record["delegation_id"]]["source"]["tasks"][0]["goal"] = "tampered"
    ad._registry_path().write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ad._store.RegistryError, match="integrity"):
        ad._store.read_registry()


def test_malformed_record_is_preserved_while_valid_sibling_recovers(monkeypatch):
    bad = _running_record(delegation_id="bad")
    good = _running_record(delegation_id="good")
    ad._write_registry_for_tests(
        {
            "schema_version": 1,
            "updated_at": time.time(),
            "records": {"bad": bad, "good": good},
        }
    )
    raw = json.loads(ad._registry_path().read_text(encoding="utf-8"))
    raw["records"]["bad"]["source"]["tasks"][0]["goal"] = "tampered"
    ad._registry_path().write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    gate = threading.Event()
    summary = ad.recover_async_delegations(
        current_boot_id="200:2.0",
        runner_factory=lambda record, continuation: (
            lambda: (gate.wait(timeout=5) or {"status": "completed"})
        ),
        max_async_children=1,
    )
    after = json.loads(ad._registry_path().read_text(encoding="utf-8"))
    assert summary["claimed"] == 1
    assert summary["failed_validation"] == 1
    assert after["records"]["bad"]["source"]["tasks"][0]["goal"] == "tampered"
    assert after["records"]["bad"]["integrity"] == raw["records"]["bad"]["integrity"]
    gate.set()


def test_queued_recovery_is_not_submitted_until_worker_starts(monkeypatch):
    record = _running_record()
    _write_record(record)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    queued = []

    class QueuingExecutor:
        def submit(self, fn):
            queued.append(fn)
            return object()

    monkeypatch.setattr(ad, "_get_executor", lambda workers: QueuingExecutor())
    summary = ad.recover_async_delegations(
        current_boot_id="200:2.0",
        runner_factory=lambda record, continuation: lambda: {"status": "completed"},
        max_async_children=1,
    )
    claimed = _load()["records"][record["delegation_id"]]
    assert summary["claimed"] == 1
    assert claimed["attempt"]["submitted_at"] is None
    assert len(queued) == 1
    queued[0]()
    assert _load()["records"][record["delegation_id"]]["attempt"]["submitted_at"] is not None


def test_session_cancel_stops_queued_recovery_before_runner_starts(monkeypatch):
    record = _running_record()
    _write_record(record)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    queued = []
    ran = []

    class QueuingExecutor:
        def submit(self, fn):
            queued.append(fn)
            return object()

    monkeypatch.setattr(ad, "_get_executor", lambda workers: QueuingExecutor())
    ad.recover_async_delegations(
        current_boot_id="200:2.0",
        runner_factory=lambda record, continuation: lambda: ran.append(True),
        max_async_children=1,
    )
    ad.interrupt_for_session(parent_session_id="parent-1")
    queued[0]()
    assert ran == []
    assert _load()["records"][record["delegation_id"]]["state"] == "cancelled"


def test_session_cancel_closes_resume_disabled_durable_record():
    record = _running_record()
    record["state"] = "recoverable"
    _write_record(record)
    assert ad.interrupt_for_session(parent_session_id="parent-1") == 0
    assert _load()["records"][record["delegation_id"]]["state"] == "cancelled"


def test_unsupported_route_fails_without_redispatch(monkeypatch):
    record = _running_record()
    record["route"]["parent_session_id"] = ""
    _write_record(record)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    called = []
    summary = ad.recover_async_delegations(
        current_boot_id="200:2.0",
        runner_factory=lambda *args: called.append(args),
    )
    assert summary["claimed"] == 0
    assert summary["failed_validation"] == 1
    assert called == []
    failed = _load()["records"][record["delegation_id"]]
    assert failed["state"] == "failed"
    assert failed["terminal"]["error"] == "unsupported_route"


def test_list_async_delegations_includes_durable_terminal_records():
    record = _running_record(delegation_id="durable-only")
    record["state"] = "done"
    record["terminal"] = {"status": "completed", "completed_at": time.time()}
    _write_record(record)
    listed = {item["delegation_id"]: item for item in ad.list_async_delegations()}
    assert listed["durable-only"]["status"] == "done"
    assert listed["durable-only"]["generation"] == 0
    assert "execution" not in listed["durable-only"]


def test_secret_shaped_execution_settings_force_sync_fallback():
    spec = _spec()
    spec["execution"]["api_key"] = "must-not-land"
    result = ad.dispatch_async_delegation_batch(
        goals=["g"],
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="agent:main:telegram:dm:1",
        runner=lambda: {"status": "completed"},
        durable_spec=spec,
        current_boot_id="100:1.0",
    )
    assert result["status"] == "rejected"
    assert result["reason"] == "registry_error"
    assert not ad._registry_path().exists()


def test_recovery_of_running_dead_owner_matches_recoverable_marker(monkeypatch):
    record = _running_record()
    _write_record(record)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    gate = threading.Event()
    result = ad.recover_async_delegations(
        current_boot_id="200:2.0",
        runner_factory=lambda record, continuation: (
            lambda: (gate.wait(timeout=5) and {"status": "completed"})
        ),
        max_async_children=1,
    )
    assert result["claimed"] == 1
    assert _load()["records"][record["delegation_id"]]["attempt"]["generation"] == 1
    gate.set()


def test_batch_recovery_unit_emits_one_consolidated_terminal_event(monkeypatch):
    record = _running_record()
    record["source"]["kind"] = "batch"
    record["source"]["tasks"].append({
        "goal": "second task", "context": None, "role": "leaf", "inherit_context": False,
    })
    _write_record(record)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    result = ad.recover_async_delegations(
        current_boot_id="200:2.0",
        runner_factory=lambda record, continuation: lambda: {
            "results": [
                {"task_index": 0, "status": "completed", "summary": "one"},
                {"task_index": 1, "status": "completed", "summary": "two"},
            ],
            "total_duration_seconds": 0.1,
        },
        max_async_children=1,
    )
    assert result["claimed"] == 1
    terminal = process_registry.completion_queue.get(timeout=5)
    assert terminal["type"] == "async_delegation"
    assert terminal["is_batch"] is True
    assert len(terminal["results"]) == 2
    assert process_registry.completion_queue.empty()


def test_terminal_retention_prunes_old_acknowledged_record():
    old = _running_record(delegation_id="old")
    old["state"] = "done"
    old["updated_at"] = time.time() - ad._store.TERMINAL_RETENTION_SECONDS - 1
    old["terminal"] = {"status": "completed", "completed_at": old["updated_at"]}
    ad._write_registry_for_tests(
        {"schema_version": 1, "updated_at": time.time(), "records": {"old": old}}
    )
    result, gate = _dispatch()
    assert result["status"] == "dispatched"
    assert "old" not in _load()["records"]
    gate.set()


def test_active_record_older_than_stale_cutoff_fails_without_redispatch(monkeypatch):
    old = _running_record()
    old["created_at"] = time.time() - ad._store.ACTIVE_STALE_SECONDS - 1
    _write_record(old)
    monkeypatch.setattr(ad, "is_boot_id_alive", lambda boot: False)
    called = []
    result = ad.recover_async_delegations(
        current_boot_id="200:2.0",
        runner_factory=lambda *args: called.append(args),
    )
    assert result["claimed"] == 0
    assert called == []
    assert _load()["records"][old["delegation_id"]]["state"] == "failed"
