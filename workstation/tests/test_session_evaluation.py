from __future__ import annotations

import os
import subprocess
import sys

import pytest

from workstation.evaluation import EvaluationCase, EvaluationHarness, RegressionBudget, SoakRunner
from workstation.journal import ExecutionJournal
from workstation.session_lifecycle import (
    SessionLease,
    SessionLifecycleStore,
    SessionTemperature,
)


def test_session_lease_is_exclusive_and_lifecycle_can_cool_down(tmp_path):
    lease_path = tmp_path / "session.lock"
    first = SessionLease(lease_path, owner_id="process-a")
    second = SessionLease(lease_path, owner_id="process-b")
    assert first.acquire() is True
    with pytest.raises(RuntimeError, match="owned"):
        second.acquire()
    first.release()
    assert second.acquire() is True

    store = SessionLifecycleStore(tmp_path / "sessions.json")
    store.register("session-1", model_id="model-a", provider="provider-a")
    store.set_temperature("session-1", SessionTemperature.COLD)
    assert store.get("session-1").temperature == SessionTemperature.COLD
    second.release()


def test_session_lease_rejects_a_second_process(tmp_path):
    lease_path = tmp_path / "cross-process.lock"
    lease = SessionLease(lease_path, owner_id="parent")
    lease.acquire()
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from workstation.session_lifecycle import SessionLease; "
            "lease=SessionLease(__import__('pathlib').Path(sys.argv[1]), owner_id='child'); "
            "lease.acquire()",
            str(lease_path),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0
    assert "already owned" in probe.stderr
    lease.release()


def test_session_migration_rolls_back_on_validation_failure(tmp_path):
    store = SessionLifecycleStore(tmp_path / "sessions.json")
    store.register("session-2", model_id="model-a", provider="provider-a")
    with pytest.raises(ValueError, match="validation"):
        store.migrate(
            "session-2",
            target_version=2,
            transform=lambda record: {**record, "model_id": "model-b"},
            validate=lambda record: False,
        )
    restored = store.get("session-2")
    assert restored.model_id == "model-a"
    assert restored.schema_version == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "sessions.json.1.bak").exists()


def test_session_compaction_is_explicit_and_persisted(tmp_path):
    store = SessionLifecycleStore(tmp_path / "sessions.json")
    store.register("session-compact")
    store.update_context_usage("session-compact", context_tokens=950, context_token_limit=1000)
    assert store.should_compact("session-compact", reserve_tokens=60)
    compacted = store.mark_compacted("session-compact", resulting_tokens=300)
    assert compacted.compaction_count == 1
    assert compacted.last_compacted_at
    restored = SessionLifecycleStore(tmp_path / "sessions.json").get("session-compact")
    assert restored.context_tokens == 300


def test_session_store_surfaces_corrupt_persistence_as_diagnostic(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not-json", encoding="utf-8")
    store = SessionLifecycleStore(path)
    assert store.diagnostics()["load_error"]


def test_model_independent_eval_and_regression_budget():
    harness = EvaluationHarness()
    case = EvaluationCase(name="search", input="find shoes")
    baseline = harness.run(case, lambda _: {"success": True, "actions": 3, "latency_seconds": 1.0, "tokens": 100, "cost_usd": 0.01})
    candidate = harness.run(case, lambda _: {"success": True, "actions": 4, "latency_seconds": 1.1, "tokens": 105, "cost_usd": 0.011})
    comparison = harness.compare(
        baseline,
        candidate,
        RegressionBudget(max_action_increase=1, max_latency_increase=0.2, max_token_increase=10, max_cost_increase=0.002),
    )
    assert comparison.accepted is True
    assert comparison.model_independent is True


def test_evaluation_metrics_can_be_recorded_in_canonical_journal(tmp_path):
    journal = ExecutionJournal("task-eval", "session-eval", file_path=tmp_path / "journal.jsonl")
    harness = EvaluationHarness(journal)
    result = harness.run(EvaluationCase("routing", "input"), lambda _: {"success": True, "actions": 2, "tokens": 20})
    assert result.success
    assert journal.read_events()[0].metadata["actions"] == 2


def test_soak_runner_reports_failures_and_metrics():
    result = SoakRunner().run(3, lambda index: {"success": index != 1, "actions": index + 1, "latency_seconds": 0.01})
    assert result.iterations == 3
    assert result.failures == 1
    assert result.success_rate == pytest.approx(2 / 3)


def test_soak_runner_supports_bounded_duration_and_iteration_evidence():
    result = SoakRunner().run_for_duration(0, lambda _: {"success": True}, max_iterations=10)
    assert result.iterations == 10
    assert result.completed_iterations == 0
    assert result.timed_out is True
    assert result.duration_seconds >= 0

    with pytest.raises(ValueError, match="non-negative"):
        SoakRunner().run_for_duration(-1, lambda _: {"success": True})
