from __future__ import annotations

import time

import pytest

from workstation.runtime import (
    BudgetExceeded,
    BudgetTracker,
    CancellationToken,
    ExecutionStatus,
    EvidenceState,
    EvidenceStateStore,
    ResourceRegistry,
    RuntimeEvent,
    RuntimeEventBus,
    HumanHandoffManager,
    ModelRouter,
    RoutingCandidate,
    TypedResource,
    run_with_deadline,
)


def test_running_requires_live_evidence_and_degrades_when_stale():
    state = EvidenceState(task_id="task-1", session_id="session-1")
    state.transition(ExecutionStatus.RUNNING, now="2026-09-10T12:00:00+00:00")
    assert state.status == ExecutionStatus.STALLED

    state.transition(ExecutionStatus.RUNNING, now="2026-09-10T12:00:01+00:00")
    state.add_evidence(
        "worker_id",
        "worker-1",
        ttl_seconds=30,
        now="2026-09-10T12:00:01+00:00",
    )
    assert state.status == ExecutionStatus.RUNNING
    state.reconcile(now="2026-09-10T12:00:32+00:00")
    assert state.status == ExecutionStatus.STALLED
    assert state.recovery_strategy


def test_event_bus_isolates_slow_subscriber_and_preserves_backpressure_signal():
    bus = RuntimeEventBus()
    fast = bus.subscribe(capacity=2)
    slow = bus.subscribe(capacity=1)

    bus.publish(RuntimeEvent(type="first", task_id="task-1"))
    bus.publish(RuntimeEvent(type="second", task_id="task-1"))
    bus.publish(RuntimeEvent(type="third", task_id="task-1"))

    assert fast.get(timeout=0).type == "second"
    assert fast.get(timeout=0).type == "third"
    assert slow.get(timeout=0).type == "third"
    assert bus.dropped_events >= 1


def test_event_bus_can_mirror_operational_events_to_the_canonical_journal(tmp_path):
    from workstation.contracts import ExecutionEventKind
    from workstation.journal import ExecutionJournal

    journal = ExecutionJournal("task-event", "session-event", file_path=tmp_path / "journal.jsonl")
    bus = RuntimeEventBus(journal)
    bus.publish(RuntimeEvent(type="task.started", task_id="task-event", session_id="session-event"))
    assert journal.read_events()[0].kind == ExecutionEventKind.TASK_STARTED


def test_event_bus_exposes_journal_failures_without_blocking_subscribers():
    class BrokenJournal:
        def record(self, *args, **kwargs):
            raise OSError("journal unavailable")

    bus = RuntimeEventBus(BrokenJournal())
    subscription = bus.subscribe()
    bus.publish(RuntimeEvent(type="progress", task_id="task-error", session_id="session-error"))
    assert subscription.get(timeout=0).type == "progress"
    assert bus.journal_errors == ["journal unavailable"]


def test_deadline_and_cancellation_are_observable():
    token = CancellationToken()
    token.cancel("user stopped")
    with pytest.raises(RuntimeError, match="user stopped"):
        run_with_deadline(lambda: "never", timeout_seconds=1, cancellation=token)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        run_with_deadline(lambda: time.sleep(0.2), timeout_seconds=0.001)
    assert time.monotonic() - started < 0.1


def test_resources_are_typed_and_lineage_is_stable():
    resources = ResourceRegistry()
    resource = TypedResource(
        resource_type="browser_task",
        resource_id="browser-task-1",
        task_id="task-1",
        session_id="session-1",
        permissions=("read", "control"),
        state={"status": "waiting-for-human"},
    )
    resources.upsert(resource)
    restored = resources.get("browser_task", "browser-task-1")
    assert restored is not None
    assert restored.task_id == "task-1"
    assert restored.permissions == ("read", "control")
    assert resources.snapshot()[0]["resource_type"] == "browser_task"


def test_typed_resources_and_handoffs_survive_client_reconnect(tmp_path):
    resource_path = tmp_path / "resources.json"
    resources = ResourceRegistry(resource_path)
    resources.upsert(TypedResource("worker", "worker-1", "task-1", "session-1", state={"status": "ready"}))
    restored_resource = ResourceRegistry(resource_path).get("worker", "worker-1")
    assert restored_resource is not None
    assert restored_resource.task_id == "task-1"

    handoff_path = tmp_path / "handoffs.json"
    handoffs = HumanHandoffManager(handoff_path)
    handoff = handoffs.request(task_id="task-1", session_id="session-1", reason="2FA", scope={"tab_id": "tab-1"})
    restored_handoff = HumanHandoffManager(handoff_path).get(handoff.handoff_id)
    assert restored_handoff is not None
    assert restored_handoff.scope == {"tab_id": "tab-1"}


def test_budget_tracker_is_bounded_per_task():
    budget = BudgetTracker(max_actions=2, max_tokens=10, max_cost_usd=0.05)
    budget.consume(actions=1, tokens=4, cost_usd=0.01)
    with pytest.raises(BudgetExceeded):
        budget.consume(actions=2, tokens=0, cost_usd=0)
    with pytest.raises(BudgetExceeded):
        budget.consume(actions=0, tokens=7, cost_usd=0)


def test_human_handoff_is_explicit_and_preserves_lineage():
    handoffs = HumanHandoffManager()
    handoff = handoffs.request(
        task_id="task-handoff",
        session_id="session-handoff",
        reason="complete 2FA",
        scope={"domain": "accounts.example", "tab_id": "tab-1"},
    )
    assert handoff.status == "waiting-for-human"
    assert handoff.task_id == "task-handoff"
    assert handoffs.resume(handoff.handoff_id, returned_by="human") is True
    assert handoff.status == "ready"


def test_model_router_makes_observable_budget_aware_choice():
    router = ModelRouter(
        [
            RoutingCandidate("fast", "local", quality=0.6, cost_usd=0.0, latency_ms=20),
            RoutingCandidate("strong", "cloud", quality=0.95, cost_usd=0.04, latency_ms=400),
        ]
    )
    decision = router.choose(complexity=0.9, risk=0.9, max_cost_usd=0.05)
    assert decision.model_id == "strong"
    assert decision.reason
    assert decision.candidates_considered == ["fast", "strong"]


def test_evidence_state_store_reconciles_and_restores_projection(tmp_path):
    path = tmp_path / "evidence.json"
    store = EvidenceStateStore(path)
    state = EvidenceState("task-store", "session-store")
    state.add_evidence("worker_id", "worker-1", ttl_seconds=1, now="2026-09-11T10:00:00+00:00")
    store.upsert(state)
    restored = EvidenceStateStore(path).get("task-store")
    assert restored is not None
    assert restored.status == ExecutionStatus.RUNNING
    changed = EvidenceStateStore(path).reconcile_all(now="2026-09-11T10:02:00+00:00")
    assert changed[0].status == ExecutionStatus.STALLED
