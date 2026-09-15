from __future__ import annotations

import threading
import time

from workstation.workers import PersistentWorkerStatus, WorkerExecutionOutput, WorkerRegistry
from workstation.runtime import RuntimeEventBus


def test_persistent_worker_accepts_messages_and_steering(tmp_path):
    seen: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def execute(prompt: str) -> str:
        seen.append(prompt)
        if prompt == "first":
            started.set()
            release.wait(timeout=2)
        return f"done:{prompt}"

    registry = WorkerRegistry(storage_path=tmp_path / "workers.json")

    worker = registry.start_persistent_worker(
        worker_id="codex",
        parent_task_id="parent-1",
        session_id="session-1",
        executor_fn=execute,
    )
    registry.send_message(worker.worker_id, "first", sender_id="parent")
    assert started.wait(timeout=2)
    registry.steer(worker.worker_id, "priority correction", sender_id="parent")
    release.set()
    first = registry.wait(worker.worker_id, timeout=2)
    assert first is not None
    assert first.parent_task_id == "parent-1"
    assert first.status == "completed"

    second = registry.wait(worker.worker_id, timeout=2, after_sequence=first.sequence)
    assert second is not None
    assert second.deliverables == ["done:priority correction"]
    assert seen[:2] == ["first", "priority correction"]

    registry.stop_worker(worker.worker_id)
    assert registry.get_persistent_worker(worker.worker_id).status == PersistentWorkerStatus.STOPPED


def test_persistent_worker_can_pause_resume_and_reconstruct(tmp_path):
    registry = WorkerRegistry(storage_path=tmp_path / "workers.json")
    worker = registry.start_persistent_worker(
        worker_id="opencode",
        parent_task_id="parent-2",
        session_id="session-2",
        executor_fn=lambda prompt: prompt.upper(),
    )
    registry.pause_worker(worker.worker_id)
    registry.send_message(worker.worker_id, "queued while paused")
    time.sleep(0.05)
    assert registry.wait(worker.worker_id, timeout=0.05) is None
    registry.resume_worker(worker.worker_id)
    result = registry.wait(worker.worker_id, timeout=2)
    assert result is not None
    assert result.deliverables == ["QUEUED WHILE PAUSED"]
    completed_sequence = result.sequence
    registry.stop_worker(worker.worker_id)

    restored = WorkerRegistry(storage_path=tmp_path / "workers.json")
    reconstructed = restored.reconstruct_worker(
        worker.worker_id,
        executor_fn=lambda prompt: f"replayed:{prompt}",
    )
    assert reconstructed.parent_task_id == "parent-2"
    restored.send_message(worker.worker_id, "after restart")
    result = restored.wait(worker.worker_id, timeout=2, after_sequence=completed_sequence)
    assert result is not None
    assert result.deliverables == ["replayed:after restart"]
    restored.stop_worker(worker.worker_id)


def test_pending_messages_and_execution_metadata_survive_worker_reconstruction(tmp_path):
    path = tmp_path / "workers.json"
    registry = WorkerRegistry(storage_path=path)
    worker = registry.start_persistent_worker(
        worker_id="worker-durable",
        parent_task_id="parent-durable",
        session_id="session-durable",
        executor_fn=lambda prompt: WorkerExecutionOutput(
            deliverables=[f"done:{prompt}"],
            model="cheap-model",
            provider="local",
            usage={"tokens": 12},
            cost_usd=0.001,
            evidence=[{"kind": "artifact", "reference": "artifact-1"}],
        ),
    )
    registry.pause_worker(worker.worker_id)
    registry.send_message(worker.worker_id, "durable queued", sender_id="parent")
    registry.stop_worker(worker.worker_id)

    restored = WorkerRegistry(storage_path=path)
    reconstructed = restored.reconstruct_worker(
        worker.worker_id,
        executor_fn=lambda prompt: WorkerExecutionOutput(deliverables=[f"replayed:{prompt}"]),
    )
    result = restored.wait(worker.worker_id, timeout=2)
    assert result is not None
    assert result.deliverables == ["replayed:durable queued"]
    restored.send_message(worker.worker_id, "metadata")
    metadata_result = restored.wait(worker.worker_id, timeout=2, after_sequence=result.sequence)
    assert metadata_result is not None
    assert metadata_result.model is None
    restored.stop_worker(reconstructed.worker_id)


def test_worker_delivery_publishes_parent_wakeup_event(tmp_path):
    bus = RuntimeEventBus()
    subscription = bus.subscribe(capacity=4)
    registry = WorkerRegistry(storage_path=tmp_path / "workers.json", event_bus=bus)
    worker = registry.start_persistent_worker(
        "worker-events",
        "parent-events",
        "session-events",
        executor_fn=lambda prompt: [f"done:{prompt}"],
    )
    registry.send_message(worker.worker_id, "wake parent", sender_id="parent")
    message_event = subscription.get(timeout=2)
    result_event = subscription.get(timeout=2)
    assert message_event.type == "worker.message"
    assert result_event.type == "worker.result"
    assert result_event.task_id == "parent-events"
    registry.stop_worker(worker.worker_id)


def test_completed_worker_result_survives_restart_until_durable_ack(tmp_path):
    """An executor completion is not lost between worker and consumer restart."""
    path = tmp_path / "workers.json"
    registry = WorkerRegistry(storage_path=path)
    worker = registry.start_persistent_worker(
        "worker-result-durable",
        "parent-result-durable",
        "session-result-durable",
        executor_fn=lambda prompt: WorkerExecutionOutput(deliverables=[f"done:{prompt}"], evidence=[{"ref": "a-1"}]),
    )
    registry.send_message(worker.worker_id, "persist me")
    completed = registry.wait(worker.worker_id, timeout=2)
    assert completed is not None
    assert completed.work_item_id
    result_id = completed.result_id
    registry.stop_worker(worker.worker_id)

    restored = WorkerRegistry(storage_path=path)
    reconstructed = restored.reconstruct_worker(worker.worker_id, executor_fn=lambda prompt: f"unexpected:{prompt}")
    delivered = restored.wait(worker.worker_id, timeout=2)
    assert delivered is not None
    assert delivered.result_id == result_id
    assert delivered.deliverables == ["done:persist me"]
    assert delivered.work_item_id == completed.work_item_id

    # ACK is the persistence boundary for the consumer. Once acknowledged, a
    # second restart cannot replay the same envelope as a fresh delivery.
    assert restored.acknowledge_result(worker.worker_id, result_id) is True
    assert restored.acknowledge_result(worker.worker_id, result_id) is False
    restored.stop_worker(reconstructed.worker_id)

    final = WorkerRegistry(storage_path=path)
    final_worker = final.reconstruct_worker(worker.worker_id, executor_fn=lambda prompt: f"unexpected:{prompt}")
    assert final.wait(worker.worker_id, timeout=0.05) is None
    final.stop_worker(final_worker.worker_id)


def test_interrupted_worker_is_recoverable_not_falsely_completed(tmp_path):
    """A claimed item remains explicit recovery work after a process boundary."""
    path = tmp_path / "workers.json"
    started = threading.Event()
    release = threading.Event()
    registry = WorkerRegistry(storage_path=path)
    worker = registry.start_persistent_worker(
        "worker-in-flight",
        "parent-in-flight",
        "session-in-flight",
        executor_fn=lambda prompt: (started.set(), release.wait(2), f"done:{prompt}")[-1],
    )
    queued = registry.send_message(worker.worker_id, "side effect boundary")
    assert started.wait(timeout=2)

    # The first registry represents the dead process. Its durable record was
    # written at CLAIM, before executing arbitrary user code.
    restored = WorkerRegistry(storage_path=path)
    reconstructed = restored.reconstruct_worker(worker.worker_id, executor_fn=lambda prompt: f"must-not-auto-replay:{prompt}")
    recovery = restored.recovery_work_item(worker.worker_id)
    assert recovery is not None
    assert recovery.work_item_id == queued.work_item_id
    assert restored.wait(worker.worker_id, timeout=0.05) is None
    restored.stop_worker(reconstructed.worker_id)

    release.set()
    registry.stop_worker(worker.worker_id)


def test_result_persistence_failure_never_publishes_completed_result(tmp_path, monkeypatch):
    """EXECUTED is not COMPLETED until the durable envelope commit succeeds."""
    registry = WorkerRegistry(storage_path=tmp_path / "workers.json")
    worker = registry.start_persistent_worker(
        "worker-persist-failure",
        "parent-persist-failure",
        "session-persist-failure",
        executor_fn=lambda prompt: f"done:{prompt}",
    )

    original_persist = registry._persist_persistent_records
    calls = 0

    def fail_result_persist():
        nonlocal calls
        calls += 1
        if calls >= 3:  # enqueue → CLAIM → result envelope
            raise OSError("simulated result persistence failure")
        original_persist()

    monkeypatch.setattr(registry, "_persist_persistent_records", fail_result_persist)
    registry.send_message(worker.worker_id, "must not complete")
    deadline = time.monotonic() + 2
    while worker.status != PersistentWorkerStatus.FAILED and time.monotonic() < deadline:
        time.sleep(0.01)

    assert worker.status == PersistentWorkerStatus.FAILED
    assert registry.wait(worker.worker_id, timeout=0.05) is None
    recovery = registry.recovery_work_item(worker.worker_id)
    assert recovery is not None
    assert recovery.content == "must not complete"
