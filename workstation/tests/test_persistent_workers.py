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
    registry.stop_worker(worker.worker_id)

    restored = WorkerRegistry(storage_path=tmp_path / "workers.json")
    reconstructed = restored.reconstruct_worker(
        worker.worker_id,
        executor_fn=lambda prompt: f"replayed:{prompt}",
    )
    assert reconstructed.parent_task_id == "parent-2"
    restored.send_message(worker.worker_id, "after restart")
    result = restored.wait(worker.worker_id, timeout=2)
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
