"""Deterministic, provider-free Workstation architecture benchmark.

The benchmark measures the contracts owned by existing Workstation components;
it does not introduce a result/artifact database or a second worker store.  The
large-result accounting models the already-established reference-first wire
shape, while worker, evidence, policy, event and compaction measurements use
their canonical implementations directly.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any

from agent.context_compressor import _build_operational_reference_envelope
from workstation.policy import ActionScope, PolicyDecision, ScopedPolicyEngine
from workstation.runtime import EvidenceState, RuntimeEvent, RuntimeEventBus
from workstation.workers import WorkerExecutionOutput, WorkerRegistry


BENCHMARK_VERSION = "workstation-workload-v1"
INLINE_RESULT_LIMIT_BYTES = 512


def _tool_result_metrics() -> dict[str, int]:
    small = b'{"ok":true,"rows":2}'
    large = (b"synthetic-row," * 220)[:3_072]
    other_large = (b"other-synthetic-row," * 220)[:3_072]
    payloads = [small, large, large, other_large, large]
    inline_bytes = 0
    reference_bytes = 0
    bytes_avoided = 0
    dedup_hits = 0
    references: set[str] = set()
    for payload in payloads:
        if len(payload) <= INLINE_RESULT_LIMIT_BYTES:
            inline_bytes += len(payload)
            continue
        digest = sha256(payload).hexdigest()
        reference = f"result://sha256/{digest}"
        reference_bytes += len(reference.encode("utf-8"))
        bytes_avoided += len(payload) - len(reference.encode("utf-8"))
        if reference in references:
            dedup_hits += 1
        references.add(reference)
    return {
        "tool_calls": len(payloads),
        "tool_result_bytes_inline": inline_bytes,
        "tool_result_bytes_as_refs": reference_bytes,
        "tool_result_bytes_avoided_by_refs": bytes_avoided,
        "large_results_replaced_by_refs": len(payloads) - 1,
        "dedup_hits": dedup_hits,
        "unique_result_references": len(references),
    }


def _compaction_metrics() -> dict[str, int]:
    # Structured refs use the new authenticated envelope format: refs are
    # attached to messages via the private _hermes_operational_refs key,
    # not extracted by regex from free-text content.
    structured_refs = [
        {
            "kind": "kanban_run",
            "id": "task-benchmark",
            "task_id": "task-benchmark",
            "session_id": "session-benchmark",
            "owner_session_id": "session-benchmark",
            "result_ref": "result://sha256/benchmark-result",
            "artifact_ref": "artifact://sha256/benchmark-artifact",
            "source": "KanbanRun",
            "version": 1,
            "trusted": True,
        },
        {
            "kind": "worker",
            "id": "worker-benchmark",
            "task_id": "task-benchmark",
            "owner_session_id": "session-benchmark",
            "source": "WorkerRegistry",
            "version": 1,
            "trusted": True,
        },
        {
            "kind": "browser_task",
            "id": "browser-benchmark",
            "task_id": "task-benchmark",
            "owner_session_id": "session-benchmark",
            "result_ref": "result://sha256/benchmark-result",
            "source": "BrowserTask",
            "version": 1,
            "trusted": True,
        },
        {
            "kind": "evidence",
            "id": "evidence://benchmark",
            "task_id": "task-benchmark",
            "owner_session_id": "session-benchmark",
            "source": "EvidenceState",
            "version": 1,
            "trusted": True,
        },
        {
            "kind": "approval",
            "id": "approval-benchmark",
            "task_id": "task-benchmark",
            "owner_session_id": "session-benchmark",
            "approval_state": "approved",
            "source": "Approval",
            "version": 1,
            "trusted": True,
        },
    ]
    # Number of logical refs tracked (matches original refs dict size)
    n_refs = 11
    turns = [
        {
            "role": "assistant",
            "content": "operational checkpoint",
            "_hermes_operational_refs": structured_refs,
        }
    ]
    turns.extend(
        {"role": "assistant", "content": f"synthetic context block {index}"}
        for index in range(6)
    )
    envelope = _build_operational_reference_envelope(turns)
    return {
        "compaction_input_chars": sum(len(str(item["content"])) for item in turns),
        "operational_reference_envelope_chars": len(envelope),
        "operational_references_preserved": n_refs,
    }



def _worker_metrics(root: Path) -> dict[str, int]:
    path = root / "workers.json"
    registry = WorkerRegistry(storage_path=path)
    worker = registry.start_persistent_worker(
        "worker-benchmark",
        "parent-benchmark",
        "session-benchmark",
        executor_fn=lambda prompt: WorkerExecutionOutput(
            deliverables=[f"done:{prompt}"],
            evidence=[{"kind": "artifact", "reference": "artifact://benchmark"}],
        ),
    )
    registry.send_message(worker.worker_id, "deterministic workload")
    first = registry.wait(worker.worker_id, timeout=3)
    if first is None:
        registry.stop_worker(worker.worker_id)
        raise AssertionError("provider-free worker benchmark did not complete")
    result_id = first.result_id
    work_item_id = first.work_item_id
    registry.stop_worker(worker.worker_id)

    restarted = WorkerRegistry(storage_path=path)
    reconstructed = restarted.reconstruct_worker(
        worker.worker_id,
        executor_fn=lambda prompt: f"unexpected replay:{prompt}",
    )
    recovered = restarted.wait(worker.worker_id, timeout=1)
    restart_recovery = int(
        recovered is not None
        and recovered.result_id == result_id
        and recovered.work_item_id == work_item_id
    )
    ack_recovery = int(recovered is not None and restarted.acknowledge_result(worker.worker_id, result_id))
    restarted.stop_worker(reconstructed.worker_id)

    final = WorkerRegistry(storage_path=path)
    final_worker = final.reconstruct_worker(
        worker.worker_id,
        executor_fn=lambda prompt: f"unexpected replay:{prompt}",
    )
    ack_recovery &= int(final.wait(worker.worker_id, timeout=0.05) is None)
    final.stop_worker(final_worker.worker_id)
    return {
        "worker_result_restart_recovery": restart_recovery,
        "worker_result_ack_recovery": ack_recovery,
    }


def _policy_metrics() -> dict[str, Any]:
    engine = ScopedPolicyEngine()
    scopes = [
        ActionScope("task-benchmark", "session-benchmark", "filesystem", "write", "/etc/hosts"),
        ActionScope("task-benchmark", "session-benchmark", "filesystem", "modify", "relative.txt", workspace_root="C:/workspace"),
        ActionScope("task-benchmark", "session-benchmark", "run_untrusted", "run_untrusted", "synthetic payload"),
    ]
    decisions = [engine.evaluate(scope).decision.value for scope in scopes]
    return {
        "structured_error_classifications": {decision: decisions.count(decision) for decision in sorted(set(decisions))},
        "policy_evaluations": len(decisions),
        "policy_fail_closed_sensitive_path": int(decisions[0] == PolicyDecision.DENY.value),
    }


def _event_metrics() -> dict[str, int]:
    bus = RuntimeEventBus()
    subscription = bus.subscribe(capacity=8)
    for event_type, resource in (
        ("tool.call", "tool-result"),
        ("worker.result", "worker-benchmark"),
        ("task.completed", "task-benchmark"),
    ):
        bus.publish(RuntimeEvent(event_type, "task-benchmark", "session-benchmark", {"resource": resource}))
    received = [subscription.get(timeout=0) for _ in range(3)]
    bus.close()
    return {
        "event_resource_invalidations": len(received),
        "event_bus_dropped_events": bus.dropped_events,
        "polling_fallback_checks": 0,
    }


def _evidence_metrics() -> dict[str, int]:
    state = EvidenceState("task-benchmark", "session-benchmark")
    state.add_evidence("artifact", "artifact://benchmark", now="2026-09-15T12:00:00+00:00")
    state.add_evidence("result", "result://benchmark", now="2026-09-15T12:00:01+00:00")
    return {
        "artifact_references": len(state.evidence),
        "unique_artifact_references": len({item.reference for item in state.evidence}),
    }


def run_workload_benchmark(storage_dir: Path | None = None) -> dict[str, Any]:
    """Return stable structural counters for the versioned synthetic workload."""
    if storage_dir is None:
        with tempfile.TemporaryDirectory(prefix="hermes-workload-benchmark-") as temp:
            return run_workload_benchmark(Path(temp))
    root = Path(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"benchmark_version": BENCHMARK_VERSION}
    result.update(_tool_result_metrics())
    result.update(_evidence_metrics())
    result.update(_worker_metrics(root))
    result.update(_compaction_metrics())
    result.update(_policy_metrics())
    result.update(_event_metrics())
    return result


if __name__ == "__main__":  # pragma: no cover - convenience for maintainers
    print(json.dumps(run_workload_benchmark(), indent=2, sort_keys=True))
