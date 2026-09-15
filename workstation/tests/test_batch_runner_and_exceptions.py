from __future__ import annotations

from uuid import uuid4
import pytest

from workstation.artifacts import ArtifactStore
from workstation.batch_runner import DurableBatchRunner
from workstation.durable_tasks import DurableTaskStore


def test_batch_runner_100_items_retries_and_failures(tmp_path):
    """Scenario 5: 100 items, 95 success, 3 retry success, 2 failed."""
    task_store = DurableTaskStore(board=f"test_batch_{tmp_path.name}")
    artifact_store = ArtifactStore(root_dir=tmp_path / "artifacts")
    task_id = f"task_batch_100_{uuid4().hex[:6]}"

    items = [{"item_id": i} for i in range(1, 101)]

    # Flaky tracking: items 96, 97, 98 fail once then succeed; items 99, 100 always fail
    attempt_counters: dict[int, int] = {}

    def worker_fn(input_payload, work_item):
        idx = input_payload["item_id"]
        attempt = attempt_counters.get(idx, 0) + 1
        attempt_counters[idx] = attempt

        if idx in {96, 97, 98} and attempt == 1:
            raise RuntimeError("Transient connection timeout")
        if idx in {99, 100}:
            raise RuntimeError("Permanent 404 Not Found")

        return {"views": 1000 + idx, "likes": 50 + idx, "text": "Valid post insights"}

    runner = DurableBatchRunner(
        task_id,
        task_store=task_store,
        artifact_store=artifact_store,
        max_retries=2,
        backoff_seconds=0.01,
    )

    summary = runner.execute_batch(
        "Batch 100 Items",
        items,
        worker_fn=worker_fn,
    )

    assert summary.total_items == 100
    assert summary.success_count == 95
    assert summary.retry_success_count == 3
    assert summary.failed_count == 2
    assert summary.suspect_count == 0
    assert len(summary.anomalies) == 2
    assert {a["index"] for a in summary.anomalies} == {99, 100}

    # Verify summary artifact exists
    assert summary.summary_artifact_ref is not None
    loaded_summary = artifact_store.read_json(summary.summary_artifact_ref)
    assert loaded_summary["total"] == 100


def test_exception_driven_llm_escalation(tmp_path):
    """Scenario 6: Valid items do NOT return raw output to LLM; only SUSPECT anomalies are escalated."""
    task_store = DurableTaskStore(board=f"test_exc_{tmp_path.name}")
    artifact_store = ArtifactStore(root_dir=tmp_path / "artifacts")
    task_id = f"task_exception_driven_{uuid4().hex[:6]}"

    items = [
        {"id": 1, "text": "Normal post content with complete insights", "views": 2500, "likes": 120},
        {"id": 2, "text": "Another valid post content", "views": 1800, "likes": 90},
        {"id": 3, "text": "Too short", "views": 50, "likes": 100},  # SUSPECT: views < likes
        {"id": 4, "text": "Valid post 4", "views": 3200, "likes": 210},
    ]

    def worker_fn(input_payload, work_item):
        return input_payload

    def validator_fn(output, input_payload):
        if output["views"] < output["likes"]:
            return {
                "valid": False,
                "suspect": True,
                "reason": f"views ({output['views']}) < likes ({output['likes']})",
            }
        return {"valid": True, "suspect": False}

    runner = DurableBatchRunner(
        task_id,
        task_store=task_store,
        artifact_store=artifact_store,
    )

    summary = runner.execute_batch("Test Exception Driven", items, worker_fn=worker_fn, validator_fn=validator_fn)

    assert summary.total_items == 4
    assert summary.success_count == 3
    assert summary.suspect_count == 1
    assert len(summary.anomalies) == 1

    anomaly = summary.anomalies[0]
    assert anomaly["index"] == 3
    assert anomaly["status"] == "suspect"
    assert "views (50) < likes (100)" in anomaly["reason"]

    # Passing items (1, 2, 4) have raw outputs in ArtifactStore (Data Plane),
    # but are NOT present in the anomalies list presented to the LLM (Reasoning Plane)!
    assert 1 not in [a["index"] for a in summary.anomalies]
    assert 2 not in [a["index"] for a in summary.anomalies]
    assert 4 not in [a["index"] for a in summary.anomalies]
