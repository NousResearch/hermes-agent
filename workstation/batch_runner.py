"""Durable Batch Runner with Exception-Driven LLM Escalation.

Executes repetitive work loops outside the conversational model loop.
Atomic checkpoints are recorded per item. Successful deterministic work stays
in the Data Plane; only anomalies (SUSPECT/FAILED items) escalate to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from workstation.artifacts import ArtifactRef, ArtifactStore
from workstation.durable_tasks import (
    AtomicPersistenceViolation,
    DurableTaskStore,
    WorkItem,
    WorkItemStatus,
    WorkPlan,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchItemResult:
    item_id: str
    item_index: int
    success: bool
    status: WorkItemStatus
    raw_output_ref: Optional[str] = None
    normalized_output_ref: Optional[str] = None
    validation: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass(slots=True)
class BatchSummary:
    task_id: str
    plan_id: str
    total_items: int
    success_count: int
    retry_success_count: int
    failed_count: int
    suspect_count: int
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    summary_artifact_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "plan_id": self.plan_id,
            "total": self.total_items,
            "success": self.success_count,
            "retry_success": self.retry_success_count,
            "failed": self.failed_count,
            "suspect": self.suspect_count,
            "anomalies": [{"item_id": a["item_id"], "status": a["status"],
                           "reason": str(a.get("reason", ""))[:240],
                           "raw_ref": a.get("raw_ref"), "norm_ref": a.get("norm_ref")}
                          for a in self.anomalies[:10]],
            "needs_reasoning": len(self.anomalies),
            "completed": self.success_count + self.retry_success_count,
            "status": "needs_reasoning" if self.anomalies else "completed",
            "duration_seconds": round(self.duration_seconds, 2),
            "summary_artifact_ref": self.summary_artifact_ref,
        }


class DurableBatchRunner:
    """Deterministic batch executor for long-horizon agent tasks."""

    def __init__(
        self,
        task_id: str,
        *,
        task_store: Optional[DurableTaskStore] = None,
        artifact_store: Optional[ArtifactStore] = None,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        self.task_id = task_id
        self.store = task_store or DurableTaskStore()
        self.artifacts = artifact_store or ArtifactStore()
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def execute_batch(
        self,
        title: str,
        items: List[Dict[str, Any]],
        *,
        worker_fn: Callable[[Dict[str, Any], WorkItem], Any],
        validator_fn: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None,
        checkpoint_every: int = 1,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stop_on_exception: bool = False,
    ) -> BatchSummary:
        """Run batch processing with per-item atomic persistence and validation."""
        start_time = time.monotonic()
        plan = self.store.get_plan(self.task_id)
        if not plan:
            plan = self.store.create_plan(
                task_id=self.task_id,
                title=title,
                items=items,
                session_id=session_id,
                max_retries=self.max_retries,
                metadata=metadata,
            )

        work_items = self.store.get_work_items(plan.id)
        success_count = 0
        retry_success_count = 0
        failed_count = 0
        suspect_count = 0
        anomalies: List[Dict[str, Any]] = []

        for item in work_items:
            # Skip items already completed during a previous run
            if item.status == WorkItemStatus.COMPLETED:
                success_count += 1
                continue

            if item.status in {WorkItemStatus.FAILED, WorkItemStatus.BLOCKED, WorkItemStatus.CANCELLED,
                               WorkItemStatus.WAITING_FOR_USER} or item.validation_result.get("suspect"):
                failed_count += item.status == WorkItemStatus.FAILED
                suspect_count += item.status != WorkItemStatus.FAILED
                anomalies.append({"item_id": item.id, "index": item.item_index,
                                  "reason": item.last_error or item.validation_result.get("reason", "Review required"),
                                  "status": item.status.value, "raw_ref": item.raw_output_ref})
                if stop_on_exception:
                    break
                continue

            item_start = time.monotonic()
            item_success = False

            for attempt in range(item.attempts + 1, self.max_retries + 2):
                try:
                    self.store.update_item_checkpoint(item.id, "navigate", "ok")
                    self.store.update_item_checkpoint(item.id, "ready", "ok")

                    # Worker execution (DOM extraction, API fetch, etc.)
                    current = self.store.get_item(item.id)
                    if current.raw_output_ref:
                        captured = self.artifacts.read(current.raw_output_ref)
                        descriptor = self.artifacts.read_json(current.raw_output_ref + ".meta.json")
                        output_type = descriptor.get("summary", {}).get("output_type")
                        if output_type == "str":
                            raw_data = captured
                        elif output_type == "bytes":
                            raw_data = self.artifacts.resolve_ref(current.raw_output_ref).read_bytes()
                        else:
                            try:
                                raw_data = json.loads(captured)
                            except ValueError:
                                raw_data = captured
                    else:
                        raw_data = worker_fn(item.input_payload, item)

                    # Persist raw output to Data Plane
                    raw_ref = self.artifacts.store(
                        task_id=self.task_id,
                        name=f"raw_item_{item.item_index:04d}.json",
                        content=raw_data,
                        schema="raw_batch_capture",
                        summary={"output_type": type(raw_data).__name__},
                    )
                    self.store.mark_item_captured(item.id, raw_ref.ref)

                    # Normalized persistence
                    norm_ref = self.artifacts.store(
                        task_id=self.task_id,
                        name=f"norm_item_{item.item_index:04d}.json",
                        content=raw_data,
                        schema="normalized_batch_output",
                    )
                    self.store.mark_item_persisted(item.id, norm_ref.ref)
                    self.store.record_evidence(item.id, norm_ref.ref)

                    # Validation
                    validation: Dict[str, Any] = {"valid": True, "suspect": False}
                    if validator_fn is not None:
                        validation = validator_fn(raw_data, item.input_payload)

                    is_valid = validation.get("valid") is True
                    is_suspect = validation.get("suspect", False)
                    self.store.mark_item_validated(item.id, validation)

                    if is_valid and not is_suspect:
                        # Complete item atomically
                        self.store.complete_item(item.id)
                        item_success = True
                        if attempt > 1:
                            retry_success_count += 1
                        else:
                            success_count += 1
                        break
                    else:
                        # Suspect or invalid item: do not mark completed, escalate as anomaly
                        suspect_count += 1
                        anomalies.append({
                            "item_id": item.id,
                            "index": item.item_index,
                            "input": item.input_payload,
                            "reason": validation.get("reason", "Validation flagged item as suspect/invalid"),
                            "issues": validation.get("issues", []),
                            "raw_ref": raw_ref.ref,
                            "norm_ref": norm_ref.ref,
                            "status": "suspect" if is_suspect else "invalid",
                        })
                        item_success = False
                        break

                except InterruptedError:
                    self.store.update_plan_state(plan.id, "interrupted")
                    raise
                except Exception as exc:
                    can_retry = attempt <= self.max_retries
                    self.store.fail_item(item.id, str(exc), can_retry=can_retry)
                    if can_retry:
                        time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    else:
                        failed_count += 1
                        anomalies.append({
                            "item_id": item.id,
                            "index": item.item_index,
                            "input": item.input_payload,
                            "reason": f"Execution error: {exc}",
                            "status": "failed",
                        })
                        break

            if anomalies and stop_on_exception:
                break

        total_duration = time.monotonic() - start_time
        summary = BatchSummary(
            task_id=self.task_id,
            plan_id=plan.id,
            total_items=len(work_items),
            success_count=success_count,
            retry_success_count=retry_success_count,
            failed_count=failed_count,
            suspect_count=suspect_count,
            anomalies=anomalies,
            duration_seconds=total_duration,
        )

        # Store summary artifact
        summary_ref = self.artifacts.store(
            task_id=self.task_id,
            name="batch_summary.json",
            content={**summary.to_dict(), "anomalies": anomalies,
                     "results": [{"item_id": i.id, "status": i.status.value,
                                  "raw_ref": i.raw_output_ref, "result_ref": i.normalized_output_ref,
                                  "evidence_refs": i.evidence_refs}
                                 for i in self.store.get_work_items(plan.id)]},
            schema="batch_run_summary",
        )
        summary.summary_artifact_ref = summary_ref.ref
        self.store.update_plan_state(plan.id, "needs_reasoning" if anomalies else "completed")
        return summary
