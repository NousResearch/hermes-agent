from __future__ import annotations

import logging
import json
from typing import Any, Iterable, Optional
import sqlite3

from hermes_cli import kanban_db
from workstation.config import load_workstation_config
from workstation.contracts import BrowserTaskReport, DiscoveredTask, ExecutionEventKind, RiskLevel
from workstation.journal import ExecutionJournal
from workstation.contracts import (AcceptanceContract, AcceptanceEvaluator, MessageEnvelope, OutcomeStatus, TaskOutcome)
from dataclasses import asdict

_log = logging.getLogger(__name__)


def is_multistep_request(prompt: str) -> bool:
    """Classify if a user request is a multistep/asynchronous workflow."""
    keywords = [
        "workflow",
        "multistep",
        "step by step",
        "passo a passo",
        "first",
        "then",
        "finally",
        "primeiro",
        "depois",
        "automate",
        "automatize",
        "extract and",
        "extraia e",
        "sign in and",
        "faça login",
        "fill out",
        "preencha",
        "download and",
        "baixe e",
        "search and",
        "pesquise e",
        "pipeline",
        "batch",
        "follow-up",
        "scrape and",
        "raspe e",
        "pesquise",
        "pesquisa",
        "navegue",
        "procure",
        "browser",
        "navegador",
        "online",
        "internet",
        "site",
        "web",
        "busque",
        "investigate",
        "research",
    ]
    lower = prompt.lower()
    return any(kw in lower for kw in keywords) or ("\n" in prompt.strip() and len(prompt.strip()) > 30)


class WorkstationKanbanBridge:
    """Bridge between Workstation tasks and the canonical Kanban SQLite database."""

    def __init__(self, *, board: Optional[str] = None) -> None:
        self.board = board

    def get_connection(self) -> sqlite3.Connection:
        return kanban_db.connect(board=self.board)

    def promote_request_if_multistep(
        self,
        prompt: str,
        *,
        session_id: str,
        title: Optional[str] = None,
        force: bool = False,
        envelope: MessageEnvelope | None = None,
        acceptance_contract: AcceptanceContract | None = None,
    ) -> Optional[str]:
        """Automatically create a parent Kanban task for a multistep request."""
        if (envelope is None or not envelope.can_create_work
                or envelope.session_id != session_id or envelope.content != prompt):
            return None
        cfg = load_workstation_config()
        from workstation.work_intent import work_intent
        intent = work_intent(envelope)
        should_create = force or (
            cfg.raw.get("tasks", {}).get("create_kanban_for_multistep", True)
            and intent.requires_task
        )
        if not should_create:
            return None

        clean_title = (title or prompt.strip().split("\n")[0])[:80]
        with self.get_connection() as conn:
            task_id = kanban_db.create_task(
                conn,
                title=clean_title,
                body=prompt,
                created_by="workstation",
                session_id=session_id,
                board=self.board,
                initial_status="running",
            )
            contract = acceptance_contract or AcceptanceContract()
            conn.execute("INSERT INTO task_acceptance_contracts(task_id, contract_json) VALUES (?,?)",
                         (task_id, json.dumps(asdict(contract))))
            conn.commit()

        # Start journal for this task
        journal = ExecutionJournal(task_id, session_id)
        journal.record(
            ExecutionEventKind.TASK_CREATED,
            f"Workstation task promoted into Kanban: {clean_title}",
            metadata={"source": "automatic_multistep_promotion", "prompt": prompt},
        )
        return task_id

    def record_discovered_followup(
        self,
        parent_task_id: str,
        followup: DiscoveredTask,
    ) -> str:
        """Record a discovered child task into Kanban with parent dependency."""
        followup.validate()

        evidence = [
            {"kind": item.kind, "uri": item.uri, "summary": item.summary, "sha256": item.sha256}
            for item in followup.evidence
        ]
        body = "\n".join([
            f"Reason: {followup.reason}",
            f"Discovered by: {followup.discovered_by}",
            f"Origin session: {followup.origin_session_id}",
            "Discovery evidence:",
            json.dumps(evidence, ensure_ascii=False),
        ])

        with self.get_connection() as conn:
            child_task_id = kanban_db.create_task(
                conn,
                title=followup.title,
                body=body,
                created_by=followup.discovered_by,
                # A required child must be runnable immediately while the
                # parent waits.  Kanban links encode prerequisites, so the
                # dependency is child -> parent (not parent -> child). The
                # durable parent_task_id remains in the child body/journal
                # projection as the product hierarchy identity.
                parents=[] if followup.required_for_parent else [parent_task_id],
                session_id=followup.origin_session_id,
                board=self.board,
            )
            if followup.required_for_parent:
                try:
                    kanban_db.link_tasks(conn, child_task_id, parent_task_id)
                    kanban_db.block_task(
                        conn,
                        parent_task_id,
                        reason=f"Blocked by required follow-up child task {child_task_id}: {followup.title}",
                        kind="dependency",
                    )
                except Exception as e:
                    _log.warning("Could not transition parent task %s to blocked: %s", parent_task_id, e)

        # Update followup instance with assigned task_id
        followup.task_id = child_task_id

        # Record into journal
        journal = ExecutionJournal(parent_task_id, followup.origin_session_id)
        journal.record(
            ExecutionEventKind.FOLLOWUP_CREATED,
            f"Discovered child task {child_task_id}: {followup.title}",
            evidence=followup.evidence,
            metadata={
                "child_task_id": child_task_id,
                "reason": followup.reason,
                "discovered_by": followup.discovered_by,
                "origin_session_id": followup.origin_session_id,
                "parent_task_id": parent_task_id,
                "required_for_parent": followup.required_for_parent,
                "evidence": evidence,
            },
        )
        return child_task_id

    def complete_task_with_report(
        self,
        task_id: str,
        report: BrowserTaskReport,
        *,
        expected_run_id: Optional[int] = None,
    ) -> bool:
        """Complete Kanban task with structured Workstation report metadata."""
        metadata = report.to_kanban_metadata()
        if report.task_id != task_id:
            raise ValueError("report task identity mismatch")
        status = report.outcome_status or (
            OutcomeStatus.VERIFIED_COMPLETED if report.completed else OutcomeStatus.BLOCKED
        )
        if not report.completed and status == OutcomeStatus.VERIFIED_COMPLETED:
            status = OutcomeStatus.BLOCKED
        target_run_id = expected_run_id
        if target_run_id is None and getattr(report, "run_id", None):
            try:
                target_run_id = int(report.run_id)
            except (ValueError, TypeError):
                target_run_id = None
        outcome = TaskOutcome(
            task_id=task_id, session_id=report.session_id, objective=report.objective,
            status=status, summary=report.result, evidence_refs=report.evidence,
            verifier_results=report.verifier_results, deliverables=report.deliverables,
            pending_items=report.pending_items, uncertain_mutation=report.uncertain_mutation,
            run_id=str(target_run_id) if target_run_id is not None else None,
            operation_id=report.operation_id,
        )
        with self.get_connection() as conn:
            cur_run_id = kanban_db._current_run_id(conn, task_id)
            if target_run_id is not None and cur_run_id is not None and cur_run_id != target_run_id:
                journal = ExecutionJournal(task_id, report.session_id)
                journal.record(
                    ExecutionEventKind.PROGRESS,
                    f"Stale run {target_run_id} rejected (current run is {cur_run_id})",
                    evidence=report.evidence,
                    metadata={"boundary": "acceptance_commit_failed", "task_id": task_id, "expected_run_id": target_run_id, "current_run_id": cur_run_id},
                )
                return False
            saved = conn.execute("SELECT contract_json FROM task_acceptance_contracts WHERE task_id=?", (task_id,)).fetchone()
        contract = AcceptanceContract(**json.loads(saved[0])) if saved else AcceptanceContract()
        reasons = AcceptanceEvaluator().evaluate(outcome, contract)
        if reasons:
            if outcome.uncertain_mutation:
                outcome.status = OutcomeStatus.UNCERTAIN
            elif outcome.status == OutcomeStatus.VERIFIED_COMPLETED:
                outcome.status = OutcomeStatus.FAILED if "verifier_failed" in reasons else OutcomeStatus.BLOCKED
            with self.get_connection() as conn:
                kanban_db.block_task(
                    conn, task_id,
                    reason="; ".join(reasons),
                    kind="needs_input",
                    expected_run_id=target_run_id,
                )
            ExecutionJournal(task_id, report.session_id).record(
                ExecutionEventKind.PROGRESS, report.result, evidence=report.evidence,
                metadata={"completed": False, "outcome": outcome.to_dict(), "acceptance_reasons": reasons},
            )
            return False
        metadata["workstation"]["outcome"] = outcome.to_dict()
        metadata["workstation"]["acceptance_approved"] = True
        metadata["workstation"]["acceptance_contract"] = asdict(contract)
        journal = ExecutionJournal(task_id, report.session_id)
        journal.read_events()  # Corrupt evidence fails closed before the canonical commit.
        if report.verifier_results:
            from agent.verification_evidence import record_outcome_verifiers
            metadata["workstation"]["verification_event_ids"] = record_outcome_verifiers(
                task_id, report.session_id, report.verifier_results, environment=journal.environment)
        journal.record(ExecutionEventKind.PROGRESS, "verified outcome submitted to canonical acceptance gate",
                       evidence=report.evidence, metadata={"boundary": "acceptance", "outcome": outcome.to_dict()})
        with self.get_connection() as conn:
            success = kanban_db.complete_task(
                conn,
                task_id=task_id,
                result=report.result,
                summary=report.result,
                metadata=metadata,
                expected_run_id=target_run_id,
            )

        if not success:
            journal.record(
                ExecutionEventKind.PROGRESS,
                "Canonical completion CAS rejected (stale run or state conflict)",
                evidence=report.evidence,
                metadata={"boundary": "acceptance_commit_failed", "task_id": task_id, "expected_run_id": target_run_id},
            )
            return False
        journal = ExecutionJournal(task_id, report.session_id)
        journal.record(
            ExecutionEventKind.TASK_COMPLETED,
            f"Workstation task completed: {report.result}",
            evidence=report.evidence,
            metadata={"completed": report.completed, "sites": report.sites,
                      "outcome": outcome.to_dict(), "acceptance_approved": True,
                      "verification_event_ids": metadata["workstation"].get("verification_event_ids", [])},
        )
        if report.repeatability_hint and report.procedure_steps:
            from workstation.memory import ProceduralMemory
            from workstation.routines import RoutinePromotionService
            candidate = RoutinePromotionService(ProceduralMemory()).experience_candidate(
                outcome, contract, site=report.procedure_scope,
                steps=report.procedure_steps, repeatable=True,
                metrics={"actions": len(report.actions), "tokens": report.input_tokens + report.output_tokens,
                         "duration_seconds": report.duration_seconds})
            if candidate:
                journal.record(ExecutionEventKind.ACTION, "experience candidate created; validation required",
                               metadata={"procedure_id": candidate.id})
        return success

    def finalize_turn_candidate(self, task_id: str, session_id: str, turn_result: dict) -> dict:
        """Verify persisted execution before submitting the turn's outcome candidate."""
        from workstation.artifacts import ArtifactStore
        from workstation.contracts import EvidenceRef
        artifacts = ArtifactStore()
        evidence, verifiers, pending, handoffs = [], [], [], []
        conn = self.get_connection()
        try:
            task = kanban_db.get_task(conn, task_id)
            if task is None or task.session_id != session_id:
                raise ValueError("Outcome candidate must belong to the canonical task/session")
            if task.status == "done":
                run = kanban_db.latest_run(conn, task_id)
                metadata = run.metadata if run and isinstance(run.metadata, dict) else {}
                workstation = metadata.get("workstation") or {}
                accepted = workstation.get("acceptance_approved") is True and workstation.get("outcome", {}).get("status") == "verified_completed"
                return {"task_id": task_id, "status": "verified_completed" if accepted else "uncertain", "acceptance_approved": accepted}
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            plans = []
            if "work_plans" in tables:
                for row in conn.execute("SELECT id, status, metadata FROM work_plans WHERE session_id=?", (session_id,)):
                    metadata = json.loads(row[2] or "{}")
                    if metadata.get("canonical_task_id") == task_id:
                        plans.append((row[0], row[1], metadata))
            if not plans:
                pending.append("No persisted execution verifies the acceptance contract")
            for plan_id, plan_status, metadata in plans:
                if metadata.get("handoff", {}).get("status") == "waiting-for-human":
                    handoffs.append(metadata["handoff"]["handoff_id"])
                if plan_status != "completed":
                    pending.append(f"Plan {plan_id} remains {plan_status}")
                rows = conn.execute("SELECT id, status, normalized_output_ref, validation_result, checkpoints FROM work_items WHERE plan_id=?", (plan_id,)).fetchall()
                if not rows:
                    pending.append(f"Plan {plan_id} has no verified items")
                for row in rows:
                    validation = json.loads(row[3] or "{}")
                    checkpoints = json.loads(row[4] or "{}")
                    if row[1] != "completed" or validation.get("valid") is not True or checkpoints.get("validate") != "ok":
                        pending.append(f"Item {row[0]} is not verified")
                        continue
                    try:
                        resolved = artifacts.resolve_structured(row[2])
                        output = resolved.get("content")
                        if not isinstance(output, dict) or output.get("valid") is not True:
                            raise ValueError("Persisted output is not a verified compiler result")
                        for result in output.get("results", []):
                            if result.get("verified") is not True:
                                raise ValueError("Step result is not verified")
                            artifacts.resolve_structured(result["artifact_ref"], max_content_bytes=0)
                        evidence.append(EvidenceRef("durable_proof", row[2], sha256=resolved["sha256"]))
                        verifiers.append({"verifier": "durable_item:" + row[0], "passed": True, "evidence_ref": row[2]})
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        pending.append(f"Item {row[0]} evidence cannot be verified: {exc}")
        finally:
            conn.close()
        stopped = not turn_result.get("completed") or turn_result.get("failed") or turn_result.get("interrupted")
        if stopped:
            pending.append("Execution stopped before a verified final outcome")
        verified = not pending and not handoffs and bool(verifiers)
        status = (OutcomeStatus.VERIFIED_COMPLETED if verified else OutcomeStatus.WAITING_FOR_HUMAN if handoffs
                  else OutcomeStatus.FAILED if turn_result.get("failed") else OutcomeStatus.UNCERTAIN if stopped
                  else OutcomeStatus.BLOCKED)
        report = BrowserTaskReport(task_id, session_id, task.body or task.title,
            str(turn_result.get("final_response") or "Execution has no verified final result"), verified,
            pending_items=pending, evidence=evidence, verifier_results=verifiers,
            deliverables=[e.uri for e in evidence], outcome_status=status)
        accepted = self.complete_task_with_report(task_id, report)
        return {"task_id": task_id, "status": status.value, "acceptance_approved": accepted,
                "pending_items": pending, "planned_handoffs": handoffs}
