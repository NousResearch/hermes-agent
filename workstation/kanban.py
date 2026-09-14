from __future__ import annotations

import logging
import json
from typing import Any, Iterable, Optional
import sqlite3

from hermes_cli import kanban_db
from workstation.config import load_workstation_config
from workstation.contracts import BrowserTaskReport, DiscoveredTask, ExecutionEventKind, RiskLevel
from workstation.journal import ExecutionJournal

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
    ) -> Optional[str]:
        """Automatically create a parent Kanban task for a multistep request."""
        cfg = load_workstation_config()
        should_create = force or (
            cfg.raw.get("tasks", {}).get("create_kanban_for_multistep", True)
            and is_multistep_request(prompt)
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
    ) -> bool:
        """Complete Kanban task with structured Workstation report metadata."""
        metadata = report.to_kanban_metadata()
        with self.get_connection() as conn:
            success = kanban_db.complete_task(
                conn,
                task_id=task_id,
                result=report.result,
                summary=report.objective,
                metadata=metadata,
            )

        journal = ExecutionJournal(task_id, report.session_id)
        journal.record(
            ExecutionEventKind.TASK_COMPLETED,
            f"Workstation task completed: {report.result}",
            evidence=report.evidence,
            metadata={"completed": report.completed, "sites": report.sites},
        )
        return success
