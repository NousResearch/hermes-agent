"""Durable Task Runner, WorkPlan, and WorkItem infrastructure.

Provides persistent, atomic, resumable execution for multi-step and batch
operations outside the conversational LLM loop. State is persisted in SQLite
(via the canonical Kanban DB) and survives process restarts, model switches,
and context compaction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from hermes_cli import kanban_db

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AtomicPersistenceViolation(RuntimeError):
    """Raised when an operation attempts to complete without verified persistence and validation."""


class WorkItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CAPTURED = "captured"
    PERSISTED = "persisted"
    VALIDATED = "validated"
    COMPLETED = "completed"
    RETRYING = "retrying"
    WAITING_FOR_USER = "waiting_for_user"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class WorkItem:
    id: str
    plan_id: str
    task_id: str
    item_index: int
    status: WorkItemStatus = WorkItemStatus.PENDING
    attempts: int = 0
    input_payload: Dict[str, Any] = field(default_factory=dict)
    raw_output_ref: Optional[str] = None
    normalized_output_ref: Optional[str] = None
    validation_result: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: Dict[str, str] = field(default_factory=dict)
    retry_state: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None
    created_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    operation_id: Optional[str] = None
    run_id: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {WorkItemStatus.COMPLETED, WorkItemStatus.FAILED, WorkItemStatus.CANCELLED}

    def verify_can_complete(self) -> None:
        """Enforce strict invariant: cannot complete if persist and validate are missing."""
        persist_ok = self.checkpoints.get("persist") == "ok" or bool(self.normalized_output_ref)
        validate_ok = self.checkpoints.get("validate") == "ok" or bool(
            self.validation_result and self.validation_result.get("valid") is True
        )
        if not persist_ok:
            raise AtomicPersistenceViolation(
                f"WorkItem {self.id} cannot be marked completed: output has not been persisted to storage."
            )
        if not validate_ok:
            raise AtomicPersistenceViolation(
                f"WorkItem {self.id} cannot be marked completed: output has not been validated."
            )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkItem":
        raw_status = data.get("status", WorkItemStatus.PENDING.value)
        try:
            status = WorkItemStatus(raw_status)
        except ValueError:
            status = WorkItemStatus.PENDING

        def _json_or_dict(val: Any) -> Any:
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return {}
            return val or {}

        def _json_or_list(val: Any) -> Any:
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return []
            return val or []

        return cls(
            id=str(data["id"]),
            plan_id=str(data["plan_id"]),
            task_id=str(data["task_id"]),
            item_index=int(data.get("item_index", 0)),
            status=status,
            attempts=int(data.get("attempts", 0)),
            input_payload=_json_or_dict(data.get("input_payload")),
            raw_output_ref=data.get("raw_output_ref"),
            normalized_output_ref=data.get("normalized_output_ref"),
            validation_result=_json_or_dict(data.get("validation_result")),
            evidence_refs=_json_or_list(data.get("evidence_refs")),
            checkpoints=_json_or_dict(data.get("checkpoints")),
            retry_state=_json_or_dict(data.get("retry_state")),
            last_error=data.get("last_error"),
            created_at=str(data.get("created_at", _utc_now())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            operation_id=data.get("operation_id"),
            run_id=data.get("run_id"),
        )


@dataclass(slots=True)
class WorkPlan:
    id: str
    task_id: str
    title: str
    session_id: Optional[str] = None
    status: str = "pending"
    total_items: int = 0
    checkpoint_frequency: int = 1
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    completed_at: Optional[str] = None
    run_id: Optional[str] = None
    execution_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPlan":
        meta = data.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return cls(
            id=str(data["id"]),
            task_id=str(data["task_id"]),
            title=str(data.get("title", "Untitled WorkPlan")),
            session_id=data.get("session_id"),
            status=str(data.get("status", "pending")),
            total_items=int(data.get("total_items", 0)),
            checkpoint_frequency=int(data.get("checkpoint_frequency", 1)),
            max_retries=int(data.get("max_retries", 3)),
            metadata=meta or {},
            created_at=str(data.get("created_at", _utc_now())),
            updated_at=str(data.get("updated_at", _utc_now())),
            completed_at=data.get("completed_at"),
            run_id=data.get("run_id"),
            execution_key=data.get("execution_key"),
        )


class DurableTaskStore:
    """ACID persistence store for WorkPlans and WorkItems on SQLite."""

    def __init__(self, *, board: Optional[str] = None, conn: Optional[sqlite3.Connection] = None) -> None:
        self.board = board
        self._external_conn = conn
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._tables_ensured = False

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS work_plans (
                    id                   TEXT PRIMARY KEY,
                    task_id              TEXT NOT NULL,
                    session_id           TEXT,
                    title                TEXT NOT NULL,
                    status               TEXT NOT NULL,
                    total_items          INTEGER NOT NULL DEFAULT 0,
                    checkpoint_frequency INTEGER NOT NULL DEFAULT 1,
                    max_retries          INTEGER NOT NULL DEFAULT 3,
                    metadata             TEXT,
                    created_at           TEXT NOT NULL,
                    updated_at           TEXT NOT NULL,
                    completed_at         TEXT
                );

                CREATE TABLE IF NOT EXISTS work_items (
                    id                     TEXT PRIMARY KEY,
                    plan_id                TEXT NOT NULL,
                    task_id                TEXT NOT NULL,
                    item_index             INTEGER NOT NULL,
                    status                 TEXT NOT NULL,
                    attempts               INTEGER NOT NULL DEFAULT 0,
                    input_payload          TEXT,
                    raw_output_ref         TEXT,
                    normalized_output_ref  TEXT,
                    validation_result      TEXT,
                    evidence_refs          TEXT,
                    error                  TEXT,
                    last_error             TEXT,
                    checkpoints            TEXT,
                    retry_state            TEXT,
                    created_at             TEXT NOT NULL,
                    started_at             TEXT,
                    completed_at           TEXT,
                    FOREIGN KEY(plan_id) REFERENCES work_plans(id)
                );

                CREATE INDEX IF NOT EXISTS idx_work_plans_task ON work_plans(task_id);
                CREATE INDEX IF NOT EXISTS idx_work_plans_status ON work_plans(status);
                CREATE INDEX IF NOT EXISTS idx_work_items_plan ON work_items(plan_id, item_index);
                CREATE INDEX IF NOT EXISTS idx_work_items_status ON work_items(status);
                """
            )
            try:
                cols_wp = {row[1] for row in conn.execute("PRAGMA table_info(work_plans)").fetchall()}
                if "run_id" not in cols_wp:
                    conn.execute("ALTER TABLE work_plans ADD COLUMN run_id TEXT")
                if "execution_key" not in cols_wp:
                    conn.execute("ALTER TABLE work_plans ADD COLUMN execution_key TEXT")

                cols = {row[1] for row in conn.execute("PRAGMA table_info(work_items)").fetchall()}
                if "error" not in cols:
                    conn.execute("ALTER TABLE work_items ADD COLUMN error TEXT")
                if "last_error" not in cols:
                    conn.execute("ALTER TABLE work_items ADD COLUMN last_error TEXT")
                if "evidence_refs" not in cols:
                    conn.execute("ALTER TABLE work_items ADD COLUMN evidence_refs TEXT")
                if "operation_id" not in cols:
                    conn.execute("ALTER TABLE work_items ADD COLUMN operation_id TEXT")
                if "run_id" not in cols:
                    conn.execute("ALTER TABLE work_items ADD COLUMN run_id TEXT")
                conn.commit()
            except Exception:
                pass

    def get_connection(self) -> sqlite3.Connection:
        if self._external_conn is not None:
            conn = self._external_conn
        else:
            if self._conn is None:
                self._conn = kanban_db.connect(board=self.board)
            conn = self._conn
        if not self._tables_ensured:
            self._ensure_tables(conn)
            self._tables_ensured = True
        return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __enter__(self) -> "DurableTaskStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def create_plan(
        self,
        task_id: str,
        title: str,
        items: List[Dict[str, Any]],
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        execution_key: Optional[str] = None,
        checkpoint_frequency: int = 1,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkPlan:
        plan_id = f"plan_{uuid4().hex[:12]}"
        now = _utc_now()
        meta = metadata or {}
        actual_run_id = run_id or meta.get("run_id") or meta.get("canonical_run_id")
        actual_exec_key = execution_key or meta.get("execution_key") or meta.get("operation_key")
        plan = WorkPlan(
            id=plan_id,
            task_id=task_id,
            session_id=session_id,
            title=title,
            status="running",
            total_items=len(items),
            checkpoint_frequency=checkpoint_frequency,
            max_retries=max_retries,
            metadata=meta,
            created_at=now,
            updated_at=now,
            run_id=actual_run_id,
            execution_key=actual_exec_key,
        )

        with self._lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO work_plans (
                    id, task_id, session_id, title, status, total_items,
                    checkpoint_frequency, max_retries, metadata, created_at, updated_at,
                    run_id, execution_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.id,
                    plan.task_id,
                    plan.session_id,
                    plan.title,
                    plan.status,
                    plan.total_items,
                    plan.checkpoint_frequency,
                    plan.max_retries,
                    json.dumps(plan.metadata, ensure_ascii=False),
                    plan.created_at,
                    plan.updated_at,
                    plan.run_id,
                    plan.execution_key,
                ),
            )

            # Insert work items
            for idx, item_input in enumerate(items, start=1):
                item_id = f"{plan_id}_{idx:04d}"
                op_id = item_input.get("operation_id") or f"op_{item_id}"
                conn.execute(
                    """
                    INSERT INTO work_items (
                        id, plan_id, task_id, item_index, status, attempts,
                        input_payload, checkpoints, retry_state, created_at,
                        operation_id, run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        plan.id,
                        plan.task_id,
                        idx,
                        WorkItemStatus.PENDING.value,
                        0,
                        json.dumps(item_input, ensure_ascii=False),
                        json.dumps({}, ensure_ascii=False),
                        json.dumps({}, ensure_ascii=False),
                        now,
                        op_id,
                        plan.run_id,
                    ),
                )
            conn.commit()

        return plan

    def get_plan(self, plan_id_or_task_id: str) -> Optional[WorkPlan]:
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM work_plans WHERE id = ? OR task_id = ? ORDER BY created_at DESC LIMIT 1",
                (plan_id_or_task_id, plan_id_or_task_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            col_names = [d[0] for d in cur.description]
            return WorkPlan.from_dict(dict(zip(col_names, row)))

    def get_work_items(
        self,
        plan_id_or_task_id: str,
        *,
        status: Optional[WorkItemStatus] = None,
    ) -> List[WorkItem]:
        with self.get_connection() as conn:
            if status is not None:
                cur = conn.execute(
                    """
                    SELECT * FROM work_items
                    WHERE (plan_id = ? OR task_id = ?) AND status = ?
                    ORDER BY item_index ASC
                    """,
                    (plan_id_or_task_id, plan_id_or_task_id, status.value),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT * FROM work_items
                    WHERE plan_id = ? OR task_id = ?
                    ORDER BY item_index ASC
                    """,
                    (plan_id_or_task_id, plan_id_or_task_id),
                )
            rows = cur.fetchall()
            if not rows:
                return []
            col_names = [d[0] for d in cur.description]
            return [WorkItem.from_dict(dict(zip(col_names, row))) for row in rows]

    def get_item(self, item_id: str) -> Optional[WorkItem]:
        with self.get_connection() as conn:
            cur = conn.execute("SELECT * FROM work_items WHERE id = ?", (item_id,))
            row = cur.fetchone()
            if not row:
                return None
            col_names = [d[0] for d in cur.description]
            return WorkItem.from_dict(dict(zip(col_names, row)))

    def update_item_checkpoint(
        self,
        item_id: str,
        step: str,
        status_val: str = "ok",
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkItem:
        now = _utc_now()
        with self._lock, self.get_connection() as conn:
            cur = conn.execute("SELECT checkpoints, attempts FROM work_items WHERE id = ?", (item_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"WorkItem {item_id} not found")
            try:
                cp = json.loads(row[0] or "{}")
            except Exception:
                cp = {}
            cp[step] = status_val
            if metadata:
                cp[f"{step}_meta"] = metadata

            conn.execute(
                """
                UPDATE work_items
                SET checkpoints = ?,
                    status = CASE
                        WHEN status IN ('pending', 'retrying') THEN 'running'
                        ELSE status
                    END,
                    started_at = COALESCE(started_at, ?)
                WHERE id = ?
                """,
                (json.dumps(cp, ensure_ascii=False), now, item_id),
            )
            conn.commit()

        item = self.get_item(item_id)
        assert item is not None
        return item

    def mark_item_captured(self, item_id: str, raw_output_ref: str) -> WorkItem:
        now = _utc_now()
        with self._lock, self.get_connection() as conn:
            cur = conn.execute("SELECT checkpoints FROM work_items WHERE id = ?", (item_id,))
            row = cur.fetchone()
            try:
                cp = json.loads(row[0] or "{}") if row else {}
            except Exception:
                cp = {}
            cp["capture"] = "ok"

            conn.execute(
                """
                UPDATE work_items
                SET status = ?,
                    raw_output_ref = ?,
                    checkpoints = ?
                WHERE id = ?
                """,
                (WorkItemStatus.CAPTURED.value, raw_output_ref, json.dumps(cp, ensure_ascii=False), item_id),
            )
            conn.commit()

        item = self.get_item(item_id)
        assert item is not None
        return item

    def mark_item_persisted(self, item_id: str, normalized_output_ref: str) -> WorkItem:
        with self._lock, self.get_connection() as conn:
            cur = conn.execute("SELECT checkpoints FROM work_items WHERE id = ?", (item_id,))
            row = cur.fetchone()
            try:
                cp = json.loads(row[0] or "{}") if row else {}
            except Exception:
                cp = {}
            cp["persist"] = "ok"

            conn.execute(
                """
                UPDATE work_items
                SET status = ?,
                    normalized_output_ref = ?,
                    checkpoints = ?
                WHERE id = ?
                """,
                (WorkItemStatus.PERSISTED.value, normalized_output_ref, json.dumps(cp, ensure_ascii=False), item_id),
            )
            conn.commit()

        item = self.get_item(item_id)
        assert item is not None
        return item

    def mark_item_validated(self, item_id: str, validation_result: Dict[str, Any]) -> WorkItem:
        valid = validation_result.get("valid", False)
        status = WorkItemStatus.VALIDATED.value if valid else WorkItemStatus.BLOCKED.value
        if not valid and validation_result.get("waiting_for_human") is True:
            status = WorkItemStatus.WAITING_FOR_USER.value
        with self._lock, self.get_connection() as conn:
            cur = conn.execute("SELECT checkpoints FROM work_items WHERE id = ?", (item_id,))
            row = cur.fetchone()
            try:
                cp = json.loads(row[0] or "{}") if row else {}
            except Exception:
                cp = {}
            cp["validate"] = "ok" if valid else "failed"

            conn.execute(
                """
                UPDATE work_items
                SET status = ?,
                    validation_result = ?,
                    checkpoints = ?
                WHERE id = ?
                """,
                (status, json.dumps(validation_result, ensure_ascii=False), json.dumps(cp, ensure_ascii=False), item_id),
            )
            conn.commit()

        item = self.get_item(item_id)
        assert item is not None
        return item

    def resume_waiting_item(self, item_id: str) -> None:
        """Reset only a waiting read result after the owning human handoff returns."""
        with self._lock, self.get_connection() as conn:
            row = conn.execute("SELECT status, checkpoints FROM work_items WHERE id = ?", (item_id,)).fetchone()
            if row is None or row[0] != WorkItemStatus.WAITING_FOR_USER.value:
                raise ValueError("Work item is not waiting for human control")
            checkpoints = json.loads(row[1] or "{}")
            for key in ("persist", "normalize", "validate"):
                checkpoints.pop(key, None)
            conn.execute("UPDATE work_items SET status = ?, raw_output_ref = NULL, normalized_output_ref = NULL, "
                         "validation_result = '{}', checkpoints = ? WHERE id = ?",
                         (WorkItemStatus.PENDING.value, json.dumps(checkpoints), item_id))
            conn.commit()

    def complete_item(self, item_id: str) -> WorkItem:
        """Mark item as COMPLETED after verifying strict persistence and validation invariants."""
        item = self.get_item(item_id)
        if not item:
            raise ValueError(f"WorkItem {item_id} not found")
        item.verify_can_complete()

        now = _utc_now()
        with self._lock, self.get_connection() as conn:
            conn.execute(
                """
                UPDATE work_items
                SET status = ?,
                    completed_at = ?,
                    error = NULL,
                    last_error = NULL
                WHERE id = ?
                """,
                (WorkItemStatus.COMPLETED.value, now, item_id),
            )
            # Update plan timestamp and check if all completed
            conn.execute(
                "UPDATE work_plans SET updated_at = ? WHERE id = ?",
                (now, item.plan_id),
            )
            conn.commit()

        updated = self.get_item(item_id)
        assert updated is not None
        return updated

    def fail_item(
        self,
        item_id: str,
        error: str,
        *,
        can_retry: bool = True,
        max_retries: Optional[int] = None,
    ) -> WorkItem:
        now = _utc_now()
        item = self.get_item(item_id)
        if not item:
            raise ValueError(f"WorkItem {item_id} not found")

        plan = self.get_plan(item.plan_id)
        limit = max_retries if max_retries is not None else (plan.max_retries if plan else 3)
        next_attempts = item.attempts + 1

        retry_state = dict(item.retry_state)
        history = list(retry_state.get("history", []))
        history.append({"attempt": next_attempts, "error": error, "at": now})
        retry_state["history"] = history
        retry_state["attempts"] = next_attempts

        if can_retry and next_attempts <= limit:
            new_status = WorkItemStatus.RETRYING.value
        else:
            new_status = WorkItemStatus.FAILED.value

        with self._lock, self.get_connection() as conn:
            conn.execute(
                """
                UPDATE work_items
                SET status = ?,
                    attempts = ?,
                    error = ?,
                    last_error = ?,
                    retry_state = ?,
                    completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END
                WHERE id = ?
                """,
                (new_status, next_attempts, error, error, json.dumps(retry_state, ensure_ascii=False), new_status, now, item_id),
            )
            conn.commit()

        updated = self.get_item(item_id)
        assert updated is not None
        return updated

    def resume_plan(self, task_id: str) -> List[WorkItem]:
        """Return all items that are not yet COMPLETED for the active plan.

        If a process died while an item was RUNNING, it is safely rehydrated
        without duplicating already-completed items.
        """
        plan = self.get_plan(task_id)
        if not plan:
            return []

        self.reconcile_running_items(plan.id)

        with self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM work_items
                WHERE plan_id = ? AND status != ?
                ORDER BY item_index ASC
                """,
                (plan.id, WorkItemStatus.COMPLETED.value),
            )
            rows = cur.fetchall()
            col_names = [d[0] for d in cur.description]
            return [WorkItem.from_dict(dict(zip(col_names, row))) for row in rows]

    def get_progress_summary(self, task_id: str) -> Dict[str, Any]:
        plan = self.get_plan(task_id)
        if not plan:
            return {"task_id": task_id, "found": False}

        items = self.get_work_items(plan.id)
        counts: Dict[str, int] = {}
        for item in items:
            counts[item.status.value] = counts.get(item.status.value, 0) + 1

        completed = counts.get(WorkItemStatus.COMPLETED.value, 0)
        failed = counts.get(WorkItemStatus.FAILED.value, 0)
        running = counts.get(WorkItemStatus.RUNNING.value, 0)
        pending = counts.get(WorkItemStatus.PENDING.value, 0)
        retrying = counts.get(WorkItemStatus.RETRYING.value, 0)
        waiting = counts.get(WorkItemStatus.WAITING_FOR_USER.value, 0)

        # Count suspects: items validated but flagged as suspect
        suspects = sum(
            1 for item in items
            if item.validation_result and item.validation_result.get("suspect") is True
        )

        return {
            "task_id": task_id,
            "plan_id": plan.id,
            "title": plan.title,
            "total_items": plan.total_items,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "retrying": retrying,
            "waiting_for_user": waiting,
            "suspect": suspects,
            "progress_ratio": (completed / plan.total_items) if plan.total_items > 0 else 0.0,
            "all_done": completed == plan.total_items and plan.total_items > 0,
        }

    def to_compact_context(self, task_id: str) -> str:
        """Produce an ultra-compact structured summary for context compaction.

        Replaces verbose narrative prose with a canonical operational handle.
        """
        summary = self.get_progress_summary(task_id)
        if not summary.get("found", True):
            return f"task://{task_id} (not found)"

        total = summary["total_items"]
        completed = summary["completed"]
        suspect = summary.get("suspect", 0)
        failed = summary.get("failed", 0)
        waiting = summary.get("waiting_for_user", 0)

        parts = [f"task://{task_id}", f"[{completed}/{total} completed"]
        if suspect > 0:
            parts.append(f"{suspect} suspect")
        if failed > 0:
            parts.append(f"{failed} failed")
        if waiting > 0:
            parts.append("waiting for human")
        parts[-1] = parts[-1] + "]"
        return " ".join(parts)

    def get_delta_state(self, task_id: str, last_completed_count: int = 0) -> Dict[str, Any]:
        """Compute delta progress to avoid transmitting full state snapshots repeatedly."""
        summary = self.get_progress_summary(task_id)
        current_completed = summary.get("completed", 0)
        delta = current_completed - last_completed_count
        sign = f"+{delta}" if delta >= 0 else str(delta)
        return {
            "task_id": task_id,
            "delta": {
                "completed": sign,
                "current": f"{current_completed}/{summary.get('total_items', 0)}",
                "suspect": summary.get("suspect", 0),
                "failed": summary.get("failed", 0),
            },
        }

    def update_plan_state(self, plan_id: str, status: str) -> None:
        with self._lock, self.get_connection() as conn:
            conn.execute("UPDATE work_plans SET status=?, updated_at=?, completed_at=? WHERE id=?",
                         (status, _utc_now(), _utc_now() if status == "completed" else None, plan_id))
            if status in {"interrupted", "failed", "cancelled", "blocked"}:
                conn.execute(
                    """UPDATE work_items
                       SET status='blocked', last_error=?
                       WHERE plan_id=? AND status IN ('running', 'pending', 'ready', 'claimed', 'retrying')""",
                    ("parent stopped; reconcile dispatched effects before resume", plan_id),
                )
            conn.commit()

    def reconcile_terminal_plans(self) -> int:
        """Startup and periodic reconciliation: ensure no terminal plan has live descendants."""
        changed = 0
        with self._lock, self.get_connection() as conn:
            terminal_plans = [
                row[0] for row in conn.execute(
                    "SELECT id FROM work_plans WHERE status IN ('interrupted', 'failed', 'cancelled', 'blocked')"
                ).fetchall()
            ]
            for plan_id in terminal_plans:
                cur = conn.execute(
                    """UPDATE work_items
                       SET status='blocked', last_error='parent terminal; reconciled on startup'
                       WHERE plan_id=? AND status IN ('running', 'pending', 'ready', 'claimed', 'retrying')""",
                    (plan_id,)
                )
                changed += cur.rowcount
            conn.commit()
        return changed

    def reconcile_running_items(self, plan_id: str, *, live_item_ids: set[str] | None = None) -> int:
        """Recover durable state against handles proved live by the current runtime.

        Stored RUNNING is never proof of execution. Block rather than retry:
        a dispatched mutation might already have landed.
        """
        live = live_item_ids or set()
        changed = 0
        with self._lock, self.get_connection() as conn:
            for row in conn.execute("SELECT id FROM work_items WHERE plan_id=? AND status='running'", (plan_id,)).fetchall():
                if row[0] not in live:
                    conn.execute("UPDATE work_items SET status='blocked', last_error=? WHERE id=?",
                                 ("recovery-required: no live handle; reconcile effect", row[0]))
                    changed += 1
            conn.commit()
        return changed

    def update_plan_metadata(self, plan_id: str, values: dict) -> None:
        with self._lock, self.get_connection() as conn:
            row = conn.execute("SELECT metadata FROM work_plans WHERE id=? OR task_id=?", (plan_id, plan_id)).fetchone()
            if row is None:
                raise ValueError("Plan not found")
            metadata = {**json.loads(row[0] or "{}"), **values}
            conn.execute("UPDATE work_plans SET metadata=? WHERE id=? OR task_id=?", (json.dumps(metadata), plan_id, plan_id))
            conn.commit()

    def record_evidence(self, item_id: str, ref: str) -> None:
        with self._lock, self.get_connection() as conn:
            conn.execute("UPDATE work_items SET evidence_refs=? WHERE id=?",
                         (json.dumps([{"artifact_ref": ref}]), item_id))

    def operational_ledger(self, task_id: str) -> Dict[str, Any]:
        """Rebuild operational truth from the canonical DB, never from reasoning."""
        plan = self.get_plan(task_id)
        if plan is None:
            return {"task_id": task_id, "found": False}
        items = self.get_work_items(plan.id)
        completed = sum(i.status == WorkItemStatus.COMPLETED for i in items)
        exceptions = [i for i in items if i.status in {
            WorkItemStatus.FAILED, WorkItemStatus.BLOCKED, WorkItemStatus.WAITING_FOR_USER
        } or i.validation_result.get("suspect")]
        ledger = {
            "task_id": plan.task_id, "plan_id": plan.id, "phase": plan.status,
            "objective_ref": plan.metadata.get("objective_ref"),
            "constraints": plan.metadata.get("constraints", {}),
            "artifacts": [i.normalized_output_ref for i in items if i.normalized_output_ref][:8],
            "artifact_count": sum(bool(i.normalized_output_ref) for i in items),
            "completed": completed,
            "pending": sum(not i.is_terminal and i not in exceptions for i in items),
            "failed": sum(i.status == WorkItemStatus.FAILED for i in items),
            "blockers": [{"item_id": i.id, "status": i.status.value,
                          "code": str(i.validation_result.get("reason", i.last_error or "review_required"))[:120],
                          "evidence_ref": i.normalized_output_ref} for i in exceptions[:10]],
            "blocker_count": len(exceptions),
            "next_action": "review_exceptions" if exceptions else (
                "finish" if completed == len(items) else "continue_plan"),
        }
        graph = plan.metadata.get("graph")
        ledger["recipe"] = plan.metadata.get("recipe", {})
        if plan.metadata.get("canary_required"):
            first = next((i for i in items if i.input_payload.get("_work_phase", "fan_out") == "fan_out"), None)
            verified = bool(first and first.checkpoints.get("canary_verified") == "ok")
            uncertain = bool(first and first.validation_result.get("reason") == "uncertain_mutation_requires_review")
            ledger["canary"] = {"item_id": first.id if first else None, "verified": verified,
                "status": "verified" if verified else "uncertain" if uncertain else "failed" if first in exceptions else "pending"}
        elif plan.metadata.get("recipe", {}).get("status") == "VERIFIED":
            first = next((i for i in items if i.input_payload.get("_work_phase", "fan_out") == "fan_out"), None)
            ready = bool(first and first.status == WorkItemStatus.COMPLETED)
            ledger["canary"] = {"required": False, "verified": ready,
                                "status": "recipe_preflight_verified" if ready else "pending_preflight"}
        if graph:
            from tools.effects import READ_EFFECTS, tool_effect
            refs = sorted({i.normalized_output_ref for i in items if i.normalized_output_ref} | {
                meta["result_ref"] for i in items for key, meta in i.checkpoints.items()
                if key.endswith("_meta") and isinstance(meta, dict) and meta.get("result_ref")})
            ledger["artifacts"], ledger["artifact_count"] = refs[:8], len(refs)
            for phase in ("setup", "fan_out", "finalize"):
                group = [i for i in items if i.input_payload.get("_work_phase") == phase]
                done = sum(i.status == WorkItemStatus.COMPLETED for i in group)
                failed = sum(i.status == WorkItemStatus.FAILED for i in group)
                review = sum(i in exceptions for i in group)
                uncertain = sum(i.validation_result.get("reason") == "uncertain_mutation_requires_review" or
                    any(k.endswith("_dispatch") and not i.checkpoints.get(k.removesuffix("_dispatch") + "_meta", {}).get("result_ref")
                        and tool_effect(graph[phase][int(k.split("_")[1])]["tool"]) not in READ_EFFECTS for k in i.checkpoints)
                    for i in group if i.status != WorkItemStatus.COMPLETED)
                if phase == "fan_out":
                    ledger["items"] = {"completed": done, "total": len(group), "pending": len(group)-done-review,
                                       "needs_reasoning": review, "failed": failed, "uncertain": uncertain}
                else:
                    steps_done = sum(bool(i.checkpoints.get(f"step_{n}_meta", {}).get("result_ref"))
                                     for i in group for n in range(len(graph[phase])))
                    ledger[phase] = {"completed": steps_done, "total": len(graph[phase]),
                                     "pending": len(graph[phase])-steps_done, "needs_reasoning": review,
                                     "failed": failed, "uncertain": uncertain}
            ledger["uncertain"] = sum(ledger[p]["uncertain"] for p in ("setup", "items", "finalize"))
            ledger["phase"] = next((p for p, key in (("setup", "setup"), ("fan_out", "items"), ("finalize", "finalize"))
                                    if ledger[key]["completed"] < ledger[key]["total"]), "completed")
        return ledger
