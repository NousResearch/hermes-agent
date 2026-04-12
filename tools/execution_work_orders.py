"""Durable work-order queue for H007 direct terminal work orders.

This module intentionally stays narrow: it persists one-shot direct terminal
work orders, tracks their execution state in SQLite + JSON snapshots, and
supports reclaim/retry/resume semantics at the work-order level.

It does NOT claim process-level continuation. Resumability here means replaying
an explicit deterministic work order from durable state after failure or an
expired runner lease.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from cron.jobs import parse_schedule
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_work_orders (
    work_order_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    goal TEXT NOT NULL,
    context TEXT,
    command TEXT NOT NULL,
    timeout_seconds INTEGER,
    workdir TEXT,
    execution_path TEXT,
    status TEXT NOT NULL,
    scheduled_for REAL NOT NULL,
    schedule_input TEXT,
    schedule_display TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    attempt_count INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 1,
    retry_delay_seconds REAL DEFAULT 0,
    claim_token TEXT,
    claim_owner TEXT,
    claim_expires_at REAL,
    reclaimed_count INTEGER DEFAULT 0,
    last_error TEXT,
    last_exit_reason TEXT,
    last_summary TEXT,
    last_receipt_path TEXT,
    last_receipt_id TEXT,
    last_duration_seconds REAL,
    parent_session_id TEXT,
    worker_mode TEXT,
    worker_task_id TEXT,
    worker_runtime_id TEXT,
    worker_runtime_kind TEXT,
    worker_runtime_reused INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_execution_work_orders_status ON execution_work_orders(status);
CREATE INDEX IF NOT EXISTS idx_execution_work_orders_scheduled_for ON execution_work_orders(scheduled_for);
CREATE INDEX IF NOT EXISTS idx_execution_work_orders_updated_at ON execution_work_orders(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_work_orders_parent_session ON execution_work_orders(parent_session_id);
"""

_EXPECTED_COLUMNS = {
    "goal": "TEXT NOT NULL DEFAULT ''",
    "context": "TEXT",
    "command": "TEXT NOT NULL DEFAULT ''",
    "timeout_seconds": "INTEGER",
    "workdir": "TEXT",
    "execution_path": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'queued'",
    "scheduled_for": "REAL NOT NULL DEFAULT 0",
    "schedule_input": "TEXT",
    "schedule_display": "TEXT",
    "created_at": "REAL NOT NULL DEFAULT 0",
    "updated_at": "REAL NOT NULL DEFAULT 0",
    "started_at": "REAL",
    "completed_at": "REAL",
    "attempt_count": "INTEGER DEFAULT 0",
    "max_attempts": "INTEGER DEFAULT 1",
    "retry_delay_seconds": "REAL DEFAULT 0",
    "claim_token": "TEXT",
    "claim_owner": "TEXT",
    "claim_expires_at": "REAL",
    "reclaimed_count": "INTEGER DEFAULT 0",
    "last_error": "TEXT",
    "last_exit_reason": "TEXT",
    "last_summary": "TEXT",
    "last_receipt_path": "TEXT",
    "last_receipt_id": "TEXT",
    "last_duration_seconds": "REAL",
    "parent_session_id": "TEXT",
    "worker_mode": "TEXT",
    "worker_task_id": "TEXT",
    "worker_runtime_id": "TEXT",
    "worker_runtime_kind": "TEXT",
    "worker_runtime_reused": "INTEGER DEFAULT 0",
}

_INT_FIELDS = {
    "attempt_count",
    "max_attempts",
    "timeout_seconds",
    "reclaimed_count",
}
_FLOAT_FIELDS = {
    "scheduled_for",
    "created_at",
    "updated_at",
    "started_at",
    "completed_at",
    "retry_delay_seconds",
    "claim_expires_at",
    "last_duration_seconds",
}
_BOOL_FIELDS = {"worker_runtime_reused"}


def get_execution_work_orders_dir() -> Path:
    path = get_hermes_home() / "artifacts" / "execution-work-orders"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_execution_work_orders_index_path() -> Path:
    return get_execution_work_orders_dir() / "index.sqlite"


def _connect_index() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_execution_work_orders_index_path()), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _ensure_index_columns(conn)
    return conn


def _ensure_index_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(execution_work_orders)").fetchall()
    existing = {row[1] for row in rows}
    for column, ddl in _EXPECTED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE execution_work_orders ADD COLUMN {column} {ddl}")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_jsonish(v) for k, v in value.items() if _coerce_jsonish(v) is not None}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonish(v) for v in value if _coerce_jsonish(v) is not None]
    return str(value)


def _resolve_schedule(*, schedule: str | None, delay_seconds: float | None, now_ts: float) -> tuple[float, str | None, str]:
    if schedule and delay_seconds is not None:
        raise ValueError("Provide either schedule or delay_seconds, not both")

    if schedule:
        parsed = parse_schedule(str(schedule).strip())
        if parsed.get("kind") != "once":
            raise ValueError("execution work orders currently support only one-shot schedules")
        run_at = parsed.get("run_at")
        if not run_at:
            raise ValueError("Could not resolve one-shot run_at for the work order schedule")
        scheduled_for = datetime.fromisoformat(str(run_at)).timestamp()
        return scheduled_for, str(schedule).strip(), str(parsed.get("display") or schedule)

    if delay_seconds is not None:
        delay = float(delay_seconds)
        if delay < 0:
            raise ValueError("delay_seconds must be >= 0")
        display = "immediate" if delay == 0 else f"in {round(delay, 2)}s"
        return now_ts + delay, str(delay), display

    return now_ts, None, "immediate"


def _row_to_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    for key in _INT_FIELDS:
        if payload.get(key) is not None:
            payload[key] = int(payload[key])
    for key in _FLOAT_FIELDS:
        if payload.get(key) is not None:
            payload[key] = float(payload[key])
    for key in _BOOL_FIELDS:
        payload[key] = bool(payload.get(key))
    return payload


def _normalize_work_order(work_order: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    normalized = dict(work_order)

    goal = _coerce_optional_text(normalized.get("goal"))
    command = _coerce_optional_text(normalized.get("command"))
    if not goal:
        raise ValueError("work order goal is required")
    if not command:
        raise ValueError("work order command is required")

    normalized["work_order_id"] = _coerce_optional_text(normalized.get("work_order_id")) or f"work-order-{uuid.uuid4().hex}"
    normalized["goal"] = goal
    normalized["context"] = _coerce_optional_text(normalized.get("context"))
    normalized["command"] = command
    normalized["timeout_seconds"] = int(normalized["timeout_seconds"]) if normalized.get("timeout_seconds") is not None else None
    if normalized.get("timeout_seconds") is not None and int(normalized["timeout_seconds"]) <= 0:
        raise ValueError("timeout_seconds must be > 0")
    normalized["workdir"] = _coerce_optional_text(normalized.get("workdir"))
    normalized["execution_path"] = "direct_terminal_work_order"
    normalized["status"] = _coerce_optional_text(normalized.get("status")) or "queued"
    normalized["scheduled_for"] = float(normalized.get("scheduled_for") if normalized.get("scheduled_for") is not None else now_ts)
    normalized["schedule_input"] = _coerce_optional_text(normalized.get("schedule_input"))
    normalized["schedule_display"] = _coerce_optional_text(normalized.get("schedule_display")) or "immediate"
    normalized["created_at"] = float(normalized.get("created_at") if normalized.get("created_at") is not None else now_ts)
    normalized["updated_at"] = float(normalized.get("updated_at") if normalized.get("updated_at") is not None else now_ts)
    normalized["started_at"] = float(normalized["started_at"]) if normalized.get("started_at") is not None else None
    normalized["completed_at"] = float(normalized["completed_at"]) if normalized.get("completed_at") is not None else None
    normalized["attempt_count"] = max(0, int(normalized.get("attempt_count") or 0))
    normalized["max_attempts"] = max(1, int(normalized.get("max_attempts") or 1))
    normalized["retry_delay_seconds"] = max(0.0, float(normalized.get("retry_delay_seconds") or 0.0))
    normalized["claim_token"] = _coerce_optional_text(normalized.get("claim_token"))
    normalized["claim_owner"] = _coerce_optional_text(normalized.get("claim_owner"))
    normalized["claim_expires_at"] = float(normalized["claim_expires_at"]) if normalized.get("claim_expires_at") is not None else None
    normalized["reclaimed_count"] = max(0, int(normalized.get("reclaimed_count") or 0))
    normalized["last_error"] = _coerce_optional_text(normalized.get("last_error"))
    normalized["last_exit_reason"] = _coerce_optional_text(normalized.get("last_exit_reason"))
    normalized["last_summary"] = _coerce_optional_text(normalized.get("last_summary"))
    normalized["last_receipt_path"] = _coerce_optional_text(normalized.get("last_receipt_path"))
    normalized["last_receipt_id"] = _coerce_optional_text(normalized.get("last_receipt_id"))
    normalized["last_duration_seconds"] = float(normalized["last_duration_seconds"]) if normalized.get("last_duration_seconds") is not None else None
    normalized["parent_session_id"] = _coerce_optional_text(normalized.get("parent_session_id"))
    normalized["worker_mode"] = _coerce_optional_text(normalized.get("worker_mode"))
    normalized["worker_task_id"] = _coerce_optional_text(normalized.get("worker_task_id"))
    normalized["worker_runtime_id"] = _coerce_optional_text(normalized.get("worker_runtime_id"))
    normalized["worker_runtime_kind"] = _coerce_optional_text(normalized.get("worker_runtime_kind"))
    normalized["worker_runtime_reused"] = bool(normalized.get("worker_runtime_reused"))
    normalized["file_path"] = str(get_execution_work_orders_dir() / f"{normalized['work_order_id']}.json")
    return normalized


def _persist_normalized_work_order(conn: sqlite3.Connection, normalized: dict[str, Any]) -> str:
    file_path = Path(str(normalized["file_path"]))
    previous_content: str | None = None
    if file_path.exists():
        try:
            previous_content = file_path.read_text(encoding="utf-8")
        except OSError:
            previous_content = None

    try:
        _atomic_write_text(file_path, json.dumps(_coerce_jsonish(normalized), indent=2, ensure_ascii=False, sort_keys=True))
        conn.execute(
            """
            INSERT OR REPLACE INTO execution_work_orders (
                work_order_id, file_path, goal, context, command, timeout_seconds,
                workdir, execution_path, status, scheduled_for, schedule_input,
                schedule_display, created_at, updated_at, started_at, completed_at,
                attempt_count, max_attempts, retry_delay_seconds, claim_token,
                claim_owner, claim_expires_at, reclaimed_count, last_error,
                last_exit_reason, last_summary, last_receipt_path, last_receipt_id,
                last_duration_seconds, parent_session_id, worker_mode,
                worker_task_id, worker_runtime_id, worker_runtime_kind,
                worker_runtime_reused
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized["work_order_id"],
                normalized["file_path"],
                normalized["goal"],
                normalized.get("context"),
                normalized["command"],
                normalized.get("timeout_seconds"),
                normalized.get("workdir"),
                normalized.get("execution_path"),
                normalized.get("status"),
                normalized.get("scheduled_for"),
                normalized.get("schedule_input"),
                normalized.get("schedule_display"),
                normalized.get("created_at"),
                normalized.get("updated_at"),
                normalized.get("started_at"),
                normalized.get("completed_at"),
                normalized.get("attempt_count"),
                normalized.get("max_attempts"),
                normalized.get("retry_delay_seconds"),
                normalized.get("claim_token"),
                normalized.get("claim_owner"),
                normalized.get("claim_expires_at"),
                normalized.get("reclaimed_count"),
                normalized.get("last_error"),
                normalized.get("last_exit_reason"),
                normalized.get("last_summary"),
                normalized.get("last_receipt_path"),
                normalized.get("last_receipt_id"),
                normalized.get("last_duration_seconds"),
                normalized.get("parent_session_id"),
                normalized.get("worker_mode"),
                normalized.get("worker_task_id"),
                normalized.get("worker_runtime_id"),
                normalized.get("worker_runtime_kind"),
                1 if normalized.get("worker_runtime_reused") else 0,
            ),
        )
    except Exception:
        try:
            if previous_content is not None:
                _atomic_write_text(file_path, previous_content)
            elif file_path.exists():
                file_path.unlink()
        except OSError as cleanup_exc:
            logger.warning("Execution work-order cleanup failed for %s: %s", file_path, cleanup_exc)
        raise

    return str(file_path)


def persist_execution_work_order(work_order: dict[str, Any]) -> str:
    normalized = _normalize_work_order(work_order)
    with _connect_index() as conn:
        _persist_normalized_work_order(conn, normalized)
    return str(normalized["file_path"])


def enqueue_execution_work_order(
    *,
    goal: str,
    command: str,
    context: str | None = None,
    timeout_seconds: int | None = None,
    workdir: str | None = None,
    schedule: str | None = None,
    delay_seconds: float | None = None,
    max_attempts: int = 1,
    retry_delay_seconds: float = 0,
    parent_session_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    scheduled_for, schedule_input, schedule_display = _resolve_schedule(
        schedule=schedule,
        delay_seconds=delay_seconds,
        now_ts=now_ts,
    )
    work_order = _normalize_work_order(
        {
            "goal": goal,
            "context": context,
            "command": command,
            "timeout_seconds": timeout_seconds,
            "workdir": workdir,
            "status": "queued",
            "scheduled_for": scheduled_for,
            "schedule_input": schedule_input,
            "schedule_display": schedule_display,
            "created_at": now_ts,
            "updated_at": now_ts,
            "attempt_count": 0,
            "max_attempts": max_attempts,
            "retry_delay_seconds": retry_delay_seconds,
            "parent_session_id": parent_session_id,
        },
        now_ts,
    )
    persist_execution_work_order(work_order)
    return get_execution_work_order(str(work_order["work_order_id"])) or work_order


def query_execution_work_orders(
    *,
    limit: int = 20,
    status: str | None = None,
    parent_session_id: str | None = None,
    work_order_id: str | None = None,
) -> list[dict[str, Any]]:
    where = []
    params: list[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if parent_session_id:
        where.append("parent_session_id = ?")
        params.append(parent_session_id)
    if work_order_id:
        where.append("work_order_id = ?")
        params.append(work_order_id)

    sql = "SELECT * FROM execution_work_orders"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(max(1, int(limit)))

    with _connect_index() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_execution_work_order(work_order_id: str) -> dict[str, Any] | None:
    rows = query_execution_work_orders(work_order_id=work_order_id, limit=1)
    return rows[0] if rows else None


def execution_work_order_counts(*, now: float | None = None) -> dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    with _connect_index() as conn:
        grouped = conn.execute(
            "SELECT status, COUNT(*) AS count FROM execution_work_orders GROUP BY status"
        ).fetchall()
        due_count = conn.execute(
            "SELECT COUNT(*) FROM execution_work_orders WHERE status IN ('queued', 'retry_scheduled') AND scheduled_for <= ?",
            (now_ts,),
        ).fetchone()[0]
        stale_running = conn.execute(
            "SELECT COUNT(*) FROM execution_work_orders WHERE status = 'running' AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?",
            (now_ts,),
        ).fetchone()[0]
    status_counts = {str(row["status"]): int(row["count"]) for row in grouped}
    return {
        "status_counts": status_counts,
        "due_count": int(due_count or 0),
        "stale_running_count": int(stale_running or 0),
    }


def _load_work_order_for_update(conn: sqlite3.Connection, work_order_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM execution_work_orders WHERE work_order_id = ?",
        (work_order_id,),
    ).fetchone()
    if not row:
        return None
    state = _row_to_dict(row)
    file_path = _coerce_optional_text(state.get("file_path"))
    if file_path:
        path = Path(file_path)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                state.update(payload)
    state["file_path"] = file_path or state.get("file_path")
    return _normalize_work_order(state)


def _append_note(existing: str | None, note: str) -> str:
    base = _coerce_optional_text(existing)
    return note if not base else f"{base}; {note}"


def claim_next_due_execution_work_order(
    *,
    claim_owner: str | None = None,
    claim_ttl_seconds: float = 900,
    now: float | None = None,
) -> dict[str, Any] | None:
    now_ts = float(now if now is not None else time.time())
    ttl = max(1.0, float(claim_ttl_seconds))
    with _connect_index() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT work_order_id
            FROM execution_work_orders
            WHERE status IN ('queued', 'retry_scheduled')
              AND scheduled_for <= ?
            ORDER BY scheduled_for ASC, created_at ASC
            LIMIT 1
            """,
            (now_ts,),
        ).fetchone()
        if not row:
            conn.commit()
            return None
        state = _load_work_order_for_update(conn, str(row["work_order_id"]))
        if not state:
            conn.rollback()
            return None
        state["status"] = "running"
        state["attempt_count"] = int(state.get("attempt_count") or 0) + 1
        state["updated_at"] = now_ts
        state["started_at"] = now_ts
        state["completed_at"] = None
        state["claim_token"] = f"claim-{uuid.uuid4().hex}"
        state["claim_owner"] = _coerce_optional_text(claim_owner)
        state["claim_expires_at"] = now_ts + ttl
        _persist_normalized_work_order(conn, _normalize_work_order(state, now_ts))
        conn.commit()
        return state


def reclaim_stale_execution_work_orders(*, now: float | None = None, limit: int = 100) -> dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    effective_limit = max(1, int(limit))
    reclaimed: list[dict[str, Any]] = []
    with _connect_index() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT work_order_id
            FROM execution_work_orders
            WHERE status = 'running'
              AND claim_expires_at IS NOT NULL
              AND claim_expires_at <= ?
            ORDER BY claim_expires_at ASC
            LIMIT ?
            """,
            (now_ts, effective_limit),
        ).fetchall()
        for row in rows:
            state = _load_work_order_for_update(conn, str(row["work_order_id"]))
            if not state:
                continue
            state["status"] = "queued"
            state["scheduled_for"] = now_ts
            state["updated_at"] = now_ts
            state["claim_token"] = None
            state["claim_owner"] = None
            state["claim_expires_at"] = None
            state["reclaimed_count"] = int(state.get("reclaimed_count") or 0) + 1
            state["last_error"] = _append_note(state.get("last_error"), "claim_expired_reclaimed")
            state["last_exit_reason"] = "claim_expired_reclaimed"
            _persist_normalized_work_order(conn, _normalize_work_order(state, now_ts))
            reclaimed.append(state)
        conn.commit()
    return {
        "reclaimed_count": len(reclaimed),
        "work_orders": reclaimed,
    }


def _receipt_id_from_path(path: str | None) -> str | None:
    text = _coerce_optional_text(path)
    if not text:
        return None
    return Path(text).stem


def finish_execution_work_order(
    work_order_id: str,
    *,
    claim_token: str | None,
    result: dict[str, Any],
    now: float | None = None,
) -> dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    with _connect_index() as conn:
        conn.execute("BEGIN IMMEDIATE")
        state = _load_work_order_for_update(conn, work_order_id)
        if not state:
            conn.rollback()
            raise ValueError(f"Unknown work order: {work_order_id}")
        current_claim = _coerce_optional_text(state.get("claim_token"))
        if current_claim and claim_token and current_claim != claim_token:
            conn.rollback()
            raise ValueError(f"Claim token mismatch for work order {work_order_id}")

        result_status = _coerce_optional_text(result.get("status")) or "failed"
        success = result_status == "completed"
        state["updated_at"] = now_ts
        state["claim_token"] = None
        state["claim_owner"] = None
        state["claim_expires_at"] = None
        state["last_error"] = _coerce_optional_text(result.get("error")) or _coerce_optional_text(result.get("fallback_reason"))
        state["last_exit_reason"] = _coerce_optional_text(result.get("exit_reason")) or ("completed" if success else "failed")
        state["last_summary"] = _coerce_optional_text(result.get("summary"))
        state["last_receipt_path"] = _coerce_optional_text(result.get("execution_receipt_path"))
        state["last_receipt_id"] = _receipt_id_from_path(state.get("last_receipt_path"))
        state["last_duration_seconds"] = float(result.get("duration_seconds") or 0.0)
        state["worker_mode"] = _coerce_optional_text(result.get("worker_mode"))
        state["worker_task_id"] = _coerce_optional_text(result.get("worker_task_id"))
        state["worker_runtime_id"] = _coerce_optional_text(result.get("worker_runtime_id"))
        state["worker_runtime_kind"] = _coerce_optional_text(result.get("worker_runtime_kind"))
        state["worker_runtime_reused"] = bool(result.get("worker_runtime_reused"))

        if success:
            state["status"] = "completed"
            state["completed_at"] = now_ts
        else:
            attempts = int(state.get("attempt_count") or 0)
            max_attempts = int(state.get("max_attempts") or 1)
            if attempts < max_attempts:
                state["status"] = "retry_scheduled"
                state["scheduled_for"] = now_ts + float(state.get("retry_delay_seconds") or 0.0)
                state["completed_at"] = None
            else:
                state["status"] = "failed"
                state["completed_at"] = now_ts

        _persist_normalized_work_order(conn, _normalize_work_order(state, now_ts))
        conn.commit()
        return state


def _requeue_existing_work_order(
    work_order_id: str,
    *,
    reason: str,
    delay_seconds: float = 0,
    now: float | None = None,
    allow_running_if_expired: bool = False,
) -> dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    delay = max(0.0, float(delay_seconds))
    with _connect_index() as conn:
        conn.execute("BEGIN IMMEDIATE")
        state = _load_work_order_for_update(conn, work_order_id)
        if not state:
            conn.rollback()
            raise ValueError(f"Unknown work order: {work_order_id}")
        status = str(state.get("status") or "")
        expired_running = status == "running" and state.get("claim_expires_at") is not None and float(state.get("claim_expires_at") or 0.0) <= now_ts
        if status not in {"failed", "completed", "cancelled", "retry_scheduled"} and not (allow_running_if_expired and expired_running):
            conn.rollback()
            raise ValueError(f"Cannot {reason} work order in status={status}")
        if expired_running:
            state["reclaimed_count"] = int(state.get("reclaimed_count") or 0) + 1
        state["status"] = "queued"
        state["scheduled_for"] = now_ts + delay
        state["updated_at"] = now_ts
        state["completed_at"] = None
        state["claim_token"] = None
        state["claim_owner"] = None
        state["claim_expires_at"] = None
        state["last_exit_reason"] = reason
        _persist_normalized_work_order(conn, _normalize_work_order(state, now_ts))
        conn.commit()
        return state


def retry_execution_work_order(work_order_id: str, *, delay_seconds: float = 0, now: float | None = None) -> dict[str, Any]:
    return _requeue_existing_work_order(work_order_id, reason="manual_retry", delay_seconds=delay_seconds, now=now)


def resume_execution_work_order(work_order_id: str, *, delay_seconds: float = 0, now: float | None = None) -> dict[str, Any]:
    return _requeue_existing_work_order(
        work_order_id,
        reason="manual_resume",
        delay_seconds=delay_seconds,
        now=now,
        allow_running_if_expired=True,
    )


def cancel_execution_work_order(work_order_id: str, *, now: float | None = None) -> dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    with _connect_index() as conn:
        conn.execute("BEGIN IMMEDIATE")
        state = _load_work_order_for_update(conn, work_order_id)
        if not state:
            conn.rollback()
            raise ValueError(f"Unknown work order: {work_order_id}")
        status = str(state.get("status") or "")
        if status not in {"queued", "retry_scheduled"}:
            expired_running = status == "running" and state.get("claim_expires_at") is not None and float(state.get("claim_expires_at") or 0.0) <= now_ts
            if not expired_running:
                conn.rollback()
                raise ValueError(f"Cannot cancel work order in status={status}")
            state["reclaimed_count"] = int(state.get("reclaimed_count") or 0) + 1
        state["status"] = "cancelled"
        state["updated_at"] = now_ts
        state["completed_at"] = now_ts
        state["claim_token"] = None
        state["claim_owner"] = None
        state["claim_expires_at"] = None
        state["last_exit_reason"] = "cancelled"
        _persist_normalized_work_order(conn, _normalize_work_order(state, now_ts))
        conn.commit()
        return state
