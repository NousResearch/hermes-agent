"""Durable task-scoped delegation closeout ledger.

The creation flag gates only first-generation registration. Recovery and delivery
always interpret existing rows so disabling the feature cannot strand durable work.
This module owns the state machine; ``tools.async_delegation`` owns child execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.async_delegation import _DB_LOCK, _transaction

_DEFAULT_AGGREGATE_CHAR_BUDGET = 48_000
_MIN_AGGREGATE_CHAR_BUDGET = 1_024
_MAX_AGGREGATE_CHAR_BUDGET = 96_000
_GROUP_CLAIM_STALE_SECONDS = 300.0
_GROUP_DETAIL_RETENTION_SECONDS = 48 * 3600.0
_MAX_GROUP_DETAIL_BYTES = 64_000
_MAX_WORK_ID_BYTES = 256
_MAX_DELEGATION_ID_BYTES = 256
_MAX_OWNER_TURN_ID_BYTES = 256
_MAX_OUTCOME_COUNT = 999_999_999

_aggregate_enqueue_lock = threading.Lock()
_aggregate_enqueued_delivery_ids: set[tuple[str, str]] = set()


def initialize_closeout_schema(conn: sqlite3.Connection) -> None:
    """Add the grouped-work columns and ledger table idempotently."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")}
    for name, sql_type in (
        ("origin_work_id", "TEXT NOT NULL DEFAULT ''"),
        ("work_generation", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS async_delegation_work_groups (
            work_id TEXT PRIMARY KEY,
            origin_session TEXT NOT NULL DEFAULT '',
            origin_ui_session_id TEXT NOT NULL DEFAULT '',
            origin_session_id TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT,
            routing_json TEXT NOT NULL DEFAULT '{}',
            owner_turn_id TEXT NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            state TEXT NOT NULL CHECK (state IN ('open','sealed','closing','closed')),
            generation INTEGER NOT NULL DEFAULT 0,
            aggregate_char_budget INTEGER NOT NULL DEFAULT 0,
            closeout_delivery_id TEXT,
            closeout_payload_json TEXT,
            closeout_claim TEXT,
            closeout_claimed_at REAL,
            closeout_turn_id TEXT,
            closeout_owner_pid INTEGER,
            closeout_owner_started_at INTEGER,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            sealed_at REAL,
            closed_at REAL,
            terminal_disposition TEXT,
            terminal_diagnostics TEXT
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_async_delegations_work "
        "ON async_delegations(origin_work_id, work_generation, dispatched_at)"
    )
    group_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(async_delegation_work_groups)")
    }
    for name, sql_type in (
        ("closeout_owner_pid", "INTEGER"),
        ("closeout_owner_started_at", "INTEGER"),
    ):
        if name not in group_columns:
            conn.execute(
                f"ALTER TABLE async_delegation_work_groups ADD COLUMN {name} {sql_type}"
            )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_async_work_groups_state "
        "ON async_delegation_work_groups(state, updated_at)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_async_work_groups_delivery "
        "ON async_delegation_work_groups(closeout_delivery_id) "
        "WHERE closeout_delivery_id IS NOT NULL"
    )


def prune_grouped_details(
    conn: sqlite3.Connection,
    *,
    now: float,
    detail_retention_seconds: float,
    max_detail_bytes: int,
) -> None:
    """Compact unresolved terminal member detail without deleting group identity."""
    unresolved = conn.execute(
        """SELECT d.delegation_id, d.state, d.result_json, d.event_json,
                  d.task_json, d.updated_at, d.origin_work_id
           FROM async_delegations d
           JOIN async_delegation_work_groups g ON g.work_id=d.origin_work_id
           WHERE g.state!='closed' AND d.state NOT IN ('running','finalizing')"""
    ).fetchall()
    grouped_bytes = sum(
        len((row[2] or "").encode("utf-8"))
        + len((row[3] or "").encode("utf-8"))
        + len((row[4] or "").encode("utf-8"))
        for row in unresolved
    )
    pressure = grouped_bytes > max_detail_bytes
    for (
        delegation_id,
        state,
        result_json,
        event_json,
        task_json,
        updated_at,
        work_id,
    ) in unresolved:
        detail_bytes = (
            len((result_json or "").encode("utf-8"))
            + len((event_json or "").encode("utf-8"))
            + len((task_json or "").encode("utf-8"))
        )
        if (
            not pressure
            and updated_at >= now - detail_retention_seconds
            and detail_bytes <= max_detail_bytes
        ):
            continue
        status = str(state or "unknown")
        tombstone = {
            "status": status,
            "detail_lost": True,
            "diagnostic": "terminal detail compacted by bounded ledger retention",
            "detail_ref": f"async_delegation:{delegation_id}",
        }
        compact = json.dumps(tombstone, sort_keys=True, separators=(",", ":"))
        conn.execute(
            """UPDATE async_delegations SET result_json=?, event_json=?,
                      task_json='{}', updated_at=?
               WHERE delegation_id=? AND state NOT IN ('running','finalizing')""",
            (compact, compact, now, delegation_id),
        )
        diagnostic = "Terminal member detail was compacted; outcome detail is unknown."
        conn.execute(
            """UPDATE async_delegation_work_groups
               SET terminal_diagnostics=CASE
                 WHEN terminal_diagnostics IS NULL OR terminal_diagnostics=''
                 THEN ? ELSE terminal_diagnostics || '; ' || ? END,
                 updated_at=? WHERE work_id=? AND state!='closed'""",
            (diagnostic, diagnostic, now, work_id),
        )


def task_scoped_closeout_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Resolve the creation-only gate, defaulting closed on every failure.

    Callers recovering or draining an existing group must not consult this
    resolver.  The gate controls only creation of the first generation.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:  # noqa: BLE001 - a safety gate fails closed
            return False
    delegation = config.get("delegation") if isinstance(config, dict) else None
    return bool(
        isinstance(delegation, dict)
        and delegation.get("task_scoped_closeout", False) is True
    )


def _process_identity() -> tuple[int, Optional[int]]:
    try:
        from gateway.status import get_process_start_time

        return os.getpid(), get_process_start_time(os.getpid())
    except Exception:  # noqa: BLE001 - PID remains a useful weaker identity
        return os.getpid(), None


def _process_identity_is_live(pid: Any, started_at: Any) -> bool:
    if not pid or started_at is None:
        return False
    try:
        from gateway.status import _pid_exists, get_process_start_time

        if not _pid_exists(int(pid)):
            return False
        return get_process_start_time(int(pid)) == int(started_at)
    except Exception:  # noqa: BLE001 - recovery must fail closed
        return False


def _bounded_budget(value: Optional[int]) -> int:
    if value is None:
        return _DEFAULT_AGGREGATE_CHAR_BUDGET
    return max(_MIN_AGGREGATE_CHAR_BUDGET, min(int(value), _MAX_AGGREGATE_CHAR_BUDGET))


def _delivery_id(work_id: str, generation: int) -> str:
    digest = hashlib.sha256(f"{work_id}\0{generation}".encode()).hexdigest()
    return f"delegation-closeout-{digest[:32]}"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


_STATUS_ALIASES = {
    "complete": "completed", "completed": "completed", "success": "completed",
    "succeeded": "completed", "ok": "completed",
    "failed": "failed", "failure": "failed", "error": "failed",
    "cancelled": "cancelled", "canceled": "cancelled", "aborted": "cancelled",
    "timeout": "timeout", "timed_out": "timeout", "timed out": "timeout",
    "stalled": "stalled", "stale": "stalled", "hung": "stalled",
    "unknown": "unknown", "dropped": "dropped", "blocked": "blocked",
}


def _status_category(value: Any) -> str:
    """Map untrusted status text to one fixed, truthful outcome category."""
    normalized = str(value or "unknown").strip().lower().replace("-", "_")
    return _STATUS_ALIASES.get(normalized, "unknown")


def _bounded_count(value: Any) -> int:
    try:
        return max(0, min(int(value or 0), _MAX_OUTCOME_COUNT))
    except (TypeError, ValueError, OverflowError):
        return 0


def _mandatory_member(
    delegation_id: str, *, status: Any = "unknown", result: Optional[Dict[str, Any]] = None,
    event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical registration-tested fallback; contains no arbitrary detail text."""
    result, event = result or {}, event or {}
    errors = result.get("schema_errors", event.get("schema_errors"))
    error_count = len(errors) if isinstance(errors, list) else int(errors is not None)
    schema_valid = result.get(
        "schema_valid", result.get("valid", event.get("schema_valid", event.get("valid")))
    )
    schema_verdict = result.get(
        "schema_verdict", result.get("verdict", event.get("schema_verdict", event.get("verdict")))
    )
    category = _status_category(status)
    return {
        "delegation_id": delegation_id,
        "status": category,
        "detail_ref": f"async_delegation:{delegation_id}",
        "detail_truncated": True,
        "detail_lost": bool(result.get("detail_lost", event.get("detail_lost", False))),
        "error_present": result.get("error", event.get("error")) not in (None, ""),
        "diagnostic_present": result.get("diagnostic", event.get("diagnostic")) not in (None, ""),
        "schema_verdict_present": schema_verdict is not None,
        "schema_valid": None if schema_valid is None else bool(schema_valid),
        "schema_error_count": _bounded_count(error_count),
        "schema_retries": _bounded_count(result.get("schema_retries", event.get("schema_retries"))),
        "timed_out": category == "timeout" or any(
            result.get(key, event.get(key)) is not None
            for key in ("timeout_seconds", "timed_out_after_seconds")
        ),
        "stalled": category == "stalled" or any(
            result.get(key, event.get(key)) is not None
            for key in ("stalled_after_quiet_seconds", "stall_threshold_seconds")
        ),
    }


def _capacity_member(delegation_id: str) -> Dict[str, Any]:
    """True byte upper bound for the canonical mandatory member shape."""
    item = _mandatory_member(delegation_id, status="completed")
    item.update({
        # JSON ``false`` is one byte longer than ``true``.
        "detail_lost": False, "error_present": False, "diagnostic_present": False,
        "schema_verdict_present": False, "schema_valid": False,
        "schema_error_count": _MAX_OUTCOME_COUNT, "schema_retries": _MAX_OUTCOME_COUNT,
        "timed_out": False, "stalled": False,
    })
    return item


def _identifier_fits(value: Any, byte_limit: int) -> bool:
    return isinstance(value, str) and bool(value) and len(value.encode("utf-8")) <= byte_limit


def _minimal_envelope_size(work_id: str, generation: int, member_ids: List[str]) -> int:
    envelope = {
        "type": "async_delegation_work_closeout",
        "work_id": work_id,
        "generation": generation,
        "delivery_id": _delivery_id(work_id, generation),
        "members": [_capacity_member(member_id) for member_id in sorted(member_ids)],
    }
    return len(_json_bytes(envelope))


def _group_row(conn: sqlite3.Connection, work_id: str):
    return conn.execute(
        "SELECT * FROM async_delegation_work_groups WHERE work_id=?", (work_id,)
    ).fetchone()


def register_work_group_member(
    *,
    work_id: str,
    owner_turn_id: str,
    delegation_id: str,
    generation: int = 0,
    routing: Optional[Dict[str, Any]] = None,
    task: Optional[Dict[str, Any]] = None,
    dispatched_at: Optional[float] = None,
    aggregate_char_budget: Optional[int] = None,
    feature_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Atomically create an open group (if gated on) and register a member.

    Additional members require the exact open owner turn and generation.
    This direct ledger API is intentionally not wired into dispatch in Stage 1.
    """
    if not (
        _identifier_fits(work_id, _MAX_WORK_ID_BYTES)
        and _identifier_fits(owner_turn_id, _MAX_OWNER_TURN_ID_BYTES)
        and _identifier_fits(delegation_id, _MAX_DELEGATION_ID_BYTES)
    ):
        return False
    routing = dict(routing or {})
    task = dict(task or {})
    now = time.time()
    dispatched = float(dispatched_at or now)
    pid, started = _process_identity()
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        group = _group_row(conn, work_id)
        created_group = False
        if group is None:
            if generation != 0 or not task_scoped_closeout_enabled(feature_config):
                return False
            conn.execute(
                """INSERT INTO async_delegation_work_groups
                   (work_id, origin_session, origin_ui_session_id,
                    origin_session_id, parent_session_id, routing_json,
                    owner_turn_id, owner_pid, owner_started_at, state,
                    generation, aggregate_char_budget, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', 0, ?, ?, ?)""",
                (
                    work_id, routing.get("origin_session", ""),
                    routing.get("origin_ui_session_id", ""),
                    routing.get("origin_session_id", ""),
                    routing.get("parent_session_id"),
                    json.dumps(routing, sort_keys=True, separators=(",", ":")),
                    owner_turn_id, pid, started,
                    _bounded_budget(aggregate_char_budget), now, now,
                ),
            )
            created_group = True
        elif not (
            group["state"] == "open"
            and group["owner_turn_id"] == owner_turn_id
            and int(group["generation"]) == generation
        ):
            return False
        budget = _bounded_budget(
            aggregate_char_budget if group is None else group["aggregate_char_budget"]
        )
        existing_ids = [
            str(row[0])
            for row in conn.execute(
                "SELECT delegation_id FROM async_delegations "
                "WHERE origin_work_id=? AND work_generation=?",
                (work_id, generation),
            ).fetchall()
        ]
        if _minimal_envelope_size(
            work_id, generation, existing_ids + [delegation_id]
        ) > budget:
            if created_group:
                conn.execute(
                    "DELETE FROM async_delegation_work_groups WHERE work_id=?",
                    (work_id,),
                )
            return False
        task_json = json.dumps(task, sort_keys=True, separators=(",", ":"))
        cur = conn.execute(
            """INSERT OR IGNORE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, origin_session_id,
                origin_work_id, work_generation)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?)""",
            (
                delegation_id, routing.get("origin_session", ""),
                routing.get("origin_ui_session_id", ""),
                routing.get("parent_session_id"), dispatched, now, pid, started,
                task_json, routing.get("origin_session_id", ""), work_id, generation,
            ),
        )
        if cur.rowcount:
            conn.execute(
                "UPDATE async_delegation_work_groups SET updated_at=? WHERE work_id=?",
                (now, work_id),
            )
        elif created_group:
            conn.execute(
                "DELETE FROM async_delegation_work_groups WHERE work_id=?",
                (work_id,),
            )
        return cur.rowcount == 1


def _unregister_unsubmitted_work_group_member(delegation_id: str) -> None:
    """Undo registration when executor submission never accepted the child."""
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT origin_work_id, work_generation FROM async_delegations "
            "WHERE delegation_id=? AND state='running'",
            (delegation_id,),
        ).fetchone()
        if row is None:
            return
        work_id = str(row["origin_work_id"] or "")
        generation = int(row["work_generation"] or 0)
        conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))
        if work_id:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM async_delegations WHERE origin_work_id=? "
                "AND work_generation=?",
                (work_id, generation),
            ).fetchone()[0]
            if not remaining:
                conn.execute(
                    "DELETE FROM async_delegation_work_groups WHERE work_id=? "
                    "AND state='open' AND generation=?",
                    (work_id, generation),
                )


def seal_work_group(work_id: str, owner_turn_id: str) -> bool:
    """Seal membership only for the exact owning turn."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegation_work_groups
               SET state='sealed', sealed_at=?, updated_at=?
               WHERE work_id=? AND state='open' AND owner_turn_id=?""",
            (now, now, work_id, owner_turn_id),
        )
        return cur.rowcount == 1


def seal_work_group_result(
    work_id: str, owner_turn_id: str, *, diagnostics: Optional[str] = None
) -> Dict[str, Any]:
    """Seal an owning turn and return an explicit, fail-closed diagnosis.

    The boolean helper remains for compatibility, but turn finalization must
    distinguish an idempotent already-sealed group from a missing group or an
    owner mismatch.  Otherwise a bad identity silently becomes an eternal
    "waiting" turn.
    """
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = _group_row(conn, work_id)
        if row is None:
            return {"ok": False, "code": "work_group_missing", "work_id": work_id}
        if row["state"] != "open":
            if row["state"] in {"sealed", "closing", "closed"}:
                return {"ok": True, "code": "already_sealed", "state": row["state"]}
            return {"ok": False, "code": "invalid_group_state", "state": row["state"]}
        if str(row["owner_turn_id"] or "") != str(owner_turn_id or ""):
            return {
                "ok": False,
                "code": "owner_turn_mismatch",
                "expected_owner_turn_id": str(row["owner_turn_id"] or ""),
                "actual_owner_turn_id": str(owner_turn_id or ""),
            }
        cur = conn.execute(
            """UPDATE async_delegation_work_groups
               SET state='sealed', sealed_at=?, updated_at=?,
                   terminal_diagnostics=COALESCE(?, terminal_diagnostics)
               WHERE work_id=? AND state='open' AND owner_turn_id=?""",
            (now, now, diagnostics, work_id, owner_turn_id),
        )
        if cur.rowcount != 1:
            return {"ok": False, "code": "seal_cas_failed"}
        return {"ok": True, "code": "sealed", "state": "sealed"}


def persist_group_member_completion(
    delegation_id: str, event: Dict[str, Any], result: Dict[str, Any]
) -> bool:
    """Persist a terminal member and report duplicate-safe sealed readiness.

    Stage 1 deliberately does not publish or suppress the legacy completion
    event. Integration will choose which persistence helper to call later.
    """
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        member = conn.execute(
            "SELECT origin_work_id, work_generation FROM async_delegations "
            "WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()
        if member is None or not member["origin_work_id"]:
            return False
        conn.execute(
            """UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
               event_json=?, result_json=?, delivery_state='pending'
               WHERE delegation_id=?""",
            (
                event.get("status", result.get("status", "completed")),
                event.get("completed_at", now), now,
                json.dumps(event, sort_keys=True), json.dumps(result, sort_keys=True),
                delegation_id,
            ),
        )
        return _group_ready(conn, member["origin_work_id"], int(member["work_generation"]))


def _group_ready(conn: sqlite3.Connection, work_id: str, generation: int) -> bool:
    group = conn.execute(
        "SELECT state, generation, terminal_diagnostics FROM async_delegation_work_groups "
        "WHERE work_id=?",
        (work_id,),
    ).fetchone()
    if group is None or group[0] != "sealed" or int(group[1]) != generation:
        return False
    counts = conn.execute(
        """SELECT COUNT(*),
                  SUM(CASE WHEN state IN ('running','finalizing') THEN 1 ELSE 0 END),
                  SUM(CASE WHEN state NOT IN ('running','finalizing')
                            AND event_json IS NULL AND result_json IS NULL THEN 1 ELSE 0 END)
           FROM async_delegations WHERE origin_work_id=? AND work_generation=?""",
        (work_id, generation),
    ).fetchone()
    total, live, missing = (int(counts[0]), int(counts[1] or 0), int(counts[2] or 0))
    return total > 0 and live == 0 and (missing == 0 or bool(group[2]))


def group_is_ready(work_id: str) -> bool:
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT generation FROM async_delegation_work_groups WHERE work_id=?",
            (work_id,),
        ).fetchone()
        return bool(row and _group_ready(conn, work_id, int(row[0])))


_PRESERVED_RESULT_KEYS = (
    "status", "timeout_seconds", "timed_out_after_seconds",
    "timeout_phase", "stalled_after_quiet_seconds", "stall_threshold_seconds",
    "stall_phase", "stall_grace_seconds", "exit_reason", "schema_valid",
    "schema_retries", "detail_lost", "diagnostic", "api_calls",
    "duration_seconds", "model",
)


def _compact_scalar(value: Any, limit: int = 160) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    else:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _schema_metadata(result: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    verdict = result.get(
        "schema_verdict",
        result.get("verdict", event.get("schema_verdict", event.get("verdict"))),
    )
    valid = result.get(
        "schema_valid",
        result.get("valid", event.get("schema_valid", event.get("valid"))),
    )
    errors = result.get("schema_errors", event.get("schema_errors"))
    metadata: Dict[str, Any] = {}
    if valid is not None:
        metadata["schema_valid"] = bool(valid)
    if errors is not None:
        metadata["schema_error_count"] = len(errors) if isinstance(errors, list) else 1
    if verdict is not None:
        if isinstance(verdict, dict):
            compact = {}
            for key in ("valid", "status", "verdict", "reason", "error_count", "retries"):
                if key in verdict:
                    compact[key] = _compact_scalar(verdict[key], 80)
            metadata["schema_verdict"] = compact or "detail_truncated"
        else:
            metadata["schema_verdict"] = _compact_scalar(verdict, 80)
    return metadata


def _build_group_envelope(
    work_id: str, generation: int, delivery_id: str, budget: int, rows: List[sqlite3.Row]
) -> Dict[str, Any]:
    members: List[Dict[str, Any]] = []
    mandatory_by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task = json.loads(row["task_json"] or "{}")
        event = json.loads(row["event_json"] or "{}")
        result = json.loads(row["result_json"] or "{}")
        status = result.get("status", event.get("status", row["state"] or "unknown"))
        delegation_id = str(row["delegation_id"])
        item = _mandatory_member(
            delegation_id, status=status, result=result, event=event
        )
        mandatory_by_id[delegation_id] = dict(item)
        item["detail_truncated"] = False
        item["dispatch_index"] = int(task.get("dispatch_index", task.get("task_index", 0)) or 0)
        item["task_index"] = int(task.get("task_index", 0) or 0)
        for key in _PRESERVED_RESULT_KEYS:
            if key in {"status", "detail_lost"}:
                continue
            if key in result:
                item[key] = _compact_scalar(result[key])
            elif key in event:
                item[key] = _compact_scalar(event[key])
        item.update(_schema_metadata(result, event))
        errors = result.get("schema_errors", event.get("schema_errors"))
        if errors is not None and "schema_error_count" not in item:
            item["schema_error_count"] = len(errors) if isinstance(errors, list) else 1
        for key, value in (
            ("goal", task.get("goal", event.get("goal"))),
            ("error", result.get("error", event.get("error"))),
        ):
            if value not in (None, ""):
                item[key] = _compact_scalar(value, 240)
        summary = result.get("summary", event.get("summary"))
        if summary is not None:
            item["summary"] = _compact_scalar(summary, 480)
        members.append(item)
    members.sort(key=lambda item: (item["dispatch_index"], item["task_index"], item["delegation_id"]))
    envelope = {
        "type": "async_delegation_work_closeout",
        "work_id": work_id,
        "generation": generation,
        "delivery_id": delivery_id,
        "members": members,
    }
    # Fail down deterministically to the registration-tested mandatory form.
    if len(_json_bytes(envelope)) > budget:
        optional_order = (
            "summary", "goal", "error", "schema_verdict", "model",
            "duration_seconds", "api_calls", "exit_reason", "timeout_phase",
            "stall_phase", "dispatch_index", "task_index",
        )
        for key in optional_order:
            for item in reversed(members):
                item.pop(key, None)
            if len(_json_bytes(envelope)) <= budget:
                break
    if len(_json_bytes(envelope)) > budget:
        # This is the exact representation registration sized. Rebuild it rather
        # than pruning rich data in place, so no untrusted scalar can leak into
        # the mandatory envelope.
        envelope["members"] = [
            mandatory_by_id[item["delegation_id"]] for item in members
        ]
    if len(_json_bytes(envelope)) > budget:
        # Only corrupt/legacy rows can reach this state: admitted Stage-1 rows
        # were checked against the shape above. Return an honest bounded marker
        # instead of crashing the claimant or emitting oversized bytes.
        envelope = {
            "type": "async_delegation_work_closeout_tombstone",
            "work_ref": hashlib.sha256(work_id.encode("utf-8")).hexdigest()[:32],
            "generation": generation,
            "delivery_id": delivery_id,
            "member_count": len(rows),
            "member_identities_lost": True,
            "detail_lost": True,
        }
    return envelope


def claim_ready_work_group(work_id: str, consumer: str) -> Optional[Dict[str, Any]]:
    """Atomically claim one sealed-ready generation and persist its envelope."""
    now = time.time()
    claim = f"{consumer}:{os.getpid()}:{uuid.uuid4().hex}"
    claim_pid, claim_started = _process_identity()
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        group = _group_row(conn, work_id)
        if group is None or not _group_ready(conn, work_id, int(group["generation"])):
            return None
        generation = int(group["generation"])
        rows = conn.execute(
            """SELECT * FROM async_delegations
               WHERE origin_work_id=? AND work_generation=?
               ORDER BY dispatched_at, delegation_id""",
            (work_id, generation),
        ).fetchall()
        delivery_id = _delivery_id(work_id, generation)
        envelope = _build_group_envelope(
            work_id, generation, delivery_id,
            _bounded_budget(group["aggregate_char_budget"]), rows,
        )
        payload_bytes = _json_bytes(envelope)
        if len(payload_bytes) > _bounded_budget(group["aggregate_char_budget"]):
            return None
        payload = payload_bytes.decode("utf-8")
        cur = conn.execute(
            """UPDATE async_delegation_work_groups
               SET state='closing', closeout_delivery_id=?,
                   closeout_payload_json=?, closeout_claim=?,
                   closeout_claimed_at=?, closeout_owner_pid=?,
                   closeout_owner_started_at=?, updated_at=?
               WHERE work_id=? AND state='sealed' AND generation=?
                 AND closeout_payload_json IS NULL""",
            (
                delivery_id, payload, claim, now, claim_pid, claim_started,
                now, work_id, generation,
            ),
        )
        if cur.rowcount != 1:
            return None
        return {
            "envelope": envelope,
            "claim_id": claim,
            "routing": {
                "session_key": group["origin_session"],
                "origin_ui_session_id": group["origin_ui_session_id"],
                "origin_session_id": group["origin_session_id"],
                "parent_session_id": group["parent_session_id"],
            },
        }


def _aggregate_ready_event(claimed: Dict[str, Any]) -> Dict[str, Any]:
    """Build the one typed queue item used by live and recovery producers."""
    envelope = dict(claimed["envelope"])
    routing = dict(claimed.get("routing") or {})
    return {
        "type": "async_delegation_work_closeout",
        "delivery_id": envelope["delivery_id"],
        "origin_work_id": envelope["work_id"],
        "work_generation": envelope["generation"],
        "claim_id": claimed["claim_id"],
        "envelope": envelope,
        # Internal-only provenance for profile-scoped ledger mutations.  This
        # is injected by the ledger itself from the active runtime scope, never
        # accepted from delegation/user payloads.
        "_ledger_profile_home": str(get_hermes_home().resolve()),
        **routing,
    }


def _enqueue_claimed_work_group(
    claimed: Dict[str, Any], *, target_queue: Any = None
) -> Optional[Dict[str, Any]]:
    """Publish a claimed durable envelope; release ownership on queue failure."""
    event = _aggregate_ready_event(claimed)
    delivery_id = event["delivery_id"]
    delivery_key = (str(event["_ledger_profile_home"]), str(delivery_id))
    with _aggregate_enqueue_lock:
        if delivery_key in _aggregate_enqueued_delivery_ids:
            return None
        _aggregate_enqueued_delivery_ids.add(delivery_key)
    try:
        if target_queue is None:
            from tools.process_registry import process_registry

            target_queue = process_registry.completion_queue
        target_queue.put(event)
    except Exception:
        with _aggregate_enqueue_lock:
            _aggregate_enqueued_delivery_ids.discard(delivery_key)
        release_work_group_claim(event["origin_work_id"], event["claim_id"])
        raise
    return event


def claim_and_enqueue_ready_work_group(
    work_id: str, *, consumer: str = "async-delegation-producer"
) -> Optional[Dict[str, Any]]:
    """Atomically claim a sealed-ready group and enqueue its aggregate once."""
    claimed = claim_ready_work_group(work_id, consumer)
    return _enqueue_claimed_work_group(claimed) if claimed is not None else None


def seal_and_enqueue_work_group(
    work_id: str, owner_turn_id: str, *, consumer: str = "turn-seal",
    diagnostics: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Later turn gates call this to seal membership and publish if ready.

    Duplicate seal callers are harmless: a group that is already sealed may
    still be claimed, while a closing group has already won that atomic claim.
    """
    sealed = seal_work_group_result(
        work_id, owner_turn_id, diagnostics=diagnostics
    )
    if not sealed["ok"]:
        return {"type": "async_delegation_work_seal_error", **sealed}
    return claim_and_enqueue_ready_work_group(work_id, consumer=consumer)


def release_work_group_claim(work_id: str, claim_id: str) -> bool:
    """Release scheduling ownership without deleting the durable envelope."""
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegation_work_groups
               SET closeout_claim=NULL, closeout_claimed_at=NULL,
                   closeout_owner_pid=NULL, closeout_owner_started_at=NULL,
                   updated_at=?
               WHERE work_id=? AND state='closing' AND closeout_claim=?
                 AND closeout_turn_id IS NULL""",
            (time.time(), work_id, claim_id),
        )
        return cur.rowcount == 1


def release_enqueued_work_group_event(event: Dict[str, Any]) -> bool:
    """Release a failed aggregate injection so the same envelope can retry.

    The gateway calls this only after the synthetic turn unwinds. Clearing the
    exact bound turn prevents a transient failure from stranding a closing
    group while the gateway PID remains healthy.
    """
    work_id = str(event.get("origin_work_id") or "")
    delivery_id = str(event.get("delivery_id") or "")
    claim_id = str(event.get("claim_id") or "")
    if not (work_id and delivery_id and claim_id):
        return False
    delivery_key = (
        str(event.get("_ledger_profile_home") or get_hermes_home().resolve()),
        delivery_id,
    )
    with _aggregate_enqueue_lock:
        _aggregate_enqueued_delivery_ids.discard(delivery_key)
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE async_delegation_work_groups
               SET closeout_turn_id=NULL, updated_at=?
               WHERE work_id=? AND state='closing'
                 AND closeout_delivery_id=? AND closeout_claim=?""",
            (now, work_id, delivery_id, claim_id),
        )
    return release_work_group_claim(work_id, claim_id)


def reclaim_stale_work_group_claim(work_id: str, consumer: str) -> Optional[Dict[str, Any]]:
    """Take a stale/unowned closing claim while retaining the same envelope."""
    now = time.time()
    claim = f"{consumer}:{os.getpid()}:{uuid.uuid4().hex}"
    claim_pid, claim_started = _process_identity()
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = _group_row(conn, work_id)
        if row is None or row["state"] != "closing" or not row["closeout_payload_json"]:
            return None
        claimed_at = row["closeout_claimed_at"]
        bound_live = bool(
            row["closeout_turn_id"]
            and _process_identity_is_live(
                row["closeout_owner_pid"], row["closeout_owner_started_at"]
            )
        )
        if bound_live:
            return None
        was_bound = bool(row["closeout_turn_id"])
        unbound_owner_known = row["closeout_owner_pid"] is not None
        unbound_owner_live = bool(
            unbound_owner_known
            and _process_identity_is_live(
                row["closeout_owner_pid"], row["closeout_owner_started_at"]
            )
        )
        if not was_bound and row["closeout_claim"]:
            if unbound_owner_live:
                return None
            if (
                not unbound_owner_known
                and claimed_at
                and claimed_at >= now - _GROUP_CLAIM_STALE_SECONDS
            ):
                return None
        old_claim = row["closeout_claim"]
        old_turn = row["closeout_turn_id"]
        old_delivery = row["closeout_delivery_id"]
        old_generation = int(row["generation"])
        old_owner_pid = row["closeout_owner_pid"]
        old_owner_started = row["closeout_owner_started_at"]
        cur = conn.execute(
            """UPDATE async_delegation_work_groups
               SET closeout_claim=?, closeout_claimed_at=?, closeout_turn_id=NULL,
                   closeout_owner_pid=?, closeout_owner_started_at=?, updated_at=?
               WHERE work_id=? AND state='closing'
                 AND generation=? AND closeout_delivery_id IS ?
                 AND closeout_claim IS ? AND closeout_turn_id IS ?
                 AND closeout_owner_pid IS ?
                 AND closeout_owner_started_at IS ?""",
            (
                claim, now, claim_pid, claim_started, now,
                work_id, old_generation, old_delivery, old_claim, old_turn,
                old_owner_pid, old_owner_started,
            ),
        )
        if cur.rowcount != 1:
            return None
        return {
            "envelope": json.loads(row["closeout_payload_json"]),
            "claim_id": claim,
            "routing": {
                "session_key": row["origin_session"],
                "origin_ui_session_id": row["origin_ui_session_id"],
                "origin_session_id": row["origin_session_id"],
                "parent_session_id": row["parent_session_id"],
            },
        }


def release_bound_work_group_closeout(
    work_id: str, generation: int, delivery_id: str, claim_id: str,
    closeout_turn_id: str,
) -> bool:
    """Release only the exact bound turn, retaining its durable envelope.

    A closeout turn calls this from its outer ``finally`` when it did not
    durably close the generation or reopen it with replacement work. Rotating
    away from the old claim fences the failed actor even though its process is
    still alive; recovery can then claim and enqueue the unchanged envelope.
    """
    pid, started = _process_identity()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegation_work_groups
               SET closeout_claim=NULL, closeout_claimed_at=NULL,
                   closeout_turn_id=NULL,
                   closeout_owner_pid=NULL, closeout_owner_started_at=NULL,
                   updated_at=?
               WHERE work_id=? AND state='closing' AND generation=?
                 AND closeout_delivery_id=? AND closeout_claim=?
                 AND closeout_turn_id=? AND closeout_owner_pid=?
                 AND (closeout_owner_started_at IS ?
                      OR closeout_owner_started_at=?)""",
            (
                time.time(), work_id, generation, delivery_id, claim_id,
                closeout_turn_id, pid, started, started,
            ),
        )
    if cur.rowcount == 1:
        with _aggregate_enqueue_lock:
            _aggregate_enqueued_delivery_ids.discard(
                (str(get_hermes_home().resolve()), delivery_id)
            )
        return True
    return False


def bind_work_group_closeout_turn(
    work_id: str, delivery_id: str, claim_id: str, closeout_turn_id: str
) -> bool:
    pid, started = _process_identity()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegation_work_groups SET closeout_turn_id=?,
                      closeout_owner_pid=?, closeout_owner_started_at=?, updated_at=?
               WHERE work_id=? AND state='closing' AND closeout_delivery_id=?
                 AND closeout_claim=? AND closeout_turn_id IS NULL""",
            (closeout_turn_id, pid, started, time.time(), work_id, delivery_id, claim_id),
        )
        return cur.rowcount == 1


def renew_work_group_claim(
    work_id: str, generation: int, delivery_id: str, claim_id: str,
    closeout_turn_id: str,
) -> bool:
    """Heartbeat only the exact live bound closeout identity."""
    pid, started = _process_identity()
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegation_work_groups
               SET closeout_claimed_at=?, updated_at=?
               WHERE work_id=? AND state='closing' AND generation=?
                 AND closeout_delivery_id=? AND closeout_claim=?
                 AND closeout_turn_id=? AND closeout_owner_pid=?
                 AND (closeout_owner_started_at IS ? OR closeout_owner_started_at=?)""",
            (now, now, work_id, generation, delivery_id, claim_id,
             closeout_turn_id, pid, started, started),
        )
        return cur.rowcount == 1


def reopen_work_group_with_member(
    *, work_id: str, generation: int, delivery_id: str, claim_id: str,
    closeout_turn_id: str,
    delegation_id: str, task: Optional[Dict[str, Any]] = None,
    dispatched_at: Optional[float] = None,
) -> bool:
    """Reopen closing N as open N+1 with its first replacement atomically."""
    if not (
        _identifier_fits(work_id, _MAX_WORK_ID_BYTES)
        and _identifier_fits(closeout_turn_id, _MAX_OWNER_TURN_ID_BYTES)
        and _identifier_fits(delegation_id, _MAX_DELEGATION_ID_BYTES)
    ):
        return False
    now = time.time()
    pid, started = _process_identity()
    task_json = json.dumps(task or {}, sort_keys=True, separators=(",", ":"))
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        group = _group_row(conn, work_id)
        if group is None or not (
            group["state"] == "closing"
            and int(group["generation"]) == generation
            and group["closeout_delivery_id"] == delivery_id
            and group["closeout_claim"] == claim_id
            and group["closeout_turn_id"] == closeout_turn_id
        ):
            return False
        if _minimal_envelope_size(
            work_id, generation + 1, [delegation_id]
        ) > _bounded_budget(group["aggregate_char_budget"]):
            return False
        cur = conn.execute(
            """INSERT OR IGNORE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, owner_pid, owner_started_at, task_json,
                origin_session_id, origin_work_id, work_generation)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', ?, ?, ?, ?, ?, ?)""",
            (
                delegation_id, group["origin_session"], group["origin_ui_session_id"],
                group["parent_session_id"], dispatched_at or now, now, pid, started,
                task_json, group["origin_session_id"], work_id, generation + 1,
            ),
        )
        if cur.rowcount != 1:
            return False
        conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=?, updated_at=?
               WHERE origin_work_id=? AND work_generation=?""",
            (now, now, work_id, generation),
        )
        conn.execute(
            """UPDATE async_delegation_work_groups
               SET state='open', generation=?, owner_turn_id=?, owner_pid=?,
                   owner_started_at=?, closeout_delivery_id=NULL,
                   closeout_payload_json=NULL, closeout_claim=NULL,
                   closeout_claimed_at=NULL, closeout_turn_id=NULL,
                   closeout_owner_pid=NULL, closeout_owner_started_at=NULL,
                   sealed_at=NULL, updated_at=? WHERE work_id=?""",
            (generation + 1, closeout_turn_id, pid, started, now, work_id),
        )
    with _aggregate_enqueue_lock:
        _aggregate_enqueued_delivery_ids.discard(
            (str(get_hermes_home().resolve()), delivery_id)
        )
    return True


def close_work_group(
    work_id: str, generation: int, delivery_id: str, claim_id: str,
    closeout_turn_id: str,
    *, disposition: str = "success", diagnostics: Optional[str] = None,
) -> bool:
    """Commit closeout only after the caller confirms transcript persistence."""
    if disposition not in {"success", "blocked", "failed", "cancelled", "dropped"}:
        return False
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """UPDATE async_delegation_work_groups
               SET state='closed', terminal_disposition=?, terminal_diagnostics=?,
                   closed_at=?, updated_at=?, closeout_claim=NULL,
                   closeout_claimed_at=NULL, closeout_owner_pid=NULL,
                   closeout_owner_started_at=NULL
               WHERE work_id=? AND state='closing' AND generation=?
                 AND closeout_delivery_id=? AND closeout_claim=?
                 AND closeout_turn_id=?""",
            (disposition, diagnostics, now, now, work_id, generation,
             delivery_id, claim_id, closeout_turn_id),
        )
        if cur.rowcount != 1:
            return False
        conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=?, updated_at=?
               WHERE origin_work_id=? AND work_generation=?""",
            (now, now, work_id, generation),
        )
    with _aggregate_enqueue_lock:
        _aggregate_enqueued_delivery_ids.discard(
            (str(get_hermes_home().resolve()), delivery_id)
        )
    return True


def close_work_groups_for_session(
    *, origin_session: str = "", origin_ui_session_id: str = "",
    parent_session_id: str = "", disposition: str = "cancelled",
    diagnostics: str = "session boundary",
) -> int:
    """Apply an explicit terminal session-boundary disposition."""
    if disposition not in {"cancelled", "dropped"}:
        return 0
    clauses, values = [], []
    for column, value in (
        ("origin_session", origin_session),
        ("origin_ui_session_id", origin_ui_session_id),
        ("parent_session_id", parent_session_id),
    ):
        if value:
            clauses.append(f"{column}=?")
            values.append(value)
    if not clauses:
        return 0
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            "UPDATE async_delegation_work_groups SET state='closed', "
            "terminal_disposition=?, terminal_diagnostics=?, closed_at=?, updated_at=?, "
            "closeout_claim=NULL, closeout_claimed_at=NULL, closeout_turn_id=NULL, "
            "closeout_owner_pid=NULL, closeout_owner_started_at=NULL "
            "WHERE work_id<>'' AND state IN ('open','sealed','closing') AND ("
            + " OR ".join(clauses) + ")",
            (disposition, diagnostics, now, now, *values),
        )
        return cur.rowcount


def _reveal_closed_closeout_provisionals(conn: sqlite3.Connection) -> int:
    """Reveal durable final rows left hidden by a crash after group close."""
    try:
        rows = conn.execute(
            "SELECT id, display_metadata FROM messages "
            "WHERE display_kind='delegation_closeout_provisional' ORDER BY id"
        ).fetchall()
    except sqlite3.OperationalError:
        # The standalone delegation ledger tests intentionally create no
        # transcript schema.
        return 0
    revealed = 0
    revealed_deliveries: set[tuple[str, str]] = set()
    for row in rows:
        try:
            metadata = json.loads(row["display_metadata"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        work_id = str(metadata.get("work_id") or "")
        delivery_id = str(metadata.get("delivery_id") or "")
        if not (work_id and delivery_id):
            continue
        group = conn.execute(
            "SELECT state, closeout_delivery_id FROM async_delegation_work_groups "
            "WHERE work_id=?",
            (work_id,),
        ).fetchone()
        if (
            group is None
            or group["state"] != "closed"
            or str(group["closeout_delivery_id"] or "") != delivery_id
        ):
            continue
        identity = (work_id, delivery_id)
        if identity in revealed_deliveries:
            # Legacy crash/replay races may already have appended duplicates.
            # Keep only the oldest row canonical and presentation-hidden.
            continue
        cur = conn.execute(
            "UPDATE messages SET display_kind=NULL, display_metadata=NULL "
            "WHERE id=? AND display_kind='delegation_closeout_provisional'",
            (row["id"],),
        )
        revealed += cur.rowcount
        if cur.rowcount:
            revealed_deliveries.add(identity)
    return revealed


def find_closeout_provisional(
    work_id: str, delivery_id: str,
) -> Optional[Dict[str, Any]]:
    """Return the canonical durable provisional for one delivery identity."""
    if not (work_id and delivery_id):
        return None
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, content, display_metadata FROM messages "
                "WHERE display_kind='delegation_closeout_provisional' "
                "ORDER BY id"
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        for row in rows:
            try:
                metadata = json.loads(row["display_metadata"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if (
                isinstance(metadata, dict)
                and str(metadata.get("work_id") or "") == work_id
                and str(metadata.get("delivery_id") or "") == delivery_id
            ):
                return {"row_id": int(row["id"]), "content": row["content"]}
    return None


def reconcile_closed_closeout_provisionals() -> int:
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        return _reveal_closed_closeout_provisionals(conn)


def recover_work_groups() -> List[Dict[str, Any]]:
    """Recover ready sealed/closing work and diagnose dead open owners."""
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        def _pid_exists(_pid: int) -> bool:
            return False

        def get_process_start_time(_pid: int) -> Optional[int]:
            return None
    now = time.time()
    recovered: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        _reveal_closed_closeout_provisionals(conn)
        groups = conn.execute(
            "SELECT * FROM async_delegation_work_groups WHERE state IN ('open','sealed','closing')"
        ).fetchall()
        for group in groups:
            state = group["state"]
            if state == "open":
                pid, started = group["owner_pid"], group["owner_started_at"]
                live = bool(pid and _pid_exists(int(pid)))
                if live and started is not None:
                    live = get_process_start_time(int(pid)) == int(started)
                if not live:
                    diagnostic = "Owner process/turn cannot resume; membership sealed with outcome unknown."
                    conn.execute(
                        """UPDATE async_delegation_work_groups
                           SET state='sealed', sealed_at=?, updated_at=?,
                               terminal_diagnostics=? WHERE work_id=? AND state='open'""",
                        (now, now, diagnostic, group["work_id"]),
                    )
                    conn.execute(
                        """UPDATE async_delegations SET state='unknown', completed_at=?,
                                  updated_at=?, result_json=?
                           WHERE origin_work_id=? AND work_generation=?
                             AND state IN ('running','finalizing')""",
                        (
                            now, now, json.dumps({"status": "unknown", "error": diagnostic}),
                            group["work_id"], group["generation"],
                        ),
                    )
                    state = "sealed"
            if state == "closing" and group["closeout_payload_json"]:
                recovered.append({
                    "work_id": group["work_id"], "state": "closing",
                    "delivery_id": group["closeout_delivery_id"],
                    "envelope": json.loads(group["closeout_payload_json"]),
                    "claim_id": group["closeout_claim"],
                    "routing": {
                        "session_key": group["origin_session"],
                        "origin_ui_session_id": group["origin_ui_session_id"],
                        "origin_session_id": group["origin_session_id"],
                        "parent_session_id": group["parent_session_id"],
                    },
                })
            elif state == "sealed" and _group_ready(conn, group["work_id"], int(group["generation"])):
                recovered.append({"work_id": group["work_id"], "state": "sealed_ready"})
    return recovered


def recover_and_enqueue_work_groups(
    *, consumer: str = "async-delegation-recovery", target_queue: Any = None
) -> List[Dict[str, Any]]:
    """Publish recoverable envelopes through the same idempotent aggregate rail."""
    enqueued: List[Dict[str, Any]] = []
    for item in recover_work_groups():
        delivery_id = item.get("delivery_id")
        if delivery_id:
            delivery_key = (
                str(get_hermes_home().resolve()),
                str(delivery_id),
            )
            with _aggregate_enqueue_lock:
                if delivery_key in _aggregate_enqueued_delivery_ids:
                    continue
        claimed: Optional[Dict[str, Any]]
        if item["state"] == "sealed_ready":
            claimed = claim_ready_work_group(item["work_id"], consumer)
        else:
            # A recoverable closing row must be reclaimed through the same
            # PID+process-start fenced CAS regardless of whether the stale
            # actor left a claim id behind. The helper skips live bound or
            # live unbound owners, and rotates dead claims while clearing the
            # old closeout_turn_id before the replacement event is enqueued.
            claimed = reclaim_stale_work_group_claim(item["work_id"], consumer)
        if claimed is not None:
            event = _enqueue_claimed_work_group(claimed, target_queue=target_queue)
            if event is not None:
                enqueued.append(event)
    return enqueued
