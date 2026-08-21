"""Normalized durable authority for asynchronous delegation lifecycle state."""

from __future__ import annotations
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional
from hermes_state import apply_wal_with_fallback, ensure_state_schema

_ACTIVE_STATES = {"starting", "running", "finalizing", "interrupt_requested"}
_TERMINAL_DELIVERY_STATES = {"delivered", "consumed", "suppressed"}
_DELIVERY_STATES = _TERMINAL_DELIVERY_STATES | {"pending", "held_by_wait", "delivering"}
_TERMINAL_ATTEMPT_STATES = {"completed", "error", "interrupted", "timeout"}
_SCHEMA_READY: set[str] = set()
_MATERIALIZED_READY: set[str] = set()
_BOOTSTRAP_LOCK = threading.Lock()


def _json(raw: Any, default: Any) -> Any:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return value if isinstance(value, type(default)) else default


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _authority_is_tombstoned(spec_json: Any) -> bool:
    spec = _json(spec_json, {})
    authority = spec.get("authority")
    if authority is None:
        return False
    if not isinstance(authority, dict):
        return True
    state = authority.get("state")
    if not isinstance(state, dict):
        return True
    return bool(state.get("revoked")) or bool(state.get("cleaned"))


def _attempt_state(value: Any) -> str:
    state = str(value or "").lower()
    if state in _ACTIVE_STATES:
        return state
    if state in {"completed", "success"}:
        return "completed"
    if state in {"error", "failed", "budget_exhausted"}:
        return "error"
    if state in {"interrupted", "cancelled"}:
        return "interrupted"
    if state == "timeout":
        return "timeout"
    return "unknown"


def _external_authority_projection(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Remove canonical authority from a lifecycle snapshot.

    Repository storage remains the trusted resume authority.  Every ordinary
    snapshot is instead shaped into an audit-only view whose cleanup state is
    derived from the current durable attempt, not from the immutable logical
    spec captured when the child was admitted.
    """

    def delegation_authority_audit_view(raw):
        return {"outcome": {}, "state": {}}

    children = snapshot.get("children")
    if not isinstance(children, dict):
        return snapshot
    for child in children.values():
        if not isinstance(child, dict) or "authority" not in child:
            continue
        raw = child.pop("authority")
        if isinstance(raw, Mapping):
            audit = delegation_authority_audit_view(raw)
        else:
            audit = {"outcome": {}, "state": {}}
        status = str(child.get("status") or "unknown")
        terminal = status in _TERMINAL_ATTEMPT_STATES
        cleanup_failed = child.get("exit_reason") == "cleanup_error"
        raw_state = raw.get("state") if isinstance(raw, Mapping) else None
        raw_state = raw_state if isinstance(raw_state, Mapping) else {}
        effective_revoked = bool(raw_state.get("revoked")) or terminal
        effective_cleaned = bool(raw_state.get("cleaned")) or (
            terminal and not cleanup_failed
        )
        if cleanup_failed:
            cleanup_outcome = "failed"
        elif effective_cleaned:
            cleanup_outcome = "succeeded"
        elif effective_revoked:
            cleanup_outcome = "revoked_pending"
        else:
            cleanup_outcome = "pending"
        outcome = audit.get("outcome")
        outcome = dict(outcome) if isinstance(outcome, Mapping) else {}
        outcome.update(execution=status, cleanup=cleanup_outcome)
        audit["outcome"] = outcome
        audit["state"] = {
            "revoked": effective_revoked,
            "cleaned": effective_cleaned,
        }
        child["authority_audit"] = audit
    return snapshot


def _new_id(kind: str) -> str:
    return f"{kind}_{uuid.uuid4().hex}"


class DelegationRepository:
    """Own connections, migration, lifecycle CAS operations, and projection."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.db_path.resolve())
        for attempt in range(20):
            conn = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=10000")
            try:
                apply_wal_with_fallback(conn, db_label=str(self.db_path))
                conn.execute("PRAGMA foreign_keys=ON")
                if key not in _SCHEMA_READY or key not in _MATERIALIZED_READY:
                    with _BOOTSTRAP_LOCK:
                        if key not in _SCHEMA_READY:
                            ensure_state_schema(conn)
                            _SCHEMA_READY.add(key)
                        if key not in _MATERIALIZED_READY:
                            conn.execute("BEGIN IMMEDIATE")
                            try:
                                ids = conn.execute("SELECT delegation_id FROM async_delegations WHERE lifecycle_version=1").fetchall()
                                for row in ids:
                                    self._materialize(conn, str(row[0]))
                                conn.commit()
                                _MATERIALIZED_READY.add(key)
                            except BaseException:
                                conn.rollback()
                                raise
                return conn
            except sqlite3.OperationalError as exc:
                conn.close()
                if "locked" not in str(exc).lower() or attempt == 19:
                    raise
                time.sleep(0.02 * (attempt + 1))
        raise AssertionError("unreachable")

    @contextmanager
    def write_txn(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _materialize(self, conn: sqlite3.Connection, delegation_id: str) -> str:
        row = conn.execute("SELECT * FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone()
        if row is None:
            return "not_found"
        if int(row["lifecycle_version"] or 1) == 2:
            return "already_materialized"
        run_id = _new_id("run")
        delivery = str(row["delivery_state"] or "pending")
        if delivery not in _DELIVERY_STATES:
            delivery = "pending"
        if delivery == "pending" and row["delivery_claim"] is not None:
            delivery = "delivering"
        conn.execute(
            "INSERT INTO delegation_runs\n               (run_id,delegation_id,run_number,kind,created_at,completed_at,\n                event_json,result_json,delivery_state,delivery_claim,\n                delivery_claimed_at,delivery_attempts,delivered_at)\n               VALUES (?,?,1,'initial',?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                delegation_id,
                float(row["dispatched_at"]),
                row["completed_at"],
                row["event_json"],
                row["result_json"],
                delivery,
                row["delivery_claim"],
                row["delivery_claimed_at"],
                int(row["delivery_attempts"] or 0),
                row["delivered_at"],
            ),
        )
        roots = [v for v in _json(row["root_subagent_ids_json"], []) if isinstance(v, str)]
        children = _json(row["children_json"], {})
        logical_ids = list(dict.fromkeys(roots + list(children)))
        for logical_id in logical_ids:
            child = children.get(logical_id) if isinstance(children.get(logical_id), dict) else {}
            conn.execute(
                "INSERT INTO delegation_logical_subagents\n                   (logical_id,delegation_id,parent_logical_id,root_ordinal,\n                    spec_json,created_at,updated_at) VALUES (?,?,NULL,?,?,?,?)",
                (
                    logical_id,
                    delegation_id,
                    roots.index(logical_id) if logical_id in roots else None,
                    _dump(child),
                    float(row["dispatched_at"]),
                    float(row["updated_at"]),
                ),
            )
            state = _attempt_state(child.get("status"))
            active = state in _ACTIVE_STATES
            conn.execute(
                "INSERT INTO delegation_attempts\n                   (attempt_id,logical_id,run_id,attempt_number,physical_worker_id,\n                    state,owner_pid,owner_started_at,created_at,started_at,\n                    completed_at,updated_at,metadata_json,interrupt_reason,\n                    interrupt_requested_at)\n                   VALUES (?,?,?,1,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _new_id("attempt"),
                    logical_id,
                    run_id,
                    logical_id,
                    state,
                    row["owner_pid"] if active else None,
                    row["owner_started_at"] if active else None,
                    float(row["dispatched_at"]),
                    child.get("started_at"),
                    None if active else row["completed_at"] or row["updated_at"],
                    float(row["updated_at"]),
                    "{}",
                    child.get("interrupt_reason"),
                    row["updated_at"] if state == "interrupt_requested" else None,
                ),
            )
        known = set(logical_ids)
        for logical_id in logical_ids:
            child = children.get(logical_id) if isinstance(children.get(logical_id), dict) else {}
            parent = child.get("parent_id")
            if parent in known:
                conn.execute("UPDATE delegation_logical_subagents SET parent_logical_id=? WHERE logical_id=?", (parent, logical_id))
        conn.execute("UPDATE async_delegations SET lifecycle_version=2 WHERE delegation_id=? AND lifecycle_version=1", (delegation_id,))
        return "materialized"

    def register_initial_dispatch(
        self, record: Dict[str, Any], *, owner_pid: Optional[int] = None, owner_started_at: Optional[int] = None
    ) -> Dict[str, Any]:
        roots = [v for v in record.get("root_subagent_ids", []) if isinstance(v, str) and v]
        run_id = str(record.get("run_id") or _new_id("run"))
        supplied_mapping = record.get("attempt_ids_by_logical_id")
        if supplied_mapping is not None:
            if (
                not isinstance(supplied_mapping, dict)
                or set(supplied_mapping) != set(roots)
                or not all(
                    isinstance(value, str) and value
                    for value in supplied_mapping.values()
                )
                or len(set(supplied_mapping.values())) != len(roots)
            ):
                raise ValueError(
                    "attempt_ids_by_logical_id must contain one unique physical attempt ID per root"
                )
            attempt_ids = [supplied_mapping[root] for root in roots]
            physical_worker_ids = list(attempt_ids)
        else:
            supplied_attempts = record.get("attempt_ids")
            attempt_ids = (
                [str(value) for value in supplied_attempts]
                if isinstance(supplied_attempts, list)
                and len(supplied_attempts) == len(roots)
                else [_new_id("attempt") for _ in roots]
            )
            # Legacy/ordinary dispatch keeps its conversational worker identity.
            physical_worker_ids = list(roots)
        authority_by_logical_id = record.get("authority_by_logical_id")
        if authority_by_logical_id is not None and (
            not isinstance(authority_by_logical_id, dict)
            or set(authority_by_logical_id) != set(roots)
            or not all(isinstance(value, dict) for value in authority_by_logical_id.values())
        ):
            raise ValueError(
                "authority_by_logical_id must contain one immutable authority object per root"
            )
        now = time.time()
        dispatched = float(record.get("dispatched_at") or now)
        task = {key: record.get(key) for key in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch") if key in record}
        try:
            with self.write_txn() as conn:
                conn.execute(
                    "INSERT INTO async_delegations\n                       (delegation_id,origin_session,origin_ui_session_id,\n                        parent_session_id,state,dispatched_at,updated_at,\n                        delivery_state,delivery_attempts,task_json,lifecycle_version)\n                       VALUES (?,?,?,?,'running',?,?,'pending',0,?,2)",
                    (
                        record["delegation_id"],
                        record.get("session_key", ""),
                        record.get("origin_ui_session_id", ""),
                        record.get("parent_session_id"),
                        dispatched,
                        now,
                        _dump(task),
                    ),
                )
                conn.execute(
                    "INSERT INTO delegation_runs (run_id,delegation_id,run_number,kind,created_at) VALUES (?,?,1,'initial',?)",
                    (run_id, record["delegation_id"], dispatched),
                )
                raw_goals = task.get("goals")
                goals: List[Any] = raw_goals if isinstance(raw_goals, list) else []
                for ordinal, (logical_id, attempt_id, physical_worker_id) in enumerate(
                    zip(roots, attempt_ids, physical_worker_ids)
                ):
                    spec = dict(task)
                    if ordinal < len(goals):
                        spec["goal"] = goals[ordinal]
                    if authority_by_logical_id is not None:
                        spec["authority"] = dict(authority_by_logical_id[logical_id])
                    conn.execute(
                        "INSERT INTO delegation_logical_subagents\n                           (logical_id,delegation_id,root_ordinal,spec_json,created_at,updated_at)\n                           VALUES (?,?,?,?,?,?)",
                        (logical_id, record["delegation_id"], ordinal, _dump(spec), dispatched, now),
                    )
                    conn.execute(
                        "INSERT INTO delegation_attempts\n                           (attempt_id,logical_id,run_id,attempt_number,physical_worker_id,\n                            state,owner_pid,owner_started_at,created_at,updated_at,metadata_json)\n                           VALUES (?,?,?,1,?,'starting',?,?,?,?,'{}')",
                        (attempt_id, logical_id, run_id, physical_worker_id, owner_pid, owner_started_at, dispatched, now),
                    )
        except sqlite3.IntegrityError:
            conn = self._connect()
            try:
                exists = conn.execute(
                    "SELECT 1 FROM async_delegations WHERE delegation_id=?",
                    (record["delegation_id"],),
                ).fetchone()
            finally:
                conn.close()
            return {"status": "already_exists" if exists else "conflict"}
        attempts = [
            {"attempt_id": attempt_id, "logical_id": logical_id, "attempt_number": 1} for logical_id, attempt_id in zip(roots, attempt_ids)
        ]
        return {"status": "registered", "run_id": run_id, "attempts": attempts}

    def reserve_resumed_attempt(
        self,
        logical_id: str,
        *,
        physical_worker_id: Optional[str] = None,
        owner_pid: Optional[int] = None,
        owner_started_at: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attempt_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        attempt_id = attempt_id or _new_id("attempt")
        run_id = run_id or _new_id("run")
        now = time.time()
        with self.write_txn() as conn:
            logical = conn.execute(
                "SELECT delegation_id,spec_json FROM delegation_logical_subagents "
                "WHERE logical_id=?",
                (logical_id,),
            ).fetchone()
            if logical is None:
                return {"status": "not_found"}
            if _authority_is_tombstoned(logical["spec_json"]):
                return {"status": "authority_revoked"}
            active = conn.execute(
                "SELECT attempt_id FROM delegation_attempts WHERE logical_id=? AND state IN ('starting','running','finalizing','interrupt_requested')",
                (logical_id,),
            ).fetchone()
            if active is not None:
                return {"status": "already_running", "attempt_id": active[0]}
            delegation_id = str(logical[0])
            run_number = int(
                conn.execute(
                    "SELECT COALESCE(MAX(run_number),0)+1 FROM delegation_runs WHERE delegation_id=?", (delegation_id,)
                ).fetchone()[0]
            )
            attempt_number = int(
                conn.execute(
                    "SELECT COALESCE(MAX(attempt_number),0)+1 FROM delegation_attempts WHERE logical_id=?", (logical_id,)
                ).fetchone()[0]
            )
            conn.execute(
                "INSERT INTO delegation_runs (run_id,delegation_id,run_number,kind,created_at) VALUES (?,?,?,'resumed',?)",
                (run_id, delegation_id, run_number, now),
            )
            conn.execute(
                "INSERT INTO delegation_attempts\n                   (attempt_id,logical_id,run_id,attempt_number,physical_worker_id,\n                    state,owner_pid,owner_started_at,created_at,updated_at,metadata_json)\n                   VALUES (?,?,?,?,?,'starting',?,?,?,?,?)",
                (
                    attempt_id,
                    logical_id,
                    run_id,
                    attempt_number,
                    physical_worker_id or attempt_id,
                    owner_pid,
                    owner_started_at,
                    now,
                    now,
                    _dump(metadata or {}),
                ),
            )
        return {
            "status": "reserved",
            "delegation_id": delegation_id,
            "run_id": run_id,
            "run_number": run_number,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
        }

    def enqueue_steer(
        self,
        delegation_id: str,
        logical_id: str,
        session_key: str,
        message: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Append one ordered steer to the current authorized attempt."""
        now = time.time()
        mailbox_id = _new_id("steer")
        with self.write_txn() as conn:
            attempt = conn.execute(
                "SELECT a.attempt_id,a.run_id,a.state,a.attempt_number,l.spec_json "
                "FROM delegation_attempts a "
                "JOIN delegation_logical_subagents l ON l.logical_id=a.logical_id "
                "JOIN async_delegations d ON d.delegation_id=l.delegation_id "
                "WHERE d.delegation_id=? AND d.origin_session=? AND l.logical_id=? "
                "ORDER BY a.attempt_number DESC LIMIT 1",
                (delegation_id, session_key, logical_id),
            ).fetchone()
            if attempt is None:
                return {"status": "not_found"}
            if _authority_is_tombstoned(attempt["spec_json"]):
                return {"status": "authority_revoked"}
            if attempt["state"] not in _ACTIVE_STATES:
                return {
                    "status": "already_terminal",
                    "terminal_status": str(attempt["state"]),
                    "attempt_id": str(attempt["attempt_id"]),
                    "run_id": str(attempt["run_id"]),
                }
            sequence = int(
                conn.execute(
                    "SELECT COALESCE(MAX(sequence_number),0)+1 "
                    "FROM delegation_steer_mailbox WHERE attempt_id=?",
                    (attempt["attempt_id"],),
                ).fetchone()[0]
            )
            conn.execute(
                "INSERT INTO delegation_steer_mailbox "
                "(mailbox_id,delegation_id,logical_id,attempt_id,sequence_number,"
                "message,force,status,created_at) VALUES (?,?,?,?,?,?,?,'pending',?)",
                (
                    mailbox_id,
                    delegation_id,
                    logical_id,
                    attempt["attempt_id"],
                    sequence,
                    message,
                    1 if force else 0,
                    now,
                ),
            )
        return {
            "status": "accepted",
            "mailbox_id": mailbox_id,
            "attempt_id": str(attempt["attempt_id"]),
            "run_id": str(attempt["run_id"]),
            "attempt_state": str(attempt["state"]),
            "sequence_number": sequence,
            "force": bool(force),
        }

    def pending_steers(self, attempt_id: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT mailbox_id,message,sequence_number,force FROM delegation_steer_mailbox "
                "WHERE attempt_id=? AND status='pending' ORDER BY sequence_number",
                (attempt_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def claim_steer(self, attempt_id: str, mailbox_id: str) -> Dict[str, Any]:
        """Claim only the oldest unresolved item for one still-current attempt."""
        now = time.time()
        with self.write_txn() as conn:
            row = conn.execute(
                "SELECT m.*,a.state FROM delegation_steer_mailbox m "
                "JOIN delegation_attempts a ON a.attempt_id=m.attempt_id "
                "WHERE m.mailbox_id=? AND m.attempt_id=?",
                (mailbox_id, attempt_id),
            ).fetchone()
            if row is None:
                return {"status": "not_found"}
            if row["state"] not in _ACTIVE_STATES:
                conn.execute(
                    "UPDATE delegation_steer_mailbox SET status='too_late_after_completion',resolved_at=? "
                    "WHERE mailbox_id=? AND status IN ('pending','forwarding','forwarded')",
                    (now, mailbox_id),
                )
                return {"status": "too_late_after_completion"}
            if row["status"] != "pending":
                return {"status": str(row["status"])}
            earlier = conn.execute(
                "SELECT 1 FROM delegation_steer_mailbox WHERE attempt_id=? "
                "AND sequence_number<? AND status IN ('pending','forwarding') LIMIT 1",
                (attempt_id, row["sequence_number"]),
            ).fetchone()
            if earlier is not None:
                return {"status": "blocked"}
            changed = conn.execute(
                "UPDATE delegation_steer_mailbox SET status='forwarding' "
                "WHERE mailbox_id=? AND status='pending'",
                (mailbox_id,),
            ).rowcount
        return (
            {
                "status": "claimed",
                "message": str(row["message"]),
                "force": bool(row["force"]),
            }
            if changed == 1
            else {"status": "stale"}
        )

    def mark_steer_forwarded(self, mailbox_id: str) -> Dict[str, Any]:
        now = time.time()
        with self.write_txn() as conn:
            changed = conn.execute(
                "UPDATE delegation_steer_mailbox SET status='forwarded',forwarded_at=? "
                "WHERE mailbox_id=? AND status='forwarding'",
                (now, mailbox_id),
            ).rowcount
            row = conn.execute(
                "SELECT status FROM delegation_steer_mailbox WHERE mailbox_id=?",
                (mailbox_id,),
            ).fetchone()
        return {
            "status": "forwarded" if changed == 1 else str(row[0] if row else "not_found")
        }

    def resolve_steer(self, mailbox_id: str, outcome: str) -> Dict[str, Any]:
        if outcome not in {
            "injected",
            "superseded_by_interrupt",
            "too_late_after_completion",
            "foreground_wait",
            "force_background_failed",
        }:
            raise ValueError("invalid steer outcome")
        now = time.time()
        with self.write_txn() as conn:
            changed = conn.execute(
                "UPDATE delegation_steer_mailbox SET status=?,resolved_at=? "
                "WHERE mailbox_id=? AND status IN ('pending','forwarding','forwarded')",
                (outcome, now, mailbox_id),
            ).rowcount
            row = conn.execute(
                "SELECT status FROM delegation_steer_mailbox WHERE mailbox_id=?",
                (mailbox_id,),
            ).fetchone()
        return {"status": outcome if changed == 1 else str(row[0] if row else "not_found")}

    def inspect_steer(self, mailbox_id: str) -> Dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM delegation_steer_mailbox WHERE mailbox_id=?",
                (mailbox_id,),
            ).fetchone()
            return {"status": "not_found"} if row is None else dict(row)
        finally:
            conn.close()

    def transition_attempt(
        self,
        attempt_id: str,
        expected_states: Iterable[str],
        new_state: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        expected = set(expected_states)
        now = time.time()
        with self.write_txn() as conn:
            row = conn.execute("SELECT state,metadata_json FROM delegation_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                return {"status": "not_found"}
            if row[0] not in expected or row[0] not in _ACTIVE_STATES:
                return {"status": "stale", "state": row[0]}
            merged = _json(row[1], {})
            if metadata:
                merged.update(metadata)
            terminal_at = completed_at
            if new_state not in _ACTIVE_STATES and terminal_at is None:
                terminal_at = now
            changed = conn.execute(
                "UPDATE delegation_attempts SET state=?,metadata_json=?,\n                   started_at=COALESCE(?,started_at),completed_at=?,updated_at=?\n                   WHERE attempt_id=? AND state=?",
                (new_state, _dump(merged), started_at, terminal_at, now, attempt_id, row[0]),
            ).rowcount
            if changed == 1 and new_state not in _ACTIVE_STATES:
                mailbox_outcome = (
                    "superseded_by_interrupt"
                    if new_state in {"interrupted", "abandoned"}
                    else "too_late_after_completion"
                )
                conn.execute(
                    "UPDATE delegation_steer_mailbox SET status=?,resolved_at=? "
                    "WHERE attempt_id=? AND status IN ('pending','forwarding','forwarded')",
                    (mailbox_outcome, terminal_at or now, attempt_id),
                )
        return {"status": "updated" if changed == 1 else "stale"}

    def request_interrupt(self, attempt_id: str, reason: str = "") -> Dict[str, Any]:
        now = time.time()
        with self.write_txn() as conn:
            row = conn.execute(
                "SELECT a.state,a.interrupt_requested_at,l.spec_json "
                "FROM delegation_attempts a JOIN delegation_logical_subagents l "
                "ON l.logical_id=a.logical_id WHERE a.attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                return {"status": "not_found"}
            if _authority_is_tombstoned(row["spec_json"]):
                return {"status": "authority_revoked"}
            if row[0] == "interrupt_requested" or row[1] is not None:
                return {"status": "already_requested"}
            if row[0] not in _ACTIVE_STATES:
                return {"status": "already_terminal"}
            changed = conn.execute(
                "UPDATE delegation_attempts SET state='interrupt_requested',\n                   interrupt_reason=?,interrupt_requested_at=?,updated_at=?\n                   WHERE attempt_id=? AND state=? AND interrupt_requested_at IS NULL",
                (reason or None, now, now, attempt_id, row[0]),
            ).rowcount
            if changed == 1:
                conn.execute(
                    "UPDATE delegation_steer_mailbox "
                    "SET status='superseded_by_interrupt',resolved_at=? "
                    "WHERE attempt_id=? AND status='pending'",
                    (now, attempt_id),
                )
        return {"status": "interrupt_requested" if changed == 1 else "stale"}

    def tombstone_logical_authority(
        self,
        logical_id: str,
        *,
        revoked: bool = False,
        cleaned: bool = False,
    ) -> Dict[str, Any]:
        """Durably narrow protected authority without rewriting immutable fields."""

        if not revoked and not cleaned:
            return {"status": "unchanged"}
        now = time.time()
        with self.write_txn() as conn:
            row = conn.execute(
                "SELECT spec_json FROM delegation_logical_subagents WHERE logical_id=?",
                (logical_id,),
            ).fetchone()
            if row is None:
                return {"status": "not_found"}
            spec = _json(row["spec_json"], {})
            authority = spec.get("authority")
            if not isinstance(authority, dict):
                return {"status": "ordinary"}
            state = authority.get("state")
            if not isinstance(state, dict):
                return {"status": "malformed"}
            immutable_before = {
                key: value for key, value in authority.items() if key != "state"
            }
            authority_after = {
                **authority,
                "state": {
                    "revoked": bool(state.get("revoked")) or bool(revoked),
                    "cleaned": bool(state.get("cleaned")) or bool(cleaned),
                },
            }
            if {
                key: value for key, value in authority_after.items() if key != "state"
            } != immutable_before:
                raise RuntimeError("protected authority immutable snapshot changed")
            spec["authority"] = authority_after
            conn.execute(
                "UPDATE delegation_logical_subagents SET spec_json=?,updated_at=? "
                "WHERE logical_id=?",
                (_dump(spec), now, logical_id),
            )
        return {"status": "tombstoned"}

    def tombstone_delegation_authorities(
        self,
        delegation_id: str,
        *,
        revoked: bool = False,
        cleaned: bool = False,
    ) -> Dict[str, Any]:
        """Retain protected tombstones with the logical delegation records."""

        conn = self._connect()
        try:
            logical_ids = [
                str(row[0])
                for row in conn.execute(
                    "SELECT logical_id FROM delegation_logical_subagents "
                    "WHERE delegation_id=? ORDER BY logical_id",
                    (delegation_id,),
                ).fetchall()
            ]
        finally:
            conn.close()
        outcomes = {
            logical_id: self.tombstone_logical_authority(
                logical_id, revoked=revoked, cleaned=cleaned
            )["status"]
            for logical_id in logical_ids
        }
        return {"status": "tombstoned", "children": outcomes}

    def take_interrupt(self, attempt_id: str) -> Dict[str, Any]:
        now = time.time()
        with self.write_txn() as conn:
            row = conn.execute("SELECT interrupt_reason FROM delegation_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                return {"status": "not_found"}
            changed = conn.execute(
                "UPDATE delegation_attempts SET interrupt_taken_at=?,updated_at=?\n                   WHERE attempt_id=? AND state='interrupt_requested'\n                     AND interrupt_requested_at IS NOT NULL AND interrupt_taken_at IS NULL",
                (now, now, attempt_id),
            ).rowcount
        return {"status": "taken", "reason": str(row[0] or "")} if changed == 1 else {"status": "not_pending"}

    def rollback_interrupt_request(self, attempt_id: str) -> Dict[str, Any]:
        """Roll back an exact interrupt marker only if no worker consumed it."""
        now = time.time()
        with self.write_txn() as conn:
            changed = conn.execute(
                "UPDATE delegation_attempts SET state='running',interrupt_reason=NULL,\n                   interrupt_requested_at=NULL,interrupt_taken_at=NULL,updated_at=?\n                   WHERE attempt_id=? AND state='interrupt_requested'\n                     AND interrupt_requested_at IS NOT NULL AND interrupt_taken_at IS NULL",
                (now, attempt_id),
            ).rowcount
            if changed == 1:
                conn.execute(
                    "UPDATE delegation_steer_mailbox SET status='pending',resolved_at=NULL "
                    "WHERE attempt_id=? AND status='superseded_by_interrupt'",
                    (attempt_id,),
                )
        return {"status": "rolled_back" if changed == 1 else "stale"}

    def complete_run(self, run_id: str, event: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        completed_at = float(event.get("completed_at") or time.time())
        with self.write_txn() as conn:
            run = conn.execute(
                "SELECT delegation_id,completed_at FROM delegation_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if run is None:
                return {"status": "not_found"}
            if run[1] is not None:
                return {"status": "stale"}
            changed = conn.execute(
                "UPDATE delegation_runs SET completed_at=?,event_json=?,result_json=?\n                   WHERE run_id=? AND completed_at IS NULL",
                (completed_at, _dump(event), _dump(result), run_id),
            ).rowcount
            if changed != 1:
                return {"status": "stale"}
            conn.execute(
                "UPDATE async_delegations SET completed_at=?,event_json=?,result_json=?,"
                "state=?,updated_at=? WHERE delegation_id=?",
                (
                    completed_at,
                    _dump(event),
                    _dump(result),
                    _attempt_state(event.get("status")),
                    completed_at,
                    run[0],
                ),
            )
            attempts = conn.execute(
                "SELECT a.attempt_id,a.metadata_json,l.root_ordinal\n                   FROM delegation_attempts a JOIN delegation_logical_subagents l\n                     ON l.logical_id=a.logical_id WHERE a.run_id=?\n                   ORDER BY l.root_ordinal IS NULL,l.root_ordinal,a.attempt_number",
                (run_id,),
            ).fetchall()
            raw_results = result.get("results")
            child_results: List[Any] = raw_results if isinstance(raw_results, list) else []
            for index, attempt in enumerate(attempts):
                child = child_results[index] if index < len(child_results) and isinstance(child_results[index], dict) else result
                metadata = _json(attempt[1], {})
                metadata.update(child)
                conn.execute(
                    "UPDATE delegation_attempts SET state=?,completed_at=?,updated_at=?,metadata_json=?\n                       WHERE attempt_id=? AND state IN\n                         ('starting','running','finalizing','interrupt_requested')",
                    (_attempt_state(child.get("status") or event.get("status")), completed_at, completed_at, _dump(metadata), attempt[0]),
                )
                conn.execute(
                    "UPDATE delegation_steer_mailbox "
                    "SET status='too_late_after_completion',resolved_at=? "
                    "WHERE attempt_id=? AND status IN ('pending','forwarding','forwarded')",
                    (completed_at, attempt[0]),
                )
        return {"status": "completed", "delegation_id": run[0], "run_id": run_id}

    @staticmethod
    def _resolve_run(conn: sqlite3.Connection, delegation_id: str, run_id: Optional[str]) -> Dict[str, Any]:
        if run_id:
            row = conn.execute("SELECT * FROM delegation_runs WHERE run_id=? AND delegation_id=?", (run_id, delegation_id)).fetchone()
            return {"status": "found", "row": row} if row else {"status": "not_found"}
        rows = conn.execute("SELECT * FROM delegation_runs WHERE delegation_id=? LIMIT 2", (delegation_id,)).fetchall()
        if not rows:
            return {"status": "not_found"}
        if len(rows) != 1:
            return {"status": "ambiguous_run"}
        return {"status": "found", "row": rows[0]}

    def inspect_delivery(self, delegation_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        conn = self._connect()
        try:
            resolved = self._resolve_run(conn, delegation_id, run_id)
            return {"status": "found", **dict(resolved["row"])} if resolved["status"] == "found" else {"status": resolved["status"]}
        finally:
            conn.close()

    def resolve_run_id(self, delegation_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        outcome = self.inspect_delivery(delegation_id, run_id)
        return {"status": "found", "run_id": outcome["run_id"]} if outcome["status"] == "found" else outcome

    def claim_run_delivery(
        self,
        delegation_id: str,
        run_id: Optional[str],
        token: str,
        *,
        now: Optional[float] = None,
        delivering_stale_seconds: float = 300,
        wait_stale_seconds: float = 360,
    ) -> Dict[str, Any]:
        claimed_at = float(time.time() if now is None else now)
        with self.write_txn() as conn:
            resolved = self._resolve_run(conn, delegation_id, run_id)
            if resolved["status"] != "found":
                return {"status": resolved["status"]}
            row = resolved["row"]
            if row["completed_at"] is None or row["event_json"] is None:
                return {"status": "not_ready"}
            disposition = str(row["delivery_state"])
            stale_before = claimed_at - delivering_stale_seconds
            wait_stale_before = claimed_at - wait_stale_seconds
            claimable = (
                disposition == "pending"
                or (disposition == "delivering" and (row["delivery_claimed_at"] is None or row["delivery_claimed_at"] < stale_before))
                or (
                    disposition == "held_by_wait" and (row["delivery_claimed_at"] is None or row["delivery_claimed_at"] < wait_stale_before)
                )
            )
            if not claimable:
                return {"status": "held" if disposition in {"delivering", "held_by_wait"} else "stale"}
            changed = conn.execute(
                "UPDATE delegation_runs SET delivery_state='delivering',delivery_claim=?,\n                   delivery_claimed_at=?,delivery_attempts=delivery_attempts+1\n                   WHERE run_id=? AND delivery_state=?",
                (token, claimed_at, row["run_id"], disposition),
            ).rowcount
        return {"status": "claimed", "token": token, "run_id": row["run_id"]} if changed == 1 else {"status": "held"}

    def release_run_delivery(self, run_id: str, token: str) -> Dict[str, Any]:
        with self.write_txn() as conn:
            changed = conn.execute(
                "UPDATE delegation_runs SET delivery_state='pending',delivery_claim=NULL,\n                   delivery_claimed_at=NULL WHERE run_id=? AND delivery_state='delivering'\n                   AND delivery_claim=?",
                (run_id, token),
            ).rowcount
        return {"status": "released" if changed == 1 else "stale"}

    def commit_run_delivery(
        self, run_id: str, token: str, *, disposition: str = "delivered", now: Optional[float] = None
    ) -> Dict[str, Any]:
        if disposition not in _TERMINAL_DELIVERY_STATES:
            raise ValueError("terminal delivery disposition required")
        completed = float(time.time() if now is None else now)
        with self.write_txn() as conn:
            changed = conn.execute(
                "UPDATE delegation_runs SET delivery_state=?,delivered_at=?,\n                   delivery_claim=NULL,delivery_claimed_at=NULL\n                   WHERE run_id=? AND delivery_state IN ('delivering','held_by_wait')\n                     AND delivery_claim=?",
                (disposition, completed, run_id, token),
            ).rowcount
        return {"status": disposition if changed == 1 else "stale"}

    def acknowledge_pending(self, delegation_id: str, *, now: Optional[float] = None) -> Dict[str, Any]:
        completed = float(time.time() if now is None else now)
        with self.write_txn() as conn:
            resolved = self._resolve_run(conn, delegation_id, None)
            if resolved["status"] != "found":
                return {"status": resolved["status"]}
            changed = conn.execute(
                "UPDATE delegation_runs SET delivery_state='delivered',delivered_at=?\n                   WHERE run_id=? AND delivery_state='pending'",
                (completed, resolved["row"]["run_id"]),
            ).rowcount
        return {"status": "delivered" if changed == 1 else "stale"}

    def hold_for_wait(
        self,
        delegation_id: str,
        session_key: str,
        token: str,
        *,
        run_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Bind one waiter to an exact run without later retargeting it."""
        claimed = float(time.time() if now is None else now)
        with self.write_txn() as conn:
            owner = conn.execute(
                "SELECT 1 FROM async_delegations WHERE delegation_id=? AND origin_session=?",
                (delegation_id, session_key),
            ).fetchone()
            if owner is None:
                return {"status": "not_found"}
            if run_id:
                row = conn.execute(
                    "SELECT * FROM delegation_runs WHERE delegation_id=? AND run_id=?",
                    (delegation_id, run_id),
                ).fetchone()
            else:
                # FIFO terminal delivery wins. With none available, bind to the
                # latest active run at call start and never jump to a later run.
                row = conn.execute(
                    """SELECT * FROM delegation_runs WHERE delegation_id=?
                       AND completed_at IS NOT NULL AND delivery_state='pending'
                       ORDER BY completed_at,run_number LIMIT 1""",
                    (delegation_id,),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        """SELECT r.* FROM delegation_runs r
                           WHERE r.delegation_id=? AND EXISTS (
                             SELECT 1 FROM delegation_attempts a WHERE a.run_id=r.run_id
                               AND a.state IN ('starting','running','finalizing','interrupt_requested'))
                           ORDER BY r.run_number DESC LIMIT 1""",
                        (delegation_id,),
                    ).fetchone()
                if row is None:
                    row = conn.execute(
                        "SELECT * FROM delegation_runs WHERE delegation_id=? ORDER BY run_number DESC LIMIT 1",
                        (delegation_id,),
                    ).fetchone()
            if row is None:
                return {"status": "not_found"}
            if row["delivery_state"] == "held_by_wait" and row["delivery_claim"] == token:
                return {"status": "held", "run_id": row["run_id"]}
            changed = conn.execute(
                "UPDATE delegation_runs SET delivery_state='held_by_wait',delivery_claim=?,\n                   delivery_claimed_at=? WHERE run_id=? AND delivery_state='pending'",
                (token, claimed, row["run_id"]),
            ).rowcount
            if changed == 1:
                return {"status": "held", "run_id": row["run_id"]}
            return {
                "status": "bound",
                "run_id": row["run_id"],
                "delivery_state": row["delivery_state"],
            }

    def consume_wait_hold(
        self, delegation_id: str, session_key: str, token: str, *,
        run_id: Optional[str] = None, now: Optional[float] = None,
    ) -> Dict[str, Any]:
        completed = float(time.time() if now is None else now)
        with self.write_txn() as conn:
            owner = conn.execute(
                "SELECT 1 FROM async_delegations WHERE delegation_id=? AND origin_session=?", (delegation_id, session_key)
            ).fetchone()
            if owner is None:
                return {"status": "not_found"}
            row = conn.execute(
                "SELECT run_id FROM delegation_runs WHERE delegation_id=? AND delivery_claim=?"
                + (" AND run_id=?" if run_id else ""),
                (delegation_id, token, run_id) if run_id else (delegation_id, token),
            ).fetchone()
            if row is None:
                return {"status": "stale"}
            changed = conn.execute(
                "UPDATE delegation_runs SET delivery_state='consumed',delivered_at=?,\n                   delivery_claim=NULL,delivery_claimed_at=NULL\n                   WHERE run_id=? AND completed_at IS NOT NULL AND event_json IS NOT NULL\n                     AND delivery_state='held_by_wait' AND delivery_claim=?",
                (completed, row["run_id"], token),
            ).rowcount
        return {"status": "consumed" if changed == 1 else "stale"}

    def release_wait_hold(
        self, delegation_id: str, session_key: str, token: str, *,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.write_txn() as conn:
            owner = conn.execute(
                "SELECT 1 FROM async_delegations WHERE delegation_id=? AND origin_session=?", (delegation_id, session_key)
            ).fetchone()
            if owner is None:
                return {"status": "not_found"}
            row = conn.execute(
                "SELECT run_id FROM delegation_runs WHERE delegation_id=? AND delivery_claim=?"
                + (" AND run_id=?" if run_id else ""),
                (delegation_id, token, run_id) if run_id else (delegation_id, token),
            ).fetchone()
            if row is None:
                return {"status": "stale"}
            changed = conn.execute(
                "UPDATE delegation_runs SET delivery_state='pending',delivery_claim=NULL,\n                   delivery_claimed_at=NULL WHERE run_id=? AND delivery_state='held_by_wait'\n                     AND delivery_claim=?",
                (row["run_id"], token),
            ).rowcount
        return {"status": "released" if changed == 1 else "stale"}

    def suppress_delivery(self, delegation_id: str, session_key: str, reason: str = "") -> Dict[str, Any]:
        """Suppress every still-pending run delivery for one logical delegation."""
        with self.write_txn() as conn:
            container = conn.execute(
                "SELECT abandon_reason FROM async_delegations WHERE delegation_id=? AND origin_session=?",
                (delegation_id, session_key),
            ).fetchone()
            if container is None:
                return {"status": "not_found"}
            now = time.time()
            changed = conn.execute(
                """UPDATE delegation_runs SET delivery_state='suppressed',delivered_at=?,
                     delivery_claim=NULL,delivery_claimed_at=NULL
                   WHERE delegation_id=? AND delivery_state IN ('pending','held_by_wait')""",
                (now, delegation_id),
            ).rowcount
            remaining = conn.execute(
                """SELECT
                     SUM(CASE WHEN delivery_state='suppressed' THEN 1 ELSE 0 END) AS suppressed,
                     SUM(CASE WHEN delivery_state IN ('pending','held_by_wait') THEN 1 ELSE 0 END) AS pending
                   FROM delegation_runs WHERE delegation_id=?""",
                (delegation_id,),
            ).fetchone()
            conn.execute(
                "UPDATE async_delegations SET abandon_reason=? WHERE delegation_id=?",
                (reason or container["abandon_reason"], delegation_id),
            )
            if changed:
                return {"status": "suppressed", "count": int(changed)}
            if int(remaining["suppressed"] or 0) > 0 and int(remaining["pending"] or 0) == 0:
                return {"status": "already_suppressed", "count": 0}
            return {"status": "too_late", "count": 0}

    def recover_stale_wait_holds(self, *, cutoff: float, delegation_id: Optional[str] = None) -> int:
        where = " AND delegation_id=?" if delegation_id else ""
        params: List[Any] = [cutoff]
        if delegation_id:
            params.append(delegation_id)
        with self.write_txn() as conn:
            changed = conn.execute(
                "UPDATE delegation_runs SET delivery_state='pending',delivery_claim=NULL,\n                   delivery_claimed_at=NULL WHERE delivery_state='held_by_wait'\n                     AND (delivery_claimed_at IS NULL OR delivery_claimed_at < ?)"
                + where,
                params,
            ).rowcount
        return changed

    def recover_orphaned_attempts(self, owner_alive: Callable[[int, Optional[int]], bool]) -> Dict[str, Any]:
        recovered: List[Dict[str, str]] = []
        now = time.time()
        with self.write_txn() as conn:
            rows = conn.execute(
                "SELECT a.*,l.delegation_id FROM delegation_attempts a\n                   JOIN delegation_logical_subagents l ON l.logical_id=a.logical_id\n                   WHERE a.state IN ('starting','running','finalizing','interrupt_requested')\n                     AND a.attempt_number=(SELECT MAX(x.attempt_number)\n                         FROM delegation_attempts x WHERE x.logical_id=a.logical_id)"
            ).fetchall()
            for row in rows:
                pid = row["owner_pid"]
                if pid and owner_alive(int(pid), row["owner_started_at"]):
                    continue
                changed = conn.execute(
                    "UPDATE delegation_attempts SET state='unknown',completed_at=?,updated_at=?\n                       WHERE attempt_id=? AND state=? AND owner_pid IS ? AND owner_started_at IS ?",
                    (now, now, row["attempt_id"], row["state"], pid, row["owner_started_at"]),
                ).rowcount
                if changed == 1:
                    recovered.append({key: str(row[key]) for key in (
                        "attempt_id", "run_id", "delegation_id"
                    )})
        return {"status": "recovered", "attempts": recovered}

    def snapshot_for_attempt(
        self,
        delegation_id: str,
        attempt_id: str,
        *,
        session_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return one authorized historical attempt projected through its run."""
        conn = self._connect()
        try:
            params: List[Any] = [delegation_id, attempt_id]
            owner_clause = ""
            if session_key is not None:
                owner_clause = " AND d.origin_session=?"
                params.append(session_key)
            row = conn.execute(
                """SELECT a.run_id,a.logical_id FROM delegation_attempts a
                   JOIN delegation_logical_subagents l ON l.logical_id=a.logical_id
                   JOIN async_delegations d ON d.delegation_id=l.delegation_id
                   WHERE d.delegation_id=? AND a.attempt_id=?"""
                + owner_clause,
                params,
            ).fetchone()
            if row is None:
                return None
            snapshot = self._snapshot_with_conn(
                conn,
                delegation_id,
                session_key=session_key,
                run_id=str(row["run_id"]),
            )
            if snapshot is None:
                return None
            logical_id = str(row["logical_id"])
            child = snapshot.get("children", {}).get(logical_id)
            if child is None:
                return None
            snapshot["children"] = {logical_id: child}
            snapshot["root_subagent_ids"] = [logical_id]
            return _external_authority_projection(snapshot)
        finally:
            conn.close()

    def resume_metadata_candidates(
        self, delegation_id: str, logical_id: str
    ) -> List[Dict[str, Any]]:
        """Return newest-first durable attempt metadata for transcript recovery."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT a.attempt_id,a.attempt_number,a.state,a.metadata_json "
                "FROM delegation_attempts a "
                "JOIN delegation_logical_subagents l ON l.logical_id=a.logical_id "
                "WHERE a.logical_id=? AND l.delegation_id=? "
                "ORDER BY a.attempt_number DESC",
                (logical_id, delegation_id),
            ).fetchall()
            return [
                {
                    "attempt_id": str(row["attempt_id"]),
                    "attempt_number": int(row["attempt_number"]),
                    "state": str(row["state"]),
                    "metadata": _json(row["metadata_json"], {}),
                }
                for row in rows
            ]
        finally:
            conn.close()

    def snapshot(
        self, delegation_id: str, *, session_key: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a backing-redacted lifecycle projection."""
        conn = self._connect()
        try:
            snapshot = self._snapshot_with_conn(
                conn, delegation_id, session_key=session_key, run_id=run_id
            )
            return (
                _external_authority_projection(snapshot)
                if snapshot is not None
                else None
            )
        finally:
            conn.close()

    def trusted_snapshot(
        self, delegation_id: str, *, session_key: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return canonical authority for internal resume and recovery only."""
        conn = self._connect()
        try:
            return self._snapshot_with_conn(
                conn, delegation_id, session_key=session_key, run_id=run_id
            )
        finally:
            conn.close()

    def _snapshot_with_conn(
        self, conn: sqlite3.Connection, delegation_id: str, *,
        session_key: Optional[str] = None, run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        params: List[Any] = [delegation_id]
        owner_clause = ""
        if session_key is not None:
            owner_clause = " AND origin_session=?"
            params.append(session_key)
        container = conn.execute(
            "SELECT * FROM async_delegations WHERE delegation_id=?" + owner_clause, params
        ).fetchone()
        if container is None:
            return None
        run = conn.execute(
            "SELECT * FROM delegation_runs WHERE delegation_id=?"
            + (" AND run_id=?" if run_id else " ORDER BY run_number DESC LIMIT 1"),
            (delegation_id, run_id) if run_id else (delegation_id,),
        ).fetchone()
        run_summary = conn.execute(
            """SELECT
                 SUM(CASE WHEN completed_at IS NOT NULL AND delivery_state IN
                   ('pending','held_by_wait','delivering') THEN 1 ELSE 0 END)
                   AS pending_count,
                 MAX(run_number) AS latest_number
               FROM delegation_runs WHERE delegation_id=?""",
            (delegation_id,),
        ).fetchone()
        active_run = conn.execute(
            """SELECT r.run_id FROM delegation_runs r
               WHERE r.delegation_id=? AND EXISTS (
                 SELECT 1 FROM delegation_attempts a WHERE a.run_id=r.run_id
                   AND a.state IN ('starting','running','finalizing','interrupt_requested'))
               ORDER BY r.run_number DESC LIMIT 1""",
            (delegation_id,),
        ).fetchone()
        if run_id and run is None:
            return None
        attempt_clause = (
            "a.run_id=?" if run_id else
            "a.attempt_number=(SELECT MAX(x.attempt_number) FROM delegation_attempts x WHERE x.logical_id=a.logical_id)"
        )
        attempts = conn.execute(
            "SELECT a.*,l.parent_logical_id,l.root_ordinal,l.spec_json,\n"
            "  (SELECT CASE WHEN first.physical_worker_id != first.logical_id "
            "    THEN 1 ELSE 0 END FROM delegation_attempts first "
            "    WHERE first.logical_id=a.logical_id "
            "    ORDER BY first.attempt_number LIMIT 1) AS protected_origin\n"
            "  FROM delegation_attempts a JOIN delegation_logical_subagents l\n"
            "    ON l.logical_id=a.logical_id\n"
            "  WHERE l.delegation_id=? AND " + attempt_clause +
            " ORDER BY l.root_ordinal IS NULL,l.root_ordinal,l.created_at,l.logical_id",
            (delegation_id, run_id) if run_id else (delegation_id,),
        ).fetchall()
        steer_rows = conn.execute(
            "SELECT m.mailbox_id,m.logical_id,m.attempt_id,m.sequence_number,"
            "m.status,m.created_at,m.forwarded_at,m.resolved_at "
            "FROM delegation_steer_mailbox m "
            "JOIN delegation_attempts a ON a.attempt_id=m.attempt_id "
            "WHERE m.delegation_id=?" + (" AND a.run_id=?" if run_id else "") +
            " ORDER BY m.logical_id,m.sequence_number",
            (delegation_id, run_id) if run_id else (delegation_id,),
        ).fetchall()
        steers_by_child: Dict[str, List[Dict[str, Any]]] = {}
        for steer_row in steer_rows:
            steers_by_child.setdefault(str(steer_row["logical_id"]), []).append(
                {
                    key: steer_row[key]
                    for key in (
                        "mailbox_id",
                        "attempt_id",
                        "sequence_number",
                        "status",
                        "created_at",
                        "forwarded_at",
                        "resolved_at",
                    )
                    if steer_row[key] is not None
                }
            )
        children: Dict[str, Dict[str, Any]] = {}
        roots: List[str] = []
        active_states: List[str] = []
        owner_pid = owner_started_at = None
        interrupts: Dict[str, str] = {}
        for attempt in attempts:
            child = {
                **_json(attempt["spec_json"], {}),
                **_json(attempt["metadata_json"], {}),
                "subagent_id": attempt["logical_id"],
                "parent_id": attempt["parent_logical_id"],
                "status": attempt["state"],
                "attempt_id": attempt["attempt_id"],
                "physical_worker_id": attempt["physical_worker_id"],
                "protected_execution": bool(attempt["protected_origin"]),
                "attempt_number": attempt["attempt_number"],
                "run_id": attempt["run_id"],
                "steers": steers_by_child.get(str(attempt["logical_id"]), []),
            }
            child["resume_available"] = (
                attempt["state"] not in _ACTIVE_STATES
                and isinstance(child.get("child_session_id"), str)
                and bool(child.get("child_session_id"))
            )
            if child["resume_available"]:
                child["suggested_action"] = "resume"
            children[attempt["logical_id"]] = child
            if attempt["root_ordinal"] is not None:
                roots.append(attempt["logical_id"])
            if attempt["state"] in _ACTIVE_STATES:
                active_states.append(attempt["state"])
                if owner_pid is None:
                    owner_pid, owner_started_at = (attempt["owner_pid"], attempt["owner_started_at"])
            if attempt["interrupt_requested_at"] is not None and attempt["interrupt_taken_at"] is None:
                interrupts[attempt["logical_id"]] = str(attempt["interrupt_reason"] or "")
        if active_states:
            state = next(
                (candidate for candidate in ("interrupt_requested", "finalizing", "running", "starting") if candidate in active_states)
            )
        elif run and run["completed_at"] is not None:
            state = _attempt_state(_json(run["event_json"], {}).get("status"))
        elif attempts:
            state = attempts[0]["state"]
        else:
            state = "running" if run and run["completed_at"] is None else "unknown"
        aggregate_state = "running" if state == "starting" else state
        event = _json(run["event_json"], {}) if run and run["event_json"] else None
        result = _json(run["result_json"], {}) if run and run["result_json"] else None
        delivery = str(run["delivery_state"] if run else "pending")
        return {
            **_json(container["task_json"], {}),
            "delegation_id": delegation_id,
            "origin_session": container["origin_session"],
            "session_key": container["origin_session"],
            "origin_ui_session_id": container["origin_ui_session_id"],
            "parent_session_id": container["parent_session_id"],
            "lifecycle_version": 2,
            "state": aggregate_state,
            "worker_status": aggregate_state,
            "status": aggregate_state,
            "dispatched_at": container["dispatched_at"],
            "completed_at": run["completed_at"] if run else None,
            "updated_at": max(
                [float(container["updated_at"])]
                + [float(a["updated_at"]) for a in attempts]
                + ([float(run["completed_at"] or run["created_at"])] if run else [])
            ),
            "event": event,
            "result": result,
            "delivery_state": delivery,
            "delivery_disposition": delivery,
            "delivery_attempts": int(run["delivery_attempts"] if run else 0),
            "delivered_at": run["delivered_at"] if run else None,
            "delivery_claim": run["delivery_claim"] if run else None,
            "delivery_claimed_at": run["delivery_claimed_at"] if run else None,
            "run_id": run["run_id"] if run else None,
            "latest_run_id": run["run_id"] if run else None,
            "active_run_id": active_run["run_id"] if active_run else None,
            "pending_run_count": int(run_summary["pending_count"] or 0),
            "root_subagent_ids": roots,
            "children": children,
            "interrupt_requests": interrupts,
            "interrupt_reason": container["interrupt_reason"],
            "abandon_reason": container["abandon_reason"],
            "owner_pid": owner_pid,
            "owner_started_at": owner_started_at,
        }

    def list_snapshots(self, *, session_keys: Optional[List[str]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if session_keys == [] or limit <= 0:
            return []
        conn = self._connect()
        try:
            params: List[Any] = []
            where = ""
            if session_keys is not None:
                owners = list(dict.fromkeys(session_keys))
                where = f"WHERE origin_session IN ({','.join(('?' for _ in owners))})"
                params.extend(owners)
            params.append(max(0, min(int(limit), 100)))
            ids = [
                str(row[0])
                for row in conn.execute(
                    f"SELECT delegation_id FROM async_delegations {where} ORDER BY dispatched_at DESC,delegation_id LIMIT ?", params
                )
            ]
            return [
                _external_authority_projection(snap) for item in ids
                if (snap := self._snapshot_with_conn(conn, item)) is not None
            ]
        finally:
            conn.close()

    def prune(self, *, cutoff: float, max_terminal: int) -> Dict[str, Any]:
        with self.write_txn() as conn:
            candidates = conn.execute(
                "SELECT d.delegation_id,d.updated_at FROM async_delegations d\n                   WHERE NOT EXISTS (SELECT 1 FROM delegation_attempts a\n                       JOIN delegation_logical_subagents l ON l.logical_id=a.logical_id\n                       WHERE l.delegation_id=d.delegation_id\n                         AND a.state IN ('starting','running','finalizing','interrupt_requested'))\n                     AND NOT EXISTS (SELECT 1 FROM delegation_runs r\n                       WHERE r.delegation_id=d.delegation_id\n                         AND (r.completed_at IS NULL OR r.delivery_state NOT IN\n                           ('delivered','consumed','suppressed')))\n                   ORDER BY d.updated_at DESC"
            ).fetchall()
            keep = max(0, int(max_terminal))
            delete_ids = [str(row[0]) for index, row in enumerate(candidates) if float(row[1]) < cutoff or index >= keep]
            if delete_ids:
                conn.executemany("DELETE FROM async_delegations WHERE delegation_id=?", [(delegation_id,) for delegation_id in delete_ids])
        return {"status": "pruned", "deleted": len(delete_ids)}

    def delete(self, delegation_id: str) -> bool:
        with self.write_txn() as conn:
            return conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,)).rowcount == 1

    def pending_events(self, *, delivery_state: str = "pending") -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT run_id,event_json FROM delegation_runs\n                   WHERE completed_at IS NOT NULL AND delivery_state=? AND event_json IS NOT NULL\n                   ORDER BY completed_at,run_id",
                (delivery_state,),
            ).fetchall()
            return [{"run_id": row["run_id"], "event": _json(row["event_json"], {})} for row in rows]
        finally:
            conn.close()

    @staticmethod
    def _attempt_for_worker(conn: sqlite3.Connection, worker_id: str) -> Optional[sqlite3.Row]:
        return conn.execute(
            "SELECT a.*,l.delegation_id,l.logical_id AS stable_id,l.spec_json\n               FROM delegation_attempts a JOIN delegation_logical_subagents l\n                 ON l.logical_id=a.logical_id\n               WHERE a.physical_worker_id=? OR a.logical_id=?\n               ORDER BY a.attempt_number DESC LIMIT 1",
            (worker_id, worker_id),
        ).fetchone()

    def register_subagent(self, record: Dict[str, Any]) -> Optional[Dict[str, str]]:
        worker_id = record.get("subagent_id")
        if not isinstance(worker_id, str) or not worker_id:
            return None
        parent_id = record.get("parent_id")
        supplied_attempt_id = record.get("delegation_attempt_id")
        has_supplied_attempt = "delegation_attempt_id" in record
        now = time.time()
        with self.write_txn() as conn:
            if has_supplied_attempt:
                if not isinstance(supplied_attempt_id, str) or not supplied_attempt_id:
                    return None
                attempt = conn.execute(
                    "SELECT a.*,l.delegation_id,l.logical_id AS stable_id,l.spec_json\n                       FROM delegation_attempts a JOIN delegation_logical_subagents l\n                         ON l.logical_id=a.logical_id\n                       WHERE a.attempt_id=?\n                         AND (a.physical_worker_id=? OR a.logical_id=?)",
                    (supplied_attempt_id, worker_id, worker_id),
                ).fetchone()
                supplied_run_id = record.get("delegation_run_id")
                if attempt is not None and supplied_run_id is not None and (
                    not isinstance(supplied_run_id, str)
                    or supplied_run_id != attempt["run_id"]
                ):
                    attempt = None
            else:
                attempt = self._attempt_for_worker(conn, worker_id)
            if attempt is None and not has_supplied_attempt and isinstance(parent_id, str) and parent_id:
                parent = self._attempt_for_worker(conn, parent_id)
                if parent is None:
                    return None
                stable = {key: record.get(key) for key in ("parent_id", "depth", "goal", "model") if key in record}
                conn.execute(
                    "INSERT INTO delegation_logical_subagents\n                       (logical_id,delegation_id,parent_logical_id,spec_json,created_at,updated_at)\n                       VALUES (?,?,?,?,?,?)",
                    (worker_id, parent["delegation_id"], parent["stable_id"], _dump(stable), now, now),
                )
                attempt_id = str(record.get("delegation_attempt_id") or _new_id("attempt"))
                runtime = {key: value for key, value in record.items() if key != "agent" and key not in stable}
                conn.execute(
                    "INSERT INTO delegation_attempts\n                       (attempt_id,logical_id,run_id,attempt_number,physical_worker_id,state,\n                        owner_pid,owner_started_at,created_at,updated_at,metadata_json)\n                       VALUES (?,?,?,1,?,'starting',?,?,?,?,?)",
                    (
                        attempt_id,
                        worker_id,
                        parent["run_id"],
                        worker_id,
                        parent["owner_pid"],
                        parent["owner_started_at"],
                        now,
                        now,
                        _dump(runtime),
                    ),
                )
                attempt = self._attempt_for_worker(conn, worker_id)
            if attempt is None:
                return None
            if attempt["state"] not in _ACTIVE_STATES:
                return None
            metadata = _json(attempt["metadata_json"], {})
            spec = _json(attempt["spec_json"], {})
            metadata.update({key: value for key, value in record.items() if key != "agent" and (key not in spec or spec[key] != value)})
            requested = attempt["interrupt_requested_at"] is not None
            requested_state = "interrupt_requested" if requested else _attempt_state(record.get("status") or attempt["state"])
            if attempt["state"] in _ACTIVE_STATES:
                conn.execute(
                    "UPDATE delegation_attempts SET state=?,metadata_json=?,updated_at=? WHERE attempt_id=?",
                    (requested_state, _dump(metadata), now, attempt["attempt_id"]),
                )
            return {"delegation_id": str(attempt["delegation_id"]), "attempt_id": str(attempt["attempt_id"])}

    def find_attempt(
        self,
        worker_id: str,
        *,
        attempt_id: Optional[str] = None,
        run_id: Optional[str] = None,
        delegation_id: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            clauses = ["(a.physical_worker_id=? OR a.logical_id=?)"]
            params: List[Any] = [worker_id, worker_id]
            if attempt_id is not None:
                clauses.append("a.attempt_id=?")
                params.append(attempt_id)
            if run_id is not None:
                clauses.append("a.run_id=?")
                params.append(run_id)
            if delegation_id is not None:
                clauses.append("l.delegation_id=?")
                params.append(delegation_id)
            if session_key is not None:
                clauses.append("d.origin_session=?")
                params.append(session_key)
            row = conn.execute(
                "SELECT a.*,l.delegation_id FROM delegation_attempts a\n                   JOIN delegation_logical_subagents l ON l.logical_id=a.logical_id\n                   JOIN async_delegations d ON d.delegation_id=l.delegation_id WHERE "
                + " AND ".join(clauses)
                + " ORDER BY a.attempt_number DESC LIMIT 1",
                params,
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
