#!/usr/bin/env python3
"""
Async (background) delegation registry.

Backs ``delegate_task(background=true)``: the parent agent dispatches a
subagent that runs on a module-level daemon executor and returns a handle
immediately, so the user and the model can keep working while the child runs.

When the child finishes, a completion event is pushed onto the SHARED
``process_registry.completion_queue`` with ``type="async_delegation"``. The
CLI (``cli.py`` process_loop) and gateway (``_run_process_watcher`` /
``completion_queue`` drain) already poll that queue while the agent is idle
and forge a fresh user/internal turn from each event. We deliberately reuse
that rail rather than reaching into a running agent loop:

  - completions surface as a NEW turn when the agent is idle, never spliced
    between a tool result and an assistant message. That keeps strict
    message-role alternation legal and the prompt cache intact (hard
    invariant: never mutate past context).
  - we inherit the queue's de-dup, crash-recovery checkpoint, and the
    existing CLI + gateway drain wiring for free — no new drain loops in the
    two largest files in the repo.

The completion payload carries a RICH, self-contained task-source block (the
original goal, the context the parent supplied, toolsets, model, dispatch
time, status, and the full result summary). When the result re-enters the
conversation the parent may be deep in unrelated context and won't remember
why the subagent existed; the block lets it either use the result or
re-dispatch if the world has moved on.

This module owns ONLY the async lifecycle. The actual child build + run is
delegated back to ``delegate_tool._run_single_child`` via an injected
runner, so all the credential leasing, heartbeat, timeout, and result-shaping
logic stays in one place.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger(__name__)

# Back-compat alias — the daemon executor now lives in tools.daemon_pool so
# other subsystems (tool_executor, memory_manager, delegate_tool, skills_hub)
# can share it. Existing imports of ``_DaemonThreadPoolExecutor`` keep working.
_DaemonThreadPoolExecutor = DaemonThreadPoolExecutor


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# A persistent daemon executor (NOT a `with ThreadPoolExecutor()` block, which
# would join on exit and defeat the whole point of async). Workers are daemon
# threads so a hard process exit doesn't hang on an in-flight child.
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_executor_max_workers: int = 0

_records_lock = threading.Lock()
# delegation_id -> record dict. Kept for the lifetime of the run plus a short
# tail after completion so `list_async_delegations()` can show recent results.
_records: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MAX_ASYNC_CHILDREN = 3
# How many completed records to retain for status queries before pruning.
_MAX_RETAINED_COMPLETED = 50
_DURABLE_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_DURABLE_PENDING = 1000
# A pending completion whose delivery keeps failing is retried across claim
# cycles (and across restarts via restore_undelivered_completions). Cap the
# attempts so an unroutable row converges to a terminal 'dropped' state
# instead of replaying on every restart forever.
_MAX_DELIVERY_ATTEMPTS = 8
_DB_LOCK = threading.Lock()
_RECONCILIATION_CLAIM_LEASE_SECONDS = 300
_MAX_RECONCILIATION_BATCH_ROWS = 10
_COMPLETION_PERSIST_IMMEDIATE_ATTEMPTS = 3
_COMPLETION_PERSIST_RETRY_BASE_SECONDS = 0.05
_COMPLETION_PERSIST_RETRY_MAX_SECONDS = 5.0


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")
    for attempt in range(6):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS async_delegations (
                    delegation_id TEXT PRIMARY KEY,
                    origin_session TEXT NOT NULL,
                    origin_ui_session_id TEXT NOT NULL DEFAULT '',
                    parent_session_id TEXT,
                    state TEXT NOT NULL,
                    dispatched_at REAL NOT NULL,
                    completed_at REAL,
                    updated_at REAL NOT NULL,
                    event_json TEXT,
                    result_json TEXT,
                    delivery_state TEXT NOT NULL DEFAULT 'pending',
                    delivery_attempts INTEGER NOT NULL DEFAULT 0,
                    delivered_at REAL,
                    owner_pid INTEGER,
                    owner_started_at INTEGER,
                    task_json TEXT,
                    delivery_claim TEXT,
                    delivery_claimed_at REAL,
                    goal_id TEXT NOT NULL DEFAULT '',
                    requires_goal_join INTEGER NOT NULL DEFAULT 0,
                    parent_delegation_id TEXT NOT NULL DEFAULT '',
                    goal_owner_session_id TEXT NOT NULL DEFAULT '',
                    reconciliation_state TEXT NOT NULL DEFAULT 'not_required',
                    reconciliation_claim TEXT,
                    reconciliation_claimed_at REAL,
                    reconciliation_attempts INTEGER NOT NULL DEFAULT 0,
                    reconciled_at REAL
                )"""
            )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(async_delegations)")
            }
            for name, sql_type in (
                ("owner_pid", "INTEGER"),
                ("owner_started_at", "INTEGER"),
                ("task_json", "TEXT"),
                ("delivery_claim", "TEXT"),
                ("delivery_claimed_at", "REAL"),
                ("goal_id", "TEXT NOT NULL DEFAULT ''"),
                ("requires_goal_join", "INTEGER NOT NULL DEFAULT 0"),
                ("parent_delegation_id", "TEXT NOT NULL DEFAULT ''"),
                ("goal_owner_session_id", "TEXT NOT NULL DEFAULT ''"),
                ("reconciliation_state", "TEXT NOT NULL DEFAULT 'not_required'"),
                ("reconciliation_claim", "TEXT"),
                ("reconciliation_claimed_at", "REAL"),
                ("reconciliation_attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("reconciled_at", "REAL"),
            ):
                if name not in columns:
                    try:
                        conn.execute(
                            f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}"
                        )
                    except sqlite3.OperationalError as exc:
                        # Another process may have completed the same migration
                        # after our table-info read.
                        if "duplicate column name" not in str(exc).lower():
                            raise
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_async_delegations_goal_reconciliation
                   ON async_delegations(
                       goal_id, requires_goal_join, reconciliation_state, state
                   )"""
            )
            conn.commit()
            return conn
        except sqlite3.OperationalError as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            if "locked" not in str(exc).lower() or attempt == 5:
                conn.close()
                raise
            # WAL setup and additive DDL can race across CLI, gateway, and TUI
            # processes even with SQLite's busy timeout. Re-read the schema on
            # each bounded retry rather than assuming which DDL won.
            time.sleep(min(0.05 * (2**attempt), 1.0))
        except Exception:
            conn.close()
            raise
    conn.close()
    raise RuntimeError("async delegation schema initialization exhausted retries")


def _goal_owner_is_current_in_transaction(
    conn: sqlite3.Connection, owner_session_id: str, goal_id: str
) -> bool:
    """Validate required ownership on the same snapshot used by insertion."""
    if not owner_session_id or not goal_id:
        return False
    row = conn.execute(
        "SELECT value FROM state_meta WHERE key=?", (f"goal:{owner_session_id}",)
    ).fetchone()
    if row is None:
        return False
    try:
        state = json.loads(row[0])
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(state, dict)
        and state.get("status") == "active"
        and str(state.get("goal_id") or "") == goal_id
    )


def _persist_dispatch(record: Dict[str, Any]) -> None:
    now = time.time()
    try:
        from gateway.status import get_process_start_time
        owner_started_at = get_process_start_time(__import__("os").getpid())
    except Exception:
        owner_started_at = None
    task_payload = {
        key: record.get(key)
        for key in (
            "goal", "goals", "context", "toolsets", "role", "model", "is_batch",
            "goal_id", "requires_goal_join", "parent_delegation_id",
            "goal_owner_session_id",
        )
        if key in record
    }
    with _DB_LOCK, _connect() as conn:
        # BEGIN IMMEDIATE serializes this read+insert with goal replacement and
        # clear writes. If insertion wins, their later sweep sees the row; if
        # lifecycle mutation wins, stale required work cannot be inserted.
        conn.execute("BEGIN IMMEDIATE")
        if record.get("requires_goal_join") and not _goal_owner_is_current_in_transaction(
            conn,
            str(record.get("goal_owner_session_id") or ""),
            str(record.get("goal_id") or ""),
        ):
            raise RuntimeError("required async delegation goal owner is no longer current")
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, goal_id, requires_goal_join,
                parent_delegation_id, goal_owner_session_id, reconciliation_state)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record["delegation_id"], record.get("session_key", ""),
             record.get("origin_ui_session_id", ""), record.get("parent_session_id"),
             record["dispatched_at"], now, __import__("os").getpid(),
             owner_started_at, json.dumps(task_payload), record.get("goal_id", ""),
             int(bool(record.get("requires_goal_join"))),
             record.get("parent_delegation_id", ""),
             record.get("goal_owner_session_id", ""),
             "pending" if record.get("requires_goal_join") else "not_required"),
        )
    _prune_durable_records()


def _delete_durable_delegation(delegation_id: str) -> None:
    with _DB_LOCK, _connect() as conn:
        conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))


def _prune_durable_records() -> None:
    """Bound terminal history, preferring delivered records for deletion."""
    now = time.time()
    cutoff = now - _DURABLE_RETENTION_SECONDS
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            """DELETE FROM async_delegations
               WHERE delivery_state='delivered' AND updated_at < ?
                 AND (requires_goal_join=0 OR reconciliation_state IN
                      ('not_required','reconciled','abandoned'))""",
            (cutoff,),
        )
        terminal_count = conn.execute(
            """SELECT COUNT(*) FROM async_delegations
               WHERE state NOT IN ('running','finalizing')
                 AND (requires_goal_join=0 OR reconciliation_state IN
                      ('not_required','reconciled','abandoned'))"""
        ).fetchone()[0]
        excess = max(0, terminal_count - _MAX_RETAINED_COMPLETED)
        if excess:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing')
                       AND (requires_goal_join=0 OR reconciliation_state IN
                            ('not_required','reconciled','abandoned'))
                     ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                              updated_at ASC LIMIT ?
                   )""",
                (excess,),
            )
        pending_count = conn.execute(
            """SELECT COUNT(*) FROM async_delegations
               WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                 AND (requires_goal_join=0 OR reconciliation_state IN
                      ('not_required','reconciled','abandoned'))"""
        ).fetchone()[0]
        overflow = max(0, pending_count - _MAX_DURABLE_PENDING)
        if overflow:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                       AND (requires_goal_join=0 OR reconciliation_state IN
                            ('not_required','reconciled','abandoned'))
                     ORDER BY updated_at ASC LIMIT ?
                   )""",
                (overflow,),
            )


def _persist_completion_once(event: Dict[str, Any], result: Dict[str, Any]) -> None:
    now = time.time()
    conn = _connect()
    try:
        with _DB_LOCK, conn:
            conn.execute(
                """UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
                   event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""",
                (event.get("status", "completed"), event.get("completed_at", now), now,
                 json.dumps(event), json.dumps(result), event["delegation_id"]),
            )
    finally:
        conn.close()


def _persist_completion(
    event: Dict[str, Any],
    result: Dict[str, Any],
    *,
    on_persisted: Optional[Callable[[], None]] = None,
) -> bool:
    """Persist terminal state, continuing autonomously after bounded retries."""
    last_error: Optional[sqlite3.OperationalError] = None
    for attempt in range(_COMPLETION_PERSIST_IMMEDIATE_ATTEMPTS):
        try:
            _persist_completion_once(event, result)
            return True
        except sqlite3.OperationalError as exc:
            last_error = exc
            if attempt + 1 < _COMPLETION_PERSIST_IMMEDIATE_ATTEMPTS:
                # The operation helper has released _DB_LOCK before we back off.
                time.sleep(_COMPLETION_PERSIST_RETRY_BASE_SECONDS * (2**attempt))

    saved_event = dict(event)
    saved_result = dict(result)

    def _retry_until_persisted() -> None:
        delay = _COMPLETION_PERSIST_RETRY_BASE_SECONDS
        while True:
            time.sleep(delay)
            try:
                _persist_completion_once(saved_event, saved_result)
            except sqlite3.OperationalError:
                delay = min(delay * 2, _COMPLETION_PERSIST_RETRY_MAX_SECONDS)
                continue
            except Exception:
                logger.exception(
                    "Async delegation %s terminal persistence retry failed permanently",
                    saved_event.get("delegation_id"),
                )
                return
            if on_persisted is not None:
                on_persisted()
            return

    logger.warning(
        "Async delegation %s terminal persistence exhausted immediate retries; "
        "continuing in background: %s",
        event.get("delegation_id"), last_error,
    )
    threading.Thread(
        target=propagate_context_to_thread(_retry_until_persisted),
        name=f"async-persist-{event.get('delegation_id', 'unknown')}",
        daemon=True,
    ).start()
    return False


def _note_delivery_attempt(delegation_id: str) -> None:
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            "UPDATE async_delegations SET delivery_attempts=delivery_attempts+1, updated_at=? WHERE delegation_id=?",
            (time.time(), delegation_id),
        )


def recover_abandoned_delegations() -> int:
    """Classify records whose owning process disappeared as outcome unknown."""
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        return 0
    now = time.time()
    recovered = 0
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            """SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, goal_id, requires_goal_join,
                      parent_delegation_id
               FROM async_delegations WHERE state IN ('running','finalizing')"""
        ).fetchall()
        for row in rows:
            (
                delegation_id,
                session_key,
                origin_ui,
                parent_id,
                dispatched_at,
                pid,
                started,
                task_json,
                goal_id,
                requires_goal_join,
                parent_delegation_id,
            ) = row
            live = False
            if pid:
                live = _pid_exists(int(pid))
                if live and started is not None:
                    live = get_process_start_time(int(pid)) == int(started)
            if live:
                continue
            task = json.loads(task_json or "{}")
            event = {
                "type": "async_delegation", "delegation_id": delegation_id,
                "session_key": session_key, "origin_ui_session_id": origin_ui,
                "parent_session_id": parent_id, "goal": task.get("goal", ""),
                "goal_id": goal_id or "",
                "goal_owner_session_id": str(task.get("goal_owner_session_id") or ""),
                "requires_goal_join": bool(requires_goal_join),
                "parent_delegation_id": parent_delegation_id or "",
                "goals": task.get("goals"), "context": task.get("context"),
                "toolsets": task.get("toolsets"), "role": task.get("role"),
                "model": task.get("model"), "is_batch": bool(task.get("is_batch")),
                "status": "unknown", "summary": None,
                "error": "Delegation owner exited before recording a terminal result; outcome unknown.",
                "dispatched_at": dispatched_at, "completed_at": now,
            }
            result = {"status": "unknown", "summary": None, "error": event["error"]}
            conn.execute(
                """UPDATE async_delegations SET state='unknown', completed_at=?,
                   updated_at=?, event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""",
                (now, now, json.dumps(event), json.dumps(result), delegation_id),
            )
            recovered += 1
    return recovered


def restore_undelivered_completions(target_queue) -> int:
    """Enqueue durable pending completions as fresh turns after process start.

    Every restored event is stamped ``restored=True`` (in-memory only — the
    stamp is added after the durable payload is deserialized and is never
    persisted). Restored events originate from a *previous* process, so no
    consumer in THIS process implicitly owns them: drain paths that run
    without an ownership filter (the legacy single-session behavior) must
    leave them queued for a consumer that can positively prove ownership,
    otherwise a brand-new session adopts a dead session's delegation
    results seconds after boot (#64484).
    """
    recover_abandoned_delegations()
    with _DB_LOCK, _connect() as conn:
        conn.execute(
            """UPDATE async_delegations
               SET reconciliation_state='pending', reconciliation_claimed_at=NULL,
                   updated_at=?
               WHERE requires_goal_join=1 AND reconciliation_state='claimed'
                 AND reconciliation_claimed_at < ?""",
            (time.time(), time.time() - _RECONCILIATION_CLAIM_LEASE_SECONDS),
        )
        rows = conn.execute(
            """SELECT delegation_id, event_json, delivery_state FROM async_delegations
               WHERE state NOT IN ('running','finalizing') AND event_json IS NOT NULL
                 AND (
                    delivery_state='pending'
                    OR (requires_goal_join=1 AND reconciliation_state='pending')
                 )
               ORDER BY completed_at, delegation_id"""
        ).fetchall()
        for _delegation_id, payload, delivery_state in rows:
            evt = json.loads(payload)
            if isinstance(evt, dict):
                evt["restored"] = True
                if delivery_state != "pending":
                    evt["semantic_recovery"] = True
            target_queue.put(evt)
    return len(rows)


def mark_completion_delivered(delegation_id: str) -> bool:
    """Atomically acknowledge successful injection of a durable completion."""
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered', delivered_at=?, updated_at=?
               WHERE delegation_id=? AND delivery_state!='delivered'""",
            (now, now, delegation_id),
        )
        return cur.rowcount == 1


def claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Claim one pending completion across competing consumers/processes."""
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()
        if row is None:
            return True  # legacy event created before durable dispatch
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_claim=?, delivery_claimed_at=?,
                      delivery_attempts=delivery_attempts+1, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
            (claim_id, now, now, delegation_id, now - 300),
        )
        return cur.rowcount == 1


def claim_event_delivery(evt: Dict[str, Any], consumer: str) -> Optional[str]:
    """Claim a durable delegation event; non-durable events need no token."""
    if evt.get("type") != "async_delegation":
        return ""
    if evt.get("semantic_recovery"):
        return "semantic-recovery"
    delegation_id = str(evt.get("delegation_id") or "")
    if not delegation_id:
        return ""
    claim_id = f"{consumer}:{__import__('os').getpid()}:{uuid.uuid4().hex}"
    return claim_id if claim_completion_delivery(delegation_id, claim_id) else None


def release_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Release a failed delivery claim so another consumer may retry.

    Attempts are counted at claim time, so a row that keeps being claimed and
    released has burned real delivery attempts. Once the budget is exhausted
    the row converges to a terminal ``dropped`` state instead of returning to
    ``pending`` — otherwise an undeliverable completion replays on every
    gateway restart forever (restore_undelivered_completions only restores
    pending rows).
    """
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        capped = conn.execute(
            """UPDATE async_delegations SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=? AND delivery_attempts>=?""",
            (now, delegation_id, claim_id, _MAX_DELIVERY_ATTEMPTS),
        )
        if capped.rowcount == 1:
            logger.warning(
                "Async delegation %s exhausted its %d delivery attempts; "
                "marking terminally dropped (result remains queryable).",
                delegation_id, _MAX_DELIVERY_ATTEMPTS,
            )
            return True
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def drop_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Terminally drop a claimed completion that can never be delivered.

    Used when the delivery target is permanently gone — the spawning session
    ended at an explicit user boundary (/new, reset) rather than a compression
    rotation. Marking the row ``dropped`` (not ``delivered``) keeps the ack
    honest, and (not ``pending``) keeps restart recovery from replaying a
    completion that will be fail-closed dropped again every time.
    """
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='dropped',
                      updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def complete_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Acknowledge acceptance for the consumer holding this claim."""
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=?, updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""",
            (now, now, delegation_id, claim_id),
        )
        return cur.rowcount == 1


def complete_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    if claim_id and claim_id != "semantic-recovery" and evt.get("type") == "async_delegation":
        complete_completion_delivery(str(evt.get("delegation_id") or ""), claim_id)


def release_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    if claim_id and claim_id != "semantic-recovery" and evt.get("type") == "async_delegation":
        release_completion_delivery(str(evt.get("delegation_id") or ""), claim_id)


def get_durable_delegation(delegation_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK, _connect() as conn:
        row = conn.execute(
            """SELECT origin_session, state, dispatched_at, completed_at,
                      result_json, delivery_state, delivery_attempts,
                      goal_id, requires_goal_join, parent_delegation_id,
                      goal_owner_session_id,
                      reconciliation_state, reconciliation_claim,
                      reconciliation_attempts, reconciled_at
               FROM async_delegations WHERE delegation_id=?""", (delegation_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "delegation_id": delegation_id, "origin_session": row[0], "state": row[1],
        "dispatched_at": row[2], "completed_at": row[3],
        "result": json.loads(row[4]) if row[4] else None,
        "delivery_state": row[5], "delivery_attempts": row[6],
        "goal_id": row[7] or "", "requires_goal_join": bool(row[8]),
        "parent_delegation_id": row[9] or "",
        "goal_owner_session_id": row[10] or "",
        "reconciliation_state": row[11], "reconciliation_claim": row[12],
        "reconciliation_attempts": row[13], "reconciled_at": row[14],
    }


def _result_metrics(result: Any) -> Dict[str, Any]:
    """Return bounded observability totals from one stored result payload."""
    totals = {
        "api_calls": 0,
        "tokens": {"input": 0, "output": 0, "reasoning": 0},
        "cost_usd": 0.0,
    }
    if not isinstance(result, dict):
        return totals
    children = result.get("results")
    if isinstance(children, list):
        for child in children[:1000]:
            metrics = _result_metrics(child)
            totals["api_calls"] += metrics["api_calls"]
            totals["cost_usd"] += metrics["cost_usd"]
            for key in totals["tokens"]:
                totals["tokens"][key] += metrics["tokens"][key]
        return totals
    try:
        totals["api_calls"] = max(0, int(result.get("api_calls", 0) or 0))
    except (TypeError, ValueError):
        pass
    tokens = result.get("tokens")
    if isinstance(tokens, dict):
        for key in totals["tokens"]:
            try:
                totals["tokens"][key] = max(0, int(tokens.get(key, 0) or 0))
            except (TypeError, ValueError):
                pass
    try:
        totals["cost_usd"] = max(0.0, float(result.get("cost_usd", 0.0) or 0.0))
    except (TypeError, ValueError):
        pass
    return totals


def _result_failure_summary(result: Any) -> str:
    """Return the first terminal failure, including failures inside a batch."""
    if not isinstance(result, dict):
        return ""
    children = result.get("results")
    if isinstance(children, list):
        for child in children[:1000]:
            failure = _result_failure_summary(child)
            if failure:
                return failure
    status = str(result.get("status") or "").strip().lower()
    if status and status not in {"completed", "success", "done"}:
        return str(result.get("error") or result.get("summary") or status)[:200]
    return ""


def _json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def get_goal_work_snapshot(goal_id: str) -> Dict[str, Any]:
    """Summarize durable required work for one logical standing goal."""
    goal_id = str(goal_id or "").strip()
    snapshot = {
        "goal_id": goal_id,
        "available": True,
        "required_count": 0,
        "running_count": 0,
        "completed_count": 0,
        "terminal_unreconciled_count": 0,
        "claimed_count": 0,
        "reconciled_count": 0,
        "abandoned_count": 0,
        "failed_count": 0,
        "delegation_ids": [],
        "api_calls": 0,
        "tokens": {"input": 0, "output": 0, "reasoning": 0},
        "cost_usd": 0.0,
        "overflow": False,
        "max_attempt": 0,
        "last_error": "",
    }
    if not goal_id:
        return snapshot
    with _DB_LOCK, _connect() as conn:
        rows = conn.execute(
            """SELECT delegation_id, state, reconciliation_state, result_json,
                      reconciliation_attempts
               FROM async_delegations
               WHERE goal_id=? AND requires_goal_join=1
               ORDER BY updated_at, delegation_id""",
            (goal_id,),
        ).fetchall()
    for delegation_id, state, reconciliation_state, result_json, attempts in rows:
        snapshot["required_count"] += 1
        state = str(state or "unknown").lower()
        result = _json_object(result_json)
        if reconciliation_state == "abandoned":
            snapshot["abandoned_count"] += 1
        elif reconciliation_state == "reconciled":
            snapshot["reconciled_count"] += 1
        elif state in {"running", "finalizing"}:
            snapshot["running_count"] += 1
        elif reconciliation_state == "claimed":
            snapshot["claimed_count"] += 1
        else:
            snapshot["terminal_unreconciled_count"] += 1
        if reconciliation_state != "abandoned" and state not in {"running", "finalizing"}:
            if state in {"completed", "success", "done"}:
                snapshot["completed_count"] += 1
            failure = _result_failure_summary(result)
            if state not in {"completed", "success", "done"} or failure:
                snapshot["failed_count"] += 1
                snapshot["last_error"] = str(
                    failure or result.get("error") or result.get("summary") or state
                )[:200]
        if (
            reconciliation_state not in {"reconciled", "abandoned"}
            and len(snapshot["delegation_ids"]) < 20
        ):
            snapshot["delegation_ids"].append(str(delegation_id))
        try:
            metrics = _result_metrics(json.loads(result_json)) if result_json else _result_metrics(None)
        except (TypeError, ValueError, json.JSONDecodeError):
            metrics = _result_metrics(None)
        snapshot["api_calls"] += metrics["api_calls"]
        snapshot["cost_usd"] += metrics["cost_usd"]
        snapshot["max_attempt"] = max(int(snapshot["max_attempt"]), int(attempts or 0))
        for key in snapshot["tokens"]:
            snapshot["tokens"][key] += metrics["tokens"][key]
    snapshot["overflow"] = (
        snapshot["running_count"]
        + snapshot["terminal_unreconciled_count"]
        + snapshot["claimed_count"]
        > _MAX_DURABLE_PENDING
    )
    return snapshot


def claim_goal_reconciliation(goal_id: str, consumer: str) -> Optional[Dict[str, Any]]:
    """Atomically claim a bounded batch of terminal pending rows for ``goal_id``."""
    goal_id = str(goal_id or "").strip()
    if not goal_id:
        return None
    now = time.time()
    claim_id = f"recon_{uuid.uuid4().hex}"
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """UPDATE async_delegations
               SET reconciliation_state='pending', reconciliation_claimed_at=NULL,
                   updated_at=?
               WHERE goal_id=? AND requires_goal_join=1
                 AND reconciliation_state='claimed'
                 AND reconciliation_claimed_at < ?""",
            (now, goal_id, now - _RECONCILIATION_CLAIM_LEASE_SECONDS),
        )
        in_flight = conn.execute(
            """SELECT 1 FROM async_delegations
               WHERE goal_id=? AND requires_goal_join=1
                 AND reconciliation_state='claimed' LIMIT 1""",
            (goal_id,),
        ).fetchone()
        if in_flight:
            conn.commit()
            return None
        rows = conn.execute(
            """SELECT delegation_id FROM async_delegations
               WHERE goal_id=? AND requires_goal_join=1
                 AND state NOT IN ('running','finalizing')
                 AND reconciliation_state='pending'
               ORDER BY completed_at, delegation_id LIMIT ?""",
            (goal_id, _MAX_RECONCILIATION_BATCH_ROWS),
        ).fetchall()
        if not rows:
            conn.commit()
            return None
        delegation_ids = [str(row[0]) for row in rows]
        placeholders = ",".join("?" for _ in delegation_ids)
        conn.execute(
            f"""UPDATE async_delegations
                SET reconciliation_state='claimed', reconciliation_claim=?,
                    reconciliation_claimed_at=?,
                    reconciliation_attempts=reconciliation_attempts+1, updated_at=?
                WHERE delegation_id IN ({placeholders})
                  AND goal_id=? AND requires_goal_join=1
                  AND state NOT IN ('running','finalizing')
                  AND reconciliation_state='pending'""",
            (claim_id, now, now, *delegation_ids, goal_id),
        )
        claimed = conn.execute(
            """SELECT delegation_id, state, task_json, result_json, event_json,
                      reconciliation_attempts, parent_delegation_id
               FROM async_delegations WHERE reconciliation_claim=?
                 AND reconciliation_state='claimed'
               ORDER BY completed_at, delegation_id""",
            (claim_id,),
        ).fetchall()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    delegations = []
    for row in claimed:
        delegations.append({
            "delegation_id": row[0],
            "status": row[1],
            "task": _json_object(row[2]),
            "result": _json_object(row[3]),
            "event": _json_object(row[4]),
            "attempts": int(row[5] or 0),
            "parent_delegation_id": row[6] or "",
        })
    metrics = _result_metrics({
        "results": [item["result"] for item in delegations],
    })
    return {
        "claim_id": claim_id,
        "goal_id": goal_id,
        "consumer": consumer,
        "claimed_at": now,
        "attempt": max((item["attempts"] for item in delegations), default=1),
        "delegation_ids": [item["delegation_id"] for item in delegations],
        "delegations": delegations,
        "api_calls": metrics["api_calls"],
        "tokens": metrics["tokens"],
        "cost_usd": metrics["cost_usd"],
    }


def complete_goal_reconciliation(claim_id: str) -> bool:
    """Mark a fenced reconciliation claim consumed; repeated completion is safe."""
    claim_id = str(claim_id or "").strip()
    if not claim_id:
        return False
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        row = conn.execute(
            """SELECT reconciliation_state FROM async_delegations
               WHERE reconciliation_claim=? LIMIT 1""",
            (claim_id,),
        ).fetchone()
        if row is None:
            return False
        if row[0] == "reconciled":
            return True
        if row[0] != "claimed":
            return False
        conn.execute(
            """UPDATE async_delegations
               SET reconciliation_state='reconciled', reconciled_at=?, updated_at=?
               WHERE reconciliation_claim=? AND reconciliation_state='claimed'""",
            (now, now, claim_id),
        )
        return True


def release_goal_reconciliation(
    claim_id: str,
    *,
    decrement_attempt: bool = False,
) -> bool:
    """Release a fenced claim for retry; stale/wrong tokens cannot steal rows."""
    claim_id = str(claim_id or "").strip()
    if not claim_id:
        return False
    with _DB_LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT reconciliation_state FROM async_delegations WHERE reconciliation_claim=? LIMIT 1",
            (claim_id,),
        ).fetchone()
        if row is None:
            return False
        if row[0] == "claimed":
            conn.execute(
                """UPDATE async_delegations
                   SET reconciliation_state='pending', reconciliation_claimed_at=NULL,
                       reconciliation_attempts=CASE WHEN ?=1
                           THEN MAX(0, reconciliation_attempts-1)
                           ELSE reconciliation_attempts END,
                       updated_at=?
                   WHERE reconciliation_claim=? AND reconciliation_state='claimed'""",
                (1 if decrement_attempt else 0, time.time(), claim_id),
            )
        return row[0] in {"claimed", "pending"}


def abandon_goal_work(goal_id: str, reason: str = "") -> int:
    """Detach all still-required rows when a goal is cleared or replaced."""
    goal_id = str(goal_id or "").strip()
    if not goal_id:
        return 0
    with _DB_LOCK, _connect() as conn:
        cur = conn.execute(
            """UPDATE async_delegations
               SET reconciliation_state='abandoned', reconciliation_claim=NULL,
                   reconciliation_claimed_at=NULL, updated_at=?
               WHERE goal_id=? AND requires_goal_join=1
                 AND reconciliation_state NOT IN ('reconciled','abandoned')""",
            (time.time(), goal_id),
        )
        if cur.rowcount:
            logger.info("Abandoned %d async goal row(s) for %s: %s", cur.rowcount, goal_id, reason)
        return cur.rowcount


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    """Lazily create (or grow) the shared daemon executor.

    We never shrink — ThreadPoolExecutor can't resize — but if the configured
    cap grows between calls we rebuild a larger pool. Existing in-flight
    futures keep running on the old pool until it's garbage collected.
    """
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            # Daemon threads: thread_name_prefix aids debugging in stack dumps.
            _executor = _DaemonThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-delegate",
            )
            _executor_max_workers = max_workers
        return _executor


def active_count() -> int:
    """Number of async delegations currently running."""
    with _records_lock:
        return sum(1 for r in _records.values() if r.get("status") in {"running", "finalizing"})


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def new_delegation_id() -> str:
    """Reserve-format a delegation id before child runners receive lineage."""
    return _new_delegation_id()


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the retention cap.

    Caller must hold ``_records_lock``.
    """
    completed = [
        (rid, r)
        for rid, r in _records.items()
        if r.get("status") not in {"running", "finalizing"}
    ]
    if len(completed) <= _MAX_RETAINED_COMPLETED:
        return
    # Oldest-first by completion time (fall back to dispatch time).
    completed.sort(key=lambda kv: kv[1].get("completed_at") or kv[1].get("dispatched_at") or 0)
    for rid, _ in completed[: len(completed) - _MAX_RETAINED_COMPLETED]:
        _records.pop(rid, None)


def dispatch_async_delegation(
    *,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    parent_session_id: Optional[str] = None,
    goal_id: str = "",
    requires_goal_join: bool = False,
    parent_delegation_id: str = "",
    goal_owner_session_id: str = "",
    delegation_id: Optional[str] = None,
    runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "",
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
) -> Dict[str, Any]:
    """Spawn ``runner`` on the daemon executor and return a handle immediately.

    Parameters
    ----------
    goal, context, toolsets, role, model
        The dispatch-time task spec, captured verbatim for the rich
        completion block.
    session_key
        The gateway session_key (from ``tools.approval.get_current_session_key``)
        captured on the parent thread BEFORE dispatch, because the daemon
        worker thread won't carry the contextvar. Used to route the
        completion back to the originating session.
    parent_session_id
        The durable ``state.db`` session id of the parent agent that spawned
        the delegation. Carried on the completion event so the gateway can
        pin routing to the spawning session instead of recovering the latest
        ``ended_at IS NULL`` row for the peer tuple (#57498).
    runner
        Zero-arg callable that builds + runs the child and returns the same
        result dict ``_run_single_child`` produces. Runs on the worker thread.
    interrupt_fn
        Optional callable to signal the child to stop (used on shutdown /
        explicit cancel).
    max_async_children
        Concurrency cap. When at capacity the dispatch is REJECTED (the caller
        should fall back to sync or tell the user) rather than queued, so a
        runaway model can't pile up unbounded background work.

    Returns
    -------
    dict
        ``{"status": "dispatched", "delegation_id": ...}`` on success, or
        ``{"status": "rejected", "error": ...}`` when at capacity.
    """
    delegation_id = str(delegation_id or _new_delegation_id())
    dispatched_at = time.time()
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": goal,
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "origin_ui_session_id": origin_ui_session_id,
        "parent_session_id": parent_session_id,
        "goal_id": str(goal_id or ""),
        "requires_goal_join": bool(requires_goal_join and goal_id),
        "parent_delegation_id": str(parent_delegation_id or ""),
        "goal_owner_session_id": str(goal_owner_session_id or ""),
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
    }
    # Capacity check and record insert under ONE lock hold — checking
    # active_count() separately would let two concurrent dispatches (e.g.
    # from different gateway sessions) both pass the check and exceed the cap.
    with _records_lock:
        if delegation_id in _records:
            return {
                "status": "rejected",
                "error": f"Async delegation id already exists: {delegation_id}",
            }
        active = sum(
            1
            for r in _records.values()
            if r.get("status") in {"running", "finalizing"}
        )
        if active >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"active). Wait for one to finish (its result will re-enter "
                    f"the chat), or run this task synchronously "
                    f"(background=false). Raise delegation.max_concurrent_children in "
                    f"config.yaml to allow more concurrent background subagents."
                ),
            }
        _records[delegation_id] = record

    try:
        _persist_dispatch(record)
    except Exception as exc:  # noqa: BLE001 — dispatch must not leave a phantom slot
        with _records_lock:
            _records.pop(delegation_id, None)
        logger.exception("Failed to persist async delegation %s", delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to persist async delegation: {exc}",
        }
    executor = _get_executor(max_async_children)

    def _worker() -> None:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            status = result.get("status") or "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation %s crashed", delegation_id)
            result = {
                "status": "error",
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
                "api_calls": 0,
                "duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize(delegation_id, result, status)

    try:
        # Propagate the dispatching profile so the detached child resolves
        # get_hermes_home() under the right profile.
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover — pool submit failure is rare
        with _records_lock:
            _records.pop(delegation_id, None)
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation: {exc}",
        }

    logger.info(
        "Dispatched async delegation %s (session_key=%s): %s",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    dispatch: Dict[str, Any] = {
        "status": "dispatched",
        "delegation_id": delegation_id,
    }
    if goal_id and requires_goal_join:
        dispatch.update({
            "goal_id": goal_id,
            "requires_goal_join": True,
            "parent_delegation_id": str(parent_delegation_id or ""),
            "goal_owner": {
                "goal_id": goal_id,
                "requires_goal_join": True,
            },
        })
    return dispatch


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        # Stay active until durable persistence and queue publication finish;
        # otherwise process shutdown can kill this daemon worker in the narrow
        # gap after status flips but before SQLite is committed.
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None  # drop the closure; child is done
        event_record = dict(record)

    _push_completion_event(event_record, result, status)


def _finish_in_memory_completion(delegation_id: str, status: str) -> None:
    with _records_lock:
        record = _records.get(delegation_id)
        if record is not None:
            record["status"] = status
        _prune_completed_locked()


def _publish_persisted_completion(
    delegation_id: str, status: str, event: Dict[str, Any], target_queue: Any
) -> None:
    try:
        target_queue.put(event)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s: failed to enqueue completion event; "
            "leaving the result durable for recovery: %s",
            delegation_id,
            exc,
        )
    finally:
        _finish_in_memory_completion(delegation_id, status)


def _push_completion_event(
    record: Dict[str, Any], result: Dict[str, Any], status: str
) -> None:
    """Push a type='async_delegation' event onto the shared completion queue.

    Terminal state is persisted before best-effort queue publication. A queue
    or registry failure must not crash the worker or strand goal-owned work;
    the durable event remains available for recovery, and we log loudly.
    """
    summary = result.get("summary")
    error = result.get("error")
    dispatched_at = record.get("dispatched_at") or time.time()
    completed_at = record.get("completed_at") or time.time()

    evt = {
        "type": "async_delegation",
        "delegation_id": record.get("delegation_id"),
        # session_key routes the completion back to the originating gateway
        # session; empty string => CLI (single-session) path.
        "session_key": record.get("session_key", ""),
        "origin_ui_session_id": record.get("origin_ui_session_id", ""),
        "parent_session_id": record.get("parent_session_id"),
        "goal_id": record.get("goal_id", ""),
        "requires_goal_join": bool(record.get("requires_goal_join")),
        "parent_delegation_id": record.get("parent_delegation_id", ""),
        "goal_owner_session_id": record.get("goal_owner_session_id", ""),
        "goal": record.get("goal", ""),
        "context": record.get("context"),
        "toolsets": record.get("toolsets"),
        "role": record.get("role"),
        "model": result.get("model") or record.get("model"),
        "status": status,
        "summary": summary,
        "error": error,
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": result.get(
            "duration_seconds", round(completed_at - dispatched_at, 2)
        ),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "exit_reason": result.get("exit_reason"),
        "tokens": result.get("tokens"),
        "cost_usd": result.get("cost_usd"),
    }
    def publish() -> None:
        try:
            from tools.process_registry import process_registry
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Async delegation %s persisted but process_registry import failed; "
                "leaving the result durable for recovery: %s",
                record.get("delegation_id"),
                exc,
            )
            _finish_in_memory_completion(
                str(record.get("delegation_id") or ""), status
            )
            return
        _publish_persisted_completion(
            str(record.get("delegation_id") or ""),
            status,
            evt,
            process_registry.completion_queue,
        )

    if _persist_completion(evt, result, on_persisted=publish):
        publish()


def dispatch_async_delegation_batch(
    *,
    goals: List[str],
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    parent_session_id: Optional[str] = None,
    goal_id: str = "",
    requires_goal_join: bool = False,
    parent_delegation_id: str = "",
    goal_owner_session_id: str = "",
    runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "",
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    delegation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatch a WHOLE fan-out batch as ONE background unit.

    Unlike ``dispatch_async_delegation`` (which backs a single subagent),
    ``runner`` here runs the entire batch — it builds and joins on every child
    in parallel and returns the combined ``{"results": [...],
    "total_duration_seconds": N}`` dict that the synchronous path would have
    returned. We occupy ONE async slot for the whole batch (the in-batch
    parallelism is bounded separately by ``max_concurrent_children``), so a
    single ``delegate_task`` fan-out never exhausts the async pool by itself.

    When the batch finishes, a SINGLE completion event is pushed onto the
    shared ``process_registry.completion_queue`` carrying the full per-task
    ``results`` list, so the consolidated summaries re-enter the conversation
    as one message once every child is done — the chat is never blocked while
    they run.

    Returns ``{"status": "dispatched", "delegation_id": ...}`` on success or
    ``{"status": "rejected", "error": ...}`` when the async pool is at
    capacity.
    """
    delegation_id = delegation_id or _new_delegation_id()
    dispatched_at = time.time()
    n = len(goals)
    # A combined goal label for status listings / the completion header.
    combined_goal = (
        goals[0] if n == 1 else f"{n} parallel subagents: " + "; ".join(g[:40] for g in goals)
    )
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": combined_goal,
        "goals": list(goals),
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "origin_ui_session_id": origin_ui_session_id,
        "parent_session_id": parent_session_id,
        "goal_id": str(goal_id or ""),
        "requires_goal_join": bool(requires_goal_join and goal_id),
        "parent_delegation_id": str(parent_delegation_id or ""),
        "goal_owner_session_id": str(goal_owner_session_id or ""),
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "is_batch": True,
    }
    with _records_lock:
        if delegation_id in _records:
            return {
                "status": "rejected",
                "error": f"Async delegation id already exists: {delegation_id}",
            }
        active = sum(
            1
            for r in _records.values()
            if r.get("status") in {"running", "finalizing"}
        )
        if active >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"active). Wait for one to finish (its result will re-enter "
                    f"the chat), or raise delegation.max_concurrent_children in "
                    f"config.yaml to allow more concurrent background units."
                ),
            }
        _records[delegation_id] = record

    try:
        _persist_dispatch(record)
    except Exception as exc:  # noqa: BLE001 — dispatch must not leave a phantom slot
        with _records_lock:
            _records.pop(delegation_id, None)
        logger.exception("Failed to persist async delegation batch %s", delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to persist async delegation batch: {exc}",
        }
    executor = _get_executor(max_async_children)

    def _worker() -> None:
        combined: Dict[str, Any] = {}
        status = "error"
        try:
            combined = runner() or {}
            # Batch status: completed unless every child errored/was interrupted.
            child_results = combined.get("results") or []
            if child_results and all(
                (r.get("status") not in ("completed", "success"))
                for r in child_results
            ):
                status = "error"
            else:
                status = "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation batch %s crashed", delegation_id)
            combined = {
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
                "total_duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize_batch(delegation_id, combined, status)

    try:
        # Propagate the dispatching profile to the detached batch children.
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover
        with _records_lock:
            _records.pop(delegation_id, None)
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation batch: {exc}",
        }

    logger.info(
        "Dispatched async delegation batch %s (%d task(s), session_key=%s)",
        delegation_id, n, session_key or "<cli>",
    )
    dispatch: Dict[str, Any] = {
        "status": "dispatched",
        "delegation_id": delegation_id,
    }
    if goal_id and requires_goal_join:
        dispatch.update({
            "goal_id": goal_id,
            "requires_goal_join": True,
            "parent_delegation_id": str(parent_delegation_id or ""),
        })
    return dispatch


def _finalize_batch(
    delegation_id: str, combined: Dict[str, Any], status: str
) -> None:
    """Mark a batch record complete and push ONE combined completion event."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None
        event_record = dict(record)

    dispatched_at = event_record.get("dispatched_at") or time.time()
    completed_at = event_record.get("completed_at") or time.time()
    evt = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": event_record.get("session_key", ""),
        "origin_ui_session_id": event_record.get("origin_ui_session_id", ""),
        "parent_session_id": event_record.get("parent_session_id"),
        "goal_id": event_record.get("goal_id", ""),
        "requires_goal_join": bool(event_record.get("requires_goal_join")),
        "parent_delegation_id": event_record.get("parent_delegation_id", ""),
        "goal_owner_session_id": event_record.get("goal_owner_session_id", ""),
        "goal": event_record.get("goal", ""),
        "goals": event_record.get("goals"),
        "context": event_record.get("context"),
        "toolsets": event_record.get("toolsets"),
        "role": event_record.get("role"),
        "model": event_record.get("model"),
        "status": status,
        "is_batch": True,
        # The full per-task results list — the formatter renders a
        # consolidated multi-task block from this.
        "results": combined.get("results") or [],
        # Per-task live transcript log paths (cache/delegation/live/...).
        # They persist after completion and double as the full-fidelity
        # operational record of each child's run.
        "live_transcripts": combined.get("live_transcripts"),
        "error": combined.get("error"),
        "total_duration_seconds": combined.get("total_duration_seconds"),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
    }
    def publish() -> None:
        try:
            from tools.process_registry import process_registry
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Async delegation batch %s persisted but process_registry import "
                "failed; leaving the result durable for recovery: %s",
                delegation_id,
                exc,
            )
            _finish_in_memory_completion(delegation_id, status)
            return
        _publish_persisted_completion(
            delegation_id, status, evt, process_registry.completion_queue
        )

    if _persist_completion(evt, combined, on_persisted=publish):
        publish()


def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed).

    Safe to call from any thread. Excludes the non-serialisable interrupt_fn.
    """
    with _records_lock:
        return [
            {k: v for k, v in r.items() if k != "interrupt_fn"}
            for r in _records.values()
        ]


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every running async delegation to stop. Returns how many.

    Used on ``/stop`` and gateway shutdown so a dangling background subagent
    can't keep burning tokens with no one listening. The child still emits a
    completion event (status='interrupted') via the normal finalize path.
    """
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values() if r.get("status") == "running"
        ]
    for r in targets:
        fn = r.get("interrupt_fn")
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception as exc:
                logger.debug(
                    "interrupt_all: %s interrupt failed: %s",
                    r.get("delegation_id"), exc,
                )
    if count:
        logger.info("Interrupted %d async delegation(s) (%s)", count, reason)
    return count


def _goal_reconciliation_retry_event(
    goal_id: str,
    session_id: str,
    source: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "type": "async_delegation",
        "semantic_recovery": True,
        "delegation_id": f"recon_retry_{uuid.uuid4().hex[:12]}",
        "goal_id": goal_id,
        "requires_goal_join": True,
        "session_key": source.get("session_key") or session_id,
        "origin_ui_session_id": source.get("origin_ui_session_id"),
        "parent_session_id": source.get("parent_session_id") or session_id,
        "parent_delegation_id": source.get("parent_delegation_id") or "",
        "goal_owner_session_id": source.get("goal_owner_session_id") or session_id,
        "status": "reconciliation_retry",
        "task": {"goal": "Retry standing-goal async result reconciliation"},
    }


def recover_stale_goal_reconciliation_claims(target_queue) -> int:
    """Release expired semantic claims and enqueue one fenced retry per claim."""
    now = time.time()
    cutoff = now - _RECONCILIATION_CLAIM_LEASE_SECONDS
    conn = _connect()
    recovered_events = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        claims = conn.execute(
            """SELECT reconciliation_claim, goal_id, event_json, parent_session_id
               FROM async_delegations
               WHERE requires_goal_join=1 AND reconciliation_state='claimed'
                 AND reconciliation_claimed_at < ?
               GROUP BY reconciliation_claim, goal_id
               ORDER BY MIN(reconciliation_claimed_at)""",
            (cutoff,),
        ).fetchall()
        for claim_id, goal_id, event_json, parent_session_id in claims:
            updated = conn.execute(
                """UPDATE async_delegations
                   SET reconciliation_state='pending', reconciliation_claimed_at=NULL,
                       updated_at=?
                   WHERE reconciliation_claim=? AND reconciliation_state='claimed'
                     AND reconciliation_claimed_at < ?""",
                (now, claim_id, cutoff),
            ).rowcount
            if updated:
                recovered_events.append(
                    _goal_reconciliation_retry_event(
                        str(goal_id or ""),
                        str(parent_session_id or ""),
                        _json_object(event_json),
                    )
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    for event in recovered_events:
        target_queue.put(event)
    return len(recovered_events)


def build_goal_reconciliation_retry_event(
    goal_id: str,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Build a queue-only wakeup for pending semantic reconciliation work."""
    goal_id = str(goal_id or "").strip()
    if not goal_id:
        return None
    with _DB_LOCK, _connect() as conn:
        row = conn.execute(
            """SELECT event_json FROM async_delegations
               WHERE goal_id=? AND requires_goal_join=1
                 AND reconciliation_state='pending'
                 AND state NOT IN ('running','finalizing')
               ORDER BY completed_at DESC, delegation_id DESC LIMIT 1""",
            (goal_id,),
        ).fetchone()
    if row is None:
        return None
    source = _json_object(row[0])
    return _goal_reconciliation_retry_event(goal_id, session_id, source)


def build_goal_reconciliation_followup_event(
    claim_id: str,
) -> Optional[Dict[str, Any]]:
    """Wake the next bounded batch after a claim reconciles successfully."""
    claim_id = str(claim_id or "").strip()
    if not claim_id:
        return None
    with _DB_LOCK, _connect() as conn:
        owner = conn.execute(
            """SELECT goal_id, parent_session_id
               FROM async_delegations WHERE reconciliation_claim=? LIMIT 1""",
            (claim_id,),
        ).fetchone()
    if owner is None:
        return None
    return build_goal_reconciliation_retry_event(
        str(owner[0] or ""),
        str(owner[1] or ""),
    )


def interrupt_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
    reason: str = "session_end",
) -> int:
    """Signal running async delegations owned by ONE session to stop.

    A delegation's lifecycle is bound to the session that spawned it: when
    that session ends, its in-flight background subagents must end with it —
    a completed orphan would otherwise sit on the shared completion queue
    with no live owner, either leaking into another chat or burning tokens
    with no one listening (#55578).

    Selectors (any matching field claims the record):
    - ``origin_ui_session_id``: the live TUI tab/window that commissioned it.
    - ``session_key``: the durable routing key captured at dispatch.
    - ``parent_session_id``: the spawning agent's durable session-db id —
      the right selector for gateway chats, whose ``session_key`` (the
      platform conversation key) SURVIVES a ``/new`` reset while the
      session id rotates.

    Returns how many were interrupted.
    """
    if not session_key and not origin_ui_session_id and not parent_session_id:
        return 0
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values()
            if r.get("status") == "running"
            and (
                (origin_ui_session_id and str(r.get("origin_ui_session_id") or "") == origin_ui_session_id)
                or (session_key and str(r.get("session_key") or "") == session_key)
                or (parent_session_id and str(r.get("parent_session_id") or "") == parent_session_id)
            )
        ]
    for r in targets:
        fn = r.get("interrupt_fn")
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception as exc:
                logger.debug(
                    "interrupt_for_session: %s interrupt failed: %s",
                    r.get("delegation_id"), exc,
                )
    if count:
        logger.info(
            "Interrupted %d async delegation(s) for ending session (%s)",
            count, reason,
        )
    return count


def _reset_for_tests() -> None:
    """Test-only: clear all state and tear down the executor."""
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None
        _executor_max_workers = 0
    with _records_lock:
        _records.clear()
