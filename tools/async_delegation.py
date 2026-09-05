#!/usr/bin/env python3
"""Async (background) delegation registry behind ``delegate_task(background=true)``.

The parent dispatches a subagent on a module-level daemon executor and returns a handle
immediately. On completion a ``type="async_delegation"`` event (self-contained task-source
block) is pushed onto the SHARED ``process_registry.completion_queue`` the CLI/gateway drain
while idle, so results surface as a NEW turn (never mid-turn) and inherit its de-dup and
crash-recovery wiring. Only the async lifecycle lives here; the child run is an injected ``runner``."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────────────────────
# Persistent daemon executor (never a `with ThreadPoolExecutor()` block, which
# would join on exit and defeat async); daemon workers can't hang a hard exit.
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_executor_max_workers: int = 0

_records_lock = threading.Lock()
# delegation_id -> record dict; kept for the run plus a short completed tail.
_records: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MAX_ASYNC_CHILDREN = 3
# Completed records retained (in memory and in the ledger) for status queries.
_MAX_RETAINED_COMPLETED = 50
_DURABLE_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_DURABLE_PENDING = 1000
# Cap retried deliveries so an unroutable row converges to terminal 'dropped'.
_MAX_DELIVERY_ATTEMPTS = 8
# Pending completions older than this are dropped on restart replay instead of
# re-run as a full-context turn; 48h keeps weekend results deliverable.
_MAX_COMPLETION_REPLAY_AGE_S = 48 * 3600.0
_DB_LOCK = threading.Lock()

# ── Stale-delegation detection (progress-based, on by default) ──────────────
# A runner wedged before returning never reaches its finalizer, so it would show
# "dispatched" forever. No wall-clock timeout (heavy work must never be killed for
# taking long): one monitor thread samples per-dispatch PROGRESS via an injected
# ``progress_fn``; a frozen child is interrupted, given a grace window to unwind via
# the normal finalize path, and only force-finalized (terminal ``stalled`` event) if
# it never returns. Thresholds mirror delegate_tool's sync heartbeat monitor.
_STALE_CHECK_INTERVAL = 30.0
_STALE_IDLE_SECONDS = 450.0
_STALE_IN_TOOL_SECONDS = 1200.0
_STALL_GRACE_SECONDS = 120.0

_monitor_lock = threading.Lock()
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()

_LIVE_STATES = {"running", "stalling", "finalizing"}
_ACTIVE_STATES = ("running", "stalling")
# Routing origin persisted at dispatch so a restart-recovered completion can
# reconstruct a full SessionSource (scope_id drives relay tenant egress).
_ROUTING_KEYS = ("scope_id", "user_id", "user_name")
# Structured stall metadata — additive, present only on stall finalizations.
_STALL_META_KEYS = ("stalled_after_quiet_seconds", "stall_threshold_seconds", "stall_phase", "stall_grace_seconds")
# Private stall bookkeeping on the record -> public field in list_async_delegations().
_STALL_FIELD_MAP = (("_stall_quiet_seconds", "stalled_after_quiet_seconds"),
                    ("_stall_threshold_seconds", "stall_threshold_seconds"), ("_stall_in_tool", "stall_in_tool"))


# ── Durable ledger (state.db / async_delegations) ───────────────────────────
def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        conn.close()  # don't leak the connection on PRAGMA/DDL failure
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state_repair import apply_durability_barriers
    # Preserve the journal mode SessionDB configured on state.db: forcing WAL from
    # every short-lived connection collides with live transcript/FTS writers.
    apply_durability_barriers(conn)
    conn.execute("""CREATE TABLE IF NOT EXISTS async_delegations (
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
            origin_session_id TEXT NOT NULL DEFAULT ''
        )""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")}
    # origin_session_id: raw api_server session id of the ORIGINATING request
    # (wake target); without it restart-recovered completions are unroutable there.
    for name, sql_type in (("owner_pid", "INTEGER"), ("owner_started_at", "INTEGER"), ("task_json", "TEXT"),
                           ("delivery_claim", "TEXT"), ("delivery_claimed_at", "REAL"), ("origin_session_id", "TEXT")):
        if name not in columns:
            conn.execute(f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}")


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it (``with conn:``
    alone leaks the connection and WAL/SHM fds until GC).

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the transaction; they do not
    close the connection. Using ``with _connect()`` alone therefore leaks a connection — and its WAL/SHM
    file descriptors — on every durable dispatch, completion, and delivery-claim, deferring the close to the
    garbage collector. On a long-running gateway that exhausts ``RLIMIT_NOFILE`` (the cron-ledger sibling of
    this bug was #69567 / PR #69594).
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _capture_routing_origin() -> Dict[str, Any]:
    """Snapshot scope_id/user_id/user_name on the PARENT thread (the daemon worker
    has no contextvars) so a restart-replayed completion can rebuild a SessionSource.
    Best-effort: empty values are omitted."""
    try:
        from gateway.session_context import get_session_env
        return {k: v for k in _ROUTING_KEYS if (v := get_session_env(f"HERMES_SESSION_{k.upper()}", ""))}
    except Exception:  # noqa: BLE001 - routing origin is additive, never fatal
        return {}


def _persist_dispatch(record: Dict[str, Any]) -> None:
    now = time.time()
    try:
        from gateway.status import get_process_start_time
        owner_started_at = get_process_start_time(os.getpid())
    except Exception:
        owner_started_at = None
    task_payload = {
        key: record.get(key)
        for key in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch", *_ROUTING_KEYS)
        if key in record}
    with _DB_LOCK, _transaction() as conn:
        conn.execute("""INSERT OR REPLACE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, origin_session_id)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (record["delegation_id"], record.get("session_key", ""), record.get("origin_ui_session_id", ""),
             record.get("parent_session_id"), record["dispatched_at"], now, os.getpid(), owner_started_at,
             json.dumps(task_payload), record.get("origin_session_id", "")))
    _prune_durable_records()


def _prune_durable_records() -> None:
    """Bound terminal history, preferring delivered records for deletion."""
    cutoff = time.time() - _DURABLE_RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < ?", (cutoff,))
        terminal_count = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing')").fetchone()[0]
        if terminal_count > _MAX_RETAINED_COMPLETED:
            conn.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing')
                     ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                              updated_at ASC LIMIT ?
                   )""", (terminal_count - _MAX_RETAINED_COMPLETED,))
        pending_count = conn.execute("""SELECT COUNT(*) FROM async_delegations
               WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'""").fetchone()[0]
        if pending_count > _MAX_DURABLE_PENDING:
            conn.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     ORDER BY updated_at ASC LIMIT ?
                   )""", (pending_count - _MAX_DURABLE_PENDING,))


def _persist_completion(event: Dict[str, Any], result: Dict[str, Any]) -> None:
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute("""UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
               event_json=?, result_json=?, delivery_state='pending'
               WHERE delegation_id=?""",
            (event.get("status", "completed"), event.get("completed_at", now), now,
             json.dumps(event), json.dumps(result), event["delegation_id"]))


def recover_abandoned_delegations() -> int:
    """Classify records whose owning process disappeared as outcome unknown."""
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        return 0
    now, recovered = time.time(), 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute("""SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, origin_session_id
               FROM async_delegations WHERE state IN ('running','finalizing')""").fetchall()
        for row in rows:
            delegation_id, session_key, origin_ui, parent_id, dispatched_at, pid, started, task_json, origin_sid = row
            if pid and _pid_exists(int(pid)) and (started is None or get_process_start_time(int(pid)) == int(started)):
                continue
            task = json.loads(task_json or "{}")
            event = {
                "type": "async_delegation", "delegation_id": delegation_id, "session_key": session_key,
                "origin_ui_session_id": origin_ui, "origin_session_id": origin_sid or "",
                "parent_session_id": parent_id, "goal": task.get("goal", ""), "goals": task.get("goals"),
                "context": task.get("context"), "toolsets": task.get("toolsets"), "role": task.get("role"),
                "model": task.get("model"), "is_batch": bool(task.get("is_batch")),
                "status": "unknown", "summary": None,
                "error": "Delegation owner exited before recording a terminal result; outcome unknown.",
                "dispatched_at": dispatched_at, "completed_at": now,
                **{k: task[k] for k in _ROUTING_KEYS if task.get(k)}}
            result = {"status": "unknown", "summary": None, "error": event["error"]}
            conn.execute("""UPDATE async_delegations SET state='unknown', completed_at=?,
                   updated_at=?, event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""", (now, now, json.dumps(event), json.dumps(result), delegation_id))
            recovered += 1
    return recovered


def restore_undelivered_completions(target_queue) -> int:
    """Enqueue durable pending completions as fresh turns after process start.
    Restored events are stamped ``restored=True`` in memory only: they came from a PREVIOUS
    process, so drains without an ownership filter must leave them for a consumer that can
    prove ownership. Rows older than ``_MAX_COMPLETION_REPLAY_AGE_S`` are terminally dropped
    instead of replaying a turn nobody is waiting on.

    Every restored event is stamped ``restored=True`` (in-memory only — the stamp is added after the durable
    payload is deserialized and is never persisted). Restored events originate from a *previous* process, so
    no consumer in THIS process implicitly owns them: drain paths that run without an ownership filter (the
    legacy single-session behavior) must leave them queued for a consumer that can positively prove
    ownership, otherwise a brand-new session adopts a dead session's delegation results seconds after boot
    (#64484).
    """
    recover_abandoned_delegations()
    now, restored = time.time(), 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at
               FROM async_delegations
               WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
               ORDER BY completed_at, delegation_id""").fetchall()
        for delegation_id, payload, completed_at, dispatched_at in rows:
            age_basis = completed_at or dispatched_at
            if age_basis and (now - age_basis) > _MAX_COMPLETION_REPLAY_AGE_S:
                conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                              delivery_claim=NULL, delivery_claimed_at=NULL,
                              updated_at=?
                       WHERE delegation_id=? AND delivery_state='pending'""", (now, delegation_id))
                logger.warning("Async delegation %s: pending completion is %.1fh old "
                               "(cap %.1fh); terminally dropping the replay (result remains queryable).",
                               delegation_id, (now - age_basis) / 3600.0, _MAX_COMPLETION_REPLAY_AGE_S / 3600.0)
                continue
            evt = json.loads(payload)
            if isinstance(evt, dict):
                evt["restored"] = True
            target_queue.put(evt)
            restored += 1
    return restored


def _update_delivery(sql: str, params: tuple) -> bool:
    """Run one UPDATE on the ledger; True iff exactly one row changed."""
    with _DB_LOCK, _transaction() as conn:
        return conn.execute(sql, params).rowcount == 1


def mark_completion_delivered(delegation_id: str) -> bool:
    """Atomically acknowledge successful injection of a durable completion."""
    now = time.time()
    return _update_delivery(
        """UPDATE async_delegations SET delivery_state='delivered', delivered_at=?, updated_at=?
           WHERE delegation_id=? AND delivery_state!='delivered'""", (now, now, delegation_id))


def claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Claim one pending completion across competing consumers/processes."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone()
        if row is None:
            return True  # legacy event created before durable dispatch
        cur = conn.execute("""UPDATE async_delegations SET delivery_claim=?, delivery_claimed_at=?,
                      delivery_attempts=delivery_attempts+1, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
            (claim_id, now, now, delegation_id, now - 300))
        return cur.rowcount == 1


def is_interim_delegation_event(evt: Dict[str, Any]) -> bool:
    """An early per-task notice for a batch that is still running. It shares the batch's
    ``delegation_id`` but is NOT the durable completion: it must never claim, acknowledge or
    dedup against the final result's row (independent review reproduced exactly that loss)."""
    return evt.get("type") == "async_delegation" and bool(evt.get("task_failure_notice"))


def claim_event_delivery(evt: Dict[str, Any], consumer: str) -> Optional[str]:
    """Claim a durable delegation event; non-durable events (and interim notices) need no token."""
    if is_interim_delegation_event(evt):
        return ""
    delegation_id = str(evt.get("delegation_id") or "") if evt.get("type") == "async_delegation" else ""
    if not delegation_id:
        return ""
    claim_id = f"{consumer}:{os.getpid()}:{uuid.uuid4().hex}"
    return claim_id if claim_completion_delivery(delegation_id, claim_id) else None


def release_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Release a failed delivery claim so another consumer may retry. Attempts are
    counted at claim time; once the budget is exhausted the row converges to
    terminal ``dropped`` (only pending rows replay on restart)."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        capped = conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                      delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=? AND delivery_attempts>=?""",
            (now, delegation_id, claim_id, _MAX_DELIVERY_ATTEMPTS))
        if capped.rowcount == 1:
            logger.warning("Async delegation %s exhausted its %d delivery attempts; "
                           "marking terminally dropped (result remains queryable).",
                           delegation_id, _MAX_DELIVERY_ATTEMPTS)
            return True
        cur = conn.execute("""UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, updated_at=?
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""", (now, delegation_id, claim_id))
        return cur.rowcount == 1


def drop_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Terminally drop a claimed completion whose target is permanently gone (the
    spawning session ended at an explicit user boundary such as /new or reset).
    ``dropped`` — not ``delivered`` — keeps the ack honest; not ``pending`` keeps
    restart recovery from replaying it into a fail-closed drop forever."""
    return _update_delivery("""UPDATE async_delegations SET delivery_state='dropped',
                  updated_at=?, delivery_claim=NULL,
                  delivery_claimed_at=NULL
           WHERE delegation_id=? AND delivery_state='pending'
             AND delivery_claim=?""", (time.time(), delegation_id, claim_id))


def complete_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Acknowledge acceptance for the consumer holding this claim."""
    now = time.time()
    return _update_delivery("""UPDATE async_delegations SET delivery_state='delivered',
                  delivered_at=?, updated_at=?, delivery_claim=NULL,
                  delivery_claimed_at=NULL
           WHERE delegation_id=? AND delivery_state='pending'
             AND delivery_claim=?""", (now, now, delegation_id, claim_id))


def complete_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    _event_delivery(complete_completion_delivery, evt, claim_id)


def release_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    _event_delivery(release_completion_delivery, evt, claim_id)


def _event_delivery(fn, evt: Dict[str, Any], claim_id: str) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        fn(str(evt.get("delegation_id") or ""), claim_id)


def get_durable_delegation(delegation_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute("""SELECT origin_session, state, dispatched_at, completed_at,
                      result_json, delivery_state, delivery_attempts,
                      origin_session_id
               FROM async_delegations WHERE delegation_id=?""", (delegation_id,)).fetchone()
    return None if row is None else {
        "delegation_id": delegation_id, "origin_session": row[0], "state": row[1], "dispatched_at": row[2],
        "completed_at": row[3], "result": json.loads(row[4]) if row[4] else None, "delivery_state": row[5],
        "delivery_attempts": row[6], "origin_session_id": row[7] or ""}


# ── In-memory registry queries ──────────────────────────────────────────────
def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    """Lazily create (or grow, never shrink) the shared daemon executor; in-flight
    futures keep running on a replaced pool until it is collected."""
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            _executor = DaemonThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="async-delegate")
            _executor_max_workers = max_workers
        return _executor


def active_count() -> int:
    """Number of live async delegation UNITS (a whole batch counts as ONE slot)."""
    with _records_lock:
        return sum(1 for r in _records.values() if r.get("status") in _LIVE_STATES)


def active_task_count() -> int:
    """Number of running child subagents (a batch of N contributes N; a batch with
    no goal list counts 1) — the truthful observability figure, unlike slots."""
    with _records_lock:
        return sum(
            len(r["goals"]) if r.get("is_batch") and isinstance(r.get("goals"), (list, tuple)) and r["goals"] else 1
            for r in _records.values() if r.get("status") in {"running", "finalizing"})


def _session_records(statuses, session_key: str, origin_ui_session_id: str, parent_session_id: str) -> list:
    """Records in ``statuses`` owned by a session: any non-empty selector claims the
    record — ``origin_ui_session_id`` (TUI tab), ``session_key`` (routing key at
    dispatch), or ``parent_session_id`` (spawner's durable id — the right one for
    gateway chats, whose session_key survives ``/new`` while the session id rotates)."""
    selectors = [(field, wanted) for field, wanted in (
        ("origin_ui_session_id", origin_ui_session_id), ("session_key", session_key),
        ("parent_session_id", parent_session_id)) if wanted]
    if not selectors:
        return []
    with _records_lock:
        return [r for r in _records.values() if r.get("status") in statuses
                and any(str(r.get(field) or "") == wanted for field, wanted in selectors)]


def has_live_for_session(session_key: str = "", origin_ui_session_id: str = "", parent_session_id: str = "") -> bool:
    """Whether a session still owns any live (running/stalling/finalizing) delegation."""
    return bool(_session_records(_LIVE_STATES, session_key, origin_ui_session_id, parent_session_id))


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the cap. Caller holds ``_records_lock``."""
    completed = [(rid, r) for rid, r in _records.items() if r.get("status") != "running"]
    completed.sort(key=lambda kv: kv[1].get("completed_at") or kv[1].get("dispatched_at") or 0)
    for rid, _ in completed[: max(0, len(completed) - _MAX_RETAINED_COMPLETED)]:
        _records.pop(rid, None)


def _current_origin_session_id() -> str:
    """Raw session id of the ORIGINATING api_server request, or ``""``. ``HERMES_SESSION_ID``
    is unsafe here: building the child agent calls ``set_current_session_id(child.session_id)``
    just before dispatch, so the wake would self-post into the subagent's own session. The
    request-scoped ``HERMES_SESSION_CHAT_ID`` (raw X-Hermes-Session-Id on api_server) survives
    child construction; on push platforms chat_id is a chat, not a session => ``""``."""
    try:
        from gateway.session_context import get_session_env
        is_api = get_session_env("HERMES_SESSION_PLATFORM", "") == "api_server"
        return (get_session_env("HERMES_SESSION_CHAT_ID", "") or "") if is_api else ""
    except Exception:
        return ""


# ── Dispatch ────────────────────────────────────────────────────────────────
def _single_crash(error: str, duration: float) -> Dict[str, Any]:
    return {"status": "error", "summary": None, "error": error, "api_calls": 0, "duration_seconds": duration}


def _batch_crash(error: str, duration: float) -> Dict[str, Any]:
    return {"results": [], "error": error, "total_duration_seconds": duration}


def _batch_status(combined: Dict[str, Any]) -> str:
    """Batch status: completed unless every child errored/was interrupted."""
    child_results = combined.get("results") or []
    ok = ("completed", "success")
    return "error" if child_results and all(r.get("status") not in ok for r in child_results) else "completed"


def _dispatch(
    *, delegation_id: str, goal: str, goals: Optional[List[str]], context: Optional[str],
    toolsets: Optional[List[str]], role: str, model: Optional[str], session_key: str,
    parent_session_id: Optional[str], runner: Callable[[], Dict[str, Any]], origin_ui_session_id: str,
    origin_session_id: str, interrupt_fn: Optional[Callable[[], None]], max_async_children: int,
    progress_fn: Optional[Callable[[], tuple]], capacity_error: str,
) -> Dict[str, Any]:
    """Shared dispatch core for single (``goals is None``) and batch units. Capacity check +
    record insert happen under ONE lock hold so concurrent dispatches can't both pass the check
    and exceed the cap. At capacity the dispatch is REJECTED (never queued) so a runaway model
    can't pile up unbounded background work."""
    is_batch = goals is not None
    label = " batch" if is_batch else ""
    classify = _batch_status if is_batch else (lambda r: r.get("status") or "completed")
    crash_result = _batch_crash if is_batch else _single_crash
    dispatched_at = time.time()
    record: Dict[str, Any] = {
        "delegation_id": delegation_id, "goal": goal, **({"goals": list(goals)} if is_batch else {}),
        "context": context, "toolsets": list(toolsets) if toolsets else None, "role": role, "model": model,
        "session_key": session_key, "origin_ui_session_id": origin_ui_session_id,
        "origin_session_id": origin_session_id, "parent_session_id": parent_session_id,
        **_capture_routing_origin(),
        "status": "running", "dispatched_at": dispatched_at, "completed_at": None,
        "interrupt_fn": interrupt_fn, **({"is_batch": True} if is_batch else {}), "progress_fn": progress_fn,
        # Stale-monitor bookkeeping (see _stale_monitor_loop).
        "_progress_token": None, "_progress_ts": dispatched_at, "_interrupted_at": None}
    with _records_lock:
        running = sum(1 for r in _records.values() if r.get("status") in _ACTIVE_STATES)
        if running >= max_async_children:
            return {"status": "rejected", "error": capacity_error}
        _records[delegation_id] = record
    _persist_dispatch(record)
    executor = _get_executor(max_async_children)

    def _worker() -> None:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            status = classify(result)
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception(f"Async delegation{label} %s crashed", delegation_id)
            result = crash_result(f"{type(exc).__name__}: {exc}", round(time.time() - dispatched_at, 2))
        finally:
            _finalize(delegation_id, result, status)

    try:
        # Propagate the dispatching profile so the detached child resolves get_hermes_home() correctly.
        executor.submit(propagate_context_to_thread(_worker))
    except Exception as exc:  # pragma: no cover — pool submit failure is rare
        with _records_lock:
            _records.pop(delegation_id, None)
        with _DB_LOCK, _transaction() as conn:
            conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))
        return {"status": "rejected", "error": f"Failed to schedule async delegation{label}: {exc}"}
    if progress_fn is not None:
        _ensure_stale_monitor()
    return {"status": "dispatched", "delegation_id": delegation_id}


def dispatch_async_delegation(
    *, goal: str, context: Optional[str], toolsets: Optional[List[str]], role: str, model: Optional[str],
    session_key: str, parent_session_id: Optional[str] = None, runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "", origin_session_id: str = "", interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN, progress_fn: Optional[Callable[[], tuple]] = None,
) -> Dict[str, Any]:
    """Spawn ``runner`` on the daemon executor and return a handle immediately.
    ``session_key``/``parent_session_id`` are captured on the parent thread (the worker carries
    no contextvars) and route the completion back to the spawning session.
    ``progress_fn() -> (token, in_tool)`` enables stale monitoring; omitted = unmonitored.
    Returns ``{"status": "dispatched", "delegation_id"}`` or ``{"status": "rejected", "error"}``."""
    delegation_id = _new_delegation_id()
    handle = _dispatch(
        delegation_id=delegation_id, goal=goal, goals=None, context=context,
        toolsets=toolsets, role=role, model=model, session_key=session_key,
        parent_session_id=parent_session_id, runner=runner,
        origin_ui_session_id=origin_ui_session_id, origin_session_id=origin_session_id,
        interrupt_fn=interrupt_fn, max_async_children=max_async_children, progress_fn=progress_fn,
        capacity_error=(
            f"Async delegation capacity reached ({max_async_children} running). Wait for one to finish "
            "(its result will re-enter the chat), or run this task synchronously (background=false). "
            "Raise delegation.max_concurrent_children in config.yaml to allow more concurrent background subagents."))
    if handle["status"] == "dispatched":
        logger.info("Dispatched async delegation %s (session_key=%s): %s",
                    delegation_id, session_key or "<cli>", (goal or "")[:80])
    return handle


def dispatch_async_delegation_batch(
    *, goals: List[str], context: Optional[str], toolsets: Optional[List[str]], role: str, model: Optional[str],
    session_key: str, parent_session_id: Optional[str] = None, runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "", origin_session_id: str = "", interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN, delegation_id: Optional[str] = None,
    progress_fn: Optional[Callable[[], tuple]] = None,
) -> Dict[str, Any]:
    """Dispatch a WHOLE fan-out batch as ONE background unit: ``runner`` runs the
    entire batch and returns the combined ``{"results": [...], "total_duration_seconds": N}``
    dict. The batch occupies ONE async slot (in-batch parallelism is bounded
    separately) and produces a SINGLE completion event carrying per-task ``results``."""
    delegation_id = delegation_id or _new_delegation_id()
    n = len(goals)
    combined_goal = goals[0] if n == 1 else f"{n} parallel subagents: " + "; ".join(g[:40] for g in goals)
    handle = _dispatch(
        delegation_id=delegation_id, goal=combined_goal, goals=goals, context=context,
        toolsets=toolsets, role=role, model=model, session_key=session_key,
        parent_session_id=parent_session_id, runner=runner,
        origin_ui_session_id=origin_ui_session_id, origin_session_id=origin_session_id,
        interrupt_fn=interrupt_fn, max_async_children=max_async_children, progress_fn=progress_fn,
        capacity_error=(
            f"Async delegation capacity reached ({max_async_children} running). Wait for one to finish "
            "(its result will re-enter the chat), or raise delegation.max_concurrent_children in "
            "config.yaml to allow more concurrent background units."))
    if handle["status"] == "dispatched":
        logger.info("Dispatched async delegation batch %s (%d task(s), session_key=%s)",
                    delegation_id, n, session_key or "<cli>")
    return handle


# ── Finalization + completion events ────────────────────────────────────────
def _finalize(delegation_id: str, result: Any, status: str) -> None:
    """Atomically claim terminal delivery, push the completion event, then mark ``status``.
    ``result`` is a dict or a callable receiving the record snapshot (stall path). The record
    stays active ("finalizing") until durable persistence and queue publication finish; otherwise
    process shutdown can kill this daemon worker after status flips but before SQLite commits.
    A second call for the same id (late runner return after a forced stall) is a no-op."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("status") not in _ACTIVE_STATES:
            return
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None  # drop the closure; child is done
        record["progress_fn"] = None  # stop stale-monitor sampling
        snapshot = dict(record)
    _push_completion_event(snapshot, result(snapshot) if callable(result) else result, status)
    with _records_lock:
        if delegation_id in _records:
            _records[delegation_id]["status"] = status
        _prune_completed_locked()


def _push_completion_event(record: Dict[str, Any], result: Dict[str, Any], status: str) -> None:
    """Push a type='async_delegation' event onto the shared completion queue. Batch records
    (``is_batch``) carry the per-task ``results`` list (plus live transcript paths, the
    full-fidelity record of each child's run) instead of a single summary. Best-effort: failure
    must not crash the worker, but it WOULD mean a silently-lost result, so we log loudly."""
    is_batch = bool(record.get("is_batch"))
    label = " batch" if is_batch else ""
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(f"Async delegation{label} %s finished but process_registry import failed; "
                     "result lost: %s", record.get("delegation_id"), exc)
        return
    dispatched_at = record.get("dispatched_at") or time.time()
    completed_at = record.get("completed_at") or time.time()
    if is_batch:
        payload = {
            "is_batch": True, "results": result.get("results") or [],
            "live_transcripts": result.get("live_transcripts"), "error": result.get("error"),
            "total_duration_seconds": result.get("total_duration_seconds")}
    else:
        payload = {
            "summary": result.get("summary"), "error": result.get("error"), "api_calls": result.get("api_calls", 0),
            "duration_seconds": result.get("duration_seconds", round(completed_at - dispatched_at, 2))}
    evt = {
        "type": "async_delegation", "delegation_id": record.get("delegation_id"),
        # session_key routes back to the originating gateway session; "" => CLI.
        "session_key": record.get("session_key", ""),
        "origin_ui_session_id": record.get("origin_ui_session_id", ""),
        "origin_session_id": record.get("origin_session_id", ""),
        "parent_session_id": record.get("parent_session_id"),
        "goal": record.get("goal", ""), **({"goals": record.get("goals")} if is_batch else {}),
        "context": record.get("context"), "toolsets": record.get("toolsets"), "role": record.get("role"),
        "model": record.get("model") if is_batch else (result.get("model") or record.get("model")),
        "status": status, **payload, "dispatched_at": dispatched_at, "completed_at": completed_at,
        **({} if is_batch else {"exit_reason": result.get("exit_reason")}),
        **{k: record[k] for k in _ROUTING_KEYS if record.get(k)},
        **{k: result[k] for k in _STALL_META_KEYS if k in result}}
    _persist_completion(evt, result)
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(f"Async delegation{label} %s: failed to enqueue completion event; "
                     "result lost: %s", record.get("delegation_id"), exc)


def push_task_failure_notice(delegation_id: str, entry: Dict[str, Any], *, n_tasks: int) -> None:
    """Surface ONE failed child of a still-running detached batch to the parent now, instead of
    when the slowest sibling finishes. In a 1,393-agent run every wave-1 child died in a 401 storm
    at 08:29 and the parent learned of it at 09:36, when the batch's "unknown outcome" block finally
    arrived: 66 minutes of a dead wave with nothing running. The notice rides the same
    ``type="async_delegation"`` event shape as the batch result (so every drain/route/format path
    treats it identically) with ``task_failure_notice=True`` and a single-entry ``results`` list; the
    batch record is NOT finalized and its consolidated result still arrives as before."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("status") not in _ACTIVE_STATES:
            return
        snapshot = dict(record)
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error("Async delegation batch %s: task failure notice dropped (process_registry import): %s", delegation_id, exc)
        return
    evt = {
        "type": "async_delegation", "task_failure_notice": True, "is_batch": True, "n_tasks": n_tasks,
        "delegation_id": delegation_id, "results": [entry],
        "session_key": snapshot.get("session_key", ""),
        "origin_ui_session_id": snapshot.get("origin_ui_session_id", ""),
        "origin_session_id": snapshot.get("origin_session_id", ""),
        "parent_session_id": snapshot.get("parent_session_id"),
        "goal": snapshot.get("goal", ""), "goals": snapshot.get("goals"), "context": snapshot.get("context"),
        "toolsets": snapshot.get("toolsets"), "role": snapshot.get("role"), "model": snapshot.get("model"),
        "status": "running", "dispatched_at": snapshot.get("dispatched_at") or time.time(), "completed_at": time.time(),
        **{k: snapshot[k] for k in _ROUTING_KEYS if snapshot.get(k)}}
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error("Async delegation batch %s: failed to enqueue task failure notice: %s", delegation_id, exc)


# ── Stale monitor ───────────────────────────────────────────────────────────
def _ensure_stale_monitor() -> None:
    """Start (once) the stale-delegation monitor thread. One daemon thread serves
    every dispatch; it exits when no monitorable records remain and is restarted
    by the next dispatch with a ``progress_fn``."""
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return
        _monitor_stop.clear()
        _monitor_thread = threading.Thread(
            target=_stale_monitor_loop, name="async-delegate-stale-monitor", daemon=True)
        _monitor_thread.start()


def _sweep_stale_locked(now: float):
    """One monitor pass over ``_records``; caller holds ``_records_lock``. Returns
    ``(stalled, expired, any_monitorable)``: newly-stalling ``(delegation_id, quiet_for, in_tool)``
    tuples, stalling ids past the grace window, and whether anything is left to monitor."""
    stalled, expired, any_monitorable = [], [], False  # (delegation_id, quiet_for, in_tool) / ids past grace
    for record in _records.values():
        status = record.get("status")
        if status == "stalling":
            any_monitorable = True
            if now - (record.get("_interrupted_at") or now) >= _STALL_GRACE_SECONDS:
                expired.append(record["delegation_id"])
            continue
        progress_fn = record.get("progress_fn")
        if status != "running" or progress_fn is None:
            continue
        any_monitorable = True
        try:
            token, in_tool = progress_fn()
        except Exception:
            # An unreadable child must not look permanently healthy —
            # keep the last timestamp running instead of refreshing it.
            token, in_tool = record.get("_progress_token"), False
        if token != record.get("_progress_token"):
            record.update(_progress_token=token, _progress_ts=now)
            continue
        quiet_for = now - (record.get("_progress_ts") or now)
        limit = _STALE_IN_TOOL_SECONDS if in_tool else _STALE_IDLE_SECONDS
        if quiet_for >= limit:
            # Stall context feeds the terminal event and status listings.
            record.update(
                status="stalling", _interrupted_at=now, _stall_quiet_seconds=round(quiet_for, 2),
                _stall_threshold_seconds=limit, _stall_in_tool=bool(in_tool))
            stalled.append((record["delegation_id"], quiet_for, in_tool))
    return stalled, expired, any_monitorable


def _call_interrupt(fn, msg: str, *args) -> bool:
    """Invoke an ``interrupt_fn``; True on success, else debug-log ``msg`` (+ exc)."""
    if not callable(fn):
        return False
    try:
        fn()
        return True
    except Exception as exc:
        logger.debug(msg, *args, exc)
        return False


def _stale_monitor_loop() -> None:
    """Sweep running delegations for stalled progress. A changed progress token refreshes the
    record's timestamp; a frozen token past the idle/in-tool threshold marks the record
    ``stalling`` and calls ``interrupt_fn``; a ``stalling`` record still unreturned after the
    grace window is force-finalized with a terminal ``stalled`` event."""
    while not _monitor_stop.wait(_STALE_CHECK_INTERVAL):
        now = time.time()
        with _records_lock:
            stalled, expired, any_monitorable = _sweep_stale_locked(now)
        for delegation_id, quiet_for, in_tool in stalled:
            logger.warning("Async delegation %s made no progress for %.0fs "
                           "(in_tool=%s) — interrupting; grace window %.0fs",
                           delegation_id, quiet_for, in_tool, _STALL_GRACE_SECONDS)
            with _records_lock:
                fn = (_records.get(delegation_id) or {}).get("interrupt_fn")
            _call_interrupt(fn, "Async delegation %s stall interrupt failed: %s", delegation_id)
        for delegation_id in expired:
            _finalize(delegation_id, lambda rec, d=delegation_id: _stalled_result(d, rec), "stalled")
        if not any_monitorable:
            return


def _stalled_result(delegation_id: str, event_record: Dict[str, Any]) -> Dict[str, Any]:
    """Synthetic terminal result for a stalling delegation whose runner never returned."""
    completed_at = event_record.get("completed_at") or time.time()
    duration = round(completed_at - (event_record.get("dispatched_at") or completed_at), 2)
    error = (
        f"Async delegation {delegation_id} stalled: the detached subagent stopped making progress "
        "(no new API calls, tool activity, or streamed tokens), did not respond to interruption, and never "
        "produced a completion event. The worker may be wedged inside a model API call — this is a known "
        "failure mode of long-lived gateway processes (#60203). Re-dispatch the task if it is still needed.")
    logger.error("Async delegation %s force-finalized as stalled after %.0fs", delegation_id, duration)
    # Structured stall metadata lets parents/UIs distinguish a stall-monitor
    # kill from other failures without parsing the error string.
    stall_in_tool = event_record.get("_stall_in_tool")
    stall_meta = {
        "stalled_after_quiet_seconds": event_record.get("_stall_quiet_seconds"),
        "stall_threshold_seconds": event_record.get("_stall_threshold_seconds"),
        "stall_phase": "in_tool" if stall_in_tool else "idle" if stall_in_tool is not None else None,
        "stall_grace_seconds": _STALL_GRACE_SECONDS}
    if event_record.get("is_batch"):
        return {**_batch_crash(error, duration), **stall_meta}
    return {**_single_crash(error, duration), "status": "stalled", "exit_reason": "stalled", **stall_meta}


# ── Observability + control ─────────────────────────────────────────────────
def _children_activity_from_token(token: Any, now: float) -> Optional[List]:
    """Parse a progress token into per-child activity dicts (best-effort): delegate_tool
    emits one ``(api_call_count, current_tool, last_activity_ts)`` tuple per child;
    foreign token shapes degrade to ``None`` entries."""
    try:
        parts = list(token)
    except TypeError:
        return None
    out: List[Optional[Dict[str, Any]]] = []
    for part in parts:
        if not (isinstance(part, (list, tuple)) and len(part) >= 2):
            out.append(None)
            continue
        entry: Dict[str, Any] = {"api_calls": part[0], "current_tool": part[1]}
        if len(part) >= 3 and isinstance(part[2], (int, float)):
            entry["seconds_since_activity"] = round(max(0.0, now - float(part[2])), 1)
        out.append(entry)
    return out


def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed) without callables or private
    monitor bookkeeping; adds computed live fields for UIs (``seconds_since_progress``,
    ``children_activity``/``in_tool`` sampled from ``progress_fn``) and stall context once tripped.

    Safe to call from any thread. See #51690.
    """
    now = time.time()
    samplers: Dict[str, Callable] = {}
    with _records_lock:
        items = []
        for r in _records.values():
            item = {k: v for k, v in r.items() if k not in {"interrupt_fn", "progress_fn"} and not k.startswith("_")}
            status = r.get("status")
            if status in _ACTIVE_STATES:
                if r.get("_progress_ts"):
                    item["seconds_since_progress"] = round(now - r["_progress_ts"], 1)
                if callable(r.get("progress_fn")):
                    samplers[r["delegation_id"]] = r["progress_fn"]
            if status in ("stalling", "stalled"):
                for src, dst in _STALL_FIELD_MAP:
                    if r.get(src) is not None:
                        item[dst] = r.get(src)
            items.append(item)
    # Sample OUTSIDE the lock — progress_fn reads child-agent attributes and a
    # slow/broken sampler must not block every dispatch/finalize.
    for item in items:
        fn = samplers.get(item.get("delegation_id"))
        if fn is None:
            continue
        try:
            token, in_tool = fn()
        except Exception:
            continue
        activity = _children_activity_from_token(token, now)
        if activity is not None:
            item["children_activity"] = activity
        item["in_tool"] = bool(in_tool)
    return items


def _interrupt_records(targets: List[Dict[str, Any]], caller: str, reason: str, msg: str) -> int:
    """Call ``interrupt_fn`` on each record; log ``msg`` once; returns how many succeeded."""
    count = sum(
        _call_interrupt(r.get("interrupt_fn"), "%s: %s interrupt failed: %s", caller, r.get("delegation_id"))
        for r in targets)
    if count:
        logger.info(msg, count, reason)
    return count


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every running async delegation to stop (``/stop``, shutdown). Returns how
    many. The child still emits a completion event (status='interrupted') via the
    normal finalize path."""
    with _records_lock:
        targets = [r for r in _records.values() if r.get("status") in _ACTIVE_STATES]
    return _interrupt_records(targets, "interrupt_all", reason, "Interrupted %d async delegation(s) (%s)")


def interrupt_for_session(
    session_key: str = "", origin_ui_session_id: str = "", parent_session_id: str = "", reason: str = "session_end",
) -> int:
    """Signal running async delegations owned by ONE ending session to stop (any
    selector matches, see ``_session_records``). Returns how many."""
    targets = _session_records(_ACTIVE_STATES, session_key, origin_ui_session_id, parent_session_id)
    return _interrupt_records(
        targets, "interrupt_for_session", reason, "Interrupted %d async delegation(s) for ending session (%s)")


def _reset_for_tests() -> None:
    """Test-only: clear all state and tear down the executor + monitor."""
    global _executor, _executor_max_workers, _monitor_thread
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None
        _executor_max_workers = 0
    _monitor_stop.set()
    with _monitor_lock:
        thread, _monitor_thread = _monitor_thread, None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    with _records_lock:
        _records.clear()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def active_for_session(origin_ui_session_id: str) -> int:
    """Number of live async delegations owned by one UI session."""
    if not origin_ui_session_id:
        return 0
    with _records_lock:
        return sum(
            1
            for r in _records.values()
            if r.get("status") in {"running", "stalling", "finalizing"}
            and str(r.get("origin_ui_session_id") or "")
            == origin_ui_session_id
        )
# ---- END PLUGIN-COMPAT ----
