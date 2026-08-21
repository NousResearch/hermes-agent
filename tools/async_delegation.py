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
_admission_condition = threading.Condition(_records_lock)
# delegation_id -> record dict. Kept for the lifetime of the run plus a short
# tail after completion so `list_async_delegations()` can show recent results.
_records: Dict[str, Dict[str, Any]] = {}
_lingering_resource_slots: Dict[str, int] = {}
_shutdown_event = threading.Event()
# Manager generation: bumped on test reset / shutdown so workers that captured
# a record from an earlier generation cannot push completions into the shared
# queue after the manager has been torn down (see _push_completion_event).
_manager_generation = 0

_DEFAULT_MAX_ASYNC_CHILDREN = 10
_DEFAULT_MAX_QUEUED_DELEGATIONS = 8
_DEFAULT_QUEUE_TIMEOUT_SECONDS = 3600.0
_ADMISSION_RECHECK_SECONDS = 1.0
# How many completed records to retain for status queries before pruning.
_MAX_RETAINED_COMPLETED = 50
_DURABLE_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_DURABLE_PENDING = 1000
# A pending completion whose delivery keeps failing is retried across claim
# cycles (and across restarts via restore_undelivered_completions). Cap the
# attempts so an unroutable row converges to a terminal 'dropped' state
# instead of replaying on every restart forever.
_MAX_DELIVERY_ATTEMPTS = 8
# Staleness cap for restart replay: a pending completion older than this is
# terminally dropped instead of re-run as a fresh full-context turn (see
# restore_undelivered_completions). 48h keeps overnight/weekend results
# deliverable while stopping weeks-old sessions from replaying after upgrades.
_MAX_COMPLETION_REPLAY_AGE_S = 48 * 3600.0
_DB_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Stale-delegation detection (progress-based, on by default)
# ---------------------------------------------------------------------------
# A detached runner that wedges before returning (e.g. stuck inside its first
# model API call — #60203) never reaches its ``finally`` finalizer, so no
# completion event is ever published: the delegation shows "dispatched"
# forever and the owning session looks silent until a process restart. We do
# NOT fix this with a wall-clock timeout — legitimate heavy subagent work
# (deep reviews, research fan-outs, slow reasoning models) must never be
# killed for taking long (see delegate_tool.DEFAULT_CHILD_TIMEOUT rationale).
# Instead a single monitor thread watches per-dispatch PROGRESS (api-call
# count + current tool, via an injected ``progress_fn``): a child that is
# advancing is left alone forever; a child with NO progress past the stale
# threshold is interrupted, given a grace window to unwind and deliver its
# partial results through the normal finalize path, and only force-finalized
# with a terminal ``stalled`` event if it never returns.
#
# Thresholds mirror the sync-path heartbeat staleness monitor in
# delegate_tool: idle (not inside a tool) stays tight so a wedged first API
# call is caught quickly; in-tool is much higher so legitimately slow tools
# (long terminal commands, big fetches) get time to finish.
_STALE_CHECK_INTERVAL = 30.0  # seconds between monitor sweeps
_STALE_IDLE_SECONDS = 450.0  # no progress, no current tool → stalled
_STALE_IN_TOOL_SECONDS = 1200.0  # no progress while inside a tool → stalled
_STALL_GRACE_SECONDS = 120.0  # after interrupt, time for the runner to return

_monitor_lock = threading.Lock()
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (async_delegation)")
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
            origin_session_id TEXT NOT NULL DEFAULT ''
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")}
    for name, sql_type in (
        ("owner_pid", "INTEGER"),
        ("owner_started_at", "INTEGER"),
        ("task_json", "TEXT"),
        ("delivery_claim", "TEXT"),
        ("delivery_claimed_at", "REAL"),
        # Raw api_server session id (X-Hermes-Session-Id) of the ORIGINATING
        # request — the wake self-post target. Without persisting it,
        # completions recovered after a process restart are unroutable on
        # api_server (the in-memory record that carried it is gone).
        ("origin_session_id", "TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE async_delegations ADD COLUMN {name} {sql_type}")


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every durable dispatch, completion, and delivery-claim, deferring the close
    to the garbage collector. On a long-running gateway that exhausts
    ``RLIMIT_NOFILE`` (the cron-ledger sibling of this bug was #69567 / PR #69594).
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _capture_routing_origin() -> Dict[str, Any]:
    """Snapshot the dispatching turn's routing origin for the completion event.

    Captured on the PARENT thread at dispatch time (the daemon worker doesn't
    carry the contextvars) and persisted with the durable record, so a
    completion replayed after a restart can reconstruct a full SessionSource
    even when the session-store origin and in-memory source cache are gone.
    scope_id matters most: on a relay-fronted deployment the connector's
    fail-closed egress guard needs the tenant discriminator (or a user
    binding) to route a scoped reply; without it, post-restart scoped
    completions bounce with "target not routed to an onboarded tenant"
    (staging 2026-08-09 defect #4). Best-effort — empty values are simply
    omitted so CLI/contextvar-unaware paths persist nothing new.
    """
    origin: Dict[str, Any] = {}
    try:
        from gateway.session_context import get_session_env

        for evt_key, env_name in (
            ("scope_id", "HERMES_SESSION_SCOPE_ID"),
            ("user_id", "HERMES_SESSION_USER_ID"),
            ("user_name", "HERMES_SESSION_USER_NAME"),
        ):
            value = get_session_env(env_name, "")
            if value:
                origin[evt_key] = value
    except Exception:  # noqa: BLE001 - routing origin is additive, never fatal
        pass
    return origin


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
            # Routing origin (scope_id/user_id/user_name): persisted so a
            # restart-recovered completion can reconstruct a full
            # SessionSource — see _capture_routing_origin.
            "scope_id", "user_id", "user_name",
        )
        if key in record
    }
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, origin_session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (record["delegation_id"], record.get("session_key", ""),
             record.get("origin_ui_session_id", ""), record.get("parent_session_id"),
             record.get("status", "running"), record["dispatched_at"], now,
             __import__("os").getpid(), owner_started_at, json.dumps(task_payload),
             record.get("origin_session_id", "")),
        )
    _prune_durable_records()


def _delete_durable_delegation(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))


def _prune_durable_records() -> None:
    """Bound terminal history, preferring delivered records for deletion.

    Live non-terminal states are ``('running','finalizing','queued')`` — a
    queued row is still pending admission and must never be pruned: its
    terminal outcome (timeout/cancel/start) has not been recorded yet.
    """
    now = time.time()
    cutoff = now - _DURABLE_RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < ?",
            (cutoff,),
        )
        terminal_count = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing','queued')"
        ).fetchone()[0]
        excess = max(0, terminal_count - _MAX_RETAINED_COMPLETED)
        if excess:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing','queued')
                     ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                              updated_at ASC LIMIT ?
                   )""",
                (excess,),
            )
        pending_count = conn.execute(
            """SELECT COUNT(*) FROM async_delegations
              WHERE state NOT IN ('running','finalizing','queued') AND delivery_state='pending'"""
        ).fetchone()[0]
        overflow = max(0, pending_count - _MAX_DURABLE_PENDING)
        if overflow:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing','queued') AND delivery_state='pending'
                     ORDER BY updated_at ASC LIMIT ?
                   )""",
                (overflow,),
            )


def _persist_completion(event: Dict[str, Any], result: Dict[str, Any]) -> None:
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
               event_json=?, result_json=?, delivery_state='pending'
               WHERE delegation_id=?""",
            (event.get("status", "completed"), event.get("completed_at", now), now,
             json.dumps(event), json.dumps(result), event["delegation_id"]),
        )


def _persist_state(delegation_id: str, state: str) -> None:
    """Durably mirror an in-memory lifecycle transition."""
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET state=?, updated_at=? WHERE delegation_id=?",
            (state, time.time(), delegation_id),
        )


def _note_delivery_attempt(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET delivery_attempts=delivery_attempts+1, updated_at=? WHERE delegation_id=?",
            (time.time(), delegation_id),
        )


def recover_abandoned_delegations() -> int:
    """Terminalize durable records whose owning process disappeared.

    A dead-owner queued record provably never began and is reported interrupted;
    running/finalizing records remain unknown because side effects may have begun.
    """
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        return 0
    now = time.time()
    recovered = 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, origin_session_id, state
               FROM async_delegations
               WHERE state IN ('queued','running','finalizing')"""
        ).fetchall()
        for row in rows:
            (delegation_id, session_key, origin_ui, parent_id, dispatched_at,
             pid, started, task_json, origin_session_id, prior_state) = row
            live = False
            if pid:
                live = _pid_exists(int(pid))
                if live and started is not None:
                    live = get_process_start_time(int(pid)) == int(started)
            if live:
                continue
            task = json.loads(task_json or "{}")
            terminal_status = "interrupted" if prior_state == "queued" else "unknown"
            terminal_error = (
                "Queued delegation was cancelled before start because its owner exited."
                if prior_state == "queued"
                else "Delegation owner exited before recording a terminal result; outcome unknown."
            )
            event = {
                "type": "async_delegation", "delegation_id": delegation_id,
                "session_key": session_key, "origin_ui_session_id": origin_ui,
                # Restore the durable wake target so completions recovered
                # after a restart remain routable to api_server sessions.
                "origin_session_id": origin_session_id or "",
                "parent_session_id": parent_id, "goal": task.get("goal", ""),
                "goals": task.get("goals"), "context": task.get("context"),
                "toolsets": task.get("toolsets"), "role": task.get("role"),
                "model": task.get("model"), "is_batch": bool(task.get("is_batch")),
                "status": terminal_status, "summary": None,
                "error": terminal_error,
                "dispatched_at": dispatched_at, "completed_at": now,
            }
            # Routing origin persisted at dispatch (see _capture_routing_origin):
            # restores scope_id/user_id for the reconstructed SessionSource so
            # relay egress priming works after a restart.
            for _k in ("scope_id", "user_id", "user_name"):
                if task.get(_k):
                    event[_k] = task[_k]
            result = {
                "status": terminal_status, "summary": None, "error": terminal_error
            }
            conn.execute(
                """UPDATE async_delegations SET state=?, completed_at=?,
                   updated_at=?, event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""",
                (
                    terminal_status, now, now, json.dumps(event), json.dumps(result),
                    delegation_id,
                ),
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

    Staleness cap: a pending completion older than
    ``_MAX_COMPLETION_REPLAY_AGE_S`` is terminally dropped instead of
    replayed. Replaying a weeks-old completion re-runs its parent session as
    a full-context turn (a July session replayed in August burned a
    102K-token context on the staging fleet) for a result nobody is waiting
    on anymore; the payload stays queryable on the dropped row.
    """
    recover_abandoned_delegations()
    now = time.time()
    restored = 0
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, event_json, completed_at, dispatched_at
               FROM async_delegations
               WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
               ORDER BY completed_at, delegation_id"""
        ).fetchall()
        for delegation_id, payload, completed_at, dispatched_at in rows:
            age_basis = completed_at or dispatched_at
            if age_basis and (now - age_basis) > _MAX_COMPLETION_REPLAY_AGE_S:
                conn.execute(
                    """UPDATE async_delegations SET delivery_state='dropped',
                              delivery_claim=NULL, delivery_claimed_at=NULL,
                              updated_at=?
                       WHERE delegation_id=? AND delivery_state='pending'""",
                    (now, delegation_id),
                )
                logger.warning(
                    "Async delegation %s: pending completion is %.1fh old "
                    "(cap %.1fh); terminally dropping the replay (result "
                    "remains queryable).",
                    delegation_id, (now - age_basis) / 3600.0,
                    _MAX_COMPLETION_REPLAY_AGE_S / 3600.0,
                )
                continue
            evt = json.loads(payload)
            if isinstance(evt, dict):
                evt["restored"] = True
            target_queue.put(evt)
            restored += 1
    return restored


def mark_completion_delivered(delegation_id: str) -> bool:
    """Atomically acknowledge successful injection of a durable completion."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
        cur = conn.execute(
            """UPDATE async_delegations SET delivery_state='delivered', delivered_at=?, updated_at=?
               WHERE delegation_id=? AND delivery_state!='delivered'""",
            (now, now, delegation_id),
        )
        return cur.rowcount == 1


def claim_completion_delivery(delegation_id: str, claim_id: str) -> bool:
    """Claim one pending completion across competing consumers/processes."""
    now = time.time()
    with _DB_LOCK, _transaction() as conn:
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
    with _DB_LOCK, _transaction() as conn:
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
    with _DB_LOCK, _transaction() as conn:
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
    with _DB_LOCK, _transaction() as conn:
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
    if claim_id and evt.get("type") == "async_delegation":
        complete_completion_delivery(str(evt.get("delegation_id") or ""), claim_id)


def release_event_delivery(evt: Dict[str, Any], claim_id: str) -> None:
    if claim_id and evt.get("type") == "async_delegation":
        release_completion_delivery(str(evt.get("delegation_id") or ""), claim_id)


def get_durable_delegation(delegation_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            """SELECT origin_session, state, dispatched_at, completed_at,
                      result_json, delivery_state, delivery_attempts,
                      origin_session_id
               FROM async_delegations WHERE delegation_id=?""", (delegation_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "delegation_id": delegation_id, "origin_session": row[0], "state": row[1],
        "dispatched_at": row[2], "completed_at": row[3],
        "result": json.loads(row[4]) if row[4] else None,
        "delivery_state": row[5], "delivery_attempts": row[6],
        "origin_session_id": row[7] or "",
    }


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
    """Number of async delegation UNITS currently running.

    A unit is one dispatch: a single subagent or one fan-out batch. This is an
    observability count of dispatch records, not slot accounting; a running batch
    reserves one child slot per goal. Use ``active_task_count()`` for the expanded
    child count.
    """
    with _records_lock:
        return sum(
            1 for r in _records.values()
            if r.get("status") in {"running", "stalling", "finalizing"}
        )


def active_for_session(origin_ui_session_id: str) -> int:
    """Number of live queued or running delegations owned by one UI session."""
    if not origin_ui_session_id:
        return 0
    with _records_lock:
        return sum(
            1
            for r in _records.values()
            if r.get("status") in {"queued", "running", "stalling", "finalizing"}
            and str(r.get("origin_ui_session_id") or "")
            == origin_ui_session_id
        )


def active_task_count() -> int:
    """Number of async delegation TASKS (child subagents) currently running.

    Unlike ``active_count()`` (units/slots), this expands a batch to its child
    count: a running batch of N tasks contributes N, a single subagent
    contributes 1. This is the truthful "how many subagents are actually
    working right now" figure for observability, where a 3-task batch shown as
    "1" undercounts real concurrent work. Falls back to counting a batch as 1
    if its goal list is missing.
    """
    with _records_lock:
        total = 0
        for r in _records.values():
            if r.get("status") not in {"running", "finalizing"}:
                continue
            if r.get("is_batch"):
                goals = r.get("goals")
                total += len(goals) if isinstance(goals, (list, tuple)) and goals else 1
            else:
                total += 1
        return total


def _matches_session_selectors(
    record: Dict[str, Any],
    *,
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
) -> bool:
    return (
        (origin_ui_session_id and str(record.get("origin_ui_session_id") or "") == origin_ui_session_id)
        or (session_key and str(record.get("session_key") or "") == session_key)
        or (parent_session_id and str(record.get("parent_session_id") or "") == parent_session_id)
    )


def has_live_for_session(
    session_key: str = "",
    origin_ui_session_id: str = "",
    parent_session_id: str = "",
) -> bool:
    """Whether a session still owns any live async delegation.

    Live = queued / running / stalling / finalizing — the states whose owner
    must be kept routable until the work is cancelled or reaches a terminal state.
    """
    if not session_key and not origin_ui_session_id and not parent_session_id:
        return False
    with _records_lock:
        return any(
            r.get("status") in {"queued", "running", "stalling", "finalizing"}
            and _matches_session_selectors(
                r,
                session_key=session_key,
                origin_ui_session_id=origin_ui_session_id,
                parent_session_id=parent_session_id,
            )
            for r in _records.values()
        )


def _current_cgroup_v2_path() -> Optional[str]:
    """Return this process's cgroup-v2 directory, if mounted conventionally."""
    if os.name != "posix":
        return None
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as handle:
            for line in handle:
                hierarchy, _controllers, path = line.rstrip("\n").split(":", 2)
                if hierarchy == "0":
                    return os.path.join("/sys/fs/cgroup", path.lstrip("/"))
    except (OSError, ValueError):
        logger.debug("Cgroup v2 path probe failed", exc_info=True)
    return None


def _effective_available_memory_bytes() -> Optional[int]:
    """Return host availability capped by this process's cgroup headroom."""
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception:
        logger.debug("Available-memory probe failed", exc_info=True)
        return None

    cgroup_dir = _current_cgroup_v2_path()
    if cgroup_dir is None:
        return available
    try:
        with open(
            os.path.join(cgroup_dir, "memory.current"), encoding="utf-8"
        ) as handle:
            current = int(handle.read().strip())
        for filename in ("memory.high", "memory.max"):
            with open(
                os.path.join(cgroup_dir, filename), encoding="utf-8"
            ) as handle:
                raw = handle.read().strip()
            if raw != "max":
                available = min(available, max(0, int(raw) - current))
    except (OSError, ValueError):
        # No cgroup v2 limit (or a transient read race): host availability is
        # still a valid cross-platform admission signal.
        pass
    return available


def _memory_psi_avg10() -> Optional[float]:
    """Return Linux memory-pressure ``some avg10`` or ``None`` if unavailable."""
    if os.name != "posix":
        return None
    try:
        with open("/proc/pressure/memory", encoding="utf-8") as handle:
            for line in handle:
                fields = line.split()
                if fields and fields[0] == "some":
                    for field in fields[1:]:
                        if field.startswith("avg10="):
                            return float(field.split("=", 1)[1])
    except (OSError, ValueError):
        logger.debug("Memory PSI probe failed", exc_info=True)
    return None


def _resources_available(record: Dict[str, Any]) -> bool:
    was_blocked = bool(record.get("_resource_blocked", False))
    minimum = max(0, int(record.get("min_available_memory_bytes", 0)))
    resume_minimum = max(
        minimum, int(record.get("resume_available_memory_bytes", minimum))
    )
    memory_floor = resume_minimum if was_blocked else minimum

    max_psi = max(0.0, float(record.get("max_memory_psi_avg10", 0.0)))
    resume_value = float(record.get("resume_memory_psi_avg10", max_psi))
    # Defense-in-depth: a non-positive explicit resume ceiling must follow the
    # stop ceiling (same convention as delegate_tool's getter). Without this,
    # a direct dispatch_async_delegation* caller passing 0 with max_psi > 0
    # would clamp the resume ceiling to 0 and stay blocked until PSI is
    # exactly 0.
    if resume_value <= 0.0:
        resume_value = max_psi
    resume_psi = max(0.0, min(max_psi, resume_value))
    psi_ceiling = resume_psi if was_blocked else max_psi

    memory_ready = True
    if memory_floor:
        available = _effective_available_memory_bytes()
        # An explicitly configured memory floor is fail-closed when the probe
        # is unavailable; otherwise the safety control would silently vanish.
        memory_ready = available is not None and available >= memory_floor

    pressure_ready = True
    if max_psi:
        pressure = _memory_psi_avg10()
        # PSI is Linux-only and optional; unavailable PSI falls back to the
        # memory-headroom gate for cross-platform compatibility.
        pressure_ready = pressure is None or pressure <= psi_ceiling

    ready = memory_ready and pressure_ready
    record["_resource_blocked"] = not ready
    return ready


def _queue_head_locked() -> Optional[str]:
    """Return the oldest queued delegation id. Caller holds ``_records_lock``."""
    for record in _records.values():
        if record.get("status") == "queued":
            return record.get("delegation_id")
    return None


def _running_task_slots_locked() -> int:
    """Return admitted and still-live child slots. Caller holds record lock."""
    running = sum(
        max(1, int(record.get("required_slots", 1)))
        for record in _records.values()
        if record.get("status") in {"running", "stalling", "finalizing"}
    )
    return running + sum(_lingering_resource_slots.values())


def _register_lingering_resources(
    delegation_id: str, futures: List[Any]
) -> None:
    """Retain slots for timed-out child threads until their futures truly exit."""
    live = [future for future in futures if future is not None and not future.done()]
    if not live:
        return
    with _admission_condition:
        _lingering_resource_slots[delegation_id] = len(live)
        _admission_condition.notify_all()

    def _release_one(_future: Any) -> None:
        with _admission_condition:
            remaining = _lingering_resource_slots.get(delegation_id, 0) - 1
            if remaining > 0:
                _lingering_resource_slots[delegation_id] = remaining
            else:
                _lingering_resource_slots.pop(delegation_id, None)
            _admission_condition.notify_all()

    for future in live:
        future.add_done_callback(_release_one)


def _wait_for_admission(delegation_id: str) -> bool:
    """Block a daemon worker until its FIFO position can claim all child slots."""
    admitted = False
    expired = False
    with _admission_condition:
        while True:
            record = _records.get(delegation_id)
            if record is None:
                return False
            if record.get("status") == "running":
                return True
            if record.get("status") != "queued":
                return False
            queue_timeout = max(0.0, float(record.get("queue_timeout_seconds", 0)))
            queued_at = float(record.get("_queued_monotonic", time.monotonic()))
            if queue_timeout and time.monotonic() - queued_at >= queue_timeout:
                expired = True
                break
            required = max(1, int(record.get("required_slots", 1)))
            limit = max(1, int(record.get("max_async_children", 1)))
            is_head = _queue_head_locked() == delegation_id
            slots_ready = _running_task_slots_locked() + required <= limit
            resources_ready = is_head and slots_ready and _resources_available(record)
            if not is_head:
                record["queue_reason"] = "fifo"
            elif not slots_ready:
                record["queue_reason"] = "capacity"
            elif not resources_ready:
                record["queue_reason"] = "resources"
            if is_head and slots_ready and resources_ready:
                record["status"] = "running"
                admitted = True
                _admission_condition.notify_all()
                break
            wait_for = _ADMISSION_RECHECK_SECONDS
            if queue_timeout:
                remaining = queue_timeout - (time.monotonic() - queued_at)
                wait_for = min(wait_for, max(0.001, remaining))
            _admission_condition.wait(timeout=wait_for)
    if expired:
        _finalize_queued_terminal(
            delegation_id,
            "timeout",
            "Queued delegation timed out before resources became available.",
        )
        return False
    if admitted:
        # The stale monitor may have exited while this record sat queued (its
        # sweeps only monitor running work). Restart it so an admitted-then-
        # wedged child is still force-finalized instead of leaking its slot.
        _ensure_stale_monitor()
        try:
            _persist_state(delegation_id, "running")
        except Exception:
            # Best-effort mirror: the durable row already exists from
            # _persist_dispatch, and the terminal completion persist will
            # overwrite state. A failure here must NOT kill the admission
            # thread — the worker still needs to be submitted or the slot
            # leaks with a record stuck "running" and no runner.
            logger.exception(
                "Failed to persist running state for admitted delegation %s",
                delegation_id,
            )
    return admitted


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the retention cap.

    Caller must hold ``_records_lock``.
    """
    completed = [
        (rid, r)
        for rid, r in _records.items()
        if r.get("status") not in {"queued", "running", "stalling", "finalizing"}
    ]
    if len(completed) <= _MAX_RETAINED_COMPLETED:
        return
    # Oldest-first by completion time (fall back to dispatch time).
    completed.sort(key=lambda kv: kv[1].get("completed_at") or kv[1].get("dispatched_at") or 0)
    for rid, _ in completed[: len(completed) - _MAX_RETAINED_COMPLETED]:
        _records.pop(rid, None)


def _current_origin_session_id() -> str:
    """Raw session id of the ORIGINATING api_server request, or ``""``.

    The obvious source — ``HERMES_SESSION_ID`` via ``get_session_env`` — is
    NOT safe to read at dispatch time: constructing a child agent
    (``agent/agent_init.py``) calls ``set_current_session_id(child.session_id)``,
    clobbering that ContextVar *and* ``os.environ`` with the subagent's
    internal ``{timestamp}_{uuid}`` id moments before the dispatch code reads
    it, so the completion wake would self-post into the subagent's own
    (unread) session instead of the spawner's.

    The request-scoped ``HERMES_SESSION_CHAT_ID`` binding survives child
    construction: ``_bind_api_server_session`` binds ``chat_id`` to the raw
    ``X-Hermes-Session-Id``, and its only writer is ``set_session_vars`` —
    ``set_current_session_id`` never touches it. Gate on the platform: on
    push platforms ``chat_id`` is a chat, not a session, so yield ``""``
    there.
    """
    try:
        from gateway.session_context import get_session_env

        if get_session_env("HERMES_SESSION_PLATFORM", "") != "api_server":
            return ""
        return get_session_env("HERMES_SESSION_CHAT_ID", "") or ""
    except Exception:
        return ""


def dispatch_async_delegation(
    *,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    parent_session_id: Optional[str] = None,
    runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "",
    origin_session_id: str = "",
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    max_queued_delegations: int = _DEFAULT_MAX_QUEUED_DELEGATIONS,
    min_available_memory_bytes: int = 0,
    resume_available_memory_bytes: int = 0,
    max_memory_psi_avg10: float = 0.0,
    resume_memory_psi_avg10: float = 0.0,
    queue_timeout_seconds: float = _DEFAULT_QUEUE_TIMEOUT_SECONDS,
    progress_fn: Optional[Callable[[], tuple]] = None,
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
    progress_fn
        Optional zero-arg callable returning ``(token, in_tool)`` where
        ``token`` is any comparable snapshot of the child's progress (api
        call count + current tool) and ``in_tool`` says whether the child is
        currently inside a tool call. Sampled by the stale monitor; a frozen
        token past the stale threshold marks the delegation stuck (see the
        stale-detection block at the top of this module). When omitted, the
        delegation is not monitored.
    max_async_children
        Global child-slot cap. Work that cannot claim all required slots enters
        the bounded FIFO; queue overflow is rejected without synchronous fallback.

    Returns
    -------
    dict
        A ``dispatched`` or ``queued`` handle, or an explicit ``rejected`` error.
    """
    delegation_id = _new_delegation_id()
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
        "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id,
        **_capture_routing_origin(),
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "progress_fn": progress_fn,
        "required_slots": 1,
        "max_async_children": max_async_children,
        "min_available_memory_bytes": min_available_memory_bytes,
        "resume_available_memory_bytes": max(
            min_available_memory_bytes, resume_available_memory_bytes
        ),
        "max_memory_psi_avg10": max_memory_psi_avg10,
        "resume_memory_psi_avg10": resume_memory_psi_avg10,
        "queue_timeout_seconds": queue_timeout_seconds,
        "_queued_monotonic": time.monotonic(),
        # Stale-monitor bookkeeping (see _stale_monitor_loop).
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
        "_generation": _manager_generation,
    }
    # Capacity check and record insert under ONE lock hold — checking
    # active_count() separately would let two concurrent dispatches (e.g.
    # from different gateway sessions) both pass the check and exceed the cap.
    with _records_lock:
        if _shutdown_event.is_set():
            return {
                "status": "rejected",
                "error": "Async delegation manager is shutting down.",
            }
        capacity_ready = (
            _queue_head_locked() is None
            and _running_task_slots_locked() + record["required_slots"]
            <= max_async_children
        )
        resources_ready = _resources_available(record)
        queued = not (capacity_ready and resources_ready)
        queue_reason = "capacity" if not capacity_ready else "resources"
        if queued:
            pending = sum(1 for r in _records.values() if r.get("status") == "queued")
            queue_limit = max(0, int(max_queued_delegations))
            if pending >= queue_limit:
                return {
                    "status": "rejected",
                    "error": f"Async delegation queue is full ({queue_limit} pending).",
                }
            record["status"] = "queued"
        _records[delegation_id] = record

    try:
        _persist_dispatch(record)
    except Exception as exc:
        logger.exception("Failed to persist async delegation %s", delegation_id)
        with _admission_condition:
            _records.pop(delegation_id, None)
            _admission_condition.notify_all()
        return {
            "status": "rejected",
            "error": f"Failed to persist async delegation: {exc}",
        }
    try:
        executor = _get_executor(max_async_children)
    except Exception as exc:
        with _admission_condition:
            _records.pop(delegation_id, None)
            _admission_condition.notify_all()
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to initialize async delegation executor: {exc}",
        }

    def _worker() -> None:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            lingering = result.pop("_lingering_futures", [])
            if isinstance(lingering, list):
                _register_lingering_resources(delegation_id, lingering)
            status = result.get("status") or "completed"
        except InterruptedError as exc:
            result = {
                "status": "interrupted",
                "summary": None,
                "error": str(exc),
                "api_calls": 0,
                "duration_seconds": round(time.time() - dispatched_at, 2),
                "exit_reason": "interrupted",
            }
            status = "interrupted"
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

    # Capture the request/profile context now; queued launchers may run much later.
    worker_with_context = propagate_context_to_thread(_worker)

    def _launch_when_admitted() -> None:
        if not _wait_for_admission(delegation_id):
            return
        try:
            executor.submit(worker_with_context)
        except Exception as exc:  # executor may be shutting down
            logger.exception("Failed to launch queued delegation %s", delegation_id)
            _finalize_queued_terminal(
                delegation_id,
                "error",
                f"Failed to launch queued delegation: {exc}",
            )

    try:
        if queued:
            threading.Thread(
                target=propagate_context_to_thread(_launch_when_admitted),
                name=f"async-delegate-admission-{delegation_id}",
                daemon=True,
            ).start()
        else:
            executor.submit(worker_with_context)
    except Exception as exc:  # pragma: no cover — thread/pool start failure is rare
        with _records_lock:
            _records.pop(delegation_id, None)
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation: {exc}",
        }
    if progress_fn is not None:
        _ensure_stale_monitor()

    logger.info(
        "%s async delegation %s (session_key=%s): %s",
        "Queued" if queued else "Dispatched",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    return {
        "status": "queued" if queued else "dispatched",
        "delegation_id": delegation_id,
        **({"queue_reason": queue_reason} if queued else {}),
    }


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn = claimed

    try:
        _push_completion_event(event_record, result, status)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Async delegation %s (%s): failed to publish completion event; "
            "releasing slot (%s)",
            delegation_id, status, exc,
        )
    finally:
        _finish_finalization(delegation_id, status)


def _begin_finalization(
    delegation_id: str,
    allowed_statuses: Optional[tuple] = None,
) -> Optional[tuple[Dict[str, Any], Optional[Callable[[], None]]]]:
    """Atomically claim terminal delivery while keeping the record active.

    ``allowed_statuses`` restricts which in-flight states may be claimed. The
    queued-interruption path passes ``("queued",)`` so a record that was
    promoted to ``running`` between the caller's snapshot and this claim is
    NOT terminalized here — the caller must instead fall back to
    ``interrupt_fn`` so the worker self-cancels before starting.
    """
    if allowed_statuses is None:
        allowed_statuses = ("queued", "running", "stalling")
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("status") not in allowed_statuses:
            return
        # Stay active until durable persistence and queue publication finish;
        # otherwise process shutdown can kill this daemon worker in the narrow
        # gap after status flips but before SQLite is committed.
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        interrupt_fn = record.get("interrupt_fn")
        record["interrupt_fn"] = None  # drop the closure; child is done
        record["progress_fn"] = None  # stop stale-monitor sampling
        event_record = dict(record)

    return event_record, interrupt_fn


def _finish_finalization(delegation_id: str, status: str) -> None:
    with _admission_condition:
        record = _records.get(delegation_id)
        if record is not None:
            record["status"] = status
        _prune_completed_locked()
        _admission_condition.notify_all()


def _push_completion_event(
    record: Dict[str, Any], result: Dict[str, Any], status: str
) -> None:
    """Push a type='async_delegation' event onto the shared completion queue.

    Best-effort: a failure here must not crash the worker, but it WOULD mean a
    silently-lost result, so we log loudly.

    Generation guard: the record was captured by the worker thread at
    finalization time. If the manager was torn down (test reset or process
    shutdown) since dispatch, ``_manager_generation`` has moved on and the
    completion must be dropped — a late worker from a previous generation
    must never contaminate the shared queue or the durable ledger.
    """
    if record.get("_generation", _manager_generation) != _manager_generation:
        logger.info(
            "Dropping stale-generation completion for %s (generation %s != %s)",
            record.get("delegation_id"),
            record.get("_generation"),
            _manager_generation,
        )
        return
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s finished but process_registry import failed; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )
        return

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
        "origin_session_id": record.get("origin_session_id", ""),
        "parent_session_id": record.get("parent_session_id"),
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
    }
    # Routing origin captured at dispatch (see _capture_routing_origin):
    # additive, lets the gateway reconstruct a full SessionSource (incl.
    # scope_id for relay tenant egress) when its own caches are cold.
    for _k in ("scope_id", "user_id", "user_name"):
        if record.get(_k):
            evt[_k] = record[_k]
    # Structured stall metadata (#51690) — additive, present only on
    # stall-monitor finalizations.
    for _k in (
        "stalled_after_quiet_seconds",
        "stall_threshold_seconds",
        "stall_phase",
        "stall_grace_seconds",
    ):
        if _k in result:
            evt[_k] = result[_k]
    _persist_completion(evt, result)
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s: failed to enqueue completion event; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )


def dispatch_async_delegation_batch(
    *,
    goals: List[str],
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    parent_session_id: Optional[str] = None,
    runner: Callable[[], Dict[str, Any]],
    origin_ui_session_id: str = "",
    origin_session_id: str = "",
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    max_queued_delegations: int = _DEFAULT_MAX_QUEUED_DELEGATIONS,
    min_available_memory_bytes: int = 0,
    resume_available_memory_bytes: int = 0,
    max_memory_psi_avg10: float = 0.0,
    resume_memory_psi_avg10: float = 0.0,
    queue_timeout_seconds: float = _DEFAULT_QUEUE_TIMEOUT_SECONDS,
    delegation_id: Optional[str] = None,
    progress_fn: Optional[Callable[[], tuple]] = None,
) -> Dict[str, Any]:
    """Dispatch a WHOLE fan-out batch as ONE background unit.

    Unlike ``dispatch_async_delegation`` (which backs a single subagent),
    ``runner`` here runs the entire batch — it builds and joins on every child
    in parallel and returns the combined ``{"results": [...],
    "total_duration_seconds": N}`` dict that the synchronous path would have
    returned. The batch reserves one admission slot per child, so concurrent
    singles and batches together never exceed ``max_async_children``.

    When the batch finishes, a SINGLE completion event is pushed onto the
    shared ``process_registry.completion_queue`` carrying the full per-task
    ``results`` list, so the consolidated summaries re-enter the conversation
    as one message once every child is done — the chat is never blocked while
    they run.

    Returns ``{"status": "dispatched", "delegation_id": ...}`` when started,
    ``status="queued"`` when admitted to the bounded FIFO, or
    ``status="rejected"`` when the FIFO is full.
    """
    delegation_id = delegation_id or _new_delegation_id()
    dispatched_at = time.time()
    n = len(goals)
    slot_limit = max(1, int(max_async_children))
    if n == 0:
        return {"status": "rejected", "error": "Delegation batch is empty."}
    if n > slot_limit:
        return {
            "status": "rejected",
            "error": (
                f"Delegation batch requires {n} child slots, but "
                f"max_concurrent_children is {slot_limit}."
            ),
        }
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
        "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id,
        **_capture_routing_origin(),
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "is_batch": True,
        "progress_fn": progress_fn,
        "required_slots": n,
        "max_async_children": max_async_children,
        "min_available_memory_bytes": min_available_memory_bytes,
        "resume_available_memory_bytes": max(
            min_available_memory_bytes, resume_available_memory_bytes
        ),
        "max_memory_psi_avg10": max_memory_psi_avg10,
        "resume_memory_psi_avg10": resume_memory_psi_avg10,
        "queue_timeout_seconds": queue_timeout_seconds,
        "_queued_monotonic": time.monotonic(),
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
        "_generation": _manager_generation,
    }
    with _records_lock:
        if _shutdown_event.is_set():
            return {
                "status": "rejected",
                "error": "Async delegation manager is shutting down.",
            }
        capacity_ready = (
            _queue_head_locked() is None
            and _running_task_slots_locked() + record["required_slots"]
            <= max_async_children
        )
        resources_ready = _resources_available(record)
        queued = not (capacity_ready and resources_ready)
        queue_reason = "capacity" if not capacity_ready else "resources"
        if queued:
            pending = sum(1 for r in _records.values() if r.get("status") == "queued")
            queue_limit = max(0, int(max_queued_delegations))
            if pending >= queue_limit:
                return {
                    "status": "rejected",
                    "error": f"Async delegation queue is full ({queue_limit} pending).",
                }
            record["status"] = "queued"
        _records[delegation_id] = record

    try:
        _persist_dispatch(record)
    except Exception as exc:
        logger.exception("Failed to persist async delegation %s", delegation_id)
        with _admission_condition:
            _records.pop(delegation_id, None)
            _admission_condition.notify_all()
        return {
            "status": "rejected",
            "error": f"Failed to persist async delegation: {exc}",
        }
    try:
        executor = _get_executor(max_async_children)
    except Exception as exc:
        with _admission_condition:
            _records.pop(delegation_id, None)
            _admission_condition.notify_all()
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to initialize async delegation executor: {exc}",
        }

    def _worker() -> None:
        combined: Dict[str, Any] = {}
        status = "error"
        try:
            combined = runner() or {}
            lingering = combined.pop("_lingering_futures", [])
            if isinstance(lingering, list):
                _register_lingering_resources(delegation_id, lingering)
            requested_status = combined.get("status")
            if requested_status in {"interrupted", "timeout", "error"}:
                status = requested_status
            else:
                # Batch status: completed unless every child errored/interrupted.
                child_results = combined.get("results") or []
                if child_results and all(
                    (r.get("status") not in ("completed", "success"))
                    for r in child_results
                ):
                    status = "error"
                else:
                    status = "completed"
        except InterruptedError as exc:
            combined = {
                "status": "interrupted",
                "results": [],
                "error": str(exc),
                "total_duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "interrupted"
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

    worker_with_context = propagate_context_to_thread(_worker)

    def _launch_when_admitted() -> None:
        if not _wait_for_admission(delegation_id):
            return
        try:
            executor.submit(worker_with_context)
        except Exception as exc:  # executor may be shutting down
            logger.exception("Failed to launch queued batch %s", delegation_id)
            _finalize_queued_terminal(
                delegation_id,
                "error",
                f"Failed to launch queued delegation batch: {exc}",
            )

    try:
        if queued:
            threading.Thread(
                target=propagate_context_to_thread(_launch_when_admitted),
                name=f"async-delegate-admission-{delegation_id}",
                daemon=True,
            ).start()
        else:
            executor.submit(worker_with_context)
    except Exception as exc:  # pragma: no cover
        with _records_lock:
            _records.pop(delegation_id, None)
        _delete_durable_delegation(delegation_id)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation batch: {exc}",
        }
    if progress_fn is not None:
        _ensure_stale_monitor()

    logger.info(
        "%s async delegation batch %s (%d task(s), session_key=%s)",
        "Queued" if queued else "Dispatched",
        delegation_id, n, session_key or "<cli>",
    )
    return {
        "status": "queued" if queued else "dispatched",
        "delegation_id": delegation_id,
        **({"queue_reason": queue_reason} if queued else {}),
    }


def _finalize_batch(
    delegation_id: str, combined: Dict[str, Any], status: str
) -> None:
    """Mark a batch record complete and push ONE combined completion event."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn = claimed

    try:
        _push_batch_completion_event(event_record, combined, status)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Async batch delegation %s (%s): failed to publish completion "
            "event; releasing slot (%s)",
            delegation_id, status, exc,
        )
    finally:
        _finish_finalization(delegation_id, status)


def _push_batch_completion_event(
    event_record: Dict[str, Any], combined: Dict[str, Any], status: str
) -> None:
    """Push a combined async-delegation batch completion event."""
    if (
        event_record.get("_generation", _manager_generation)
        != _manager_generation
    ):
        logger.info(
            "Dropping stale-generation batch completion for %s (generation %s != %s)",
            event_record.get("delegation_id"),
            event_record.get("_generation"),
            _manager_generation,
        )
        return
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s finished but process_registry import "
            "failed; result lost: %s",
            event_record.get("delegation_id"), exc,
        )
        return

    dispatched_at = event_record.get("dispatched_at") or time.time()
    completed_at = event_record.get("completed_at") or time.time()
    evt = {
        "type": "async_delegation",
        "delegation_id": event_record.get("delegation_id"),
        "session_key": event_record.get("session_key", ""),
        "origin_ui_session_id": event_record.get("origin_ui_session_id", ""),
        "origin_session_id": event_record.get("origin_session_id", ""),
        "parent_session_id": event_record.get("parent_session_id"),
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
    # Routing origin captured at dispatch (see _capture_routing_origin).
    for _k in ("scope_id", "user_id", "user_name"):
        if event_record.get(_k):
            evt[_k] = event_record[_k]
    # Structured stall metadata (#51690) — additive, present only on
    # stall-monitor finalizations.
    for _k in (
        "stalled_after_quiet_seconds",
        "stall_threshold_seconds",
        "stall_phase",
        "stall_grace_seconds",
    ):
        if _k in combined:
            evt[_k] = combined[_k]
    _persist_completion(evt, combined)
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s: failed to enqueue completion event; "
            "result lost: %s",
            event_record.get("delegation_id"), exc,
        )


def _ensure_stale_monitor() -> None:
    """Start (once) the module-level stale-delegation monitor thread.

    One daemon thread serves every dispatch; it exits on its own when no
    monitorable records remain, and is restarted by the next dispatch that
    carries a ``progress_fn``.
    """
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return
        _monitor_stop.clear()
        _monitor_thread = threading.Thread(
            target=_stale_monitor_loop,
            name="async-delegate-stale-monitor",
            daemon=True,
        )
        _monitor_thread.start()


def _stale_monitor_loop() -> None:
    """Sweep running delegations for stalled progress.

    Per sweep, for every running record with a ``progress_fn``:

    - Sample ``(token, in_tool)``. A changed token refreshes the record's
      progress timestamp — a child that keeps advancing is never touched, no
      matter how long it runs.
    - A frozen token past the idle/in-tool threshold marks the record
      ``stalling``: we call ``interrupt_fn`` so a responsive-but-slow child
      can unwind and deliver its (partial) result through the normal
      ``_finalize`` path with full fidelity.
    - A ``stalling`` record whose runner still hasn't returned after the
      grace window is force-finalized with one terminal ``stalled`` event so
      the owning session hears an outcome and the async slot frees. A late
      runner return after that is ignored by ``_begin_finalization``.
    """
    while not _monitor_stop.wait(_STALE_CHECK_INTERVAL):
        now = time.time()
        stalled: List[tuple] = []  # (delegation_id, is_batch, quiet_for, in_tool)
        expired: List[str] = []  # stalling past grace → force-finalize
        any_monitorable = False
        with _records_lock:
            for record in _records.values():
                status = record.get("status")
                if status == "stalling":
                    any_monitorable = True
                    interrupted_at = record.get("_interrupted_at") or now
                    if now - interrupted_at >= _STALL_GRACE_SECONDS:
                        expired.append(record["delegation_id"])
                    continue
                if status != "running":
                    continue
                progress_fn = record.get("progress_fn")
                if progress_fn is None:
                    continue
                any_monitorable = True
                try:
                    token, in_tool = progress_fn()
                except Exception:
                    # An unreadable child must not look permanently healthy —
                    # keep the last timestamp running instead of refreshing it.
                    token, in_tool = record.get("_progress_token"), False
                if token != record.get("_progress_token"):
                    record["_progress_token"] = token
                    record["_progress_ts"] = now
                    continue
                quiet_for = now - (record.get("_progress_ts") or now)
                limit = (
                    _STALE_IN_TOOL_SECONDS if in_tool else _STALE_IDLE_SECONDS
                )
                if quiet_for >= limit:
                    record["status"] = "stalling"
                    record["_interrupted_at"] = now
                    # Structured stall context for the terminal event and
                    # status listings (#51690): how long progress was frozen,
                    # which threshold applied, and whether the child was
                    # inside a tool when it went quiet.
                    record["_stall_quiet_seconds"] = round(quiet_for, 2)
                    record["_stall_threshold_seconds"] = limit
                    record["_stall_in_tool"] = bool(in_tool)
                    stalled.append(
                        (
                            record["delegation_id"],
                            bool(record.get("is_batch")),
                            quiet_for,
                            in_tool,
                        )
                    )
        for delegation_id, _is_batch, quiet_for, in_tool in stalled:
            logger.warning(
                "Async delegation %s made no progress for %.0fs "
                "(in_tool=%s) — interrupting; grace window %.0fs",
                delegation_id, quiet_for, in_tool, _STALL_GRACE_SECONDS,
            )
            with _records_lock:
                record = _records.get(delegation_id)
                fn = record.get("interrupt_fn") if record else None
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    logger.debug(
                        "Async delegation %s stall interrupt failed: %s",
                        delegation_id, exc,
                    )
        for delegation_id in expired:
            _finalize_stalled(delegation_id)
        if not any_monitorable:
            return


def _finalize_stalled(delegation_id: str) -> None:
    """Force-finalize a stalling delegation whose runner never returned."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn = claimed

    completed_at = event_record.get("completed_at") or time.time()
    duration = round(
        completed_at - (event_record.get("dispatched_at") or completed_at),
        2,
    )
    quiet_seconds = event_record.get("_stall_quiet_seconds")
    threshold_seconds = event_record.get("_stall_threshold_seconds")
    stall_in_tool = event_record.get("_stall_in_tool")
    error = (
        f"Async delegation {delegation_id} stalled: the detached subagent "
        "stopped making progress (no new API calls, tool activity, or "
        "streamed tokens), did not respond to interruption, and never "
        "produced a completion event. The worker may be wedged inside a "
        "model API call — this is a known failure mode of long-lived "
        "gateway processes (#60203). Re-dispatch the task if it is still "
        "needed."
    )
    logger.error(
        "Async delegation %s force-finalized as stalled after %.0fs",
        delegation_id, duration,
    )
    # Structured stall metadata (#51690): lets parents and UIs distinguish
    # a stall-monitor kill from other failures without parsing the error
    # string, mirroring the sync path's timeout_seconds/timed_out_after_
    # seconds/timeout_phase fields.
    stall_meta = {
        "stalled_after_quiet_seconds": quiet_seconds,
        "stall_threshold_seconds": threshold_seconds,
        "stall_phase": (
            "in_tool" if stall_in_tool
            else "idle" if stall_in_tool is not None
            else None
        ),
        "stall_grace_seconds": _STALL_GRACE_SECONDS,
    }
    try:
        if event_record.get("is_batch"):
            _push_batch_completion_event(
                event_record,
                {
                    "results": [],
                    "error": error,
                    "total_duration_seconds": duration,
                    **stall_meta,
                },
                "stalled",
            )
        else:
            _push_completion_event(
                event_record,
                {
                    "status": "stalled",
                    "summary": None,
                    "error": error,
                    "api_calls": 0,
                    "duration_seconds": duration,
                    "exit_reason": "stalled",
                    **stall_meta,
                },
                "stalled",
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Async delegation %s (stalled): failed to publish completion "
            "event; releasing slot (%s)",
            delegation_id, exc,
        )
    finally:
        _finish_finalization(delegation_id, "stalled")


def _children_activity_from_token(token: Any, now: float) -> Optional[List]:
    """Parse a progress token into per-child activity dicts (best-effort).

    delegate_tool's ``_batch_progress`` emits one ``(api_call_count,
    current_tool, last_activity_ts)`` tuple per child. Foreign token shapes
    (custom dispatchers) degrade to ``None`` entries rather than raising —
    the token contract is intentionally opaque to the registry.
    """
    try:
        parts = list(token)
    except TypeError:
        return None
    out: List[Optional[Dict[str, Any]]] = []
    for part in parts:
        if isinstance(part, (list, tuple)) and len(part) >= 2:
            entry: Dict[str, Any] = {
                "api_calls": part[0],
                "current_tool": part[1],
            }
            if len(part) >= 3 and isinstance(part[2], (int, float)):
                entry["seconds_since_activity"] = round(
                    max(0.0, now - float(part[2])), 1
                )
            out.append(entry)
        else:
            out.append(None)
    return out


def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed).

    Safe to call from any thread. Excludes the non-serialisable callables
    and private monitor bookkeeping, but exposes computed live-status
    fields for UIs (#51690):

    - ``seconds_since_progress``: how long the stale monitor has seen a
      frozen progress token (running/stalling records).
    - ``children_activity``: per-child ``{api_calls, current_tool,
      seconds_since_activity}`` sampled live from the dispatch's
      ``progress_fn``.
    - ``stalled_after_quiet_seconds`` / ``stall_threshold_seconds`` /
      ``stall_in_tool``: stall context once the monitor has tripped.
    """
    now = time.time()
    samplers: Dict[str, Callable] = {}
    with _records_lock:
        items = []
        for r in _records.values():
            item = {
                k: v
                for k, v in r.items()
                if k not in {"interrupt_fn", "progress_fn"}
                and not k.startswith("_")
            }
            status = r.get("status")
            if status in ("running", "stalling"):
                ts = r.get("_progress_ts")
                if ts:
                    item["seconds_since_progress"] = round(now - ts, 1)
                fn = r.get("progress_fn")
                if callable(fn):
                    samplers[r["delegation_id"]] = fn
            if status in ("stalling", "stalled"):
                for src, dst in (
                    ("_stall_quiet_seconds", "stalled_after_quiet_seconds"),
                    ("_stall_threshold_seconds", "stall_threshold_seconds"),
                    ("_stall_in_tool", "stall_in_tool"),
                ):
                    if r.get(src) is not None:
                        item[dst] = r.get(src)
            items.append(item)

    # Sample live activity OUTSIDE the lock — progress_fn reads child-agent
    # attributes and must never run under _records_lock (a slow or broken
    # sampler would block every dispatch/finalize in the process).
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


def _finalize_queued_terminal(
    delegation_id: str,
    status: str,
    error: str,
) -> bool:
    """Publish one terminal event for a delegation that never started.

    Only a record still in ``queued`` may be terminalized here. If the record
    was promoted to ``running`` between the caller's snapshot and this claim
    (interrupt racing admission), fall back to ``interrupt_fn`` so the worker
    observes the cancellation and self-cancels — terminalizing it here would
    let the worker run the delegation un-cancelled while the completion is
    silently dropped (see _begin_finalization).
    """
    claimed = _begin_finalization(delegation_id, allowed_statuses=("queued",))
    if claimed is None:
        # Either the record no longer exists (already terminal) or it was
        # promoted. Deliver the cancellation through the runner's interrupt
        # hook; the not-yet-started worker then finalizes itself with
        # exactly-once delivery. If the worker already completed (record is
        # terminal or being finalized), there is nothing left to cancel —
        # report False so callers don't over-count an interruption.
        with _records_lock:
            record = _records.get(delegation_id)
            if record is None or record.get("status") not in ("running", "stalling"):
                return False
            interrupt_fn = record.get("interrupt_fn")
        if callable(interrupt_fn):
            try:
                interrupt_fn()
                return True
            except Exception as exc:
                logger.debug(
                    "Queued interruption fallback for %s failed: %s",
                    delegation_id, exc,
                )
        return False
    event_record, _interrupt_fn = claimed
    try:
        if event_record.get("is_batch"):
            _push_batch_completion_event(
                event_record,
                {"results": [], "error": error, "total_duration_seconds": 0.0},
                status,
            )
        else:
            _push_completion_event(
                event_record,
                {
                    "status": status,
                    "summary": None,
                    "error": error,
                    "api_calls": 0,
                    "duration_seconds": 0.0,
                    "exit_reason": status,
                },
                status,
            )
    except Exception as exc:  # noqa: BLE001
        # A durable-persist failure must not strand this record in
        # "finalizing" with no actor to retry it: queued work never started,
        # so no runner thread will ever finalize it and its slot would leak
        # for the process lifetime. Log loudly, then the finally below
        # releases the slot; restart recovery re-classifies the durable row
        # conservatively (queued -> interrupted).
        logger.exception(
            "Queued delegation %s (%s): failed to publish terminal event; "
            "releasing slot (%s)",
            delegation_id, status, exc,
        )
    finally:
        _finish_finalization(delegation_id, status)
    return True


def _finalize_queued_interruption(delegation_id: str, reason: str) -> bool:
    return _finalize_queued_terminal(
        delegation_id,
        "interrupted",
        f"Queued delegation cancelled before start ({reason}).",
    )


def begin_shutdown(reason: str = "shutdown") -> int:
    """Stop new admission and cancel every queued/running delegation.

    This is process-lifecycle cleanup. Session-scoped ``/stop`` must use
    ``interrupt_for_session`` instead.
    """
    with _admission_condition:
        _shutdown_event.set()
        _admission_condition.notify_all()
    return interrupt_all(reason=reason)


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every running async delegation to stop. Returns how many.

    Used on ``/stop`` and gateway shutdown so a dangling background subagent
    can't keep burning tokens with no one listening. The child still emits a
    completion event (status='interrupted') via the normal finalize path.
    """
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values()
            if r.get("status") in ("queued", "running", "stalling")
        ]
    for r in targets:
        if r.get("status") == "queued":
            if _finalize_queued_interruption(r["delegation_id"], reason):
                count += 1
            continue
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
            if r.get("status") in ("queued", "running", "stalling")
            and _matches_session_selectors(
                r,
                session_key=session_key,
                origin_ui_session_id=origin_ui_session_id,
                parent_session_id=parent_session_id,
            )
        ]
    for r in targets:
        if r.get("status") == "queued":
            if _finalize_queued_interruption(r["delegation_id"], reason):
                count += 1
            continue
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
    """Test-only: clear all state and tear down the executor + monitor."""
    global _executor, _executor_max_workers, _monitor_thread, _manager_generation
    with _admission_condition:
        _shutdown_event.set()
        # Bump the generation BEFORE canceling so any worker that already
        # captured its record cannot push a completion into the shared queue
        # once we clear state below.
        _manager_generation += 1
        _admission_condition.notify_all()
    interrupt_all(reason="test reset")
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None
        _executor_max_workers = 0
    _monitor_stop.set()
    with _monitor_lock:
        thread = _monitor_thread
        _monitor_thread = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    with _admission_condition:
        _records.clear()
        _lingering_resource_slots.clear()
        _shutdown_event.clear()
        _admission_condition.notify_all()
