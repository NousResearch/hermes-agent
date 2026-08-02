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
# delegation_id -> record dict. Kept for the lifetime of the run plus a short
# tail after completion so `list_async_delegations()` can show recent results.
_records: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MAX_ASYNC_CHILDREN = 3
# How many completed records to retain for status queries before pruning.
_MAX_RETAINED_COMPLETED = 50
# Parent-facing controls use a bounded, in-memory live snapshot only.
_MAX_OWNER_LIVE_SNAPSHOTS = 20
_MAX_OWNER_LIVE_CURSOR = 10_000
_MAX_OWNER_GOAL_CHARS = 160
_MAX_OWNER_ROLE_CHARS = 80
_MAX_DELEGATION_ID_CHARS = 128
_MAX_STEER_MESSAGE_CHARS = 4000
_MAX_STEER_MESSAGES = 20
_MAX_STEER_TOTAL_CHARS = 12000
_NATIVE_DELEGATION_ID_LENGTH = len("deleg_") + 8
_DURABLE_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_DURABLE_PENDING = 1000
# A pending completion whose delivery keeps failing is retried across claim
# cycles (and across restarts via restore_undelivered_completions). Cap the
# attempts so an unroutable row converges to a terminal 'dropped' state
# instead of replaying on every restart forever.
_MAX_DELIVERY_ATTEMPTS = 8
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


def _persist_dispatch(record: Dict[str, Any]) -> None:
    now = time.time()
    try:
        from gateway.status import get_process_start_time
        owner_started_at = get_process_start_time(__import__("os").getpid())
    except Exception:
        owner_started_at = None
    task_payload = {
        key: record.get(key)
        for key in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch")
        if key in record
    }
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, updated_at,
                delivery_state, delivery_attempts, owner_pid,
                owner_started_at, task_json, origin_session_id)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (record["delegation_id"], record.get("session_key", ""),
             record.get("origin_ui_session_id", ""), record.get("parent_session_id"),
             record["dispatched_at"], now, __import__("os").getpid(),
             owner_started_at, json.dumps(task_payload),
             record.get("origin_session_id", "")),
        )
    _prune_durable_records()


def _delete_durable_delegation(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))


def _prune_durable_records() -> None:
    """Bound terminal history, preferring delivered records for deletion."""
    now = time.time()
    cutoff = now - _DURABLE_RETENTION_SECONDS
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < ?",
            (cutoff,),
        )
        terminal_count = conn.execute(
            "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing')"
        ).fetchone()[0]
        excess = max(0, terminal_count - _MAX_RETAINED_COMPLETED)
        if excess:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing')
                     ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                              updated_at ASC LIMIT ?
                   )""",
                (excess,),
            )
        pending_count = conn.execute(
            """SELECT COUNT(*) FROM async_delegations
               WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'"""
        ).fetchone()[0]
        overflow = max(0, pending_count - _MAX_DURABLE_PENDING)
        if overflow:
            conn.execute(
                """DELETE FROM async_delegations WHERE delegation_id IN (
                     SELECT delegation_id FROM async_delegations
                     WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
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


def _note_delivery_attempt(delegation_id: str) -> None:
    with _DB_LOCK, _transaction() as conn:
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
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, origin_session, origin_ui_session_id,
                      parent_session_id, dispatched_at, owner_pid,
                      owner_started_at, task_json, origin_session_id
               FROM async_delegations WHERE state IN ('running','finalizing')"""
        ).fetchall()
        for row in rows:
            (delegation_id, session_key, origin_ui, parent_id, dispatched_at,
             pid, started, task_json, origin_session_id) = row
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
                # Restore the durable wake target so completions recovered
                # after a restart remain routable to api_server sessions.
                "origin_session_id": origin_session_id or "",
                "parent_session_id": parent_id, "goal": task.get("goal", ""),
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
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT delegation_id, event_json FROM async_delegations
               WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
               ORDER BY completed_at, delegation_id"""
        ).fetchall()
        for _delegation_id, payload in rows:
            evt = json.loads(payload)
            if isinstance(evt, dict):
                evt["restored"] = True
            target_queue.put(evt)
    return len(rows)


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

    A unit is one dispatch: a single subagent OR a whole fan-out batch. A batch
    counts as ONE here because it occupies one async-pool slot (the capacity
    semantics ``dispatch_async_delegation_batch`` relies on). For the count of
    actual concurrent child subagents (batch expanded), use
    ``active_task_count()``.
    """
    with _records_lock:
        return sum(
            1 for r in _records.values()
            if r.get("status") in {"running", "stalling", "finalizing"}
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


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the retention cap.

    Caller must hold ``_records_lock``.
    """
    completed = [
        (rid, r)
        for rid, r in _records.items()
        if r.get("status") != "running"
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


def _current_control_owner(session_id: Optional[str] = None) -> Optional[tuple[str, str]]:
    """Return a canonical profile plus a proven compression-only session root."""
    try:
        from agent.relay_runtime import current_profile_key
        from gateway.session_context import get_session_env
        from hermes_state import SessionDB

        profile = current_profile_key()
        if session_id is None:
            session_id = get_session_env("HERMES_SESSION_ID", "")
    except Exception:
        return None
    if not isinstance(profile, str) or not isinstance(session_id, str):
        return None
    profile, session_id = profile.strip(), session_id.strip()
    if not profile or not session_id:
        return None
    try:
        db = SessionDB(get_hermes_home() / "state.db", read_only=True)
        try:
            root = db.get_compression_lineage_root(session_id)
        finally:
            db.close()
    except Exception:
        return None
    return (profile, root) if root else None


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
    steer_fn: Optional[Callable[[str], bool]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
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
        Concurrency cap. When at capacity the dispatch is REJECTED (the caller
        should fall back to sync or tell the user) rather than queued, so a
        runaway model can't pile up unbounded background work.

    Returns
    -------
    dict
        ``{"status": "dispatched", "delegation_id": ...}`` on success, or
        ``{"status": "rejected", "error": ...}`` when at capacity.
    """
    delegation_id = _new_delegation_id()
    owner = _current_control_owner(parent_session_id) if isinstance(parent_session_id, str) and parent_session_id.strip() else None
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
        "owner_profile": owner[0] if owner else "",
        "owner_session_id": owner[1] if owner else "",
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "steer_fn": steer_fn,
        "progress_fn": progress_fn,
        # Stale-monitor bookkeeping (see _stale_monitor_loop).
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
    }
    # Capacity check and record insert under ONE lock hold — checking
    # active_count() separately would let two concurrent dispatches (e.g.
    # from different gateway sessions) both pass the check and exceed the cap.
    with _records_lock:
        running = sum(
            1 for r in _records.values()
            if r.get("status") in ("running", "stalling")
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or run this task synchronously "
                    f"(background=false). Raise delegation.max_concurrent_children in "
                    f"config.yaml to allow more concurrent background subagents."
                ),
            }
        _records[delegation_id] = record

    _persist_dispatch(record)
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
    if progress_fn is not None:
        _ensure_stale_monitor()

    logger.info(
        "Dispatched async delegation %s (session_key=%s): %s",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn = claimed

    _push_completion_event(event_record, result, status)
    _finish_finalization(delegation_id, status)


def _begin_finalization(
    delegation_id: str,
) -> Optional[tuple[Dict[str, Any], Optional[Callable[[], None]]]]:
    """Atomically claim terminal delivery while keeping the record active."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("status") not in ("running", "stalling"):
            return
        # Stay active until durable persistence and queue publication finish;
        # otherwise process shutdown can kill this daemon worker in the narrow
        # gap after status flips but before SQLite is committed.
        record["status"] = "finalizing"
        record["completed_at"] = time.time()
        interrupt_fn = record.get("interrupt_fn")
        record["interrupt_fn"] = None  # drop the closure; child is done
        record["steer_fn"] = None  # controls are only valid while live
        record["progress_fn"] = None  # stop stale-monitor sampling
        event_record = dict(record)

    return event_record, interrupt_fn


def _finish_finalization(delegation_id: str, status: str) -> None:
    with _records_lock:
        record = _records.get(delegation_id)
        if record is not None:
            record["status"] = status
        _prune_completed_locked()


def _push_completion_event(
    record: Dict[str, Any], result: Dict[str, Any], status: str
) -> None:
    """Push a type='async_delegation' event onto the shared completion queue.

    Best-effort: a failure here must not crash the worker, but it WOULD mean a
    silently-lost result, so we log loudly.
    """
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
    steer_fn: Optional[Callable[[str], bool]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    delegation_id: Optional[str] = None,
    progress_fn: Optional[Callable[[], tuple]] = None,
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
    if not _is_native_delegation_id(delegation_id):
        return {"status": "rejected", "error": "Invalid async delegation id"}
    owner = _current_control_owner(parent_session_id) if isinstance(parent_session_id, str) and parent_session_id.strip() else None
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
        "origin_session_id": origin_session_id,
        "parent_session_id": parent_session_id,
        "owner_profile": owner[0] if owner else "",
        "owner_session_id": owner[1] if owner else "",
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "steer_fn": steer_fn,
        "is_batch": True,
        "progress_fn": progress_fn,
        "_progress_token": None,
        "_progress_ts": dispatched_at,
        "_interrupted_at": None,
    }
    with _records_lock:
        running = sum(
            1 for r in _records.values()
            if r.get("status") in ("running", "stalling")
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or raise delegation.max_concurrent_children in "
                    f"config.yaml to allow more concurrent background units."
                ),
            }
        _records[delegation_id] = record

    _persist_dispatch(record)
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
    if progress_fn is not None:
        _ensure_stale_monitor()

    logger.info(
        "Dispatched async delegation batch %s (%d task(s), session_key=%s)",
        delegation_id, n, session_key or "<cli>",
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize_batch(
    delegation_id: str, combined: Dict[str, Any], status: str
) -> None:
    """Mark a batch record complete and push ONE combined completion event."""
    claimed = _begin_finalization(delegation_id)
    if claimed is None:
        return
    event_record, _interrupt_fn = claimed

    _push_batch_completion_event(event_record, combined, status)
    _finish_finalization(delegation_id, status)


def _push_batch_completion_event(
    event_record: Dict[str, Any], combined: Dict[str, Any], status: str
) -> None:
    """Push a combined async-delegation batch completion event."""
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
            if record is not None:
                _interrupt_live_record(record, "stale_monitor")
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


def _owned_live_record_locked(owner: tuple[str, str], delegation_id: str) -> Optional[Dict[str, Any]]:
    record = _records.get(delegation_id)
    if (
        record is None
        or record.get("owner_profile") != owner[0]
        or record.get("owner_session_id") != owner[1]
        or record.get("status") not in ("running", "stalling")
    ):
        return None
    return record


def _control_lock_locked(record: Dict[str, Any]) -> threading.Lock:
    """Return this record's control mutex while the registry lock is held."""
    lock = record.get("_control_lock")
    if not hasattr(lock, "acquire") or not hasattr(lock, "release"):
        lock = threading.Lock()
        record["_control_lock"] = lock
    return lock


def _safe_control_text(value: Any, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _is_native_delegation_id(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _NATIVE_DELEGATION_ID_LENGTH:
        return False
    suffix = value.removeprefix("deleg_")
    return len(suffix) == 8 and all(char in "0123456789abcdef" for char in suffix)


def list_owned_async_delegations(
    owner_session_id: Optional[str] = None,
    cursor: int = 0,
) -> Dict[str, Any]:
    """Return one bounded, non-authorizing chooser page for the current owner."""
    if (
        isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or not 0 <= cursor <= _MAX_OWNER_LIVE_CURSOR
    ):
        return {
            "delegations": [],
            "total_live": 0,
            "truncated": False,
            "next_cursor": None,
        }
    owner = _current_control_owner(owner_session_id)
    if owner is None:
        return {
            "delegations": [],
            "total_live": 0,
            "truncated": False,
            "next_cursor": None,
        }
    with _records_lock:
        items = [
            {
                "delegation_id": delegation_id,
                "status": record["status"],
                "is_batch": bool(record.get("is_batch")),
                "dispatched_at": record.get("dispatched_at"),
                "goal": _safe_control_text(record.get("goal"), _MAX_OWNER_GOAL_CHARS),
                "role": _safe_control_text(record.get("role"), _MAX_OWNER_ROLE_CHARS),
            }
            for delegation_id, record in _records.items()
            if (
                _is_native_delegation_id(delegation_id)
                and _owned_live_record_locked(owner, delegation_id) is record
            )
        ]
    items.sort(
        key=lambda item: (item["dispatched_at"] or 0, item["delegation_id"]),
        reverse=True,
    )
    total_live = len(items)
    page = items[cursor: cursor + _MAX_OWNER_LIVE_SNAPSHOTS]
    next_cursor = cursor + len(page)
    return {
        "delegations": page,
        "total_live": total_live,
        "truncated": next_cursor < total_live,
        "next_cursor": next_cursor if next_cursor < total_live else None,
    }


def cancel_owned_async_delegation(
    delegation_id: str,
    owner_session_id: Optional[str] = None,
) -> Dict[str, str]:
    """Issue one cooperative interrupt request for the current owner."""
    if not _is_native_delegation_id(delegation_id):
        return {"status": "not_found"}
    owner = _current_control_owner(owner_session_id)
    if owner is None:
        return {"status": "not_found"}
    with _records_lock:
        record = _owned_live_record_locked(owner, delegation_id)
        if record is None:
            return {"status": "not_found"}
        control_lock = _control_lock_locked(record)
    if not control_lock.acquire(blocking=False):
        return {"status": "pending"}
    try:
        with _records_lock:
            if _owned_live_record_locked(owner, delegation_id) is not record:
                return {"status": "not_found"}
            state = record.get("_cancel_state")
            if state == "accepted":
                return {"status": "accepted"}
            if state in ("indeterminate", "error"):
                return {"status": "indeterminate"}
            if state == "requesting":
                return {"status": "pending"}
            callback = record.get("interrupt_fn")
            if not callable(callback):
                return {"status": "rejected"}
            record["_cancel_state"] = "requesting"
        try:
            callback()
        except Exception:  # noqa: BLE001 - issuance may have taken effect
            logger.debug(
                "Async delegation %s cancel callback failed",
                delegation_id,
                exc_info=True,
            )
            with _records_lock:
                if _owned_live_record_locked(owner, delegation_id) is not record:
                    return {"status": "not_found"}
                record["_cancel_state"] = "indeterminate"
            return {"status": "indeterminate"}
        with _records_lock:
            if _owned_live_record_locked(owner, delegation_id) is not record:
                return {"status": "not_found"}
            record["_cancel_state"] = "accepted"
        return {"status": "accepted"}
    finally:
        control_lock.release()


def steer_owned_async_delegation(
    delegation_id: str,
    message: str,
    owner_session_id: Optional[str] = None,
) -> Dict[str, str]:
    """Offer bounded guidance to one live delegation owned by this runtime."""
    if (
        not isinstance(message, str)
        or not message.strip()
        or len(message) > _MAX_STEER_MESSAGE_CHARS
    ):
        return {"status": "rejected"}
    if not _is_native_delegation_id(delegation_id):
        return {"status": "not_found"}
    owner = _current_control_owner(owner_session_id)
    if owner is None:
        return {"status": "not_found"}
    with _records_lock:
        record = _owned_live_record_locked(owner, delegation_id)
        if record is None:
            return {"status": "not_found"}
        control_lock = _control_lock_locked(record)
    if not control_lock.acquire(blocking=False):
        return {"status": "pending"}
    try:
        with _records_lock:
            if _owned_live_record_locked(owner, delegation_id) is not record:
                return {"status": "not_found"}
            if record.get("_cancel_state") in (
                "requesting",
                "accepted",
                "indeterminate",
                "error",
            ):
                return {"status": "rejected"}
            used_count = int(record.get("_steer_message_count", 0))
            used_chars = int(record.get("_steer_character_count", 0))
            if (
                used_count >= _MAX_STEER_MESSAGES
                or used_chars + len(message) > _MAX_STEER_TOTAL_CHARS
            ):
                return {"status": "rejected"}
            callback = record.get("steer_fn")
            if not callable(callback):
                return {"status": "rejected"}
        try:
            accepted = bool(callback(message))
        except Exception:  # noqa: BLE001 - child callback boundary
            logger.debug(
                "Async delegation %s steer callback failed",
                delegation_id,
                exc_info=True,
            )
            accepted = None
        with _records_lock:
            if _owned_live_record_locked(owner, delegation_id) is not record:
                return {"status": "not_found"}
            if record.get("_cancel_state") in (
                "requesting",
                "accepted",
                "indeterminate",
                "error",
            ):
                return {"status": "rejected"}
            if accepted is None:
                return {"status": "error"}
            if not accepted:
                return {"status": "rejected"}
            record["_steer_message_count"] = used_count + 1
            record["_steer_character_count"] = used_chars + len(message)
        return {"status": "accepted"}
    finally:
        control_lock.release()


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
                if k not in {
                    "interrupt_fn", "steer_fn", "progress_fn", "owner_profile", "owner_session_id"
                }
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


def _interrupt_live_record(record: Dict[str, Any], source: str) -> bool:
    """Issue one lifecycle interrupt while superseding concurrent steering."""
    with _records_lock:
        delegation_id = record.get("delegation_id")
        if (
            not isinstance(delegation_id, str)
            or _records.get(delegation_id) is not record
            or record.get("status") not in ("running", "stalling")
        ):
            return False
        if record.get("_cancel_state") in (
            "requesting",
            "accepted",
            "indeterminate",
            "error",
        ):
            return False
        callback = record.get("interrupt_fn")
        if not callable(callback):
            return False
        record["_cancel_state"] = "requesting"
    try:
        callback()
    except Exception as exc:
        with _records_lock:
            if _records.get(delegation_id) is record:
                record["_cancel_state"] = "indeterminate"
        logger.debug("%s: %s interrupt failed: %s", source, delegation_id, exc)
        return False
    with _records_lock:
        if _records.get(delegation_id) is record:
            record["_cancel_state"] = "accepted"
    return True


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
            if r.get("status") in ("running", "stalling")
        ]
    for r in targets:
        if _interrupt_live_record(r, "interrupt_all"):
            count += 1
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
            if r.get("status") in ("running", "stalling")
            and (
                (origin_ui_session_id and str(r.get("origin_ui_session_id") or "") == origin_ui_session_id)
                or (session_key and str(r.get("session_key") or "") == session_key)
                or (parent_session_id and str(r.get("parent_session_id") or "") == parent_session_id)
            )
        ]
    for r in targets:
        if _interrupt_live_record(r, "interrupt_for_session"):
            count += 1
    if count:
        logger.info(
            "Interrupted %d async delegation(s) for ending session (%s)",
            count, reason,
        )
    return count


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
        thread = _monitor_thread
        _monitor_thread = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    with _records_lock:
        _records.clear()
