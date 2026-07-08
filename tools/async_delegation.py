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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread
from utils import atomic_json_write

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

# Durable checkpoint for async delegation lifecycle.  Background subagent worker
# threads are intentionally process-local daemon threads — a hard gateway/CLI
# restart cannot adopt them — but the parent must still get a recovered failure
# notice instead of silently losing the handle/result.
_PERSISTENCE_VERSION = 1
_PERSISTENCE_FILENAME = "async_delegations.json"
_MAX_PERSISTENT_RECORDS = 200
_persistent_lock = threading.Lock()
_TERMINAL_STATUSES = {"completed", "success", "error", "failed", "timeout", "interrupted", "lost"}


def _home_path(hermes_home: str | Path | None = None) -> Path:
    return Path(hermes_home) if hermes_home is not None else get_hermes_home()


def _checkpoint_path(hermes_home: str | Path | None = None) -> Path:
    return _home_path(hermes_home) / _PERSISTENCE_FILENAME


def _recovery_dir(hermes_home: str | Path | None = None) -> Path:
    return _home_path(hermes_home) / "recovery" / "async-delegations"


def _get_process_start_time(pid: int) -> Optional[int]:
    """Return a stable per-process start-time fingerprint when available."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None

    stat_path = Path(f"/proc/{pid}/stat")
    try:
        return int(stat_path.read_text(encoding="utf-8").split()[21])
    except (FileNotFoundError, IndexError, PermissionError, ValueError, OSError):
        pass

    try:
        import psutil  # type: ignore

        return int(round(psutil.Process(pid).create_time() * 100))
    except Exception:
        return None


def _pid_exists(pid: int) -> bool:
    """Best-effort PID liveness probe that avoids false-dead on uncertainty."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(pid))
    except Exception:
        pass
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    # On Windows without psutil we cannot prove a foreign PID is dead.  Keep the
    # running checkpoint retryable rather than falsely converting a live sibling
    # process's delegation to lost.
    return True


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _owner_metadata(session_key: str | None = None) -> Dict[str, Any]:
    pid = os.getpid()
    return {
        "owner_pid": pid,
        "owner_start_time": _get_process_start_time(pid),
        "owner_hermes_home": str(get_hermes_home()),
        "owner_session_key": session_key or "",
        "owner_heartbeat_at": time.time(),
    }


def _running_owner_is_recoverably_gone(
    delegation_id: str,
    record: Dict[str, Any],
    live_running: set[str],
) -> bool:
    """Return True only when a running checkpoint is safe to surface as lost.

    A different Hermes process may legitimately own the same HERMES_HOME and be
    running the delegation.  The in-memory ``_records`` map is process-local, so
    absence from *this* process is not evidence of loss.  Only convert to lost
    when the owner is this process but no local record remains, the owner PID is
    dead/reused, or the checkpoint predates owner metadata.
    """
    if delegation_id in live_running:
        return False

    owner_pid = _int_or_none(record.get("owner_pid"))
    if owner_pid is None:
        # Legacy checkpoints have no cross-process owner metadata. Preserve the
        # previous recovery behavior for those old records so they do not remain
        # silent forever after a real crash.
        return True

    if owner_pid == os.getpid():
        # Same process, but no live in-memory record: local executor state was
        # reset or the record is stale. Daemon worker threads cannot be adopted.
        return True

    recorded_start = _int_or_none(record.get("owner_start_time"))
    live_start = _get_process_start_time(owner_pid)
    if live_start is not None:
        if recorded_start is None:
            # PID exists but the old checkpoint cannot fingerprint reuse. Avoid
            # stealing a possibly-live sibling's delegation.
            return False
        return live_start != recorded_start

    if _pid_exists(owner_pid):
        # Live but start time unreadable (permissions/Windows without psutil).
        # Conservative default: do not mark another process's work lost.
        return False
    return True


def _json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-native data for durable checkpoints."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _persistent_record(record: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "delegation_id",
        "goal",
        "goals",
        "context",
        "toolsets",
        "role",
        "model",
        "session_key",
        "status",
        "dispatched_at",
        "completed_at",
        "parent_session_id",
        "parent_turn_id",
        "child_session_ids",
        "is_batch",
        "delivered",
        "delivered_at",
        "updated_at",
        "owner_pid",
        "owner_start_time",
        "owner_hermes_home",
        "owner_session_key",
        "owner_heartbeat_at",
        "result_event",
        "recovery_packet_path",
    }
    out = {k: _json_safe(v) for k, v in record.items() if k in allowed}
    out.setdefault("delivered", False)
    out["updated_at"] = time.time()
    return out


def _read_persistent_state_unlocked(
    hermes_home: str | Path | None = None,
) -> Dict[str, Any]:
    path = _checkpoint_path(hermes_home)
    if not path.exists():
        return {"version": _PERSISTENCE_VERSION, "records": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read async delegation checkpoint %s: %s", path, exc)
        return {"version": _PERSISTENCE_VERSION, "records": {}}
    if isinstance(raw, dict):
        records = raw.get("records", {})
    elif isinstance(raw, list):  # forward-compatible with early/manual packets
        records = {str(r.get("delegation_id")): r for r in raw if isinstance(r, dict) and r.get("delegation_id")}
    else:
        records = {}
    if not isinstance(records, dict):
        records = {}
    return {"version": _PERSISTENCE_VERSION, "records": records}


def _prune_persistent_records_unlocked(state: Dict[str, Any]) -> None:
    records = state.setdefault("records", {})
    if not isinstance(records, dict) or len(records) <= _MAX_PERSISTENT_RECORDS:
        return
    removable = [
        (rid, rec)
        for rid, rec in records.items()
        if isinstance(rec, dict) and rec.get("delivered")
    ]
    removable.sort(key=lambda item: item[1].get("delivered_at") or item[1].get("completed_at") or item[1].get("updated_at") or 0)
    overflow = len(records) - _MAX_PERSISTENT_RECORDS
    for rid, _ in removable[:overflow]:
        records.pop(rid, None)


def _write_persistent_state_unlocked(
    state: Dict[str, Any],
    hermes_home: str | Path | None = None,
) -> None:
    _prune_persistent_records_unlocked(state)
    atomic_json_write(_checkpoint_path(hermes_home), state)


def _persist_record(
    record: Dict[str, Any],
    hermes_home: str | Path | None = None,
) -> None:
    delegation_id = str(record.get("delegation_id") or "")
    if not delegation_id:
        raise ValueError("async delegation record missing delegation_id")
    with _persistent_lock:
        state = _read_persistent_state_unlocked(hermes_home)
        records = state.setdefault("records", {})
        existing = records.get(delegation_id, {}) if isinstance(records.get(delegation_id), dict) else {}
        existing.update(_persistent_record(record))
        records[delegation_id] = existing
        _write_persistent_state_unlocked(state, hermes_home)


def _update_persistent_record(
    delegation_id: str,
    *,
    hermes_home: str | Path | None = None,
    **fields: Any,
) -> bool:
    if not delegation_id:
        return False
    with _persistent_lock:
        state = _read_persistent_state_unlocked(hermes_home)
        records = state.setdefault("records", {})
        rec = records.get(delegation_id)
        if not isinstance(rec, dict):
            return False
        rec.update(_json_safe(fields))
        rec["updated_at"] = time.time()
        records[delegation_id] = rec
        _write_persistent_state_unlocked(state, hermes_home)
        return True


def _remove_persistent_record(
    delegation_id: str,
    hermes_home: str | Path | None = None,
) -> None:
    if not delegation_id:
        return
    with _persistent_lock:
        state = _read_persistent_state_unlocked(hermes_home)
        records = state.setdefault("records", {})
        records.pop(delegation_id, None)
        _write_persistent_state_unlocked(state, hermes_home)


def _completion_queue_has_async_event(completion_queue: Any, delegation_id: str) -> bool:
    """Return whether an undelivered async event is already queued.

    ``queue.Queue`` intentionally has no public peek API, but Hermes owns this
    in-process queue and only needs a best-effort duplicate guard. If peeking is
    unavailable, fail open (return False) so durable recovery favours re-delivery
    over silent loss.
    """
    if not delegation_id:
        return False
    try:
        mutex = getattr(completion_queue, "mutex", None)
        backing_queue = getattr(completion_queue, "queue", None)
        if mutex is None or backing_queue is None:
            return False
        with mutex:
            return any(
                isinstance(evt, dict)
                and evt.get("type") == "async_delegation"
                and str(evt.get("delegation_id") or "") == delegation_id
                for evt in backing_queue
            )
    except Exception as exc:
        logger.debug("Could not inspect async delegation completion queue: %s", exc)
        return False


def is_delivered(
    delegation_id: str,
    *,
    hermes_home: str | Path | None = None,
) -> bool:
    """Return whether a durable async delegation result was delivered."""
    if not delegation_id:
        return False
    with _records_lock:
        rec = _records.get(delegation_id)
        if isinstance(rec, dict) and rec.get("delivered"):
            return True
    with _persistent_lock:
        state = _read_persistent_state_unlocked(hermes_home)
        records = state.get("records", {})
        if not isinstance(records, dict):
            return False
        rec = records.get(delegation_id)
        return bool(isinstance(rec, dict) and rec.get("delivered"))


def _write_recovery_packet(
    record: Dict[str, Any],
    event: Dict[str, Any],
    *,
    hermes_home: str | Path | None = None,
) -> Optional[str]:
    delegation_id = str(record.get("delegation_id") or event.get("delegation_id") or "unknown")
    try:
        path = _recovery_dir(hermes_home) / f"{delegation_id}.json"
        atomic_json_write(
            path,
            {
                "version": _PERSISTENCE_VERSION,
                "created_at": time.time(),
                "record": _json_safe(record),
                "event": _json_safe(event),
            },
        )
        return str(path)
    except Exception as exc:
        logger.warning("Failed to write async delegation recovery packet for %s: %s", delegation_id, exc)
        return None


def _build_lost_event(
    record: Dict[str, Any],
    *,
    origin: str,
    reason: Optional[str] = None,
    hermes_home: str | Path | None = None,
) -> Dict[str, Any]:
    now = time.time()
    delegation_id = record.get("delegation_id") or "unknown"
    goals = record.get("goals") or []
    if not goals and record.get("goal"):
        goals = [record.get("goal")]
    child_session_ids = [str(x) for x in (record.get("child_session_ids") or []) if x]
    message = (
        reason
        or "The async delegation checkpoint was still marked running, but this "
        "process has no live worker for it. The parent CLI/gateway likely "
        "crashed or restarted; daemon subagent threads cannot be adopted after "
        "process exit, so Hermes is surfacing a recovered lost-delegation notice "
        "instead of silently dropping the result."
    )
    if child_session_ids:
        message += " Child session transcript(s): " + ", ".join(child_session_ids) + "."
    event: Dict[str, Any] = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": record.get("session_key", ""),
        "goal": record.get("goal", ""),
        "goals": goals,
        "context": record.get("context"),
        "toolsets": record.get("toolsets"),
        "role": record.get("role"),
        "model": record.get("model"),
        "status": "lost",
        "is_batch": bool(record.get("is_batch") or len(goals) != 1),
        "results": [],
        "error": message,
        "dispatched_at": record.get("dispatched_at"),
        "completed_at": now,
        "duration_seconds": round(now - (record.get("dispatched_at") or now), 2),
        "total_duration_seconds": round(now - (record.get("dispatched_at") or now), 2),
        "parent_session_id": record.get("parent_session_id"),
        "parent_turn_id": record.get("parent_turn_id"),
        "child_session_ids": child_session_ids,
        "recovered": True,
        "recovery_origin": origin,
    }
    packet_path = _write_recovery_packet(record, event, hermes_home=hermes_home)
    if packet_path:
        event["recovery_packet_path"] = packet_path
        event["error"] = message + f" Recovery packet: {packet_path}"
    return event


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
        return sum(1 for r in _records.values() if r.get("status") == "running")


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
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    parent_turn_id: Optional[str] = None,
    child_session_ids: Optional[List[str]] = None,
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
        "parent_session_id": parent_session_id,
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "parent_turn_id": parent_turn_id,
        "child_session_ids": list(child_session_ids or []),
        "delivered": False,
        **_owner_metadata(session_key),
    }
    # Capacity check and record insert under ONE lock hold — checking
    # active_count() separately would let two concurrent dispatches (e.g.
    # from different gateway sessions) both pass the check and exceed the cap.
    with _records_lock:
        running = sum(
            1 for r in _records.values() if r.get("status") == "running"
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
        try:
            _persist_record(record)
        except Exception as exc:
            logger.error(
                "Refusing async delegation %s: durable checkpoint write failed: %s",
                delegation_id,
                exc,
            )
            return {
                "status": "rejected",
                "error": (
                    "Failed to persist the async delegation recovery record. "
                    "Hermes will not return a background handle that could be "
                    "silently lost across a crash/restart; run synchronously instead. "
                    f"Details: {exc}"
                ),
            }
        _records[delegation_id] = record

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
        try:
            _remove_persistent_record(delegation_id)
        except Exception:
            logger.debug("Failed to remove rejected async delegation checkpoint", exc_info=True)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation: {exc}",
        }

    logger.info(
        "Dispatched async delegation %s (session_key=%s): %s",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        record["status"] = status
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None  # drop the closure; child is done
        # Snapshot fields needed for the event while holding the lock.
        event_record = dict(record)
        _prune_completed_locked()

    _push_completion_event(event_record, result, status)


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
        "parent_session_id": record.get("parent_session_id"),
        "parent_turn_id": record.get("parent_turn_id"),
        "child_session_ids": record.get("child_session_ids") or [],
    }
    try:
        _update_persistent_record(
            str(record.get("delegation_id") or ""),
            status=status,
            completed_at=completed_at,
            delivered=False,
            result_event=evt,
        )
    except Exception as exc:
        logger.error(
            "Async delegation %s finished but durable result update failed: %s",
            record.get("delegation_id"),
            exc,
        )
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
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    parent_turn_id: Optional[str] = None,
    child_session_ids: Optional[List[str]] = None,
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
    delegation_id = _new_delegation_id()
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
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "is_batch": True,
        "parent_turn_id": parent_turn_id,
        "child_session_ids": list(child_session_ids or []),
        "delivered": False,
        **_owner_metadata(session_key),
    }
    with _records_lock:
        running = sum(
            1 for r in _records.values() if r.get("status") == "running"
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
        try:
            _persist_record(record)
        except Exception as exc:
            logger.error(
                "Refusing async delegation batch %s: durable checkpoint write failed: %s",
                delegation_id,
                exc,
            )
            return {
                "status": "rejected",
                "error": (
                    "Failed to persist the async delegation recovery record. "
                    "Hermes will not return a background handle that could be "
                    "silently lost across a crash/restart; run synchronously instead. "
                    f"Details: {exc}"
                ),
            }
        _records[delegation_id] = record

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
        try:
            _remove_persistent_record(delegation_id)
        except Exception:
            logger.debug("Failed to remove rejected async delegation batch checkpoint", exc_info=True)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation batch: {exc}",
        }

    logger.info(
        "Dispatched async delegation batch %s (%d task(s), session_key=%s)",
        delegation_id, n, session_key or "<cli>",
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize_batch(
    delegation_id: str, combined: Dict[str, Any], status: str
) -> None:
    """Mark a batch record complete and push ONE combined completion event."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        record["status"] = status
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None
        event_record = dict(record)
        _prune_completed_locked()

    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s finished but process_registry import "
            "failed; result lost: %s",
            delegation_id, exc,
        )
        return

    dispatched_at = event_record.get("dispatched_at") or time.time()
    completed_at = event_record.get("completed_at") or time.time()
    evt = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": event_record.get("session_key", ""),
        "origin_ui_session_id": event_record.get("origin_ui_session_id", ""),
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
        "error": combined.get("error"),
        "total_duration_seconds": combined.get("total_duration_seconds"),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "parent_session_id": event_record.get("parent_session_id"),
        "parent_turn_id": event_record.get("parent_turn_id"),
        "child_session_ids": event_record.get("child_session_ids") or [],
    }
    try:
        _update_persistent_record(
            delegation_id,
            status=status,
            completed_at=completed_at,
            delivered=False,
            result_event=evt,
        )
    except Exception as exc:
        logger.error(
            "Async delegation batch %s finished but durable result update failed: %s",
            delegation_id,
            exc,
        )
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s: failed to enqueue completion event; "
            "result lost: %s",
            delegation_id, exc,
        )


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


def recover_pending_delegations(
    origin: str = "startup",
    *,
    hermes_home: str | Path | None = None,
) -> int:
    """Requeue undelivered async delegation outcomes from the durable checkpoint.

    Completed records with a stored result event are re-enqueued while they are
    absent from the in-process notification queue until ``mark_delivered()``
    confirms a parent turn consumed them. A delivery attempt that consumes/drops
    the queue item without marking it delivered therefore remains retryable in
    the same process.
    Records still marked running become explicit ``status='lost'`` recovery
    notices only when their owner process is proven gone/reused or they are
    stale local/legacy records. A sibling process that shares HERMES_HOME but
    still owns the delegation must not have its running checkpoint stolen by
    this process's empty in-memory ``_records`` map.
    """
    with _records_lock:
        live_running = {
            rid for rid, rec in _records.items() if rec.get("status") == "running"
        }

    to_enqueue: List[Dict[str, Any]] = []
    changed = False
    with _persistent_lock:
        state = _read_persistent_state_unlocked(hermes_home)
        records = state.setdefault("records", {})
        if not isinstance(records, dict):
            return 0

        for delegation_id, rec in list(records.items()):
            if not isinstance(rec, dict):
                continue
            if rec.get("delivered"):
                continue

            status = str(rec.get("status") or "running")
            if status == "running" and delegation_id in live_running:
                continue

            event: Optional[Dict[str, Any]] = None
            if status == "running":
                if not _running_owner_is_recoverably_gone(
                    str(delegation_id), rec, live_running
                ):
                    continue
                event = _build_lost_event(rec, origin=origin, hermes_home=hermes_home)
                rec.update(
                    {
                        "status": "lost",
                        "completed_at": event.get("completed_at"),
                        "result_event": event,
                        "recovery_packet_path": event.get("recovery_packet_path"),
                        "delivered": False,
                        "updated_at": time.time(),
                    }
                )
                changed = True
            elif status in _TERMINAL_STATUSES:
                stored_event = rec.get("result_event")
                if isinstance(stored_event, dict):
                    event = stored_event
                else:
                    event = _build_lost_event(
                        rec,
                        origin=origin,
                        hermes_home=hermes_home,
                        reason=(
                            "The async delegation checkpoint reached a terminal "
                            f"status ({status}) but has no stored result event. "
                            "Hermes is surfacing this recovery notice instead of "
                            "silently dropping the delegation."
                        ),
                    )
                    rec.update(
                        {
                            "status": "lost",
                            "completed_at": event.get("completed_at"),
                            "result_event": event,
                            "recovery_packet_path": event.get("recovery_packet_path"),
                            "delivered": False,
                            "updated_at": time.time(),
                        }
                    )
                    changed = True

            if event is not None:
                to_enqueue.append(event)

        if changed:
            _write_persistent_state_unlocked(state, hermes_home)

    if not to_enqueue:
        return 0

    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Could not enqueue recovered async delegation event(s); "
            "process_registry import failed: %s",
            exc,
        )
        return 0

    enqueued = 0
    for event in to_enqueue:
        delegation_id = str(event.get("delegation_id") or "")
        if _completion_queue_has_async_event(process_registry.completion_queue, delegation_id):
            continue
        process_registry.completion_queue.put(event)
        enqueued += 1
    if not enqueued:
        return 0
    logger.warning(
        "Recovered %d undelivered async delegation event(s) from checkpoint (%s)",
        enqueued,
        origin,
    )
    return enqueued


def mark_delivered(
    delegation_id: str,
    *,
    hermes_home: str | Path | None = None,
) -> bool:
    """Mark a durable async delegation event as delivered to its parent turn."""
    if not delegation_id:
        return False
    now = time.time()
    with _records_lock:
        rec = _records.get(delegation_id)
        if rec is not None:
            rec["delivered"] = True
            rec["delivered_at"] = now
    try:
        updated = _update_persistent_record(
            delegation_id,
            hermes_home=hermes_home,
            delivered=True,
            delivered_at=now,
        )
    except Exception as exc:
        logger.debug("Could not mark async delegation %s delivered: %s", delegation_id, exc)
        updated = False
    return updated or rec is not None


def extract_delegation_ids_from_text(text: str) -> List[str]:
    """Return async delegation ids embedded in formatted notification text."""
    if not text:
        return []
    import re

    seen: set[str] = set()
    out: List[str] = []
    for match in re.finditer(r"\[ASYNC DELEGATION[^\]]+—\s*(deleg_[0-9a-fA-F]+)\]", text):
        deleg_id = match.group(1)
        if deleg_id not in seen:
            seen.add(deleg_id)
            out.append(deleg_id)
    return out


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


def _reset_for_tests(
    *,
    clear_persistent: bool = True,
    hermes_home: str | Path | None = None,
) -> None:
    """Test-only: clear all state and tear down the executor."""
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None
        _executor_max_workers = 0
    with _records_lock:
        _records.clear()
    if clear_persistent:
        try:
            _checkpoint_path(hermes_home).unlink(missing_ok=True)
        except Exception:
            pass
