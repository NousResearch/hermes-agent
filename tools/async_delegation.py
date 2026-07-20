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

import copy
import logging
import os
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit.

    Stdlib ``ThreadPoolExecutor`` workers are non-daemon. Background
    delegation is explicitly best-effort detached work, so a long child should
    be interruptible by ``/stop``/shutdown but must not keep a CLI process alive
    after the user exits.
    """

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            t.start()
            self._threads.add(t)


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
_DEFAULT_STALLED_AFTER_SECONDS = 180.0
# How many completed records to retain for status queries before pruning.
_MAX_RETAINED_COMPLETED = 50


def _is_active(record: Dict[str, Any]) -> bool:
    """Whether a record still owns runtime capacity and workspace locks."""
    return record.get("completed_at") is None


def _normalize_workspace_path(path: Optional[str]) -> Optional[str]:
    if not path or not str(path).strip():
        return None
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def _workspace_conflict_locked(
    workspace_path: Optional[str], workspace_mode: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Find an active same-workspace conflict; caller holds _records_lock."""
    if not workspace_path:
        return None
    requested_mode = "read" if workspace_mode == "read" else "write"
    for record in _records.values():
        held_path = record.get("workspace_path")
        if not _is_active(record) or not held_path:
            continue
        try:
            common_path = os.path.commonpath([workspace_path, held_path])
        except ValueError:
            continue
        paths_overlap = common_path in {workspace_path, held_path}
        if not paths_overlap:
            continue
        held_mode = "read" if record.get("workspace_mode") == "read" else "write"
        if requested_mode == "write" or held_mode == "write":
            return record
    return None


def _workspace_rejection(holder: Dict[str, Any], workspace_path: str) -> Dict[str, Any]:
    return {
        "status": "rejected",
        "reason_code": "workspace_locked",
        "holder_delegation_id": holder.get("delegation_id"),
        "workspace_path": workspace_path,
        "error": (
            f"Workspace is already in use by active delegation "
            f"{holder.get('delegation_id')}: {workspace_path}. Wait for it to "
            "finish or cancel it before starting a conflicting task."
        ),
    }


def _snapshot_record(
    record: Dict[str, Any],
    *,
    now: Optional[float] = None,
    stalled_after_seconds: float = _DEFAULT_STALLED_AFTER_SECONDS,
) -> Dict[str, Any]:
    """Return a serialisable record with live activity and derived status."""
    current_time = time.time() if now is None else float(now)
    snapshot = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in {"interrupt_fn", "activity_fn", "session_key"}
    }
    activity_fn = record.get("activity_fn")
    if _is_active(record) and callable(activity_fn):
        try:
            activity = activity_fn() or {}
            if not isinstance(activity, dict):
                activity = {}
            activity_at = activity.get("last_activity_ts")
            if isinstance(activity_at, (int, float)):
                snapshot["last_activity_at"] = float(activity_at)
            snapshot["last_activity_desc"] = activity.get("last_activity_desc")
            snapshot["current_tool"] = activity.get("current_tool")
            snapshot["api_call_count"] = activity.get("api_call_count", 0)
        except Exception as exc:
            logger.debug(
                "Async delegation %s activity snapshot failed: %s",
                record.get("delegation_id"), exc,
            )

    last_activity_at = snapshot.get("last_activity_at") or snapshot.get("dispatched_at")
    if isinstance(last_activity_at, (int, float)):
        seconds_since = max(0.0, current_time - float(last_activity_at))
        snapshot["seconds_since_activity"] = round(seconds_since, 1)
    else:
        seconds_since = 0.0

    snapshot["active"] = _is_active(record)
    if _is_active(record):
        if record.get("cancel_requested"):
            snapshot["status"] = "cancelling"
        elif seconds_since >= max(0.0, float(stalled_after_seconds)):
            snapshot["status"] = "stalled"
        else:
            snapshot["status"] = "running"
    return snapshot


def _build_delegation_record(
    *,
    delegation_id: str,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    dispatched_at: float,
    interrupt_fn: Optional[Callable[[], None]],
    activity_fn: Optional[Callable[[], Dict[str, Any]]],
    workspace_path: Optional[str],
    workspace_mode: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_workspace = _normalize_workspace_path(workspace_path)
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": goal,
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
        "activity_fn": activity_fn,
        "last_activity_at": dispatched_at,
        "cancel_requested": False,
        "workspace_path": normalized_workspace,
        "workspace_mode": (
            "read" if workspace_mode == "read" else "write"
        ) if workspace_mode is not None else None,
    }
    if extra:
        record.update(extra)
    return record


def _admit_record(
    record: Dict[str, Any], max_async_children: int
) -> Optional[Dict[str, Any]]:
    """Atomically enforce workspace/capacity limits and insert one record."""
    with _records_lock:
        workspace_path = record.get("workspace_path")
        if record.get("workspace_mode") == "write" and (
            not workspace_path or not os.path.isdir(str(workspace_path))
        ):
            return {
                "status": "rejected",
                "reason_code": "workspace_unavailable",
                "error": "Write delegation requires an authoritative existing workspace.",
            }
        holder = _workspace_conflict_locked(
            record.get("workspace_path"), record.get("workspace_mode")
        )
        if holder is not None:
            return _workspace_rejection(
                holder, str(record.get("workspace_path") or "")
            )
        running = sum(1 for candidate in _records.values() if _is_active(candidate))
        if running >= max_async_children:
            return {
                "status": "rejected",
                "reason_code": "capacity_reached",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    "running). Wait for one to finish or raise "
                    "delegation.max_async_children in config.yaml."
                ),
            }
        _records[str(record["delegation_id"])] = record
    return None


def _complete_record_locked(record: Dict[str, Any], status: str) -> str:
    """Apply shared terminal-state transitions while _records_lock is held."""
    record["status"] = status
    record["completed_at"] = time.time()
    record["interrupt_fn"] = None
    record["activity_fn"] = None
    return status


def _prewarm_executor(executor: ThreadPoolExecutor, max_workers: int) -> None:
    """Start every worker before any business WorkItem can be submitted."""
    release = threading.Event()
    ready_condition = threading.Condition()
    ready_count = 0

    def _warm_worker() -> None:
        nonlocal ready_count
        with ready_condition:
            ready_count += 1
            ready_condition.notify_all()
        release.wait()

    futures = []
    try:
        for _ in range(max_workers):
            futures.append(executor.submit(_warm_worker))
        deadline = time.monotonic() + 5.0
        with ready_condition:
            while ready_count < max_workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Timed out while prewarming async executor")
                ready_condition.wait(timeout=remaining)
    finally:
        release.set()

    for future in futures:
        future.result(timeout=5)


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    """Lazily create (or grow) a fully prewarmed shared daemon executor.

    Every worker is started before the pool is published. This avoids the
    stdlib submit edge case where a WorkItem is queued and Thread.start then
    raises: business submit calls never need to start another worker.
    """
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            candidate = _DaemonThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="async-delegate",
            )
            try:
                _prewarm_executor(candidate, max_workers)
            except Exception:
                candidate.shutdown(wait=True, cancel_futures=True)
                raise
            _executor = candidate
            _executor_max_workers = max_workers
        return _executor


def active_count() -> int:
    """Number of async delegations currently running."""
    with _records_lock:
        return sum(1 for r in _records.values() if _is_active(r))


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


def _normalize_single_result(raw: Any) -> tuple[Dict[str, Any], str]:
    """Validate and normalize a single child terminal result."""
    if not isinstance(raw, dict) or not raw:
        return {
            "status": "error",
            "summary": None,
            "error": "Invalid empty or non-object delegation result.",
        }, "error"

    result = dict(raw)
    raw_status = str(result.get("status") or "").lower()
    status_map = {
        "completed": "completed",
        "success": "completed",
        "interrupted": "interrupted",
        "cancelled": "interrupted",
        "canceled": "interrupted",
        "error": "error",
        "failed": "error",
        "failure": "error",
        "timeout": "timeout",
    }
    status = status_map.get(raw_status)
    if status is None:
        return {
            "status": "error",
            "summary": None,
            "error": "Invalid delegation result status.",
        }, "error"
    result["status"] = status
    return result, status


def _normalize_batch_result(
    raw: Any, expected_count: int
) -> tuple[Dict[str, Any], str]:
    """Validate batch shape/completeness and derive a truthful terminal status."""
    if not isinstance(raw, dict):
        return {
            "results": [],
            "error": "Invalid non-object batch result.",
        }, "error"

    combined = dict(raw)
    raw_results = combined.get("results")
    safe_results = (
        [dict(item) for item in raw_results if isinstance(item, dict)]
        if isinstance(raw_results, list)
        else []
    )
    combined["results"] = safe_results
    valid_statuses = {
        "completed", "success", "interrupted", "cancelled", "canceled",
        "error", "failed", "failure", "timeout",
    }
    indices = [item.get("task_index") for item in safe_results]
    statuses = [str(item.get("status") or "").lower() for item in safe_results]
    complete_shape = (
        isinstance(raw_results, list)
        and len(raw_results) == expected_count
        and len(safe_results) == expected_count
        and set(indices) == set(range(expected_count))
        and len(indices) == len(set(indices))
        and all(status in valid_statuses for status in statuses)
    )
    if not complete_shape:
        combined["error"] = "Invalid, incomplete, or duplicate batch results."
        return combined, "error"
    if any(status in {"error", "failed", "failure", "timeout"} for status in statuses):
        return combined, "error"
    if any(status in {"interrupted", "cancelled", "canceled"} for status in statuses):
        return combined, "interrupted"
    return combined, "completed"


def _rollback_admitted_record(
    record: Dict[str, Any], *, label: str, exc: Exception
) -> Dict[str, Any]:
    """Rollback one admitted-but-unscheduled record by object identity."""
    delegation_id = str(record.get("delegation_id") or "")
    with _records_lock:
        current = _records.get(delegation_id)
        if current is record and _is_active(record):
            _records.pop(delegation_id, None)
    return {
        "status": "rejected",
        "reason_code": "schedule_failed",
        "error": f"Failed to schedule {label}: {type(exc).__name__}",
    }


def dispatch_async_delegation(
    *,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    runner: Callable[[], Dict[str, Any]],
    interrupt_fn: Optional[Callable[[], None]] = None,
    activity_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    workspace_path: Optional[str] = None,
    workspace_mode: Optional[str] = None,
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
    record = _build_delegation_record(
        delegation_id=delegation_id,
        goal=goal,
        context=context,
        toolsets=toolsets,
        role=role,
        model=model,
        session_key=session_key,
        dispatched_at=dispatched_at,
        interrupt_fn=interrupt_fn,
        activity_fn=activity_fn,
        workspace_path=workspace_path,
        workspace_mode=workspace_mode,
    )
    rejection = _admit_record(record, max_async_children)
    if rejection is not None:
        return rejection

    try:
        executor = _get_executor(max_async_children)
    except Exception as exc:
        return _rollback_admitted_record(
            record, label="async delegation", exc=exc
        )

    def _worker() -> None:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result, status = _normalize_single_result(runner())
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
        executor.submit(_worker)
    except Exception as exc:  # pragma: no cover — pool submit failure is rare
        return _rollback_admitted_record(
            record, label="async delegation", exc=exc
        )

    logger.info(
        "Dispatched async delegation %s (session_key=%s): %s",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize(delegation_id: str, result: Dict[str, Any], status: str) -> None:
    """Mark a record complete and push the completion event onto the queue."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or not _is_active(record):
            return
        status = _complete_record_locked(record, status)
        # Snapshot fields needed for the event while holding the lock.
        event_record = copy.deepcopy(record)
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
    runner: Callable[[], Dict[str, Any]],
    interrupt_fn: Optional[Callable[[], None]] = None,
    activity_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    workspace_path: Optional[str] = None,
    workspace_mode: Optional[str] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
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
    record = _build_delegation_record(
        delegation_id=delegation_id,
        goal=combined_goal,
        context=context,
        toolsets=toolsets,
        role=role,
        model=model,
        session_key=session_key,
        dispatched_at=dispatched_at,
        interrupt_fn=interrupt_fn,
        activity_fn=activity_fn,
        workspace_path=workspace_path,
        workspace_mode=workspace_mode,
        extra={"goals": list(goals), "is_batch": True},
    )
    rejection = _admit_record(record, max_async_children)
    if rejection is not None:
        return rejection

    try:
        executor = _get_executor(max_async_children)
    except Exception as exc:
        return _rollback_admitted_record(
            record, label="async delegation batch", exc=exc
        )

    def _worker() -> None:
        combined: Dict[str, Any] = {}
        status = "error"
        try:
            combined, status = _normalize_batch_result(runner(), n)
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
        executor.submit(_worker)
    except Exception as exc:  # pragma: no cover
        return _rollback_admitted_record(
            record, label="async delegation batch", exc=exc
        )

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
        if record is None or not _is_active(record):
            return
        status = _complete_record_locked(record, status)
        event_record = copy.deepcopy(record)
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
        "results": copy.deepcopy(combined.get("results") or []),
        "error": combined.get("error"),
        "total_duration_seconds": combined.get("total_duration_seconds"),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
    }
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation batch %s: failed to enqueue completion event; "
            "result lost: %s",
            delegation_id, exc,
        )


def list_async_delegations(
    *,
    owner_session_key: Optional[str] = None,
    now: Optional[float] = None,
    stalled_after_seconds: float = _DEFAULT_STALLED_AFTER_SECONDS,
) -> List[Dict[str, Any]]:
    """Snapshot delegations, optionally restricted to one owning session."""
    with _records_lock:
        records = [
            dict(record)
            for record in _records.values()
            if owner_session_key is None
            or record.get("session_key") == owner_session_key
        ]
    return [
        _snapshot_record(
            record, now=now, stalled_after_seconds=stalled_after_seconds
        )
        for record in records
    ]


def get_async_delegation(
    delegation_id: str,
    *,
    owner_session_key: Optional[str] = None,
    now: Optional[float] = None,
    stalled_after_seconds: float = _DEFAULT_STALLED_AFTER_SECONDS,
) -> Optional[Dict[str, Any]]:
    """Return one snapshot, hiding records owned by another session."""
    with _records_lock:
        live_record = _records.get(delegation_id)
        if live_record is None or (
            owner_session_key is not None
            and live_record.get("session_key") != owner_session_key
        ):
            return None
        record = dict(live_record)
    return _snapshot_record(
        record, now=now, stalled_after_seconds=stalled_after_seconds
    )


def interrupt_async_delegation(
    delegation_id: str,
    reason: str = "explicit cancel",
    owner_session_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Idempotently request cancellation of one active async delegation."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or (
            owner_session_key is not None
            and record.get("session_key") != owner_session_key
        ):
            return {
                "delegation_id": delegation_id,
                "status": "not_found",
                "active": False,
            }
        if not _is_active(record):
            completed_record = dict(record)
            fn = None
        else:
            completed_record = None
            if record.get("cancel_requested"):
                return {
                    "delegation_id": delegation_id,
                    "status": "cancelling",
                    "active": True,
                    "already_requested": True,
                }
            fn = record.get("interrupt_fn")
            if not callable(fn):
                return {
                    "delegation_id": delegation_id,
                    "status": "unavailable",
                    "active": True,
                    "error": "This delegation does not expose an interrupt callback.",
                }
            record["cancel_requested"] = True

    if completed_record is not None:
        snapshot = _snapshot_record(completed_record)
        snapshot["active"] = False
        return snapshot

    assert callable(fn)
    callback_error: Optional[Exception] = None
    try:
        fn()
    except Exception as exc:
        callback_error = exc

    with _records_lock:
        current = _records.get(delegation_id)
        if current is None or current is not record:
            current_record = None
            terminal_record = None
        elif not _is_active(current):
            current_record = current
            terminal_record = dict(current)
        else:
            current_record = current
            terminal_record = None
            if callback_error is not None:
                current["cancel_requested"] = False

    if terminal_record is not None:
        snapshot = _snapshot_record(terminal_record)
        snapshot["active"] = False
        return snapshot
    if current_record is None:
        return {
            "delegation_id": delegation_id,
            "status": "not_found",
            "active": False,
        }
    if callback_error is not None:
        logger.debug(
            "interrupt_async_delegation(%s) failed: %s",
            delegation_id,
            callback_error,
        )
        return {
            "delegation_id": delegation_id,
            "status": "error",
            "active": True,
            "error": (
                "Cancellation callback failed "
                f"({type(callback_error).__name__})."
            ),
        }

    logger.info(
        "Cancellation requested for async delegation %s (%s)", delegation_id, reason
    )
    return {
        "delegation_id": delegation_id,
        "status": "cancelling",
        "active": True,
        "already_requested": False,
    }


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every active async delegation to stop. Returns how many."""
    count = 0
    with _records_lock:
        targets = [
            str(record.get("delegation_id"))
            for record in _records.values()
            if _is_active(record)
        ]
    for delegation_id in targets:
        result = interrupt_async_delegation(delegation_id, reason=reason)
        if result.get("status") == "cancelling" and not result.get("already_requested"):
            count += 1
    if count:
        logger.info("Interrupted %d async delegation(s) (%s)", count, reason)
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
