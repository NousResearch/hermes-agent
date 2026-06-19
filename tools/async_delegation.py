#!/usr/bin/env python3
"""
Async Delegation -- Background Subagent Architecture

Daemon-executor registry for dispatching subagents that run in the background
and return a handle immediately — the parent turn is NOT blocked.

When a subagent finishes, its result is pushed as an ``async_delegation`` event
onto the shared ``process_registry.completion_queue`` so that the existing
idle-drain rail in ``gateway/run.py`` and ``cli.py`` picks it up without any
mid-loop splice.

Capacity
--------
Async delegation is capped at ``delegation.max_async_children`` (default 3).
When at capacity, dispatches are queued (FIFO) and promoted automatically as
running slots free up — no manual retry needed.

Result Storage
--------------
Completed delegation results are stored in ``_completed`` (in-memory, keyed by
delegation_id).  Use ``get_result(delegation_id)`` or ``list_completed()`` to
retrieve them.  Results are kept until ``clear_result()`` or
``clear_completed()`` is called.

Cancel
------
``cancel(delegation_id)`` cancels a queued OR running delegation:
- queued → removed from the FIFO waiting queue immediately
- running → marked as cancel-requested; the subagent finishes naturally but
  the result is flagged ``status="cancelled"`` in the completion event

Timeout
-------
Pass ``timeout_seconds`` to ``dispatch()``.  A background watcher checks every
30 s; tasks that exceed their deadline are marked ``status="timed_out"`` and
evicted from ``_running``.

Event shape
-----------
``{"type": "async_delegation", "delegation_id": str, "session_key": str,
  "goal": str, "context": str, "status": str, "result": Any,
  "dispatch_time": float, "completion_time": float}``

The ``session_key`` field is used by the gateway watcher to route the result
back into the originating session.
"""

from __future__ import annotations

import logging
import json
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, NamedTuple, Optional, Set

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------

def _get_max_async_children() -> int:
    """Read delegation.max_async_children from the config."""
    try:
        from hermes_cli.config import CLI_CONFIG
        delegation = CLI_CONFIG.get("delegation", {})
        return int(delegation.get("max_async_children", 3))
    except Exception:
        return 3  # safe default


# ----------------------------------------------------------------------
# Queued item shape — stored in the FIFO waiting queue
# ----------------------------------------------------------------------
class QueuedDispatch(NamedTuple):
    delegation_id: str
    runner_fn: Callable[[], Any]
    task_info: Dict[str, Any]
    completion_queue: Any
    session_key: str
    parent_agent: Any
    enqueue_time: float
    timeout_seconds: Optional[float]


# ----------------------------------------------------------------------
# State stores
# ----------------------------------------------------------------------

# In-memory registry of running async delegations.
# Key = delegation_id, Value = {"thread", "stop_event", "cancel_event", ...}
_running: Dict[str, Dict[str, Any]] = {}
_running_lock = threading.Lock()

# FIFO waiting queue for at-capacity dispatches.
_waiting: Deque[QueuedDispatch] = deque()
# Completed results: key = delegation_id, value = completion event dict.
_completed: Dict[str, Dict[str, Any]] = {}
_completed_lock = threading.Lock()

# Condition variable notified whenever ANY delegation reaches a terminal state
# (completed, cancelled, timed_out, error).  Used by wait() to block efficiently.
_completion_cv = threading.Condition()

# Cancellation requests (delegation_ids that should be aborted ASAP).
# Only used for *queued* task cancellation in _promotion_loop / cancel().
# Running tasks use their per-task cancel_event (threading.Event) which is
# already thread-safe — no need to touch _cancel_requests for them.
_cancel_requests: Set[str] = set()
_cancel_requests_lock = threading.Lock()  # M1 fix: protect _cancel_requests

# Condition variable used to signal the promotion thread when a slot opens.
_promotion_cv = threading.Condition(_running_lock)

# Daemon threads — started on first dispatch, live for process lifetime.
_promotion_thread: Optional[threading.Thread] = None
_timeout_watcher_thread: Optional[threading.Thread] = None
# Startup lock — prevents two concurrent dispatch() callers from racing to
# create the same daemon thread (M2 fix).
_daemon_start_lock = threading.Lock()


# ----------------------------------------------------------------------
# Promotion thread
# ----------------------------------------------------------------------

def _start_promotion_thread() -> None:
    global _promotion_thread
    with _daemon_start_lock:
        if _promotion_thread is None or not _promotion_thread.is_alive():
            _promotion_thread = threading.Thread(
                target=_promotion_loop, daemon=True, name="async-delegation-promotion"
            )
            _promotion_thread.start()


def _start_timeout_watcher() -> None:
    global _timeout_watcher_thread
    with _daemon_start_lock:
        if _timeout_watcher_thread is None or not _timeout_watcher_thread.is_alive():
            _timeout_watcher_thread = threading.Thread(
                target=_timeout_watcher_loop, daemon=True, name="async-delegation-timeout"
            )
            _timeout_watcher_thread.start()


def _promotion_loop() -> None:
    """Daemon loop: waits for a free slot, promotes the next queued item."""
    while True:
        item: Optional[QueuedDispatch] = None
        cancelled_items: List[QueuedDispatch] = []
        with _promotion_cv:
            while len(_running) >= _get_max_async_children() or not _waiting:
                _promotion_cv.wait()
            # Skip cancelled items — collect them; push events *outside* the lock
            # to avoid lock-ordering deadlock (_running_lock → _completed_lock).
            while _waiting:
                candidate = _waiting[0]
                with _cancel_requests_lock:
                    is_cancelled = candidate.delegation_id in _cancel_requests
                    if is_cancelled:
                        _cancel_requests.discard(candidate.delegation_id)
                if is_cancelled:
                    _waiting.popleft()
                    logger.info(
                        "Async delegation %s cancelled while queued", candidate.delegation_id
                    )
                    cancelled_items.append(candidate)
                    continue
                item = _waiting.popleft()
                break

        # Push cancelled events outside the lock (C3 fix).
        for cancelled in cancelled_items:
            _push_cancelled_event(cancelled)

        if item is not None:
            _do_dispatch(item)


def _timeout_watcher_loop() -> None:
    """Daemon loop: evicts running delegations that exceed their timeout."""
    while True:
        time.sleep(30)
        now = time.time()
        with _running_lock:
            timed_out = [
                (did, info)
                for did, info in _running.items()
                if info.get("timeout_seconds") is not None
                and now - info.get("dispatch_time", now) > info["timeout_seconds"]
            ]
        for did, info in timed_out:
            logger.warning(
                "Async delegation %s timed out after %.0fs",
                did, info.get("timeout_seconds"),
            )
            # Signal cancel via cancel_event (per-task, thread-safe).
            # No need to touch _cancel_requests — the runner checks cancel_event.
            cancel_ev = info.get("cancel_event")
            if cancel_ev:
                cancel_ev.set()
            # Push TIMED_OUT event
            evt = {
                "type": "async_delegation",
                "delegation_id": did,
                "session_key": info.get("session_key", ""),
                "status": "timed_out",
                "goal": info.get("goal", ""),
                "context": info.get("context", ""),
                "toolsets": info.get("toolsets"),
                "role": info.get("role"),
                "model": info.get("model"),
                "provider": info.get("provider"),
                "result": {"error": "timeout", "status": "timed_out"},
                "dispatch_time": info.get("dispatch_time", 0),
                "completion_time": now,
                "duration_seconds": round(now - info.get("dispatch_time", now), 2),
            }
            cq = info.get("completion_queue")
            if cq is not None:
                try:
                    cq.put(evt)
                except Exception:
                    pass
            with _completed_lock:
                _completed[did] = evt
            with _completion_cv:
                _completion_cv.notify_all()
            # M3 fix: pop from _running and notify promotion thread in one lock
            # acquisition, so the slot is never briefly "gone from _running but
            # promotion not yet signalled".
            with _promotion_cv:
                _running.pop(did, None)
                _promotion_cv.notify()


# ----------------------------------------------------------------------
# Core dispatch
# ----------------------------------------------------------------------

def _push_cancelled_event(item: QueuedDispatch) -> None:
    """Push a 'cancelled' completion event for a queued item."""
    now = time.time()
    evt = {
        "type": "async_delegation",
        "delegation_id": item.delegation_id,
        "session_key": item.session_key,
        "status": "cancelled",
        "goal": item.task_info.get("goal", ""),
        "context": item.task_info.get("context", ""),
        "toolsets": item.task_info.get("toolsets"),
        "role": item.task_info.get("role"),
        "model": item.task_info.get("model"),
        "provider": item.task_info.get("provider"),
        "result": {"error": "cancelled", "status": "cancelled"},
        "dispatch_time": item.enqueue_time,
        "completion_time": now,
        "duration_seconds": round(now - item.enqueue_time, 2),
    }
    try:
        item.completion_queue.put(evt)
    except Exception:
        pass
    with _completed_lock:
        _completed[item.delegation_id] = evt
    with _completion_cv:
        _completion_cv.notify_all()


def _do_dispatch(item: QueuedDispatch) -> None:
    """Execute a single dispatch. Called from the main thread (immediate path)
    or from the promotion thread (queued path)."""
    delegation_id = item.delegation_id
    runner_fn = item.runner_fn
    task_info = item.task_info
    completion_queue = item.completion_queue
    session_key = item.session_key
    parent_agent = item.parent_agent
    timeout_seconds = item.timeout_seconds
    dispatch_time = time.time()

    cancel_event = threading.Event()

    def _runner() -> None:
        """Worker thread: runs the subagent and pushes result to completion_queue."""
        logger.info("Async delegation %s started", delegation_id)

        # C2 fix: the timeout watcher may have already written a timed_out event
        # and evicted us from _running before this thread even gets scheduled.
        # If so, skip all work — a duplicate completion event would confuse callers.
        with _completed_lock:
            if delegation_id in _completed:
                logger.info(
                    "Async delegation %s already completed by watcher — skipping runner",
                    delegation_id,
                )
                return

        # Heartbeat: periodically touch the parent so the gateway doesn't
        # think the parent agent is idle and kill it.
        _heartbeat_stop = threading.Event()
        _heartbeat_thread = threading.Thread(
            target=_async_heartbeat_loop,
            args=(parent_agent, delegation_id, _heartbeat_stop),
            daemon=True,
        )
        _heartbeat_thread.start()

        try:
            # If already cancelled before we even start, skip the work.
            # Use cancel_event (per-task, thread-safe) rather than _cancel_requests.
            if cancel_event.is_set():
                result = {"error": "cancelled", "status": "cancelled"}
            else:
                result = runner_fn()
                # If cancel was requested while running, mark the result.
                if cancel_event.is_set():
                    if isinstance(result, dict):
                        result["status"] = "cancelled"
                    else:
                        result = {"original_result": result, "status": "cancelled"}
        except Exception as exc:
            logger.exception("Async delegation %s raised: %s", delegation_id, exc)
            result = {"error": str(exc), "status": "error"}
        finally:
            _heartbeat_stop.set()
            _heartbeat_thread.join(timeout=2.0)

        completion_time = time.time()
        status = result.get("status", "completed") if isinstance(result, dict) else "completed"

        # ── Task 2: Write result to Kanban ticket (shadow clone mode) ──────────
        _kanban_ticket_id = task_info.get("kanban_ticket_id")
        if _kanban_ticket_id and task_info.get("shadow_clone"):
            try:
                from hermes_cli import kanban_db as _kanban_db
                _result_meta = {
                    "delegation_id": delegation_id,
                    "session_key": session_key,
                    "status": status,
                    "duration_seconds": round(completion_time - dispatch_time, 2),
                }
                if isinstance(result, dict):
                    _result_meta["insights"] = result.get("insights", [])
                    _result_meta["decision_trail"] = result.get("decision_trail", [])
                    _result_meta["tool_calls_count"] = result.get("tool_calls_count")
                _summary = ""
                if isinstance(result, dict):
                    _summary = result.get("summary") or result.get("error") or ""
                with _kanban_db.connect_closing() as _conn:
                    _kanban_db.complete_task(
                        _conn,
                        _kanban_ticket_id,
                        result=json.dumps(result, ensure_ascii=False, default=str)[:8000],
                        summary=_summary[:2000],
                        metadata=_result_meta,
                    )
                logger.info(
                    "Shadow clone %s result written to Kanban ticket %s",
                    delegation_id, _kanban_ticket_id,
                )
            except Exception as _kbe:
                logger.warning(
                    "Shadow clone: failed to write Kanban result for %s: %s",
                    delegation_id, _kbe,
                )
        # ──────────────────────────────────────────────────────────────────────

        # ── P1: Write final status to state.db shadow_clone_tasks ─────────────
        if task_info.get("shadow_clone"):
            try:
                from hermes_state import SessionDB
                _sdb = SessionDB()
                _sdb.update_shadow_clone_task(
                    delegation_id=delegation_id,
                    status=status,
                    result_json=json.dumps(result, default=str)[:8000] if result is not None else None,
                    completed_at=completion_time,
                )
            except Exception as _sdbe:
                logger.warning(
                    "Shadow clone: state.db update failed for %s: %s",
                    delegation_id, _sdbe,
                )
        # ──────────────────────────────────────────────────────────────────────

        evt = {
            "type": "async_delegation",
            "delegation_id": delegation_id,
            "session_key": session_key,
            "status": status,
            "goal": task_info.get("goal", ""),
            "context": task_info.get("context", ""),
            "toolsets": task_info.get("toolsets"),
            "role": task_info.get("role"),
            "model": task_info.get("model"),
            "provider": task_info.get("provider"),
            "result": result,
            "dispatch_time": dispatch_time,
            "completion_time": completion_time,
            "duration_seconds": round(completion_time - dispatch_time, 2),
            "kanban_ticket_id": _kanban_ticket_id,               # Task 2
            "shadow_clone": task_info.get("shadow_clone", False), # Task 2
            **task_info.get("routing_meta", {}),                  # Task 6
        }

        # Store in completed registry.
        with _completed_lock:
            _completed[delegation_id] = evt
        with _completion_cv:
            _completion_cv.notify_all()

        try:
            completion_queue.put(evt)
            logger.info(
                "Async delegation %s %s in %.1fs, result queued",
                delegation_id,
                status,
                evt["duration_seconds"],
            )
        except Exception:
            logger.exception(
                "Failed to push async delegation %s result to queue", delegation_id
            )
        finally:
            # Always free the running slot so the promotion thread can proceed
            # even if completion_queue.put() raised an exception (m4 fix).
            with _promotion_cv:
                _running.pop(delegation_id, None)
                _promotion_cv.notify()

    thread = threading.Thread(
        target=_runner, daemon=True, name=f"async-delegation-{delegation_id}"
    )
    with _running_lock:
        _running[delegation_id] = {
            "thread": thread,
            "cancel_event": cancel_event,
            "dispatch_time": dispatch_time,
            "goal": task_info.get("goal", "")[:80],
            "context": task_info.get("context", ""),
            "toolsets": task_info.get("toolsets"),
            "role": task_info.get("role"),
            "model": task_info.get("model"),
            "provider": task_info.get("provider"),
            "session_key": session_key,
            "completion_queue": completion_queue,
            "timeout_seconds": timeout_seconds,
        }
    thread.start()


def dispatch(
    runner_fn: Callable[..., Any],
    task_info: Dict[str, Any],
    completion_queue,  # queue.Queue — shared with process_registry
    session_key: str,
    parent_agent=None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Dispatch a subagent to run in the background.

    Returns immediately with ``{"status": "dispatched", "delegation_id": str}``.
    When the subagent finishes, an ``async_delegation`` event is pushed onto
    ``completion_queue``.

    When at capacity, the dispatch is queued (FIFO) and the return value has
    ``status="queued"`` — the caller does NOT need to retry; the promotion
    thread handles it automatically when a slot frees up.

    Parameters
    ----------
    runner_fn
        Callable that takes no arguments and returns the subagent result dict.
        Typically a lambda that captures the pre-built child agent.
    task_info
        Dict with ``goal``, ``context``, ``toolsets``, ``role``, ``model``,
        ``provider`` — stored in the completion event so the re-injected
        message carries full provenance.
    completion_queue
        The shared ``process_registry.completion_queue`` instance.
    session_key
        Opaque routing key for the gateway watcher to know which session
        to inject the result into. Typically ``f"{platform}:{chat_id}"``.
    parent_agent
        The parent AIAgent instance (for heartbeat / stale detection).
    timeout_seconds
        Optional wall-clock deadline.  If the subagent has not finished within
        this many seconds, it is marked ``status="timed_out"`` and evicted.

    Returns
    -------
    ``{"status": "dispatched" | "queued", "delegation_id": str, "mode": "background"}``
    """
    max_children = _get_max_async_children()
    delegation_id = f"deleg_{uuid.uuid4().hex[:8]}"
    task_info_copy = dict(task_info)  # shallow copy

    # ── P1: Persist dispatch record to state.db before touching any thread ────
    # Written here (outside the capacity lock) so a crash between dispatch and
    # thread-start still leaves a 'running' row that the next startup can find.
    if task_info_copy.get("shadow_clone"):
        try:
            from hermes_state import SessionDB
            _sdb_dispatch = SessionDB()
            _sdb_dispatch.insert_shadow_clone_task(
                delegation_id=delegation_id,
                session_key=session_key,
                goal=task_info_copy.get("goal", ""),
                kanban_ticket_id=task_info_copy.get("kanban_ticket_id"),
                routing_meta=task_info_copy.get("routing_meta"),
                dispatched_at=time.time(),
            )
        except Exception as _p1_e:
            logger.warning(
                "Shadow clone: state.db insert failed for %s: %s", delegation_id, _p1_e
            )
    # ─────────────────────────────────────────────────────────────────────────

    # C1 fix: capacity check + enqueue/dispatch decision in a single atomic
    # section so no other thread can slip in between the read and the write.
    queue_depth = 0
    with _running_lock:
        active = len(_running)
        at_capacity = active >= max_children
        if at_capacity:
            item = QueuedDispatch(
                delegation_id=delegation_id,
                runner_fn=runner_fn,
                task_info=task_info_copy,
                completion_queue=completion_queue,
                session_key=session_key,
                parent_agent=parent_agent,
                enqueue_time=time.time(),
                timeout_seconds=timeout_seconds,
            )
            _waiting.append(item)
            queue_depth = len(_waiting)

    if at_capacity:
        logger.info(
            "Async delegation %s queued (at capacity %d/%d). Queue depth: %d",
            delegation_id, active, max_children, queue_depth,
        )
        _start_promotion_thread()
        if timeout_seconds is not None:
            _start_timeout_watcher()
        return {
            "status": "queued",
            "delegation_id": delegation_id,
            "mode": "background",
            "queue_depth": queue_depth,
        }

    _start_promotion_thread()
    if timeout_seconds is not None:
        _start_timeout_watcher()

    item = QueuedDispatch(
        delegation_id=delegation_id,
        runner_fn=runner_fn,
        task_info=task_info_copy,
        completion_queue=completion_queue,
        session_key=session_key,
        parent_agent=parent_agent,
        enqueue_time=time.time(),
        timeout_seconds=timeout_seconds,
    )
    _do_dispatch(item)
    return {"status": "dispatched", "delegation_id": delegation_id, "mode": "background"}


# ----------------------------------------------------------------------
# Heartbeat
# ----------------------------------------------------------------------

def _async_heartbeat_loop(
    parent_agent, delegation_id: str, stop_event: threading.Event
) -> None:
    """Touch the parent agent's activity flag while the async delegation runs."""
    interval = 30.0  # match _HEARTBEAT_INTERVAL in delegate_tool.py
    while not stop_event.wait(interval):
        if parent_agent is None:
            continue
        touch = getattr(parent_agent, "_touch_activity", None)
        if not touch:
            continue
        try:
            touch(f"async_delegation: {delegation_id} running")
        except Exception:
            pass


# ----------------------------------------------------------------------
# Cancel API
# ----------------------------------------------------------------------

def cancel(delegation_id: str) -> Dict[str, Any]:
    """
    Cancel a queued or running delegation.

    - queued  → removed immediately; a ``status="cancelled"`` event is pushed
    - running → ``cancel_event`` is set; the subagent finishes naturally but
                the result is flagged ``status="cancelled"``

    Returns ``{"cancelled": True, "state": "queued"|"running"|"not_found"}``.
    """
    # Check queued first — remove from _waiting under _running_lock so the
    # promotion thread cannot race us.
    with _running_lock:
        for i, item in enumerate(_waiting):
            if item.delegation_id == delegation_id:
                del _waiting[i]
                with _cancel_requests_lock:
                    _cancel_requests.discard(delegation_id)
                break
        else:
            item = None  # type: ignore[assignment]

    if item is not None:
        _push_cancelled_event(item)
        logger.info("Async delegation %s cancelled from queue", delegation_id)
        return {"cancelled": True, "state": "queued", "delegation_id": delegation_id}

    # Check running — use cancel_event only (no _cancel_requests needed for
    # running tasks; M1 fix keeps _cancel_requests for queued tasks only).
    with _running_lock:
        info = _running.get(delegation_id)

    if info is not None:
        cancel_ev = info.get("cancel_event")
        if cancel_ev:
            cancel_ev.set()
        logger.info("Async delegation %s cancel requested (running)", delegation_id)
        return {"cancelled": True, "state": "running", "delegation_id": delegation_id}

    return {"cancelled": False, "state": "not_found", "delegation_id": delegation_id}


def interrupt_all() -> int:
    """
    Signal all running async delegations to cancel.
    Returns the number of delegations that were signalled.
    """
    with _running_lock:
        count = 0
        for did, info in list(_running.items()):
            cancel_ev = info.get("cancel_event")
            if cancel_ev:
                cancel_ev.set()
                count += 1
    return count


# ----------------------------------------------------------------------
# Result store API
# ----------------------------------------------------------------------

def wait(
    delegation_id: str,
    timeout: Optional[float] = None,
    poll_interval: float = 0.25,
) -> Optional[Dict[str, Any]]:
    """
    Block until the delegation reaches a terminal state, then return its result.

    Uses ``_completion_cv`` so no busy-polling — CPU cost is essentially zero
    while waiting.

    Parameters
    ----------
    delegation_id
        The delegation to wait for.
    timeout
        Wall-clock deadline in seconds.  Returns ``None`` on timeout (the
        delegation is still running/queued — the caller decides what to do).
    poll_interval
        Fallback re-check interval (seconds) in case a notify was missed.
        Defaults to 0.25 s — harmless overhead, prevents missed-notify stalls.

    Returns
    -------
    The completion event dict, or ``None`` if the delegation was not found
    or the timeout expired before it finished.
    """
    deadline = (time.monotonic() + timeout) if timeout is not None else None

    while True:
        # Fast-path: already completed
        with _completed_lock:
            if delegation_id in _completed:
                return _completed[delegation_id]

        # Check still active; if not found anywhere, it never existed
        with _running_lock:
            active = delegation_id in _running or any(
                item.delegation_id == delegation_id for item in _waiting
            )
        if not active:
            # One last check: might have just completed between the two locks
            with _completed_lock:
                return _completed.get(delegation_id)

        # Block on Condition with remaining timeout
        remaining = (deadline - time.monotonic()) if deadline is not None else poll_interval
        if remaining <= 0:
            return None  # timed out

        wait_secs = min(remaining, poll_interval)
        with _completion_cv:
            _completion_cv.wait(timeout=wait_secs)

        # Re-check deadline after wakeup
        if deadline is not None and time.monotonic() >= deadline:
            with _completed_lock:
                return _completed.get(delegation_id)  # may have just landed


def get_result(delegation_id: str) -> Optional[Dict[str, Any]]:
    """Return the stored completion event for a finished delegation, or None."""
    with _completed_lock:
        return _completed.get(delegation_id)


def list_completed(limit: int = 50) -> List[Dict[str, Any]]:
    """Return summary list of completed delegations (newest first)."""
    with _completed_lock:
        items = sorted(
            _completed.values(),
            key=lambda e: e.get("completion_time", 0),
            reverse=True,
        )
    result = []
    for evt in items[:limit]:
        result.append({
            "delegation_id": evt.get("delegation_id", ""),
            "status": evt.get("status", ""),
            "goal": evt.get("goal", "")[:60],
            "duration_seconds": evt.get("duration_seconds"),
            "completion_time": evt.get("completion_time"),
        })
    return result


def clear_result(delegation_id: str) -> bool:
    """Remove a single completed result. Returns True if it existed."""
    with _completed_lock:
        existed = delegation_id in _completed
        _completed.pop(delegation_id, None)
    return existed


def clear_completed() -> int:
    """Clear all completed results. Returns count cleared."""
    with _completed_lock:
        count = len(_completed)
        _completed.clear()
    return count


# ----------------------------------------------------------------------
# Status / admin helpers (used by CLI)
# ----------------------------------------------------------------------

def count_running() -> int:
    """Return the number of currently active async delegations."""
    with _running_lock:
        return len(_running)


def count_queued() -> int:
    """Return the number of queued (waiting) async delegations."""
    with _running_lock:
        return len(_waiting)


def count_completed() -> int:
    """Return the number of stored completed results."""
    with _completed_lock:
        return len(_completed)


def get_running_ids() -> list[str]:
    """Return list of active delegation_ids (for debugging/admin)."""
    with _running_lock:
        return list(_running.keys())


def get_running_details() -> List[Dict[str, Any]]:
    """Return detailed info about running delegations (for CLI/status)."""
    with _running_lock:
        return [
            {
                "delegation_id": did,
                "goal": info.get("goal", ""),
                "dispatch_time": info.get("dispatch_time", 0),
                "timeout_seconds": info.get("timeout_seconds"),
                "is_alive": info.get("thread").is_alive() if info.get("thread") else False,
            }
            for did, info in _running.items()
        ]


def get_queued_details() -> List[Dict[str, Any]]:
    """Return detailed info about queued (waiting) delegations (for CLI/status)."""
    with _running_lock:
        return [
            {
                "delegation_id": item.delegation_id,
                "goal": item.task_info.get("goal", "")[:60],
                "enqueue_time": item.enqueue_time,
                "timeout_seconds": item.timeout_seconds,
            }
            for item in _waiting
        ]


def get_detail(delegation_id: str) -> Optional[Dict[str, Any]]:
    """Return details for a specific delegation_id (running, queued, or completed)."""
    with _running_lock:
        if delegation_id in _running:
            info = _running[delegation_id]
            return {
                "delegation_id": delegation_id,
                "state": "running",
                "goal": info.get("goal", ""),
                "dispatch_time": info.get("dispatch_time", 0),
                "timeout_seconds": info.get("timeout_seconds"),
                "is_alive": info.get("thread").is_alive() if info.get("thread") else False,
            }
        for item in _waiting:
            if item.delegation_id == delegation_id:
                return {
                    "delegation_id": delegation_id,
                    "state": "queued",
                    "goal": item.task_info.get("goal", "")[:60],
                    "enqueue_time": item.enqueue_time,
                    "timeout_seconds": item.timeout_seconds,
                }
    # Check completed store
    with _completed_lock:
        if delegation_id in _completed:
            evt = _completed[delegation_id]
            return {
                "delegation_id": delegation_id,
                "state": evt.get("status", "completed"),
                "goal": evt.get("goal", ""),
                "completion_time": evt.get("completion_time"),
                "duration_seconds": evt.get("duration_seconds"),
                "result_available": True,
            }
    return None
