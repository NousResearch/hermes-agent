"""Replay a "stopped before it finished" notice that a restart lost.

When Hermes itself ends a ``notify_on_complete`` job (a session close, a backend shutdown, or a
crash that took the job down), the completion notice lives in this process's in-memory queue. A
backend restart drops that queue, and the session that promised "I'll report back" is never told.

The durable receipt (``logs/process-results``) survives. When a session's notification poller
starts (the session is open in a client again), :func:`pending_stopped_notices` finds receipts for
jobs Hermes stopped or lost on that session's behalf whose notice is not in the transcript yet, and
returns them as ordinary completion events. The transcript is the ground truth for "the user was
told": a delivered notice is a user-role row naming the job; a tool result (``process list``) is not.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, Iterable

logger = logging.getLogger("tools.process_registry")

# Ends that Hermes chose, not the user. ``agent_close:<reason>`` is a session-level close
# (a bare ``agent_close`` is a delegated child's teardown, whose summary is the deliverable).
_HERMES_STOP_PREFIX = "agent_close:"
_HERMES_STOP_SOURCES = frozenset({"gateway_shutdown", "hermes_restart"})
_REPLAYED: set[str] = set()  # one replay per job per process (a re-opened session must not double it)
_REPLAYED_LOCK = threading.Lock()


def is_hermes_stop(source: str) -> bool:
    source = str(source or "")
    return source.startswith(_HERMES_STOP_PREFIX) or source in _HERMES_STOP_SOURCES


def _told(db, owner: str, proc_id: str, since: float) -> bool:
    """Whether a notice or reply naming ``proc_id`` reached ``owner``'s conversation after ``since``."""
    ids = db.get_compression_chain(owner) or [owner]
    marks = ",".join("?" * len(ids))
    rows = db._read_all(
        f"SELECT 1 FROM messages WHERE session_id IN ({marks}) AND timestamp >= ? "
        "AND role IN ('user', 'assistant') AND content LIKE ? LIMIT 1",
        (*ids, since - 5, f"%{proc_id}%"))
    return bool(rows)


def _event(session, exited_at: float) -> dict[str, Any]:
    from tools.process_registry import _completion_output
    return {
        "type": "completion", "session_id": session.id, "session_key": session.session_key,
        "task_id": session.task_id, "owner_task_id": session.owner_task_id, "command": session.command,
        "exit_code": session.exit_code, "completion_reason": session.completion_reason,
        "termination_source": session.termination_source, **_completion_output(session),
        "started_at": session.started_at, "exited_at": exited_at, "restored": True,
    }


def record_lost(entry: dict) -> None:
    """Write a ``lost`` receipt for a checkpointed notify job whose PID died while Hermes was down
    (crash, SIGKILL, power loss: no kill path ran, so nothing recorded its end). The session's next
    poller start replays it like any other stop. No-op when a receipt already exists."""
    from hermes_constants import get_hermes_home
    from tools.process_registry import ProcessSession
    from tools.process_registry_results import save_completed_result

    sid = str(entry.get("session_id") or "")
    if not (sid and entry.get("notify_on_complete")):
        return
    if (get_hermes_home() / "logs" / "process-results" / f"{sid}.json").exists():
        return
    session = ProcessSession(
        id=sid, command=str(entry.get("command") or "unknown"), cwd=entry.get("cwd"),
        task_id=str(entry.get("task_id") or ""),
        owner_task_id=str(entry.get("owner_task_id") or entry.get("task_id") or ""),
        session_key=str(entry.get("session_key") or ""), parent_session_id=str(entry.get("parent_session_id") or ""),
        started_at=float(entry.get("started_at") or 0), notify_on_complete=True,
        exited=True, exit_code=None, completion_reason="lost", termination_source="hermes_restart")
    logger.warning("Background process %s (notify_on_complete) died while Hermes was down; recording it as lost", sid)
    save_completed_result(session)


def pending_stopped_notices(session_keys: Iterable[str], *, tracked: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Completion events for jobs Hermes stopped/lost for one of ``session_keys`` and never reported.

    Runs in the session's profile scope (receipts and transcript are profile-local). ``tracked`` is
    the registry's in-memory ids: those notices are still queued in this process. Fails closed
    (returns nothing) on any read error: a missed replay is reported by the watchdog, a wrong one
    would put a stale turn in front of the user."""
    keys = {str(k) for k in session_keys if k}
    if not keys:
        return []
    from hermes_state import SessionDB
    from tools.process_registry_results import _result_paths, load_receipt

    tracked = set(tracked)
    events: list[dict[str, Any]] = []
    db = None
    try:
        for path in _result_paths():
            if path.stem in tracked or path.stem in _REPLAYED:
                continue
            loaded = load_receipt(path)
            if loaded is None:
                continue
            session, exited_at = loaded
            owner = str(session.parent_session_id or session.session_key or "")
            if not (owner and session.notify_on_complete and is_hermes_stop(session.termination_source)):
                continue
            db = db or SessionDB(read_only=True)
            # A compressed conversation's open row is its tip; the receipt keeps the spawning id.
            if owner not in keys and db.get_compression_tip(owner) not in keys:
                continue
            if _told(db, owner, session.id, exited_at):
                continue
            with _REPLAYED_LOCK:
                if session.id in _REPLAYED:
                    continue
                _REPLAYED.add(session.id)
            logger.info("Replaying stopped-job notice proc=%s owner=%s source=%s",
                        session.id, owner, session.termination_source)
            events.append(_event(session, exited_at))
    except Exception:
        logger.warning("Could not check for unreported stopped jobs", exc_info=True)
    finally:
        if db is not None:
            with contextlib.suppress(Exception):
                db.close()
    return events


def replay_stopped_notices(session_keys: Iterable[str], registry) -> int:
    """Queue :func:`pending_stopped_notices` for a session whose notification poller is starting.
    Never raises: the poller is the session's only notification path."""
    try:
        with registry._lock:
            tracked = {*registry._running, *registry._finished}
        events = pending_stopped_notices(session_keys, tracked=tracked)
        for evt in events:
            registry.completion_queue.put(evt)
        return len(events)
    except Exception:
        logger.warning("Stopped-job notice replay failed", exc_info=True)
        return 0
