"""Durable-history refresh for resumed ``prompt.submit`` calls.

A resumed Desktop/TUI session keeps an in-memory model-history snapshot. Other
processes can continue writing the same durable session (cron is the important
case), so the display transcript may advance while the model snapshot stays at
its open-time prefix. Before a new turn is claimed, refresh an idle resumed
session from its owning state.db when the durable model projection has grown.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable


_LOG = logging.getLogger(__name__)


def _server_logger(server):
    return getattr(server, "logger", _LOG)


def refresh_resumed_history_before_submit(server, params: dict[str, Any]) -> bool:
    """Adopt newer durable model history for an idle resumed session.

    Returns ``True`` only when ``session['history']`` was replaced. The update
    is deliberately monotonic: a shorter/equal durable projection never
    replaces memory, and a local history mutation that races the DB read wins.
    This keeps the guard safe for queued/live turns while repairing the stale
    open-time snapshot produced by cross-process writers.
    """
    if not isinstance(params, dict):
        return False

    sid = str(params.get("session_id") or "")
    sessions = getattr(server, "_sessions", None)
    session = sessions.get(sid) if isinstance(sessions, dict) else None
    if not isinstance(session, dict) or not session.get("resume_session_id"):
        return False

    # Deferred Desktop resumes hydrate once in a background worker. Never race
    # that assignment: wait for its completion/error signal before taking a
    # fresh durable snapshot. A bounded wait avoids making prompt dispatch
    # permanently dependent on a stuck hydration worker.
    ready = session.get("resume_history_ready")
    if ready is not None and callable(getattr(ready, "wait", None)):
        try:
            if not ready.wait(timeout=30.0):
                _server_logger(server).warning(
                    "prompt.submit: resume history still hydrating for session %s; "
                    "skipping durable refresh",
                    sid,
                )
                return False
        except Exception:
            _server_logger(server).debug(
                "prompt.submit: failed waiting for resume history for session %s",
                sid,
                exc_info=True,
            )
            return False

    if session.get("resume_history_error"):
        return False

    lock = session.get("history_lock")
    if lock is None:
        return False

    with lock:
        # A busy submit is handled by the normal queue/interrupt path. Its
        # active turn owns the in-memory transcript and must not be overwritten
        # from a concurrent durable read.
        if session.get("running"):
            return False
        start_version = int(session.get("history_version", 0))
        start_history = list(session.get("history") or [])

    session_key = str(
        session.get("session_key") or session.get("resume_session_id") or sid
    )
    if not session_key:
        return False

    try:
        session_db = getattr(server, "_session_db")
        with session_db(session) as db:
            if db is None:
                return False
            get_resume = getattr(db, "get_resume_conversations", None)
            if callable(get_resume):
                raw_history, _display_history = get_resume(session_key)
            else:
                get_history = getattr(db, "get_messages_as_conversation", None)
                if not callable(get_history):
                    return False
                raw_history = get_history(
                    session_key,
                    repair_alternation=True,
                    include_row_ids=True,
                )
    except Exception:
        _server_logger(server).debug(
            "prompt.submit: failed refreshing durable history for resumed session %s",
            session_key,
            exc_info=True,
        )
        return False

    if not isinstance(raw_history, list):
        return False
    sanitize = getattr(server, "sanitize_replay_history", None)
    try:
        durable_history = sanitize(raw_history) if callable(sanitize) else raw_history
    except Exception:
        _server_logger(server).debug(
            "prompt.submit: failed sanitizing refreshed history for session %s",
            session_key,
            exc_info=True,
        )
        return False
    if not isinstance(durable_history, list):
        return False

    with lock:
        # A local mutation after the snapshot wins. Checking both the explicit
        # version and the value catches legacy/in-place mutations that forgot to
        # bump history_version.
        if session.get("running"):
            return False
        if int(session.get("history_version", 0)) != start_version:
            return False
        if list(session.get("history") or []) != start_history:
            return False
        if len(durable_history) <= len(start_history):
            return False

        session["history"] = list(durable_history)
        session["history_version"] = start_version + 1

    _server_logger(server).info(
        "prompt.submit: refreshed resumed session %s history %d -> %d messages "
        "from durable state",
        session_key,
        len(start_history),
        len(durable_history),
    )
    return True


def wrap_prompt_submit(server, handler: Callable) -> Callable:
    """Wrap the registered ``prompt.submit`` handler with the refresh guard."""

    @functools.wraps(handler)
    def wrapped(rid, params):
        refresh_resumed_history_before_submit(server, params)
        return handler(rid, params)

    return wrapped
