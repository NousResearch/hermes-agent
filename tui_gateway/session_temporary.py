"""Temporary chats on the TUI/Desktop backend (``session.create {temporary: true}``).

The runtime record carries ``temporary``; the row is born hidden with a ``model_config.temporary``
owner record (``_workdir_row_model_config``) and the agent is built with ``temporary=True`` (no
memory, recall or title). Teardown deletes the row — except for the backend's own reclaim reasons,
where the client still shows the chat and resumes it; the client deletes it when the user leaves,
and the startup sweep catches rows whose backend died. Bodies are rebound onto server.py's globals
(method_ctx.bind_module).
"""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module

# Backend-initiated reclaim: the chat is still on the user's screen and resumes from the row.
_TEMPORARY_KEEP_END_REASONS = frozenset({"ws_orphan_reap", "ws_disconnect", "idle_timeout", "lru_evict"})


def _session_is_temporary(session: dict | None, db=None, session_id: str | None = None, *,
                          fail_closed: bool = False) -> bool:
    """The runtime flag, else the stored row's (a resumed temporary chat); cached on the record.
    A store error reads as a normal chat (the agent build must not fail), except where the answer
    protects the transcript (branching): ``fail_closed`` then treats the unknown as temporary."""
    if isinstance(session, dict) and session.get("temporary"):
        return True
    db = db if db is not None else _get_db()
    if db is None or not session_id:
        return False
    if not hasattr(db, "is_session_temporary"):
        return False  # a store without the temporary-chat calls holds no temporary rows
    try:
        temporary = bool(db.is_session_temporary(session_id))
    except Exception:  # health: allow BLE001 -- a store error must not fail the agent build; read as a normal chat
        logger.debug("temporary-session lookup failed for %s", session_id, exc_info=True)
        return fail_closed
    if temporary and isinstance(session, dict):
        session["temporary"] = True
    return temporary


def _apply_temporary_create(record: dict, params: dict) -> None:
    """``session.create {temporary: true}``: the row is born hidden and stamped temporary on first write."""
    if params.get("temporary") is True:
        record["temporary"] = record["pending_hidden"] = True


def _parent_is_temporary(params: dict, parent_session_id: str) -> bool:
    """Seeded branch create: an unreadable parent row is refused rather than copied."""
    with _profile_db(params) as db:
        return db is None or _session_is_temporary(None, db, parent_session_id, fail_closed=True)


def _discard_temporary_session(session: dict | None, end_reason: str) -> bool:
    """Delete a closed temporary chat's row (whole compression chain). Runs after ``agent.close()``
    so no later flush can resurrect it. True when rows were removed."""
    if not isinstance(session, dict) or not session.get("temporary") or end_reason in _TEMPORARY_KEEP_END_REASONS:
        return False
    if session.get("_lease_taken_over"):
        return False  # another runtime owns the chat now
    session_id = getattr(session.get("agent"), "session_id", None) or session.get("session_key")
    if not session_id:
        return False
    from hermes_constants import get_hermes_home
    with contextlib.suppress(Exception), _session_db(session) as db:
        if db is not None:
            sessions_dir = Path(session.get("profile_home") or get_hermes_home()) / "sessions"
            deleted = bool(db.delete_temporary_session(session_id, sessions_dir=sessions_dir))
            # This profile's store may hold orphans the launch-profile startup sweep never sees.
            db.sweep_orphaned_temporary_sessions(sessions_dir=sessions_dir, keep_ids=_temporary_live_session_ids())
            return deleted
    return False


def _temporary_live_session_ids() -> set[str]:
    live: set[str] = set()
    with _sessions_lock:
        for sid, session in _sessions.items():
            live.add(str(sid))
            if isinstance(session, dict):
                live.update(str(c) for c in (getattr(session.get("agent"), "session_id", None),
                                             session.get("session_key")) if c)
    return live


def _sweep_orphaned_temporary_rows(db=None) -> list[str]:
    """Delete temporary chats a dead process left in *db* (default: the launch profile's)."""
    db = db if db is not None else _get_db()
    if db is None:
        return []
    from hermes_constants import get_hermes_home
    return db.sweep_orphaned_temporary_sessions(
        sessions_dir=get_hermes_home() / "sessions", keep_ids=_temporary_live_session_ids())


def _schedule_temporary_session_sweep() -> None:
    """Startup: clear temporary chats orphaned by a crashed or killed backend. Off the response path."""
    _reaper_daemon_timer(0.5, _sweep_orphaned_temporary_rows, "temporary-session sweep failed")


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
