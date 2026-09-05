"""Session-scoped approval notify registry (survives turn teardown).

A CLI-initiated background SDK turn between Hermes turns must still be able to page the
operator (the 2026-08-06 incident lane: ``notify_cb`` None → silent deny falsely attributed
to the user). Populated ONLY by the Hermes gateway's turn registration
(``gateway/run_turn_runner.py``) — NOT folded into ``register_gateway_notify``, because the
api_server per-run and tui_gateway per-session callers key differently and a blanket refresh
would leak dead run_id/session_id entries. Removed at conversation boundaries
(``tools.approval.clear_session``, via the gateway's boundary funnel) and at gateway shutdown
(:func:`clear_all_session_notify`). Entries are guarded by ``tools.approval._lock``.
"""

from __future__ import annotations

_session_notify_cbs: dict[str, object] = {}  # session_key → callable(approval_data)


def _lock():
    from tools import approval as _approval

    return _approval._lock


def register_session_notify(session_key: str, cb) -> None:
    """Register/refresh the SESSION-scoped approval notify callback.

    Idempotent — the gateway calls this on every turn alongside ``register_gateway_notify``;
    the latest callback wins. The callback must stay valid between turns (the gateway's
    closure is: adapter, chat id and event loop all outlive the turn).
    """
    if not session_key:
        return
    with _lock():
        _session_notify_cbs[session_key] = cb


def unregister_session_notify(session_key: str) -> None:
    """Remove the session-scoped callback (idempotent); never touches the turn registry or queues."""
    with _lock():
        _session_notify_cbs.pop(session_key, None)


def lookup_session_notify(session_key: str):
    """The session-scoped callback for *session_key*, or None."""
    with _lock():
        return _session_notify_cbs.get(session_key)


def clear_session_notify(session_key: str) -> None:
    """Conversation boundary (/new, expiry, auto-reset, …): the session-scoped approver dies with the
    conversation — the next turn's registration refreshes it. A retained entry would page the operator
    for a session the user deliberately rotated away. Caller holds ``tools.approval._lock``."""
    _session_notify_cbs.pop(session_key, None)


def clear_all_session_notify() -> None:
    """Drop every session-scoped notify callback (gateway shutdown)."""
    with _lock():
        _session_notify_cbs.clear()
