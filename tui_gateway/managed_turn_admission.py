"""Opt-in managed-turn admission guards; ordinary TUI submission is unchanged.

The receipt is written atomically with the user row by SessionDB.append_message.
A read-only lookup is not evidence that the worker exited or that usage is final.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def persist_submit(rid: str, session: dict, text: str, display_kind: str | None,
                   key: str | None, survivor_fields: dict) -> dict | None:
    """Persist admission before worker start; an unproved write must not return streaming."""
    from . import server
    if key is None:
        error = server._persist_session_row_for_submit(rid, session, text, display_kind)
    else:
        error = server._persist_session_row_for_submit(
            rid, session, text, display_kind, managed_turn_key=key)
    if error is not None or key is None:
        return error
    staged = session.get("_submit_user_row")
    if not isinstance(staged, dict) or type(staged.get("_row_id")) is not int:
        with session["history_lock"]:
            session["running"] = False
            server._clear_inflight_turn(session)
            server._release_active_session_slot(session)
        return server._err(rid, 5031, "Managed turn admission not durably confirmed")
    session["_managed_turn_key"] = key
    survivor_fields["managed_turn_key"] = key
    return None


def voice_response(rid: str, text: object, params: dict) -> dict | None:
    """Managed admission cannot silently turn into a voice-mode control action."""
    from . import server
    if params.get("managed_turn_key") is not None and server._voice_mode_enabled():
        return server._err(rid, 4002, "Managed turns are unavailable in voice mode")
    return server._typed_stop_phrase_response(rid, text)


def valid_key(key: object) -> bool:
    return isinstance(key, str) and len(key) == 32 and all(c in "0123456789abcdef" for c in key)


def profile_mismatch(params: dict, session: dict) -> bool:
    """An explicit profile must name the same home as the resolved live session."""
    from . import server
    explicit = params.get("profile")
    if not explicit:
        return False
    selected = Path(server._profile_home(explicit) or server._hermes_home).resolve()
    owner = Path(session.get("profile_home") or server._hermes_home).resolve()
    return selected != owner


def lookup_response(rid: str, params: dict) -> dict:
    from . import server
    key = params.get("managed_turn_key")
    if not valid_key(key):
        return server._err(rid, 4002, "Invalid managed turn key")
    session, err = server._sess_nowait(params, rid)
    if err:
        return err
    try:
        if profile_mismatch(params, session):
            return server._err(rid, 4031, "Profile does not own this session")
        with server._session_db(session) as db:
            if db is None:
                return server._err(rid, 5031, "Managed turn storage unavailable")
            receipt = db.get_managed_turn(server._submit_row_target_key(session), key)
    except Exception:
        logger.exception("managed turn admission lookup failed")
        return server._err(rid, 5031, "Managed turn lookup unavailable")
    return server._ok(rid, {"found": receipt is not None, **(receipt or {})})


def preflight(rid: str, params: dict, session: dict, text: object, truncation_params: tuple) -> dict | None:
    from . import server
    key = params.get("managed_turn_key")
    if key is None:
        return None
    if (not valid_key(key) or not isinstance(text, str) or not text.strip() or
            params.get("queued") or params.get("interrupted") or
            any(params.get(k) is not None for k in truncation_params) or
            params.get("_turn_author") is not None or params.get("_hosted_task") is not None or
            params.get("_hosted_terminal_callback") is not None):
        return server._err(rid, 4002, "Managed turns require a fresh plain-text submission")
    if profile_mismatch(params, session):
        return server._err(rid, 4031, "Profile does not own this session")
    with session["history_lock"]:
        if session.get("running"):
            return server._err(rid, 4009, "Managed turn cannot queue, steer or redirect a busy session")
    return None


def duplicate_or_error(rid: str, session: dict, key: str | None, turn_isolation: bool) -> dict | None:
    from . import server
    if key is None:
        return None
    if turn_isolation:
        return server._err(rid, 4002, "Managed turns do not support isolated compute dispatch")
    try:
        with server._session_db(session) as db:
            if db is None:
                return server._err(rid, 5031, "Managed turn storage unavailable")
            if db.get_managed_turn(server._submit_row_target_key(session), key):
                return server._err(rid, 4090, "Managed turn already admitted; use read-only lookup")
    except Exception:
        logger.exception("managed turn duplicate admission check failed")
        return server._err(rid, 5031, "Managed turn lookup unavailable")
    return None
