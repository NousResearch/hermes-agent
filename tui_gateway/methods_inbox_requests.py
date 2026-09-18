"""Scoped request-detail read layer for the Desktop Agent Inbox.

``inbox.requests`` returns, for a specific session identified by its durable ``session_key``,
the live request details: redacted approval payloads with actual allowed choices, and
clarification parameters including locked answers for batch clarify.

This module is READ-ONLY. It never resolves approvals or clarifications, never resumes
or hydrates sessions, and never has side effects on the server request queue or approval
queue. Only ``approval`` and ``clarify`` request types are surfaced; password, secret,
vault, sudo and other credential-shaped types are excluded.
"""

from __future__ import annotations

import logging
import os

from .method_ctx import HandlerRegistry, bind_module
from .methods_inbox import _INBOX_DENY_SOURCES, _inbox_denied_source

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = frozenset({"approval", "clarify"})


def _safe_error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}"


def _build_approval_payload(data: dict) -> dict:
    """Redacted approval payload with computed choices, matching _approval_request_payload."""
    # Use chat's authority for redaction and choice calculation; do not fork it.
    from tui_gateway.server import _approval_request_payload
    payload = _approval_request_payload(data)
    return {
        "request_id": payload.get("request_id", ""),
        "command": payload.get("command", ""),
        "description": payload.get("description", ""),
        "choices": payload.get("choices", ["deny"]),
        "allow_permanent": payload.get("allow_permanent"),
        "allow_session": payload.get("allow_session"),
        "smart_denied": payload.get("smart_denied"),
        "tool_name": payload.get("tool_name"),
    }


def _build_clarify_detail(snapshot: dict) -> dict:
    """Extract clarification details from a server request snapshot."""
    params = dict(snapshot.get("params", {}))
    qids = params.pop("questions", None)
    answers = params.pop("answers", None)
    if qids is not None:
        return {
            "request_id": snapshot.get("id", ""),
            "kind": "batch",
            "params": {
                "questions": qids,
                "answers": answers,
            },
        }
    return {
        "request_id": snapshot.get("id", ""),
        "kind": "single",
        "params": {
            "question": params.get("question"),
            "choices": params.get("choices"),
            "multi_select": params.get("multi_select"),
            "answers": answers,
        },
    }


def _gather_requests_for_session(
    sid: str, session: dict, profile_home: str | None
) -> tuple[list[dict], list[dict], list[str]]:
    """Gather approvals and clarifications for one live session.

    Returns (approvals, clarifications, errors).
    """
    from tui_gateway import server_requests

    approvals: list[dict] = []
    clarifications: list[dict] = []
    errors: list[str] = []

    session_key = str(session.get("session_key") or "")
    if not session_key:
        return approvals, clarifications, errors

    # Gather approvals from gateway queue
    try:
        from tools import approval as _approval
        raw_approvals = _approval.list_gateway_approvals(session_key)
        for raw in raw_approvals:
            approvals.append(_build_approval_payload(raw))
    except Exception as exc:
        errors.append(f"approval read failed: {_safe_error_message(exc)}")

    # Gather clarifications from server_requests
    try:
        snapshots = server_requests.open_requests(sid)
        for snap in snapshots:
            method_name = snap.get("method", "")
            if method_name not in _ALLOWED_METHODS:
                continue
            if method_name == "clarify":
                clarifications.append(_build_clarify_detail(snap))
    except Exception as exc:
        errors.append(f"clarify read failed: {_safe_error_message(exc)}")

    return approvals, clarifications, errors


def _inbox_requests(rid: dict, params: dict) -> dict:
    from tui_gateway.server import (
        _profile_home, _sessions, _sessions_lock, ProfileUnavailableError,
    )

    session_key = str(params.get("session_key") or "").strip()
    if not session_key:
        return _err(rid, 4002, "session_key is required")

    profile = (params.get("profile") or "").strip() or None
    try:
        profile_home_raw = _profile_home(profile)
        profile_home = str(profile_home_raw) if profile_home_raw is not None else None
    except ProfileUnavailableError:
        raise
    except Exception as exc:
        return _err(rid, 5031, f"profile resolution failed: {_safe_error_message(exc)}")

    # Durable identity is sessions.id; never trust only a runtime record's source.
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5031)
        row = db.get_session(session_key)
        if row is None or _inbox_denied_source(row):
            return _err(rid, 4001, "Session not found")

    # Thread-safe snapshot of live sessions
    try:
        with _sessions_lock:
            snapshot = list(_sessions.items())
    except Exception as exc:
        return _err(rid, 5036, f"could not enumerate active sessions: {_safe_error_message(exc)}")

    want_home = os.path.normcase(str(profile_home) if profile_home is not None else str(_hermes_home))
    live_sessions: list[tuple[str, dict]] = []
    for sid, record in snapshot:
        if not isinstance(record, dict):
            continue
        if record.get("_finalized"):
            continue
        if os.path.normcase(str(record.get("profile_home") or "")) != want_home:
            continue
        key = str(record.get("session_key") or "")
        if key != session_key:
            continue
        # Deny-listed sources are not human-facing
        source = str(record.get("source") or "").strip().lower()
        if source in _INBOX_DENY_SOURCES:
            continue
        live_sessions.append((sid, record))

    all_approvals: list[dict] = []
    all_clarifications: list[dict] = []
    all_errors: list[str] = []
    live_ids: list[str] = []

    for sid, record in live_sessions:
        live_ids.append(sid)
        try:
            apps, cls, errs = _gather_requests_for_session(sid, record, profile_home)
            all_approvals.extend(apps)
            all_clarifications.extend(cls)
            all_errors.extend(errs)
        except Exception as exc:
            all_errors.append(f"{sid}: request gathering failed: {_safe_error_message(exc)}")

    # Runtime presence alone provides no transcript/message anchor.
    context_anchor = "unavailable: open chat for context"

    coverage = {
        "profile": profile or _inbox_request_profile_name(profile),
        "session_key": session_key,
        "live_session_count": len(live_ids),
        "approval_count": len(all_approvals),
        "clarification_count": len(all_clarifications),
        "context_anchor": context_anchor,
        "errors": all_errors[:20],
    }

    return _ok(rid, {
        "sessions": [{
            "live_session_ids": live_ids,
            "approvals": all_approvals,
            "clarifications": all_clarifications,
        }],
        "coverage": coverage,
    })


def _inbox_request_profile_name(profile: str | None) -> str:
    """Best-effort profile name for coverage metadata."""
    if profile:
        return profile
    try:
        from tui_gateway.server import _current_profile_name
        return _current_profile_name()
    except Exception:
        return "default"


@method("inbox.requests")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Read-only scoped request details for a specific session."""
    from tui_gateway.server import ProfileUnavailableError
    try:
        return _inbox_requests(rid, params)
    except ProfileUnavailableError:
        raise
    except Exception as exc:
        logger.debug("inbox.requests failed: %s", exc, exc_info=True)
        return _err(rid, 5031, "inbox.requests failed")


def register(server) -> None:
    """Rebind this module's handlers onto the server namespace."""
    bind_module(globals(), server, skip=("_",))
