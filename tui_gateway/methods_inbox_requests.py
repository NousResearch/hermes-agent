"""Scoped request-detail read layer for the Desktop Action Center.

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
from .methods_inbox import (
    _INBOX_DENY_SOURCES, _inbox_denied_source, _inbox_home_key,
    clear_expired_request, load_expired_requests,
)

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = frozenset({"approval", "clarify"})

# Bounded transcript context for the request-detail view. The scan window is wider than
# what we return so tool-only and empty rows cannot starve the excerpt.
_CONTEXT_SCAN_LIMIT = 24
_CONTEXT_KEPT_MESSAGES = 4
_CONTEXT_MESSAGE_CHARS = 320
_CONTEXT_TOTAL_CHARS = 1200
_CONTEXT_ROLES = frozenset({"user", "assistant"})


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


def _redact_context_text(text: str) -> str:
    """Redact credential-shaped content before any transcript text leaves the gateway."""
    try:
        from agent.redact import redact_sensitive_text
        return str(redact_sensitive_text(str(text), force=True) or "")
    except Exception:
        return ""


def _build_context_excerpt(db, session_key: str) -> dict:
    """Bounded, redacted excerpt of the owning session's recent turns.

    Read-only and side-effect free: it reads PERSISTED messages directly and never
    resumes, hydrates or mutates the session, so opening the inbox cannot change the
    state of the work it describes.

    Only user/assistant text rows are surfaced. Tool rows are raw JSON payloads that
    cost context without informing a decision, and tool-call-only assistant rows carry
    no text to show. Content is redacted and hard-capped per message and in total, so
    an unbounded transcript can never be egressed through this path.
    """
    try:
        rows = db.get_messages(session_key, limit=_CONTEXT_SCAN_LIMIT, latest=True)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"transcript read failed: {_safe_error_message(exc)}",
            "messages": [],
        }

    kept: list[dict] = []
    total = 0
    for row in reversed(rows or []):  # newest first while selecting, displayed oldest first
        role = str(row.get("role") or "")
        if role not in _CONTEXT_ROLES:
            continue
        text = _redact_context_text(row.get("content") or "").strip()
        if not text:
            continue
        if len(text) > _CONTEXT_MESSAGE_CHARS:
            text = text[:_CONTEXT_MESSAGE_CHARS].rstrip() + "…"
        kept.append({
            "role": role,
            "text": text,
            "timestamp": row.get("timestamp"),
        })
        total += len(text)
        if len(kept) >= _CONTEXT_KEPT_MESSAGES or total >= _CONTEXT_TOTAL_CHARS:
            break

    if not kept:
        return {"available": False, "reason": "no displayable messages in this session", "messages": []}
    kept.reverse()  # chronological for display
    return {"available": True, "reason": None, "messages": kept}


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
        # Same connection as the identity check: one read, no second open, no side effects
        # on the session itself.
        context = _build_context_excerpt(db, session_key)
        # Expired requests survive the turn, the session close and an app restart; they are
        # read here so the panel can still say what died unanswered — and offer a Redo.
        expired_requests = load_expired_requests(db, session_key)

    # Thread-safe snapshot of live sessions
    try:
        with _sessions_lock:
            snapshot = list(_sessions.items())
    except Exception as exc:
        return _err(rid, 5036, f"could not enumerate active sessions: {_safe_error_message(exc)}")

    want_home = _inbox_home_key(profile_home)
    live_sessions: list[tuple[str, dict]] = []
    for sid, record in snapshot:
        if not isinstance(record, dict):
            continue
        if record.get("_finalized"):
            continue
        if _inbox_home_key(record.get("profile_home")) != want_home:
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

    # The anchor now reports what the excerpt actually carries. "Unavailable" stays a
    # real, named state (read failure vs. nothing displayable) so the panel never
    # renders an empty transcript as if it were the whole story.
    if context.get("available"):
        context_anchor = f"available: last {len(context['messages'])} message(s)"
    else:
        context_anchor = f"unavailable: {context.get('reason') or 'no context'}"

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
            "context": context,
            "expired_requests": expired_requests,
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


# ── expired requests: redo / dismiss ──────────────────────────────────────────
_REDO_PROMPT = (
    "[Action Center] An approval request expired before it was answered. "
    "The user asked to redo it — attempt this action again now so they can approve or deny it.\n\n"
    "Command: {command}"
)


def _inbox_live_session_for_key(profile_home: str | None, session_key: str) -> str | None:
    """Runtime id of the live session owning *session_key* in this profile, else None."""
    from tui_gateway.server import _sessions, _sessions_lock

    want_home = _inbox_home_key(profile_home)
    try:
        with _sessions_lock:
            snapshot = list(_sessions.items())
    except Exception:
        return None
    for sid, record in snapshot:
        if not isinstance(record, dict) or record.get("_finalized"):
            continue
        if _inbox_home_key(record.get("profile_home")) != want_home:
            continue
        if str(record.get("session_key") or "") == session_key:
            return sid
    return None


@method("inbox.redo")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Re-raise an expired request by asking its session to attempt the action again.

    The original waiter is long gone (the agent already received its not-approved result
    and must not be told to retry from here), so re-raising means a fresh attempt: the
    session is prompted, which raises a NEW approval the operator can answer. Requires the
    session to be live — the prompt path resumes nothing on its own — and the redo record is
    cleared only after the submit is accepted.
    """
    from tui_gateway.server import _methods, _profile_home, ProfileUnavailableError

    session_key = str(params.get("session_key") or "").strip()
    request_id = str(params.get("request_id") or "").strip()
    if not session_key or not request_id:
        return _err(rid, 4002, "session_key and request_id are required")
    profile = (params.get("profile") or "").strip() or None
    try:
        profile_home_raw = _profile_home(profile)
    except ProfileUnavailableError:
        raise
    except Exception as exc:
        return _err(rid, 5031, f"profile resolution failed: {_safe_error_message(exc)}")
    profile_home = str(profile_home_raw) if profile_home_raw is not None else None

    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5031)
        record = next(
            (entry for entry in load_expired_requests(db, session_key)
             if str(entry.get("request_id")) == request_id),
            None,
        )
    if record is None:
        return _err(rid, 4001, "expired request not found")

    live_sid = _inbox_live_session_for_key(profile_home, session_key)
    if live_sid is None:
        return _err(rid, 4009, "session is not running — open it to redo this request")

    text = _REDO_PROMPT.format(command=str(record.get("command") or "(command not recorded)"))
    # Through the composer's own choke point: role alternation, persistence and streaming
    # behave exactly as a typed message. ``queued`` never interrupts a turn in flight, and
    # ``hidden`` keeps a message the user did not type out of the transcript's bubbles.
    submitted = _methods["prompt.submit"](rid, {
        "session_id": live_sid, "text": text, "queued": True, "display_kind": "hidden",
    })
    if "error" in submitted:
        return submitted

    cleared = False
    with _profile_db(params) as db:
        if db is not None:
            cleared = clear_expired_request(db, session_key, request_id)
    return _ok(rid, {"redone": True, "session_id": live_sid, "record_cleared": cleared})


@method("inbox.dismiss")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Drop one expired-request record (the operator's decision that it is done)."""
    from tui_gateway.server import ProfileUnavailableError

    session_key = str(params.get("session_key") or "").strip()
    request_id = str(params.get("request_id") or "").strip()
    if not session_key or not request_id:
        return _err(rid, 4002, "session_key and request_id are required")
    try:
        with _profile_db(params) as db:
            if db is None:
                return _db_unavailable_error(rid, code=5031)
            dismissed = clear_expired_request(db, session_key, request_id)
    except ProfileUnavailableError:
        raise
    except Exception as exc:
        return _err(rid, 5031, f"inbox.dismiss failed: {_safe_error_message(exc)}")
    if not dismissed:
        return _err(rid, 4001, "expired request not found")
    return _ok(rid, {"dismissed": True})


def register(server) -> None:
    """Rebind this module's handlers onto the server namespace."""
    bind_module(globals(), server, skip=("_",))
