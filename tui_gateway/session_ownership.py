"""Browser ownership, observation, takeover, and mutation fencing.

Ownership is opt-in: legacy stdio/Desktop/TUI sessions keep their existing
transport semantics until a browser resumes with an explicit ``owner_id``.
"""

from __future__ import annotations

import time

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

_BROWSER_FENCE_CODE = 4093
_BROWSER_CONFIRMATION_CODE = 4094
_BROWSER_UNSAFE_TAKEOVER_CODE = 4095


def _browser_owner_id(params: dict) -> str:
    value = params.get("owner_id")
    return value.strip() if isinstance(value, str) else ""


def _transport_supports_takeover(transport) -> bool:
    """Only a live browser WS transport can be revoked safely."""
    return bool(
        transport is not None
        and getattr(transport, "supports_session_takeover", False) is True
        and not getattr(transport, "closed", getattr(transport, "_closed", False))
    )


def _browser_ownership_payload(session: dict, *, read_only: bool | None = None) -> dict:
    owner_id = str(session.get("owner_id") or "")
    if not owner_id:
        return {}
    if read_only is None:
        read_only = current_transport() is not session.get("owner_transport")
    return {
        "read_only": bool(read_only),
        "owner_id": owner_id,
        "ownership_epoch": int(session.get("ownership_epoch") or 1),
    }


def _browser_ownership_error(rid, session: dict, message: str = "SESSION_NOT_OWNED") -> dict:
    return _err(
        rid,
        _BROWSER_FENCE_CODE,
        message,
        {"reason": "SESSION_NOT_OWNED", **_browser_ownership_payload(session, read_only=True)},
    )


def _browser_resume_request_error(rid, params: dict) -> dict | None:
    """Validate explicit browser identity before resume reads or registration."""
    if "owner_id" not in params:
        return None
    if not _browser_owner_id(params):
        return _err(rid, -32602, "invalid params: owner_id must be a non-empty string")
    if not _transport_supports_takeover(current_transport()):
        return _err(rid, _BROWSER_UNSAFE_TAKEOVER_CODE, "browser ownership requires a revocable WebSocket transport")
    return None


def _initialize_browser_ownership(session: dict, params: dict, transport=None) -> None:
    """Fence a newly-minted live record when resume carries an explicit owner."""
    owner_id = _browser_owner_id(params)
    if not owner_id:
        return
    transport = transport or current_transport()
    session.update(
        owner_id=owner_id,
        ownership_epoch=max(1, int(session.get("ownership_epoch") or 0)),
        owner_transport=transport,
        transport=transport,
    )
    session.setdefault("observers", set()).discard(transport)


def _attach_browser_resume(
    rid, sid: str, session: dict, params: dict, transport,
) -> tuple[dict | None, dict]:
    """Attach a resume caller as owner or observer without implicit takeover.

    Caller holds ``_session_resume_lock``. The returned metadata is merged into
    the snapshot response; a foreign/legacy viewer receives ``read_only``.
    """
    requested_owner = _browser_owner_id(params)
    current_owner = str(session.get("owner_id") or "")

    if not current_owner:
        live_transport = session.get("transport")
        if requested_owner and live_transport is not None and live_transport is not transport:
            if transport not in (None, _stdio_transport, _detached_ws_transport):
                session.setdefault("observers", set()).add(transport)
                session.setdefault("viewers", {})[transport] = time.time()
            return None, {"read_only": True}
        if requested_owner:
            _initialize_browser_ownership(session, params, transport)
            _cancel_ws_orphan_reap(sid)
            return None, _browser_ownership_payload(session, read_only=False)
        _rebind_live_transport(sid, session, transport)
        return None, {}

    if requested_owner and requested_owner == current_owner:
        session["owner_transport"] = transport
        session.setdefault("observers", set()).discard(transport)
        _rebind_live_transport(sid, session, transport)
        return None, _browser_ownership_payload(session, read_only=False)

    # Explicitly foreign browsers and legacy clients may observe, but cannot
    # move the transport that owns asynchronous session output.
    if transport not in (None, _stdio_transport, _detached_ws_transport):
        session.setdefault("observers", set()).add(transport)
        session.setdefault("viewers", {})[transport] = time.time()
    return None, _browser_ownership_payload(session, read_only=True)


def _browser_mutation_fence(rid, params: dict, session: dict, *, transport=None) -> dict | None:
    """CAS fence for every mutation of a browser-owned session."""
    if not session.get("owner_id"):
        return None
    supplied_epoch = params.get("ownership_epoch")
    valid_epoch = type(supplied_epoch) is int and supplied_epoch == int(session.get("ownership_epoch") or 0)
    if (
        _browser_owner_id(params) != str(session.get("owner_id") or "")
        or not valid_epoch
        or (transport or current_transport()) is not session.get("owner_transport")
    ):
        return _browser_ownership_error(rid, session)
    return None


def _browser_pending_mutation_fence(rid, params: dict) -> dict | None:
    """Fence a prompt-card response before its pending registry is changed."""
    sid = str(params.get("session_id") or "")
    session = _sessions.get(sid) if sid else None
    if session is None and (request_id := str(params.get("request_id") or "")):
        with _prompt_lock:
            entry = _pending.get(request_id)
        if entry:
            session = _sessions.get(entry[0])
    return _browser_mutation_fence(rid, params, session) if session is not None else None


@method("session.takeover")
def _(rid, params: dict) -> dict:
    """Atomically transfer a live browser session after explicit confirmation."""
    if params.get("confirmed") is not True:
        return _err(rid, _BROWSER_CONFIRMATION_CODE, "session takeover requires explicit confirmation")
    new_owner = _browser_owner_id(params)
    if not new_owner:
        return _err(rid, -32602, "invalid params: owner_id must be a non-empty string")
    new_transport = current_transport()
    if not _transport_supports_takeover(new_transport):
        return _err(rid, _BROWSER_UNSAFE_TAKEOVER_CODE, "session takeover requires a revocable WebSocket transport")

    sid = str(params.get("session_id") or "")
    with _session_resume_lock, _sessions_lock:
        session = _sessions.get(sid)
        if session is None:
            return _err(rid, 4001, "session not found")
        current_owner = str(session.get("owner_id") or "")
        if not current_owner:
            return _err(rid, _BROWSER_UNSAFE_TAKEOVER_CODE, "session is not browser-owned and cannot be safely revoked")
        supplied_epoch = params.get("ownership_epoch")
        if type(supplied_epoch) is not int or supplied_epoch != int(session.get("ownership_epoch") or 0):
            return _browser_ownership_error(rid, session, "SESSION_OWNERSHIP_CHANGED")
        old_transport = session.get("owner_transport")
        if new_owner == current_owner:
            session["owner_transport"] = new_transport
            _rebind_live_transport(sid, session, new_transport)
            return _ok(rid, {"session_id": sid, **_browser_ownership_payload(session, read_only=False)})
        if not _transport_supports_takeover(old_transport):
            return _err(rid, _BROWSER_UNSAFE_TAKEOVER_CODE, "current session owner cannot be safely revoked")

        next_epoch = int(session.get("ownership_epoch") or 0) + 1
        session.update(owner_id=new_owner, ownership_epoch=next_epoch, owner_transport=new_transport)
        session.setdefault("observers", set()).discard(new_transport)
        if old_transport is not new_transport:
            session["observers"].add(old_transport)
        _rebind_live_transport(sid, session, new_transport)

    # The ownership generation is already committed. Notify the exact old
    # transport directly; routing through _emit would target the new owner.
    old_transport.write(_event_frame("session.revoked", sid, {
        "owner_id": new_owner,
        "ownership_epoch": next_epoch,
    }))
    return _ok(rid, {
        "session_id": sid,
        "owner_id": new_owner,
        "ownership_epoch": next_epoch,
        "read_only": False,
    })


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
