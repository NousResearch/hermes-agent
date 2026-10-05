"""Authenticated conditional subscription; legacy activation remains independent."""
from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def _creation_binding_fields(session: dict, profile_home) -> dict:
    origin = session.get("creation_binding")
    if origin is None:
        return {}
    store_path = str((Path(profile_home or _hermes_home) / "state.db").resolve())
    return origin.fields_for(_transport_auth_user_id(current_transport()), store_path)


def _creation_retry_result(rid, sid: str, session: dict, profile, profile_home, copy_parent_history: bool) -> dict:
    history = session["history"]
    override = session.get("model_override") or {}
    # Same result shape as fresh create: branch_stored
    # (copy_parent_history) answers messages_omitted and NEVER puts the
    # copied transcript on the wire — its result contract forbids
    # ``messages``, and serializing the parent's history through the
    # renderer is exactly what the method exists to avoid.
    return _ok(rid, {
        "session_id": sid, "stored_session_id": session["session_key"],
        **_creation_binding_fields(session, profile_home),
        "message_count": len(history),
        **({"messages_omitted": True} if copy_parent_history
           else {"messages": _history_to_messages(history, profile_home=session.get("profile_home"))}),
        "info": {**_lazy_info_route(session, override), "tools": {}, "skills": {},
                 "cwd": session["cwd"], "branch": git_probe.branch(session["cwd"]),
                 "project": _project_info_for_cwd(session["cwd"]), "lazy": True,
                 "desktop_contract": DESKTOP_BACKEND_CONTRACT,
                 "profile_name": _response_profile_name(profile)}})


def _bound_activation_receipt(session: dict, sid: str, owner: str, store_path: str, expected: dict) -> dict | None:
    from .session_creation_binding import CreationBinding

    # Built engines rotate agent.session_id independently of gateway history_lock.
    # Host children likewise publish identity asynchronously. They are unproven
    # until their identity writers participate in this boundary; refuse them.
    if session.get("agent") is not None or any(session.get(flag) for flag in (
        "running", "_compute_host_active", "_compute_host_turn_id", "inflight_turn",
    )):
        return None
    origin = session.get("creation_binding")
    if not isinstance(origin, CreationBinding) or origin.runtime_record is not session:
        return None
    wire = origin.fields_for(owner, store_path).get("creation_binding")
    current_store = str((Path(session.get("profile_home") or _hermes_home) / "state.db").resolve())
    if (wire != expected or origin.session_id != sid or current_store != origin.store_path
            or session.get("session_key") != origin.stored_session_id
            or session.get("auth_user_id") != owner):
        return None
    if any(session.get(flag) for flag in (
        "_closing", "_finalized", "_client_gone_interrupt_requested", "_turn_cancel_requested",
        "resume_hydrating", "agent_error",
    )):
        return None
    return wire


@method("session.activate_bound")
def _(rid, params: dict) -> dict:
    from pydantic import ValidationError
    from .contracts.sessions import SessionActivateBoundParams

    # The legacy dispatcher rejects unknown keys only. This boundary requires
    # complete strict validation before lookup, locks, or subscription effects.
    try:
        SessionActivateBoundParams.model_validate(params)
    except ValidationError:
        return _err(rid, 4000, "Invalid conditional activation precondition")
    sid = params["session_id"]
    peer = current_transport()
    owner = _transport_auth_user_id(peer)
    if not owner or not _transport_is_live_peer(peer):
        return _err(rid, 4007, "Conditional activation refused")
    requested_home = _profile_home(params.get("profile"))
    store_path = str((Path(requested_home or _hermes_home) / "state.db").resolve())
    # Do not hold registry while waiting for history. Recheck object membership
    # under registry after acquiring history; replacement writers use registry.
    with _sessions_lock:
        session = _sessions.get(sid)
    if session is None or session.get("history_lock") is None:
        return _err(rid, 4007, "Conditional activation refused")
    # Resume serializes grace expiry/interrupt; history serializes compression and
    # host adoption; registry excludes replacement; transport is the leaf lock.
    with _session_resume_lock, session["history_lock"], _sessions_lock, _session_transport_lock:
        receipt = _bound_activation_receipt(session, sid, owner, store_path, params["expected_binding"])
        if _sessions.get(sid) is not session or receipt is None or not _transport_is_live_peer(peer):
            return _err(rid, 4007, "Conditional activation refused")
        if not _attach_session_transport(session, peer):
            return _err(rid, 4007, "Conditional activation refused")
        session.setdefault("viewers", {})[peer] = time.time()
        _cancel_ws_orphan_reap(sid)
        # Receipt was captured inside the comparison/subscription cut. Never turn
        # a post-lock mutable snapshot into accepted identity or recovery evidence.
        return _ok(rid, {"attached": True, "accepted_binding": receipt})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
