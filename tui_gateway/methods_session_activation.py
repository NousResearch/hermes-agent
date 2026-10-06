"""Authenticated conditional subscription; legacy activation remains independent."""
from __future__ import annotations

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def _attach_built_agent(sid: str, current: dict, agent) -> bool:
    """Attach a freshly built agent to its live record (session DB row deferred to first run_conversation()).
    False when ``session.close`` popped this record mid-build: teardown saw ``agent=None`` and closed
    nothing, so the caller owns closing the orphan (#49852)."""
    # Bot Mode gate hint: the DB title lands post-first-turn but the system prompt builds at turn START.
    if _title_hint := str(current.get("pending_title") or "").strip():
        agent._session_title_hint = _title_hint
    # Under the same lock session.close takes to pop the record: no window between "still live" and "attached".
    with _sessions_lock:
        if _sessions.get(sid) is not current:
            return False
        current["agent"] = agent
        if current.get("creation_binding") is not None and "creation_engine" not in current:
            from .session_creation_binding import EngineCreationBinding
            current["creation_engine"] = EngineCreationBinding.capture(current["creation_binding"], agent)
    # A workspace move can land while construction is still in flight.
    _register_session_cwd(current)
    _session_todo_state(current)
    # Baseline for the per-turn config sync (profile home override still active).
    current["config_model_seen"] = _config_model_target()
    return True


@contextlib.contextmanager
def _activation_engine_guard(session: dict):
    from .session_creation_binding import EngineCreationBinding

    agent = session.get("agent")
    if agent is None:
        yield "creation_engine" not in session and not session.get("running") and not session.get("inflight_turn")
        return
    ready = session.get("agent_ready")
    if ready is None or not ready.is_set():
        yield False
        return
    witness = session.get("creation_engine")
    if not isinstance(witness, EngineCreationBinding) or witness.engine is not agent:
        yield False
        return
    # Engine callbacks may need gateway parent locks. Never wait here while
    # holding those locks; a busy publication/adoption is settling, not healthy.
    guard = agent.session_identity_guard()
    if not guard.acquire(blocking=False):
        yield False
        return
    try:
        yield witness.matches(session["creation_binding"], running=bool(session.get("running")))
    finally:
        guard.release()


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

    if (session.get("_compute_host_active") or session.get("_compute_host_turn_id")
            or session.get("_compute_host_ever_owned")):
        return None
    turn = session.get("inflight_turn")
    if turn is not None and (not isinstance(turn, dict) or turn.get("error")
                             or turn.get("status") not in (None, "running", "streaming")):
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
    # Route qualification is profile-owned; an unbound config read would certify
    # a secondary destined for a child using the launch profile's local policy.
    with _session_profile_runtime_scope(session, hydrate_secrets=False):
        if _session_uses_compute_host(session):
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
        with _activation_engine_guard(session) as proven:
            if not proven:
                return _err(rid, 4007, "Conditional activation refused")
            receipt = _bound_activation_receipt(session, sid, owner, store_path, params["expected_binding"])
            if _sessions.get(sid) is not session or receipt is None or not _transport_is_live_peer(peer):
                return _err(rid, 4007, "Conditional activation refused")
            if not _attach_session_transport(session, peer):
                return _err(rid, 4007, "Conditional activation refused")
            session.setdefault("bound_subscribers", {})[peer] = dict(receipt)
            # Sticky per-connection mode: runtime replacement/removal cannot restore
            # the legacy mutation/response path on a connection that opted into CAS.
            peer._conditional_session_mode = True
            session.setdefault("viewers", {})[peer] = time.time()
            _cancel_ws_orphan_reap(sid)
            # Receipt was captured inside the comparison/subscription cut. Never turn
            # a post-lock mutable snapshot into accepted identity or recovery evidence.
            return _ok(rid, {"attached": True, "accepted_binding": receipt})



def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
