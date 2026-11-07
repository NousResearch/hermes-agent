"""Live-session resume reattach (``methods_session`` split).

Moved from ``tui_gateway/methods_session.py`` (file-line ratchet): the bodies close over
server.py globals through ``method_ctx.bind_module``, published from the parent's ``register()``.
"""

from .method_ctx import bind_module


def _resume_live_unpersisted(ctx: "_Resume", live_sid: str, live: dict) -> dict:
    """Reattach a LIVE lazy session with no state.db row yet (every fresh Bot Chat; a 404 here killed messaging
    for never-spoken bots). Attach the transport and cancel the armed orphan-reap Timer (a WS drop may have
    sentinel-parked the record) or it fires against this client."""
    if ctx.owns_db:
        _release_db(ctx.db)
    with _session_resume_lock:
        if (refusal := _reattach_refusal(ctx.rid, live_sid, live)) is not None:
            return refusal
        live["last_active"] = time.time()
        if (transport := current_transport()) is not None:
            _rebind_live_transport(live_sid, live, transport)
        else:
            _cancel_ws_orphan_reap(live_sid)
    messages = ctx.messages(live.get("history") or [])  # count the wire, as every other resume path does
    # The chat's own pick, not the profile default: a warm reattach that reported `_resolve_model()` flipped the
    # Desktop picker on every reload while the session was still live, and back once it had been dropped.
    model, provider = _live_session_identity(live)
    return _ok(ctx.rid, _attach_todo_state({
        "session_id": live_sid, "stored_session_id": str(live.get("session_key") or ""),
        "message_count": len(messages), "messages": messages,
        "info": {"model": model, "provider": provider, "lazy": True,
                 "desktop_contract": DESKTOP_BACKEND_CONTRACT,
                 "profile_name": profile_name_for_home(live.get("profile_home")) or _response_profile_name(ctx.profile)}}, live))


def _resume_reuse_live(ctx: "_Resume", sid: str, session: dict) -> dict:
    """Reattach an already-live session. The resume lock covers only the grace-atomic part (client-gone check,
    transport attach, reap cancel) so expiry stays atomic; the payload is built after release because it waits
    on the session's history_lock, which the global lock must never wait on. _rebind_live_transport ATTACHES this
    caller alongside the client(s) already streaming instead of taking the slot from them."""
    with _session_resume_lock:
        if (refusal := _resume_attach_live_locked(ctx, sid, session)) is not None:
            return refusal
    return _resume_reuse_payload(ctx, sid, session)


def _resume_attach_live_locked(ctx: "_Resume", sid: str, session: dict) -> dict | None:
    """Reattach with _session_resume_lock already held (including the eager double-check): the refusal, or None
    once the transport is bound and the reap cancelled."""
    if (refusal := _reattach_refusal(ctx.rid, sid, session)) is not None:
        return refusal
    _cancel_ws_orphan_reap(sid)  # unconditionally: the fast path must never race the reap Timer
    _rebind_live_transport(sid, session, current_transport() or _stdio_transport)
    return None


def _resume_reuse_payload(ctx: "_Resume", sid: str, session: dict) -> dict:
    """Resume payload for a session already attached by _resume_attach_live_locked. Caller holds no resume lock."""
    payload = _live_session_payload(sid, session, cols=ctx.cols, touch=True, omit_messages=ctx.omit_messages,
                                    inline_images=ctx.inline_images)
    payload["resumed"] = ctx.target
    if ctx.defer_history:
        payload.update(messages=[], hydrating=bool(session.get("resume_hydrating")),
                       message_count=int(session.get("resume_message_count") or payload["message_count"]))
    # A lazy watch session never owns a run loop — overlay the child-run registry.
    if session.get("agent") is None and _child_run_active(ctx.target, ctx.profile_home):
        payload.update(running=True, status="streaming")
    return _ok(ctx.rid, payload)


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))
