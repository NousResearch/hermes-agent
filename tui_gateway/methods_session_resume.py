"""Live-unpersisted resume reattach (``methods_session`` split).

Moved verbatim from ``tui_gateway/methods_session.py`` (file-line ratchet): the body closes over
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


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))
