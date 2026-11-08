"""Retire one Desktop viewer without closing its shared socket or stored conversation."""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _viewer_detach_is_settled(sid: str, session: dict) -> bool:
    """A tab disappearing cannot withdraw the last return address for unfinished work."""
    if (session.get("running") or session.get("_compute_host_turn_id")
            or session.get("queued_prompt") or session.get("queued_prompts")
            or session.get("inflight_turn") or session.get("_auto_continue_scheduled")
            or session.get("_pending_turn_tail")
            or _session_pending_kind(sid)):
        return False
    ready = session.get("agent_ready")
    if ready is not None and not ready.is_set() and not session.get("lazy"):
        return False
    try:
        return not (_session_has_active_delegations(sid, session)
                    or _session_owns_live_wakeup_schedule(session, strict=True)
                    or _session_owns_pending_process_notification(session)
                    or _canonical_ledger_refusal(session))
    # health: allow BLE001 -- uncertain work ownership refuses retirement; it never authorizes cleanup
    except Exception:
        logger.warning("Cannot prove session %s settled; preserving its viewer", sid, exc_info=True)
        return False


@method("session.detach")
@_profile_scoped
def _detach_viewer(rid, params):
    sid, transport = params["session_id"], current_transport()
    with _session_resume_lock:
        session = _sessions.get(sid)
        if session is None or transport is None:
            return _ok(rid, {"detached": False})
        # Match resume/submit's history-before-sessions lock order. Eligibility
        # may read stores, so only membership mutations hold the leaf transport lock.
        with session["history_lock"]:
            if params.get("profile") is not None and not _live_profile_matches(session, _profile_home(params["profile"])):
                return _ok(rid, {"detached": False})
            settled = _viewer_detach_is_settled(sid, session)
            with _sessions_lock:
                with _session_transport_lock:
                    if _sessions.get(sid) is not session or not _session_transport_contains(session, transport):
                        return _ok(rid, {"detached": False})
                    if not _session_has_live_transport(session, excluding=transport) and not settled:
                        return _ok(rid, {"detached": False})
                    clientless = not _detach_session_transport(session, transport)
                    if clientless:
                        session["transport"] = _detached_ws_transport
                        session.pop("_client_gone_interrupt_requested", None)
                        session.pop("_client_gone_interrupt_polls", None)
                if clientless:
                    _schedule_ws_orphan_reap(sid)
    return _ok(rid, {"detached": True})


def register(server) -> None:
    bind_module(globals(), server)
