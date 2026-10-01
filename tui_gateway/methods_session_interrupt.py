"""Interrupt / steer / redirect session handlers (``methods_session`` split).

Moved verbatim from ``tui_gateway/methods_session.py`` (file-line ratchet #68779): the
bodies close over server.py globals through ``method_ctx.bind_module`` exactly as before —
``@method`` registration, ``_session_arg`` wrapping and module publication all still run
from the parent's ``register()`` via ``HandlerRegistry``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .compute_host_bridge import _get_compute_host_supervisor, _session_uses_compute_host
    from .server import _clear_pending, _ok, _sess_building, logger
    from .session_history import _clear_inflight_turn
    from .session_lifecycle import _owns_turn_claim

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

def _resume_wake_after_interrupt() -> None:
    """Re-arm a wake lease held by the interrupt caller or a voice capture.

    ``_wake_resume_if_owner`` no-ops unless that object holds the lease, so an
    in-progress capture owned by someone else is not stolen. Interrupt already
    silenced TTS before it can return an error; this matches that cut. A
    ``not_interrupted`` hosted-task mismatch must not call it.
    """
    with _voice_sid_lock:
        voice_owner = _voice_wake_owner
    seen = []
    for owner in (_caller_transport(), voice_owner):
        if owner is None or any(owner is item for item in seen):
            continue
        seen.append(owner)
        _wake_resume_if_owner(owner)


# ── interrupt / steer / redirect ─────────────────────────────────────
def _note_user_input(session: dict) -> None:
    """Mark the running turn as touched by the user (also covers compute-host turns, whose agent
    flag never reaches this process); popped by the turn's own finally."""
    with session["history_lock"]:
        if session.get("running"):
            session["_turn_user_input"] = True


@method("session.interrupt")
def _(rid, params: dict) -> dict:
    _tts_stream_stop()  # keypress barge-in also silences streaming TTS (voice is process-global)
    resume_wake = True
    try:
        session, err = _sess_nowait(params, rid)
        if err:
            return err
        if expected := _str_param(params, "expected_hosted_task_id"):
            with session["history_lock"]:
                task = session.get("_hosted_room_task")
                if not (session.get("running") and isinstance(task, dict) and task.get("task_id") == expected):
                    resume_wake = False
                    return _ok(rid, {"status": "not_interrupted", "interrupted": False})
        sid = str(params.get("session_id") or "")
        _note_user_input(session)
        if _session_uses_compute_host(session):
            from tools.approval import list_gateway_approvals, resolve_gateway_approval
            from tui_gateway import server_requests

            host_unreachable = False
            with session["history_lock"]:
                inflight_at_interrupt = session.get("inflight_turn")
                interrupted_claim = session.get("_turn_claim")
                interrupted_session_key = str(session.get("session_key") or "")
                pending_request_ids = {request["id"] for request in server_requests.open_requests(sid)}
                pending_approvals = list_gateway_approvals(interrupted_session_key)
            try:
                _interrupt_session_turn(sid, session, request_id=f"interrupt-{rid}")
            except Exception as exc:
                host_unreachable = True
                logger.warning("session.interrupt: compute-host interrupt failed for %s: %s", sid, exc)
            # The host operation can need history_lock, so probe it before acquiring that lock.
            pending_for_sid = True
            if host_unreachable:
                try:
                    pending_for_sid = _get_compute_host_supervisor().has_pending_turn(sid)
                except Exception as exc:
                    logger.warning(
                        "session.interrupt: compute-host pending-turn probe failed for %s: %s", sid, exc
                    )
                    pending_for_sid = True
            with session["history_lock"]:
                if not _owns_turn_claim(session, interrupted_claim):
                    return _ok(rid, {"status": "interrupted", "turn_isolation": True})
                session["_turn_cancel_requested"] = True
                session["queued_prompt"] = None
                session.pop("queued_prompts", None)
                session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
                inflight_now = session.get("inflight_turn")
                if (
                    host_unreachable
                    and not pending_for_sid
                    and session.get("running")
                    and inflight_now is not None
                    and inflight_now is inflight_at_interrupt
                ):
                    session["running"] = False
                    _clear_inflight_turn(session)
            _clear_pending(sid, request_ids=pending_request_ids)
            try:
                for pending_approval in pending_approvals:
                    with session["history_lock"]:
                        if not _owns_turn_claim(session, interrupted_claim):
                            break
                    if approval_id := pending_approval.get("request_id"):
                        resolve_gateway_approval(interrupted_session_key, "deny", request_id=approval_id)
            except Exception:
                pass
            return _ok(rid, {"status": "interrupted", "turn_isolation": True})
        # Stop must inspect the session without waiting for its deferred agent build.
        session, err = _sess_building(params, rid)
        if err:
            return err
        _interrupt_session_turn(sid, session)
        # Retire the crash-recovery marker NOW: until the run thread's finally, a backend exit looks like a crash
        # and session.resume auto-continues the turn the user just stopped (the extra key covers compression
        # rotating session_key mid-turn).
        with session["history_lock"]:
            active_marker_key = str(session.pop("_active_turn_marker_key", "") or "")
        _retire_turn_marker(session, active_marker_key)
        return _ok(rid, {"status": "interrupted"})
    finally:
        if resume_wake:
            try:
                _resume_wake_after_interrupt()
            except Exception:
                logger.debug("session.interrupt wake resume failed", exc_info=True)




def _apply_correction(rid, session: dict, verb: str, text: str, accepted_status: str) -> dict:
    """``agent.<verb>(text)``; on acceptance record it on the live turn (mid-turn resume rebuilds the bubble)
    and purge queued self-copies so post-turn drain cannot re-fire the old prompt."""
    try:
        accepted = getattr(session["agent"], verb)(text)
    except Exception as exc:
        return _err(rid, 5000, f"{verb} failed: {exc}")
    if accepted:
        with session["history_lock"]:
            _record_inflight_correction(session, text)
            # #84417: steer does not cancel the live original, but a server queue self-copy of that original
            # must still not re-fire after settle (same class as redirect).
            # #84417: purge server-queue self-duplicates of the live original so post-turn drain cannot
            # restart the pre-correction prompt.
            _drop_queued_duplicates_of_inflight_user(session)
            session["last_active"] = time.time()
    return _ok(rid, {"status": accepted_status if accepted else "rejected", "text": text})


def _correction_method(name: str, verb: str, accepted_status: str, supported, unsupported: str):
    """steer/redirect RPC: ``params.text`` (4002, checked before the session) into a live session;
    ``supported(agent)`` gates 4010."""
    @method(name)
    def _(rid, params: dict) -> dict:
        if not (text := (params.get("text") or "").strip()):
            return _err(rid, 4002, "text is required")
        session, err = _sess_nowait(params, rid)
        if err:
            return err
        _note_user_input(session)
        agent = session.get("agent")
        # Redirect during the turn-build window (running=True, agent None): queue for the next turn instead of
        # a misleading 4010 the client swallows into a lost follow-up.
        if verb == "redirect" and agent is None and session.get("running"):
            _enqueue_prompt(session, text, current_transport() or _stdio_transport)
            session["last_active"] = time.time()
            return _ok(rid, {"status": "queued", "text": text})
        # Compression in flight: queue instead of steering/redirecting. A correction that
        # reaches the provider mid-compression aborts the compression (explicit_interrupt)
        # — the follow-up kills the turn that would answer it (#61042). Queued here, it
        # drains when compression finishes (the Discord-gateway contract; mirrors the
        # interrupt→queue demotion in gateway/run_busy.py for the channel busy path).
        if _session_compression_in_flight(session):
            _enqueue_prompt(session, text, current_transport() or _stdio_transport)
            session["last_active"] = time.time()
            return _ok(rid, {"status": "queued", "text": text})
        if not supported(agent):
            return _err(rid, 4010, unsupported)
        # An idle agent accepts steer() but only the next turn drains it, spliced after an old tool
        # row (#64578). 'rejected' makes the client queue it as a normal next prompt.
        if verb == "steer" and not session.get("running"):
            return _ok(rid, {"status": "rejected", "text": text})
        return _apply_correction(rid, session, verb, text, accepted_status)


# Inject text into the next tool result without interrupting (AIAgent.steer(): no new user turn, no role
# alternation violation).
_correction_method("session.steer", "steer", "queued", lambda agent: hasattr(agent, "steer"),
                   "agent does not support steer")
# Redirect the active model turn while preserving valid work/context.
_correction_method("session.redirect", "redirect", "redirected",
                   lambda agent: getattr(agent, "_supports_active_turn_redirect", False) is True
                   and hasattr(agent, "redirect"), "agent does not support active-turn redirect")


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
