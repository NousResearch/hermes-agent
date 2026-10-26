"""Interrupt session handler (``methods_session`` split).

The steer / redirect corrections live in ``methods_corrections.py`` with their shared-session observations.

Moved verbatim from ``tui_gateway/methods_session.py`` (file-line ratchet #68779): the
bodies close over server.py globals through ``method_ctx.bind_module`` exactly as before —
``@method`` registration, ``_session_arg`` wrapping and module publication all still run
from the parent's ``register()`` via ``HandlerRegistry``.
"""

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


def _recover_unreachable_compute_host_turn(sid: str, session: dict, inflight_at_interrupt) -> None:
    """Stop on a compute host that is gone (broken pipe, respawn exhausted, #82603).

    The failed dial raised before ``_interrupt_session_turn`` purged the queue, and no ``turn.end`` will ever
    clear ``running``, so every later submit would queue forever. Clear the turn only when the supervisor owes
    no completion for it and the inflight dict is still the one this Stop saw (a successor turn keeps its own).
    The probe runs outside ``history_lock``: compute-host calls never run under it.
    """
    try:
        pending = _get_compute_host_supervisor().has_pending_turn(sid)
    except Exception as exc:  # health: allow BLE001 -- an unreadable supervisor is not proof the turn ended; fail closed
        logger.warning("session.interrupt: compute-host pending-turn probe failed for %s: %s", sid, exc)
        pending = True
    with session["history_lock"]:
        session["_turn_cancel_requested"] = True
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
        inflight_now = session.get("inflight_turn")
        if not pending and session.get("running") and inflight_now is not None and inflight_now is inflight_at_interrupt:
            session["running"] = False
            _clear_inflight_turn(session)
    _clear_pending(sid)
    with contextlib.suppress(Exception):
        _announce_cancelled_gateway_approvals(session, "interrupt", session_id=sid)
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)


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
            with session["history_lock"]:
                inflight_at_interrupt = session.get("inflight_turn")
            try:
                _interrupt_session_turn(sid, session, request_id=f"interrupt-{rid}")
            except Exception as exc:  # health: allow BLE001 -- the bridge raises transport-specific errors unknown here; logged, and Stop recovers the session instead of failing
                logger.warning("session.interrupt: compute-host interrupt failed for %s: %s", sid, exc)
                _recover_unreachable_compute_host_turn(sid, session, inflight_at_interrupt)
            return _ok(rid, {"status": "interrupted", "turn_isolation": True})
        # No agent wait: a turn parked on the deferred build already honors _turn_cancel_requested. Waiting here
        # failed Stop with 5032 once the build failed or outlived 30s (and started a build for a lazy session), so
        # the Desktop, which interrupts before session.close, could not delete such a session (#133262).
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


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
