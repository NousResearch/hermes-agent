"""Interrupt / steer / redirect session handlers (``methods_session`` split).

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


@method("session.steer_clear")
def _(rid, params: dict) -> dict:
    """Drop any pending /steer text that has not been injected yet.

    The TUI shows a pending steer as the top row of its queue strip; when the
    user deletes that row (Ctrl+X) it calls this so the agent's stashed slot is
    cleared too — otherwise the text would still be injected on the next tool
    result. ``cleared`` is True when there was a pending steer to drop.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None:
        return _err(rid, 4010, "no active agent")
    clearer = getattr(agent, "clear_pending_steer", None)
    if callable(clearer):
        try:
            cleared = bool(clearer())
        except Exception as exc:
            return _err(rid, 5000, f"steer clear failed: {exc}")
        return _ok(rid, {"status": "cleared" if cleared else "noop", "cleared": cleared})
    # Older runtimes / test stubs without the method: clear the internal slot
    # directly under its lock when present.
    _steer_lock = getattr(agent, "_pending_steer_lock", None)
    try:
        if _steer_lock is not None:
            with _steer_lock:
                had = getattr(agent, "_pending_steer", None) is not None
                agent._pending_steer = None
        else:
            had = getattr(agent, "_pending_steer", None) is not None
            setattr(agent, "_pending_steer", None)
    except Exception as exc:
        return _err(rid, 5000, f"steer clear failed: {exc}")
    return _ok(rid, {"status": "cleared" if had else "noop", "cleared": bool(had)})


@method("session.pending")
def _(rid, params: dict) -> dict:
    """Snapshot of what will run next: the pending steer (if any), a model
    redirect still waiting to rebuild the request, and every queued prompt.

    The TUI renders the steer as the pending strip, and a reconnect rebuilds
    it from here. Shared-gateway mode reads the same shape from the canonical
    authority (``gateway.session_busy_controls.pending_view``)."""
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    steer: str | None = None
    redirect: str | None = None
    if agent is not None:
        pending = getattr(agent, "_pending_steer", None)
        if isinstance(pending, str) and pending:
            steer = pending
        waiting = getattr(agent, "_pending_redirect", None)
        if isinstance(waiting, str) and waiting:
            redirect = waiting
    queue: list[str] = []
    queued = session.get("queued_prompt")
    if queued and isinstance(queued.get("text"), str):
        queue.append(queued["text"])
    for extra in session.get("queued_prompts", []) or []:
        text = extra.get("text") if isinstance(extra, dict) else None
        if isinstance(text, str):
            queue.append(text)
    result: dict = {"steer": steer, "queue": queue}
    if redirect:
        result["redirect"] = redirect
    return _ok(rid, result)


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
