"""The prompt turn: ``_run_prompt_submit`` and the per-phase helpers it drives.

Bodies are rebound onto server.py's globals at install time (method_ctx.bind_module),
so they reference server.py globals bare.  Turn shape (one fresh daemon thread):
admit -> crash marker -> bind scopes -> resolve message -> run_conversation ->
commit history / message.complete -> goal & loop hooks -> release scopes ->
post-turn follow-ups (queued prompt, goal continuation, notifications).
"""

from __future__ import annotations

import dataclasses

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _hook_failure(what: str, exc: BaseException) -> None:
    print(f"[tui_gateway] {what} failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def _is_successful_goal_turn(result: Any, status: str, raw: Any) -> bool:
    """Whether a turn produced a real response the goal judge can use."""
    return bool(
        status == "complete" and isinstance(raw, str) and raw.strip()
        and not (isinstance(result, dict) and result.get("failed"))
        and not (isinstance(result, dict) and result.get("completed") is False))


def _active_goal_manager(session: dict):
    """The session's GoalManager when a goal is active, else None."""
    from hermes_cli.goals import GoalManager
    try:
        max_turns = int((_load_cfg().get("goals") or {}).get("max_turns", 20) or 20)
    except Exception:
        max_turns = 20
    goal_mgr = GoalManager(
        session_id=str(session.get("session_key") or ""), default_max_turns=max_turns)
    return goal_mgr if goal_mgr.is_active() else None


def _plan_goal_compression_recovery(
    session: dict, result: Any, *, status: str, raw: Any) -> tuple[str | None, str | None]:
    """Bounded active-goal retry after compression exhaustion: ``(continuation, notice)``.
    Exhaustion is a failed turn (never judge input, never a spent goal turn); one fresh
    continuation is allowed, a second exhaustion pauses the goal instead of spinning."""
    if not (isinstance(result, dict) and result.get("compression_exhausted")):
        if _is_successful_goal_turn(result, status, raw):
            session.pop(_GOAL_COMPRESSION_RECOVERY_ATTEMPTS, None)
        return None, None
    if not str(session.get("session_key") or ""):
        return None, None
    if (goal_mgr := _active_goal_manager(session)) is None:
        session.pop(_GOAL_COMPRESSION_RECOVERY_ATTEMPTS, None)
        return None, None
    goal_created_at = float(getattr(goal_mgr.state, "created_at", 0.0) or 0.0)
    goal_text = getattr(goal_mgr.state, "goal", "")
    recovery_state = session.get(_GOAL_COMPRESSION_RECOVERY_ATTEMPTS)
    attempts = 0
    if (
        isinstance(recovery_state, dict)
        and recovery_state.get("goal_created_at") == goal_created_at
        and recovery_state.get("goal") == goal_text):
        with contextlib.suppress(TypeError, ValueError):
            attempts = int(recovery_state.get("attempts", 0) or 0)
    continuation_prompt = goal_mgr.next_continuation_prompt()
    if attempts < _GOAL_COMPRESSION_RECOVERY_LIMIT and continuation_prompt:
        session[_GOAL_COMPRESSION_RECOVERY_ATTEMPTS] = {
            "goal_created_at": goal_created_at, "goal": goal_text, "attempts": attempts + 1}
        return (
            continuation_prompt,
            "Context compression was exhausted. Retrying the active goal once.")
    goal_mgr.pause(reason="context compression exhausted twice consecutively")
    # A later explicit /goal resume gets a fresh bounded recovery cycle.
    session.pop(_GOAL_COMPRESSION_RECOVERY_ATTEMPTS, None)
    return None, (
        "Goal paused after context compression was exhausted twice. "
        "Run /compress, then /goal resume to continue.")


def _admit_prompt_turn(
    sid: str, session: dict, text: Any, image_paths: list[str] | None,
    queued_prompt_generation: int | None) -> tuple[list[str], Any] | None:
    """Ownership + liveness gate every turn source must cross; ``(images, agent)`` or None.
    Synthesized turns (auto-continue, wake-ups) call ``_run_prompt_submit`` directly — the
    bypass that once let a second backend run a duplicate turn."""
    # When the session already holds its lease this is a cheap dict check. See #94778.
    if (ownership_refusal := _ensure_active_session_slot(sid, session)) is not None:
        logger.info(
            "Refusing turn for session %s at _run_prompt_submit: %s",
            session.get("session_key") or sid,
            getattr(ownership_refusal, "reason", None) or "refused")
        with session["history_lock"]:
            session["running"] = False
        _emit("error", sid, {"message": str(ownership_refusal)})
        return None
    with session["history_lock"]:
        if session.get("_closing") or (
            queued_prompt_generation is not None
            and int(session.get("_queued_prompt_generation", 0)) != queued_prompt_generation):
            session["running"] = False
            return None
        images = list(session.get("attached_images", []) if image_paths is None else image_paths)
        if image_paths is None:
            session["attached_images"] = []
        inflight = session.get("inflight_turn")
        # A retained failed turn (see _fail_inflight_turn) is a stale leftover
        # by the time a new turn starts — replace it, never append onto it.
        if not isinstance(inflight, dict) or inflight.get("status") == "error":
            _start_inflight_turn(session, text)
        agent = session["agent"]
        with contextlib.suppress(Exception):
            agent.clear_interrupt()
    return images, agent


def _record_turn_marker(session: dict, text: Any) -> str:
    """Write the durable crash marker; returns the session key it was written under (compression
    can rotate session_key mid-turn).  A surviving marker means the process died mid-turn.
    The key is published before the disk write so an interrupt racing startup can retire
    it; the post-write cancel check closes the inverse race (Stop landed first, no file)."""
    marker_home = _session_home(session)
    marker_key = str(session.get("session_key") or "")
    marker_attempt = int(session.pop("_auto_continue_attempt", 0) or 0)
    marker_text = session.pop("_auto_continue_prompt", None) or text
    if isinstance(marker_text, str) and marker_text.strip():
        with session["history_lock"]:
            session["_active_turn_marker_key"] = marker_key
        record_turn_start(marker_home, marker_key, marker_text, attempts=marker_attempt)
        with session["history_lock"]:
            marker_cancelled = bool(session.get("_turn_cancel_requested"))
        if marker_cancelled:
            clear_turn_marker(marker_home, marker_key)
    return marker_key


@dataclasses.dataclass(slots=True)
class _TurnScopes:
    """Reset tokens for the thread/context scopes a turn binds (filled incrementally)."""

    approval: Any = None
    session_tokens: list = dataclasses.field(default_factory=list)
    home: Any = None  # per-turn HERMES_HOME override for a resumed remote profile
    secret: Any = None
    terminal: Any = None


def _route_turn_images(agent, prompt: Any, images: list[str]) -> Any:
    """Run message for a turn with attached images: "native" content parts, or "text" path
    references the agent analyzes in-loop (never blocking submit on vision calls).
    Decision table: agent/image_routing.py."""
    try:
        from agent.image_routing import build_native_content_parts, decide_image_input_mode
        from hermes_cli.config import load_config as _tui_load_config
        _provider, _model = _active_image_routing_identity(agent)
        mode = decide_image_input_mode(
            _provider, _model, _tui_load_config(),
            requested_provider=getattr(agent, "requested_provider", ""))
        if getattr(agent, "api_mode", "") == "codex_app_server":
            mode = "text"
    except Exception as _img_exc:
        print(f"[tui_gateway] image_routing decision failed, defaulting to text: {_img_exc}",
              file=sys.stderr)
        mode = "text"
    if mode != "native":
        return _build_image_ref_message(prompt, images)
    try:
        parts, skipped = build_native_content_parts(prompt, images)
        if skipped:
            print(
                f"[tui_gateway] native image attachment skipped {len(skipped)} unreadable path(s)",
                file=sys.stderr)
        if any(p.get("type") == "image_url" for p in parts):
            return parts
    except Exception as _img_exc:
        print(f"[tui_gateway] native attach failed, falling back to text: {_img_exc}",
              file=sys.stderr)
    return _build_image_ref_message(prompt, images)


def _start_turn_voice() -> tuple[Any, bool]:
    """Arm voice-mode turn audio; ``(tts_queue, thinking_started)``.  ``_tts_stream_begin``
    goes first: cutting a still-speaking previous turn IS this turn's barge-in, so it must
    latch before the caller consumes the latch."""
    tts_queue = _tts_stream_begin()
    if not _voice_mode_enabled():
        return tts_queue, False
    if _voice_cfg_dict().get("barge_in", True):
        _arm_full_duplex_listener()
    try:
        from tools.voice_mode import is_audio_output_active, start_thinking_sound

        def _thinking_should_play() -> bool:
            if is_audio_output_active():
                return False
            try:
                from hermes_cli.voice import is_continuous_active
                return not is_continuous_active()
            except Exception:
                return True
        return tts_queue, start_thinking_sound(should_play=_thinking_should_play)
    except Exception:
        return tts_queue, False


def _commit_turn_history(
    session: dict, result: dict, history: list, history_version: int) -> str | None:
    """Write the agent's messages back to session history; returns a client warning or None.
    If history_version moved mid-turn, the only tolerated mutation is a gateway-inserted
    pivot marker (compare content, not indices: ``_append_model_switch_marker`` strips prior
    markers in place); any other desync is surfaced, never dropped."""
    with session["history_lock"]:
        current_version = int(session.get("history_version", 0))
        if current_version == history_version:
            session["history"] = result["messages"]
            session["history_version"] = history_version + 1
            return None
        # History mutated externally during the turn. Check if the only mutation was a pivot marker the
        # gateway itself inserted mid-turn (#76870). If so the agent output is still valid — merge it into
        # the current history that now contains the marker. A personality change counts here too: unlike a
        # model switch it has no pending queue, so `/personality` during a running turn lands immediately
        # and used to read as a genuine desync, dropping the finished turn (#82756).
        # _append_model_switch_marker strips prior markers in-place then appends a new one, so the delta is
        # NOT a simple tail-slice — we must compare content, not indices.
        current_history = list(session["history"])
        history_no_markers = [e for e in history if not _is_pivot_marker(e)]
        current_no_markers = [e for e in current_history if not _is_pivot_marker(e)]
        if current_no_markers == history_no_markers and any(
                _is_pivot_marker(e) for e in current_history):
            # Auto-compression can leave the result shorter than the turn-start history.
            msgs = result["messages"]
            new_messages = msgs[len(history):] if len(msgs) > len(history) else list(msgs)
            session["history"] = current_history + new_messages
            session["history_version"] = current_version + 1
            return None
        print(
            f"[tui_gateway] prompt.submit: history_version mismatch "
            f"(expected={history_version} current={current_version}) — "
            f"agent output NOT written to session history",
            file=sys.stderr)
        return (
            "History changed during this turn — the response above is visible "
            "but was not saved to session history.")


def _result_status(result: dict) -> str:
    return (
        "interrupted" if result.get("interrupted")
        else "error" if result.get("error") else "complete")


def _turn_outcome(result: Any) -> tuple[Any, str, str | None]:
    """Reduce a run_conversation result to ``(raw_text, status, last_reasoning)``."""
    if not isinstance(result, dict):
        return str(result), "complete", None
    raw = result.get("final_response", "")
    status = _result_status(result)
    # No visible response AND a real error: surface the error as the text (classic CLI
    # parity).  An empty successful turn still renders as empty.
    if (not raw) and result.get("error") and (result.get("failed") or result.get("partial")):
        raw = f"Error: {result.get('error')}"
    # "Operation interrupted: waiting for model response (…)" is cancellation
    # metadata, not assistant prose (gateway/run.py and ACP suppress it too).
    # "Operation interrupted: waiting for model response (…)" is cancellation metadata, not assistant prose.
    # gateway/run.py and the ACP adapter already suppress this sentinel; without this the desktop paints it
    # as the agent's reply whenever a stop/steer lands mid-request (#7921).
    if status == "interrupted" and isinstance(raw, str) and raw.strip().startswith(
            INTERRUPT_WAITING_FOR_MODEL_PREFIX):
        raw = ""
    lr = result.get("last_reasoning")
    last_reasoning = lr.strip() if isinstance(lr, str) and lr.strip() else None
    return raw, status, last_reasoning


def _goal_followup_after_turn(
    sid: str, session: dict, result: Any, status: str, raw: Any) -> str | None:
    """/goal continuation (mirrors gateway/run._post_turn_goal_continuation): the prompt to
    chain once ``running`` is released, or None.  Compression failures are never judge
    input: the error text is not work toward the goal, and judging it spends a turn."""
    goal_followup = None
    compression_exhausted = bool(isinstance(result, dict) and result.get("compression_exhausted"))
    try:
        recovery_prompt, recovery_notice = _plan_goal_compression_recovery(
            session, result, status=status, raw=raw)
        if recovery_notice:
            _emit("status.update", sid, {"kind": "goal", "text": recovery_notice})
        goal_followup = recovery_prompt or None
    except Exception as _goal_recovery_exc:
        _hook_failure("goal compression recovery", _goal_recovery_exc)
    if compression_exhausted or not _is_successful_goal_turn(result, status, raw):
        return goal_followup
    try:
        if session.get("session_key") and (goal_mgr := _active_goal_manager(session)) is not None:
            try:
                from hermes_cli.goals import gather_background_processes as _gather_bg
                _bg_procs = _gather_bg()
            except Exception:
                _bg_procs = None
            decision = goal_mgr.evaluate_after_turn(
                raw, user_initiated=True, background_processes=_bg_procs)
            if verdict_msg := decision.get("message") or "":
                _emit("status.update", sid, {"kind": "goal", "text": verdict_msg})
            if decision.get("should_continue") and (
                cont_prompt := decision.get("continuation_prompt") or ""):
                goal_followup = cont_prompt
    except Exception as _goal_exc:
        _hook_failure("goal continuation hook", _goal_exc)
    return goal_followup


def _after_complete_turn(sid: str, session: dict, st: _TurnRun, raw: Any) -> None:
    """Hooks for a ``complete`` turn: /loop tick evaluation, pending title, voice fallback."""
    try:
        from hermes_cli.loops import LoopManager
        loop_sid_key = session.get("session_key") or ""
        if loop_sid_key:
            loop_mgr = LoopManager(session_id=loop_sid_key)
            loop_state = loop_mgr.state
            if loop_state is not None and loop_state.awaiting_response:
                loop_decision = loop_mgr.complete_tick(raw if isinstance(raw, str) else "")
                if loop_msg := loop_decision.get("message") or "":
                    _emit("status.update", sid, {"kind": "loop", "text": loop_msg})
    except Exception as _loop_exc:
        _hook_failure("loop completion hook", _loop_exc)
    # Apply pending_title now that the DB row exists — in the session-owned profile store.
    if _pending := session.get("pending_title"):
        _session_key = session.get("session_key") or sid
        try:
            with _session_db(session) as _pdb:
                if _pdb and _pdb.set_session_title(_session_key, _pending):
                    session["pending_title"] = None
        except ValueError as exc:
            # Invalid/duplicate title — non-retryable, drop it; auto-title takes over.
            session["pending_title"] = None
            logger.info("Dropping pending title for session %s: %s", _session_key, exc)
        except Exception:
            pass  # transient DB failure — keep pending_title for retry
    # Voice fallback when the streaming pipeline couldn't start (tts_queue already spoke
    # everything otherwise); barge-aware.
    if st.tts_queue is None and isinstance(raw, str) and raw.strip() and _voice_tts_enabled():
        try:
            threading.Thread(target=_speak_text_with_barge, args=(raw,), daemon=True).start()
        except ImportError:
            logger.warning("voice TTS skipped: hermes_cli.voice unavailable")
        except Exception as e:
            logger.warning("voice TTS dispatch failed: %s", e)


def _dispatch_followup_turn(rid, sid: str, session: dict, prompt: Any, what: str, *,
                            on_done=None, on_error=None) -> None:
    """Chain one follow-up turn (caller set ``running``); on failure run ``on_error``, log,
    release ``running``."""
    try:
        _emit("message.start", sid)
        _run_prompt_submit(rid, sid, session, prompt)
        if on_done is not None:
            on_done()
    except Exception as exc:
        if on_error is not None:
            on_error()
        _hook_failure(what, exc)
        with session["history_lock"]:
            session["running"] = False


def _run_post_turn_followups(
    rid, sid: str, session: dict, result: Any, goal_followup: str | None) -> None:
    """Chain whatever should run after ``running`` was released.  Order: a mid-turn user
    prompt wins over every auto follow-up (drain it, skip the rest); a leftover /steer is
    requeued first so it isn't dropped; then goal continuation, then completion
    notifications.  Each nested submit re-checks ``running`` under the lock."""
    steer = result.get("pending_steer") if isinstance(result, dict) else None
    if isinstance(steer, str) and steer.strip():
        with session["history_lock"]:
            _enqueue_prompt(session, steer, session.get("transport"))
    if _drain_queued_prompt(rid, sid, session):
        return
    if goal_followup:
        with session["history_lock"]:
            if session.get("running"):
                return  # user already sent something — their turn wins
            session["running"] = True
        _dispatch_followup_turn(rid, sid, session, goal_followup, "goal continuation dispatch")
    # Safety net for completion events that arrived mid-turn.  Ownership is positive-proof
    # and compression-chain aware (same fail-closed gate as the poller): session B must
    # not consume session A's event.  Unclaimable events are requeued for the poller.
    try:
        from tools.process_registry import process_registry
        drained = process_registry.drain_notifications(
            session_key=session.get("session_key", ""),
            owns_event=lambda e: _session_owns_notification_event(sid, session, e),
            skip_poll_observed=False)
        for index, (_evt, synth) in enumerate(drained):
            with session["history_lock"]:
                if session.get("running"):
                    for pending_evt, _pending_synth in drained[index:]:
                        process_registry.completion_queue.put(pending_evt)
                    break
                session["running"] = True
            from tools.async_delegation import (
                claim_event_delivery, complete_event_delivery, release_event_delivery)
            _claim = claim_event_delivery(_evt, "tui-post-turn")
            if _claim is None:
                continue
            _dispatch_followup_turn(
                rid, sid, session, synth, "completion notification dispatch",
                on_done=lambda: complete_event_delivery(_evt, _claim),
                on_error=lambda: release_event_delivery(_evt, _claim))
    except Exception as _drain_exc:
        _hook_failure("completion queue drain", _drain_exc)


@dataclasses.dataclass(slots=True)
class _TurnRun:
    """Shared state of one turn thread.  ``agent`` is bound eagerly so except/finally always
    have one; ``error_retained`` makes the finally keep the failed inflight snapshot for
    resume replay; ``error_detail`` is the "tui turn finished" failure cause."""

    agent: Any
    one_turn_restore: Any
    terminal_callback: Any
    receipt_committed: bool
    scopes: _TurnScopes = dataclasses.field(default_factory=_TurnScopes)
    result: Any = None  # read after the finally for leftover /steer
    tts_queue: Any = None
    thinking_started: bool = False
    history: list = dataclasses.field(default_factory=list)
    history_version: int = 0
    run_kwargs: Any = None
    error_retained: bool = False
    error_detail: str = ""
    prompt_text: str = ""
    marker_key: str = ""
    receipt_attempted: bool = False


def _prepare_turn_input(sid: str, session: dict, st: _TurnRun, text: Any, images: list[str]):
    """Bind scopes, sync the agent, snapshot history, build the run message; returns
    ``(prompt, run_message, cols, streamer)`` or None when @-expansion was refused.
    Scopes fill field by field so a failure midway still leaves every bound token for the
    finally; the profile's terminal policy is bound too (a failed install leaves a
    fail-closed refusal scope).  The config-model sync is skipped under a /model --once
    override (not pinned as model_override, the sync would clobber it); a model picked
    mid-turn is applied first so the explicit pick wins over a config change."""
    from tools.approval_context import set_current_session_key
    scopes = st.scopes
    scopes.approval = set_current_session_key(session["session_key"])
    scopes.session_tokens = _set_session_context(session["session_key"], ui_session_id=sid)
    profile_home = session.get("profile_home")
    if profile_home:
        scopes.home = set_hermes_home_override(profile_home)
        scopes.secret = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
        from tools.terminal_scope import install_profile_terminal_scope
        scopes.terminal = install_profile_terminal_scope(Path(profile_home))
    # The sudo password callback is thread-local: without re-wiring here, sudo prompts
    # fall through to /dev/tty and hang the headless gateway (re-run is a no-op).
    _wire_callbacks(sid)
    if not st.one_turn_restore:
        # Skip the config-model sync while a /model --once override is active: the once-model is
        # intentionally not pinned as a session model_override (it must not persist), so without this guard
        # the sync would see "agent model != config model" and clobber the once-override back to the config
        # model before the turn runs (#29923 review defect). Any config.yaml change is adopted on the NEXT
        # turn, after the finally-restore below.
        _apply_pending_model_switch(sid, session)
        _sync_agent_model_with_config(sid, session)
        _sync_agent_compression_with_config(sid, session)
    _sync_bot_capabilities(sid, session)  # Bot Chat: adopt Settings->Capabilities edits
    st.agent = agent = session["agent"]
    # Snapshot after the model sync: a deferred switch's history mutation belongs to this turn.
    with session["history_lock"]:
        st.history = list(session["history"])
        st.history_version = int(session.get("history_version", 0))
    cwd = _session_cwd(session)
    _register_session_cwd(session)
    cols = session.get("cols", 80)
    streamer = make_stream_renderer(cols)
    prompt = text
    if isinstance(prompt, str) and "@" in prompt:
        from agent.context_references import preprocess_context_references
        from agent.model_metadata import get_model_context_length
        ctx_len = get_model_context_length(
            getattr(agent, "model", "") or _resolve_model(),
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            provider=getattr(agent, "provider", "") or "",
            config_context_length=getattr(agent, "_config_context_length", None))
        ctx = preprocess_context_references(
            prompt, cwd=cwd, allowed_root=cwd, context_length=ctx_len)
        if ctx.blocked:
            _emit(
                "error", sid, {"message": "\n".join(ctx.warnings) or "Context injection refused."})
            return None
        prompt = ctx.message
    st.prompt_text = prompt if isinstance(prompt, str) else ""
    run_message: Any = _route_turn_images(agent, prompt, images) if images else prompt
    st.tts_queue, st.thinking_started = _start_turn_voice()
    # Per-turn API-message notes: barge mid-speech, reactions, HUD surface (per-turn state
    # that must not touch the byte-stable system prompt).
    from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE, take_speech_interrupted
    if take_speech_interrupted():
        run_message = _prepend_note(run_message, SPEECH_INTERRUPTED_NOTE)
    run_message = _prepend_note(run_message, _pending_reaction_notes(session))
    return prompt, _prepend_note(run_message, _hud_surface_note(session)), cols, streamer


def _invoke_agent(
    sid: str, session: dict, st: _TurnRun, prompt: Any, run_message: Any, streamer,
    images: list[str], display_kind: str | None, display_metadata: dict | None) -> None:
    """Wire the streaming callbacks and run the conversation into ``st.result``."""
    agent = st.agent

    def _stream(delta):
        with session["history_lock"]:
            _append_inflight_delta(session, delta)
        payload = {"text": delta}
        if streamer and (r := streamer.feed(delta)) is not None:
            payload["rendered"] = r
        if st.tts_queue is not None and isinstance(delta, str):
            st.tts_queue.put(delta)
        _emit("message.delta", sid, payload)

    # Interim assistant text (commentary beside tool calls, pre-nudge final answer) is sealed
    # by the desktop as its own segment instead of being lost to message.complete.
    def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
        _emit("message.interim", sid, {"text": text, "already_streamed": already_streamed})
    agent.interim_assistant_callback = (
        _interim_assistant_cb if _load_interim_assistant_messages() else None)
    # A synthesized turn is typed at turn START so a crash persist writes a timeline event,
    # not a raw user bubble; the post-turn stamp is the fallback for an older agent.
    st.run_kwargs = run_kwargs = {
        "conversation_history": list(st.history),
        "stream_callback": _stream,
        "persist_user_message": (
            _build_persist_user_message(prompt, images, run_message) if images else prompt)}
    try:
        run_params = inspect.signature(agent.run_conversation).parameters
    except (TypeError, ValueError):
        run_params = {}
    if "task_id" in run_params:
        run_kwargs["task_id"] = session["session_key"]
    if display_kind and "persist_user_display_kind" in run_params:
        run_kwargs["persist_user_display_kind"] = display_kind
        run_kwargs["persist_user_display_metadata"] = display_metadata
    # Live-rename hook: auto-titling fires inside the turn prologue.
    _title_key = session.get("session_key") or sid
    agent._on_session_title = lambda t, _src, _k=_title_key: _emit(
        "session.title", sid, {"session_id": _k, "title": t})
    _usage_stop, _usage_thread = _start_usage_ticker(sid, agent)
    try:
        st.result = agent.run_conversation(run_message, **st.run_kwargs)
    finally:
        # Stop AND join before anything emits: a tick surviving past message.complete would
        # roll the client's usage back to a stale snapshot (unbounded join: same worst case).
        _usage_stop.set()
        _usage_thread.join()


def _absorb_turn_result(
    sid: str, session: dict, st: _TurnRun, text: Any, display_kind: str | None, display_metadata
) -> str | None:
    """Stamp, restore /moa, commit history, re-sync the session key; returns the history warning."""
    result, agent = st.result, st.agent
    if display_kind and isinstance(text, str):
        # Post-turn fallback stamp of a synthesized turn's display kind (DB row + result).
        db = getattr(agent, "_session_db", None)
        current_session_id = getattr(agent, "session_id", None) or session.get("session_key")
        if db is not None:
            try:
                db.set_latest_matching_message_display_kind(
                    current_session_id, role="user", content=text, display_kind=display_kind,
                    display_metadata=display_metadata)
            except Exception:
                logger.debug("failed to stamp synthetic display kind", exc_info=True)
        if isinstance(result, dict) and isinstance(result.get("messages"), list):
            for message in reversed(result["messages"]):
                if message.get("role") == "user" and message.get("content") == text:
                    message["display_kind"] = display_kind
                    if display_metadata:
                        message["display_metadata"] = display_metadata
                    break
    if "moa_one_shot_restore" in session:
        # Undo a /moa one-shot through the switch path: resetting model_override alone
        # would leave the live client pinned to MoA after the in-place switch_model().
        _restore = session.pop("moa_one_shot_restore", None)
        # Restore the model the user was on before the /moa one-shot. See #53444.
        if isinstance(_restore, dict):
            _prev_override = _restore.get("override")
            _prev_model = _restore.get("model")
            _prev_provider = _restore.get("provider")
            if _prev_override is None:
                session.pop("model_override", None)
            else:
                session["model_override"] = _prev_override
            if _prev_model:
                _raw = (
                    f"{_prev_model} --provider {_prev_provider}" if _prev_provider else _prev_model)
                try:
                    _apply_model_switch(
                        sid, session, _raw, confirm_expensive_model=False,
                        pin_session_override=bool(_prev_override),
                        persist_override=False)  # session-internal restore, never config.yaml
                except Exception as _moa_restore_exc:
                    logger.warning("MoA one-shot model restore failed: %s", _moa_restore_exc)
        elif _restore is None:
            session.pop("model_override", None)
        else:
            session["model_override"] = _restore
    status_note = None
    if isinstance(result, dict):
        if isinstance(result.get("messages"), list):
            status_note = _commit_turn_history(session, result, st.history, st.history_version)
        # Auto-compression may have rotated agent.session_id: sync session_key before
        # title/goal/finalize use it, keep pending_title (user intent), restart the slash
        # worker so worker-backed commands target the live session.
        # Fix for #20001.
        _sync_session_key_after_compress(
            sid, session, clear_pending_title=False, restart_slash_worker=True)
    return status_note


def _complete_turn_payload(session: dict, st: _TurnRun, status_note: str | None, cols: int):
    """``(payload, raw, status)`` for message.complete; retains/clears the inflight turn and
    settles the hosted-room terminal receipt."""
    result, agent = st.result, st.agent
    raw, status, last_reasoning = _turn_outcome(result)
    payload = {"text": raw, "usage": _get_usage(agent), "status": status}
    if last_reasoning:
        payload["reasoning"] = last_reasoning
    if status_note:
        payload["warning"] = status_note
    if result.get("response_previewed"):
        payload["response_previewed"] = True
    # Structured billing-wall descriptor: the client renders recovery without re-parsing text.
    if _billing_block := result.get("billing_block"):
        payload["billing"] = _billing_block
        payload["failure_reason"] = result.get("failure_reason")
    if rendered := render_message(raw, cols):
        payload["rendered"] = rendered
    # Advisory {layer, code, retryable} descriptor; computed before the retain so resume
    # replay carries the same one.
    _error_surface = None
    if status == "error":
        try:
            from agent.error_surface import build_error_surface_from_result
            _error_surface = build_error_surface_from_result(
                result, provider=str(getattr(agent, "provider", "") or ""),
                model=str(getattr(agent, "model", "") or ""))
        except Exception:
            _error_surface = None
    error_value = result.get("error")
    with session["history_lock"]:
        if status == "error":
            # Retain the failed turn: resume's inflight payload is the only carrier of the
            # failure if this frame is lost to a disconnect.
            _fail_inflight_turn(session, error_value, error_surface=_error_surface)
            st.error_retained = True
            st.error_detail = _turn_failure_detail(
                error_value, result.get("failure_reason"), st.prompt_text)
        else:
            _clear_inflight_turn(session)
    if status == "error":
        payload["error"] = str(error_value or raw)
        payload["recoverable"] = True
        if _error_surface:
            payload["error_surface"] = _error_surface
    if st.terminal_callback is not None:
        st.receipt_attempted = True
        st.terminal_callback({
            "status": {"interrupted": "cancelled", "error": "failed"}.get(status, "settled"),
            "text": raw if isinstance(raw, str) else str(raw),
            **({"error": str(error_value or raw)} if status == "error" else {})})
        st.receipt_committed = True
    if st.receipt_committed:
        _retire_turn_marker(session, st.marker_key)
    return payload, raw, status


def _recover_turn_exception(sid: str, session: dict, st: _TurnRun, e: BaseException) -> None:
    """Except-path of the turn: crash log, history restore, terminal error frame."""
    import traceback
    with contextlib.suppress(Exception):
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== turn-dispatcher exception · "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} · sid={sid} ===\n")
            f.write(traceback.format_exc())
    print(f"[gateway-turn] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    # A finalizer exception can leave in-memory history at the turn-start snapshot.
    _restore_agent_history_after_turn_error(session, st.agent)
    if st.terminal_callback is not None and not st.receipt_attempted:
        st.receipt_attempted = True
        try:
            st.terminal_callback({"status": "failed", "text": "", "error": str(e)})
            st.receipt_committed = True
        except Exception:
            logger.exception("hosted room terminal receipt commit failed")
    try:
        # Same terminal error frame shape as the returned-error path.
        _emit_terminal_turn_error(sid, session, e, retire_marker=st.receipt_committed)
        st.error_retained = True
        st.error_detail = _turn_failure_detail(e, type(e).__name__, st.prompt_text)
    except Exception as emit_exc:
        print(
            f"[gateway-turn] terminal error emit failed: {type(emit_exc).__name__}: {emit_exc}",
            file=sys.stderr, flush=True)
        _emit("error", sid, {"message": str(e)})


def _finish_turn(sid: str, session: dict, st: _TurnRun) -> None:
    """Finally-path of the turn: release everything, then the "tui turn finished" bookend."""
    # Drop both pre-turn history snapshots before asking glibc to return pages (a test
    # inspects these two locals by name).
    history, run_kwargs = st.history, st.run_kwargs
    history.clear()
    if isinstance(run_kwargs, dict):
        run_kwargs.clear()
    try:  # while the profile HERMES_HOME override is still active (session's own config)
        from hermes_cli.mem_trim import trim_memory
        trim_memory(reason="tui turn completion")
    except Exception:
        logger.debug("post-turn memory trim failed", exc_info=True)
    if st.thinking_started:
        with contextlib.suppress(Exception):
            from tools.voice_mode import stop_thinking_sound
            stop_thinking_sound()
    if st.tts_queue is not None:
        st.tts_queue.put(None)  # end-of-text sentinel — flush + finish speaking
    if st.one_turn_restore:
        try:
            _restore_agent_model_runtime(st.agent, st.one_turn_restore)
            _restart_slash_worker(sid, session)
            _persist_live_session_runtime(session)
            _persist_live_session_system_prompt(session)
        except Exception:
            logger.debug("TUI one-turn model restore failed", exc_info=True)
    scopes = st.scopes
    with contextlib.suppress(Exception):
        if scopes.approval is not None:
            from tools.approval_context import reset_current_session_key
            reset_current_session_key(scopes.approval)
    if scopes.home is not None:
        reset_hermes_home_override(scopes.home)
    if scopes.secret is not None:
        reset_secret_scope(scopes.secret)
    if scopes.terminal is not None:
        from tools.terminal_scope import reset_terminal_scope
        reset_terminal_scope(scopes.terminal)
    _clear_session_context(scopes.session_tokens)


def _run_prompt_submit(
    rid, sid: str, session: dict, text: Any, *, display_kind: str | None = None,
    display_metadata: dict | None = None, image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None,
    terminal_callback: Callable[[dict[str, Any]], None] | None = None) -> bool:
    admitted = _admit_prompt_turn(sid, session, text, image_paths, queued_prompt_generation)
    if admitted is None:
        return False
    images, agent = admitted
    # The ONE INFO record proving a prompt was accepted by THIS process; ties ui sid,
    # session_key and the agent's live session_id together.  No prompt content is logged.
    _turn_started_monotonic = time.monotonic()
    logger.info(
        # Desktop/TUI observability (#86647): this is the ONE INFO record proving a Desktop/TUI prompt was
        # accepted by THIS process, and it ties together every id a rotation-mute trace needs — the UI
        # session id, the gateway session_key, and the agent's live session_id (which compression rotates
        # independently of the other two). Before this line a Desktop request left no trace in agent.log at
        # all ("0 platform=desktop" — see #86647), so a muted window was structurally indistinguishable from
        # a request that never arrived.
        "tui prompt accepted: ui_session=%s session_key=%s agent_session_id=%s "
        "kind=%s chars=%s images=%d",
        sid, session.get("session_key") or "", getattr(agent, "session_id", "") or "",
        display_kind or "user", len(text) if isinstance(text, str) else "-", len(images))
    _emit("message.start", sid)

    def run():
        # RPC-dispatcher ContextVars do not follow onto this thread: rebind the transport
        # before any tool can commission a child (delegate_task captures it as authority).
        transport_token = bind_transport(session.get("transport"))
        runtime_session_token = _current_runtime_session_record.set(session)
        st = _TurnRun(
            session["agent"], session.pop("one_turn_model_restore", None), terminal_callback,
            receipt_committed=terminal_callback is None)
        st.marker_key = _record_turn_marker(session, text)
        goal_followup = None
        try:
            prepared = _prepare_turn_input(sid, session, st, text, images)
            if prepared is None:
                return
            prompt, run_message, cols, streamer = prepared
            _invoke_agent(
                sid, session, st, prompt, run_message, streamer, images, display_kind,
                display_metadata)
            status_note = _absorb_turn_result(
                sid, session, st, text, display_kind, display_metadata)
            payload, raw, status = _complete_turn_payload(session, st, status_note, cols)
            _emit("message.complete", sid, payload)
            goal_followup = _goal_followup_after_turn(sid, session, st.result, status, raw)
            if status == "complete":
                _after_complete_turn(sid, session, st, raw)
        except Exception as e:
            _recover_turn_exception(sid, session, st, e)
        finally:
            _finish_turn(sid, session, st)
            _current_runtime_session_record.reset(runtime_session_token)
            reset_transport(transport_token)
            # A stale interim closure must not fire during a later turn.
            st.agent.interim_assistant_callback = None
            with session["history_lock"]:
                session["running"] = False
                session["last_active"] = time.time()
                if not st.error_retained:
                    _clear_inflight_turn(session)
            # Closing bookend of "tui prompt accepted" — exactly one per accepted prompt.
            # agent.session_id is re-read because compression may have rotated it (an
            # accepted/finished pair whose id changed IS a rotation trace).
            if isinstance(st.result, dict):
                status = _result_status(st.result)
            else:
                status = "error" if st.error_retained else "complete"
            logger.info(
                "tui turn finished: ui_session=%s session_key=%s agent_session_id=%s status=%s "
                "error_retained=%s duration=%.1fs%s",
                sid, session.get("session_key") or "", getattr(st.agent, "session_id", "") or "",
                status, st.error_retained, time.monotonic() - _turn_started_monotonic,
                st.error_detail)
            # Backstop for turns that never reached a terminal frame.
            if st.receipt_committed:
                _retire_turn_marker(session, st.marker_key)
                with session["history_lock"]:
                    if session.get("_active_turn_marker_key") == st.marker_key:
                        session.pop("_active_turn_marker_key", None)
                    session.pop("_hosted_room_task", None)
            session.pop("_auto_continue_scheduled", None)
            _emit_settled_session_info(sid, session, st.agent)
        _run_post_turn_followups(rid, sid, session, st.result, goal_followup)
    run_thread = threading.Thread(target=run, daemon=True)
    with _sessions_lock:
        registered = _sessions.get(sid)
        can_start = not session.get("_closing") and (registered is None or registered is session)
        if can_start:
            session["_run_thread"] = run_thread
            run_thread.start()
    if not can_start:
        with session["history_lock"]:
            session["running"] = False
    return can_start


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
