"""Auto-continue: resume a turn killed by a process/machine death, plus queued-prompt drain and
busy-submit handling. Bodies are rebound onto server.py's globals at install time
(method_ctx.bind_module), so they reference server.py globals bare."""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module

# A concluded turn (success, handled error, interrupt) clears its durable marker (turn_marker.py) in _run_prompt_submit's
# finally; only a process death leaves it behind, so a marker at session.resume proves the turn never finished AND the
# client never saw a terminal frame. Fresh: re-submit automatically (as the messaging gateway does). Stale: clear it
# and let the partial transcript speak.
# If the interruption is fresh, re-submit the interrupted prompt automatically (the messaging gateway has
# done this for restart-interrupted sessions since #27856); if it's stale, clear the marker and let the
# recovered partial transcript speak for itself — the user can ask to continue manually.
_AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT = 15


def _auto_continue_config() -> tuple[bool, float, int]:
    """(enabled, freshness window in seconds, max attempts) from ``desktop.auto_continue`` in config.yaml."""
    desktop = _load_cfg().get("desktop")
    cfg = desktop.get("auto_continue") if isinstance(desktop, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    try:
        minutes = float(cfg.get("freshness_minutes", _AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT))
    except (TypeError, ValueError):
        minutes = float(_AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT)
    return (is_truthy_value(cfg.get("enabled"), default=True), max(0.0, minutes) * 60.0,
            _coerce_int_config_value(cfg.get("max_attempts"), 2, min_value=0))


def _session_home(session: dict) -> Path:
    """The HERMES_HOME the session's durable state lives in (profile-aware)."""
    return Path(session.get("profile_home") or _hermes_home)


def _retire_turn_marker(session: dict, *keys: str) -> None:
    """Drop the crash marker right before the terminal frame (not at turn-thread end: post-turn work outlives the
    client's answer, and quitting in that window would leave a marker that re-runs a finished turn). Extra ``keys``
    cover a session_key that compression rotated mid-turn."""
    home = _session_home(session)
    for key in dict.fromkeys((*keys, str(session.get("session_key") or ""))):
        if key:
            clear_turn_marker(home, key)


def _auto_continue_note(prompt: str) -> str:
    # Same opening as the gateway's recovery notes (transcript tooling recognizes both). The prompt is embedded: a hard
    # crash persists nothing else of the turn.
    return (f"{_AUTO_CONTINUE_NOTE_PREFIX} — the app or its backend process stopped before the turn could finish. "
            "Some of the work may already be complete; check the current state before redoing anything, then "
            f"finish the task. The interrupted request was:]\n\n{prompt}")


def _maybe_schedule_auto_continue(sid: str, session: dict, session_key: str) -> dict | None:
    """Kick off a continuation turn for a crash-interrupted session (session.resume cold paths). Returns a descriptor
    for the resume payload when scheduled, else None. The turn runs on a background thread after the deferred agent
    build via _run_prompt_submit, so the client that just resumed streams it."""
    # Hosted room turns are recovered by their durable task/lease state machine; generic auto-continue would bypass
    # its execution generation and duplicate work.
    if session.get("source") == "bot_room":
        return None
    home = _session_home(session)
    if (marker := read_turn_marker(home, session_key)) is None:
        return None
    enabled, freshness_secs, max_attempts = _auto_continue_config()
    age = time.time() - marker["started_at"]
    if not enabled or age > freshness_secs or marker["attempts"] >= max_attempts:
        clear_turn_marker(home, session_key)  # stale/disabled/crash-looping: a manual message continues
        return None
    if session.get("_auto_continue_scheduled"):
        return None
    session["_auto_continue_scheduled"] = True
    attempt, text = marker["attempts"] + 1, _auto_continue_note(marker["prompt"])

    def kickoff() -> None:
        rid = f"__auto_continue__{int(time.time() * 1000)}"
        try:
            _start_agent_build(sid, session)
            err = _wait_agent(session, rid, timeout=120.0)
        except Exception:
            logger.warning("auto-continue agent build failed for %s", sid, exc_info=True)
            err = {"error": {"message": "agent build failed"}}
        if err:  # leave the marker: the next resume retries (bounded by attempts)
            session["_auto_continue_scheduled"] = False
            return
        with session["history_lock"]:
            if session.get("running") or session.get("_turn_cancel_requested") or session.get("_finalized"):
                session["_auto_continue_scheduled"] = False  # a real user prompt beat us; it clears the marker
                return
            session["running"] = True
            session["last_active"] = time.time()
        # Ownership admission BEFORE message.start: a sibling backend sharing this HERMES_HOME may have written the
        # marker and still be mid-turn. Leave the marker so a later resume retries.
        # Running the continuation anyway would be the double-writer this fence exists to prevent. See
        # #94778.
        if _ensure_active_session_slot(sid, session) is not None:
            logger.info("auto-continue for %s refused: session has another live owner", session_key)
            with session["history_lock"]:
                session["running"] = False
                session["_auto_continue_scheduled"] = False
            return
        with session["history_lock"]:
            # Marker inputs read back by _run_prompt_submit: attempt count (crash breaker) and the ORIGINAL prompt (no
            # nested notes). Set here, not at schedule time, so a bail above leaves nothing for a racing user turn.
            session["_auto_continue_attempt"], session["_auto_continue_prompt"] = attempt, marker["prompt"]
        try:
            _emit("status.update", sid, {"kind": "process", "text": "Resuming interrupted turn…"})
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text, display_kind="auto_continue")
        except Exception as exc:
            _notif_log_failure("auto-continue dispatch failed", exc)
            _notif_release_turn(session)  # rebound from session_notifications
    threading.Thread(target=kickoff, daemon=True).start()
    logger.info("auto-continue scheduled for session %s (attempt %d, interrupted %.0fs ago)", session_key, attempt, age)
    return {"attempt": attempt, "interrupted_at": marker["started_at"]}


def _ac_inflight_original(session: dict) -> str:
    turn = session.get("inflight_turn")
    return str(turn.get("user") or "").strip() if isinstance(turn, dict) else ""


def _enqueue_prompt(session: dict, text: Any, transport: Any, image_paths: list[str] | None = None) -> None:
    """Queue a message for the next turn. Text-only arrivals share a slot and merge losslessly (like the
    consecutive-user merge in ``repair_message_sequence``); image-bearing ones stay separate envelopes so attachment
    chronology survives. ``transport`` is pinned so the drained turn streams to its sender."""
    image_paths = list(image_paths or [])
    # Scrub live-turn self-duplicates first so the text merge below can't glue "{original}\n\n{later}" and re-fire the
    # original after a correction settles.
    # See #84417.
    _drop_queued_duplicates_of_inflight_user(session)
    text_only = not image_paths and isinstance(text, str)
    # Never queue a text-only self-copy of the live prompt: draining it would restart it.
    if text_only and text.strip() == _ac_inflight_original(session) != "":
        return
    queued = {"text": text, "transport": transport, **({"image_paths": image_paths} if image_paths else {})}
    existing = session.get("queued_prompt")
    if (existing and text_only and isinstance(existing.get("text"), str)
            and not existing.get("image_paths") and not session.get("queued_prompts")):
        prev = existing["text"]
        existing["text"] = f"{prev}\n\n{text}" if prev and text else (prev or text)
    elif existing:
        session.setdefault("queued_prompts", []).append(queued)
    else:
        session["queued_prompt"] = queued


def _sanitize_queued_entry_vs_inflight_user(entry: Any, original: str) -> dict | None:
    """Drop (``None``) a text-only self-duplicate of the live user text, or rewrite a merged slot
    ``"{original}\\n\\n{later}"`` to ``later`` so the correction survives without re-firing the original. Image-bearing
    envelopes are left alone (chronology is load-bearing).

    Returns ``None`` to drop the envelope, or a (possibly rewritten) dict to keep. A merged slot
    ``"{original}\\n\\n{later}"`` (from ``_enqueue_prompt``'s consecutive text merge) is rewritten to just
    ``later`` so a later correction is not lost and the original is not re-fired (#84417).
    """
    if not isinstance(entry, dict):
        return None
    text = entry.get("text")
    if not original or entry.get("image_paths") or not isinstance(text, str):
        return entry
    # A lossless text-merge may have glued the live original onto a later follow-up: keep the remainder.
    rest = next((text[len(original + sep):] for sep in ("\n\n", "\n") if text.startswith(original + sep)), text).strip()
    return None if not rest or rest == original else (entry if rest == text.strip() else {**entry, "text": rest})


def _drop_queued_duplicates_of_inflight_user(session: dict) -> None:
    """Remove server-queue copies of the live turn's original user text: a mid-turn ``prompt.submit`` of the same text
    queued while redirect was unavailable must not drain and restart the original.

    A mid-turn ``prompt.submit`` of the same text can land in ``queued_prompt`` when redirect is not yet
    available (model not active, build window, tool boundary). If the user then corrects the turn with a
    different prompt via redirect, that stale self-duplicate must not ``_drain_queued_prompt`` after the
    redirected turn completes — otherwise the original prompt restarts as a fresh agent turn (#84417).
    """
    if not (original := _ac_inflight_original(session)):
        return
    head = session.get("queued_prompt")
    cleaned = (_sanitize_queued_entry_vs_inflight_user(e, original)
               for e in ([head] if head else []) + list(session.get("queued_prompts") or []))
    _ac_set_queue(session, [c for c in cleaned if c is not None])


def _ac_set_queue(session: dict, entries: list) -> None:
    """Write ``entries`` back as queued_prompt (head) + queued_prompts (rest)."""
    session["queued_prompt"] = entries[0] if entries else None
    if len(entries) > 1:
        session["queued_prompts"] = entries[1:]
    else:
        session.pop("queued_prompts", None)


def _interrupt_busy_session(sid: str, session: dict, agent: Any) -> None:
    """Interrupt a busy turn on a worker thread, never under ``history_lock`` (some providers can't apply ``interrupt()``
    until a blocking call returns; inline it stalled ``session.resume``). At most one interrupt worker per session so
    repeated steering can't leak threads."""
    use_agent = agent is not None and hasattr(agent, "interrupt")
    if not use_agent and not _session_uses_compute_host(session):
        return
    with session["history_lock"]:
        if session.get("_busy_interrupt_pending"):
            return
        session["_busy_interrupt_pending"] = True

    def interrupt() -> None:
        try:
            with contextlib.suppress(Exception):
                agent.interrupt() if use_agent else _get_compute_host_supervisor().interrupt(sid)
        finally:
            with session["history_lock"]:
                session["_busy_interrupt_pending"] = False
    threading.Thread(target=interrupt, daemon=True, name=f"busy-interrupt-{sid}").start()


def _ac_try_correction(rid, session: dict, agent: Any, method: str, plain_text: str, status: str) -> dict | None:
    """Apply ``agent.<method>(plain_text)`` (steer/redirect); on acceptance record the correction, scrub stale
    self-duplicates so the live turn's original text is not re-fired after settle, and return the ``status`` reply.
    None → caller falls through to the queue path."""
    try:
        if not getattr(agent, method)(plain_text):
            return None
    except Exception:
        return None
    with session["history_lock"]:
        _record_inflight_correction(session, plain_text)
        _drop_queued_duplicates_of_inflight_user(session)
        session["last_active"] = time.time()
    return _ok(rid, {"status": status})


def _handle_busy_submit(rid, sid: str, session: dict, text: Any, transport: Any, queued: bool = False) -> dict | None:
    """Apply ``display.busy_input_mode`` to a mid-turn prompt instead of rejecting it (rejection made clients busy-retry
    and drop sends): ``interrupt`` (default) → redirect, falling back to hard interrupt + queue; ``queue`` → queue only;
    ``steer`` → inject after the current atomic action. ``queued=True`` (client queue drain) forces queue mode: a "run
    after" message must NEVER become a live correction."""
    mode = "queue" if queued else _load_busy_input_mode()
    agent = session.get("agent")
    with session["history_lock"]:
        if not session.get("running"):
            return None  # turn ended since prompt.submit's busy check; caller retries on the idle session
        image_paths = list(session.get("attached_images", []))
        if image_paths:
            session["attached_images"] = []  # claim now so a later paste isn't consumed when the turn yields
    plain_text = _coerce_message_text(text).strip() if not image_paths and _is_text_only_busy_payload(text) else ""
    # Text-only corrections steer/redirect in place when supported; media payloads and older agents fall through to
    # the proven interrupt + queue path.
    if plain_text and agent is not None:
        supported = {
            "steer": hasattr(agent, "steer"),
            "interrupt": getattr(agent, "_supports_active_turn_redirect", False) is True and hasattr(agent, "redirect")}
        method, status = {"steer": ("steer", "steered"), "interrupt": ("redirect", "redirected")}.get(mode, (None, None))
        if (method and supported[mode]
                and (resp := _ac_try_correction(rid, session, agent, method, plain_text, status)) is not None):
            return resp
    # Queue before asking the live turn to stop. Never call a provider/compute-host method under history_lock: an
    # interrupt can wait behind the op it cancels.
    with session["history_lock"]:
        if not session.get("running"):
            if image_paths:
                session["attached_images"] = image_paths + list(session.get("attached_images", []))
            return None
        _enqueue_prompt(session, text, transport, image_paths=image_paths)
        session["last_active"] = time.time()
    # Attachments need their own model invocation: queue without cancelling so the user gets both results in order.
    # ``steer`` must NEVER escalate to a hard interrupt: it would kill the live turn AND drop ``AIAgent._pending_steer``
    # (earlier accepted steers); steer fall-throughs stay FIFO-queued.
    # A burst of user messages while the agent is busy can land as a mix of accepted steers (stashed in
    # ``AIAgent._pending_steer``) and fall-through queue envelopes (payload not steerable, ``steer()``
    # rejected/raised). A hard interrupt here kills the live turn AND ``AIAgent.interrupt()`` drops the
    # pending steer buffer — silently destroying the earlier messages of the burst. See #86134.
    if mode == "interrupt" and not image_paths:
        _interrupt_busy_session(sid, session, agent)
    return _ok(rid, {"status": "queued"})


def _drain_queued_prompt(rid, sid: str, session: dict) -> bool:
    """Fire a queued next-turn prompt if one is waiting and the session is idle. True when dispatched: the caller
    skips lower-priority follow-ups this cycle (the user's message wins)."""
    with session["history_lock"]:
        if session.get("_closing") or not (queued := session.get("queued_prompt")) or session.get("running"):
            return False
        queue_generation = int(session.get("_queued_prompt_generation", 0))
        _ac_set_queue(session, session.get("queued_prompts") or [])
        session["running"] = True
        if queued.get("transport") is not None:
            session["transport"] = queued["transport"]
    use_compute_host = _session_uses_compute_host(session)
    with session["history_lock"]:
        if int(session.get("_queued_prompt_generation", 0)) != queue_generation:
            # Generation bump cancelled the claim (Stop, compress re-anchor, …): don't dispatch, but restore the
            # envelope (claimed head first, then whatever advanced into the slot) so a legitimate follow-up isn't dropped.
            # See #84417.
            advanced = session.get("queued_prompt")
            _ac_set_queue(session, [queued, *([advanced] if advanced else []), *(session.get("queued_prompts") or [])])
            session["running"] = False
            return True
    kwargs: dict = {"queued_prompt_generation": queue_generation}
    if queued.get("image_paths"):
        kwargs["image_paths"] = queued["image_paths"]
    dispatch_failed = False
    try:
        if not use_compute_host:
            _run_prompt_submit(rid, sid, session, queued["text"], **kwargs)
        elif (resp := _submit_prompt_to_compute_host(rid, sid, session, queued["text"], **kwargs)).get("error"):
            with session["history_lock"]:
                session["running"] = False
                _clear_inflight_turn(session)
            _emit("error", sid, {"message": str((resp.get("error") or {}).get("message") or "queued prompt failed")})
            dispatch_failed = True
    except Exception as exc:
        _notif_log_failure("queued prompt dispatch failed", exc)
        _notif_release_turn(session)
        dispatch_failed = True
    if dispatch_failed:
        with session["history_lock"]:
            drain_next = bool(session.get("queued_prompt")) and not session.get("_turn_cancel_requested")
        if drain_next:
            _drain_queued_prompt(rid, sid, session)
    return True


def _inflight_snapshot(session: dict) -> dict | None:
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return None
    user, assistant = str(turn.get("user") or "").strip(), str(turn.get("assistant") or "")
    streaming, error = bool(turn.get("streaming")), str(turn.get("error") or "").strip()
    if not (user or assistant or streaming or error):
        return None
    snapshot = {"assistant": assistant, "streaming": streaming, "user": user}
    raw_offsets = turn.get("correction_offsets") or []
    correction_pairs = [(str(c), raw_offsets[i] if i < len(raw_offsets) else None)
                        for i, c in enumerate(turn.get("corrections") or []) if str(c).strip()]
    if correction_pairs:
        # Mid-turn redirects alongside (not over) the original prompt so resume can rebuild every user bubble; offsets
        # only when every correction has one so clients can trust the pairing.
        snapshot["corrections"] = [c for c, _ in correction_pairs]
        if all(isinstance(offset, int) and offset >= 0 for _, offset in correction_pairs):
            snapshot["correction_offsets"] = [int(offset) for _, offset in correction_pairs]  # type: ignore[arg-type]
    if error:
        # Retained failed turn (_fail_inflight_turn): a resuming client must rebuild the failed bubble, not render the
        # partial text as a healthy reply.
        snapshot.update(error=error, status=str(turn.get("status") or "error"), recoverable=bool(turn.get("recoverable")))
        if isinstance(surface := turn.get("error_surface"), dict) and surface:
            snapshot["error_surface"] = surface
    return snapshot


def _emit_terminal_turn_error(
    sid: str, session: dict, error: Any, error_surface: Optional[dict] = None, *, retire_marker: bool = True) -> None:
    """Close a failed turn with the same ``status: "error"`` ``message.complete`` frame as the returned-error path,
    retaining the turn so a client that missed the frame recovers it from ``session.resume``'s ``inflight``.
    ``error_surface`` ({layer, code, retryable}) is classified from an exception if absent."""
    agent = session.get("agent")
    if error_surface is None and isinstance(error, BaseException):
        with contextlib.suppress(Exception):
            from agent.error_surface import build_error_surface_from_exception
            error_surface = build_error_surface_from_exception(
                error, provider=str(getattr(agent, "provider", "") or ""), model=str(getattr(agent, "model", "") or ""))
    with session["history_lock"]:
        _fail_inflight_turn(session, error, error_surface=error_surface)
        turn = session.get("inflight_turn") or {}
        message, partial = str(turn.get("error") or "turn failed"), str(turn.get("assistant") or "")
        cols = int(session.get("cols", 80))
    text = partial or f"Error: {message}"
    rendered = ""
    with contextlib.suppress(Exception):
        rendered = render_message(text, cols)
    payload = {"text": text, "usage": _get_usage(agent) if agent is not None else {}, "status": "error",
               "error": message, "recoverable": True, **({"error_surface": error_surface} if error_surface else {}),
               **({"partial": True} if partial else {}), **({"rendered": rendered} if rendered else {})}
    if retire_marker:
        _retire_turn_marker(session)
    _emit("message.complete", sid, payload)


def _restore_agent_history_after_turn_error(session: dict, agent) -> bool:
    """Keep a failed turn's working transcript: ``AIAgent`` persists its messages independently, so after a raise the
    next prompt must see them, not the pre-turn snapshot."""
    agent_messages = getattr(agent, "_session_messages", None)
    if not isinstance(agent_messages, list):
        return False
    with session["history_lock"]:
        session["history"] = list(agent_messages)
        session["history_version"] = int(session.get("history_version", 0)) + 1
    return True


def _queued_prompt_snapshot(session: dict) -> dict | None:
    """The accepted next-turn prompt without its transport handle, for the live-session projection (Desktop may
    reconnect while it is still queued)."""
    queued = session.get("queued_prompt")
    user = _inflight_text(queued.get("text")) if isinstance(queued, dict) else ""
    return {"user": user} if user else None


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
