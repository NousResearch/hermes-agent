"""Auto-continue: resume a turn killed by a process/machine death, plus queued-prompt drain and
busy-submit handling. Bodies are rebound onto server.py's globals at install time
(method_ctx.bind_module), so they reference server.py globals bare."""

from __future__ import annotations

import contextlib
import threading
import uuid
from typing import Any

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


def _retire_turn_marker(
    session: dict,
    *keys: str,
    include_current: bool = True,
    expected_turn_id: str | None = None,
) -> None:
    """Drop the crash marker right before the terminal frame (not at turn-thread end: post-turn work outlives the
    client's answer, and quitting in that window would leave a marker that re-runs a finished turn). Extra ``keys``
    cover a session_key that compression rotated mid-turn. Generation-bound callers pass every key captured while
    claiming the turn and set ``include_current=False`` so a later turn's marker can never be retired."""
    if not expected_turn_id:
        return
    home = _session_home(session)
    candidates = (*keys, str(session.get("session_key") or "")) if include_current else keys
    for key in dict.fromkeys(candidates):
        if key:
            clear_turn_marker(home, key, expected_turn_id=expected_turn_id)


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
    if not marker.get("auto_continue", True):
        return None  # The mailbox owns recovery and receipt identity for imported turns.
    enabled, freshness_secs, max_attempts = _auto_continue_config()
    age = time.time() - marker["started_at"]
    if not enabled or age > freshness_secs or marker["attempts"] >= max_attempts:
        clear_turn_marker(
            home,
            session_key,
            expected_turn_id=marker.get("turn_id"),
            expected_started_at=(
                None if marker.get("turn_id") else marker.get("started_at")
            ),
        )  # stale/disabled/crash-looping: a manual message continues
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
        from agent.turn_authorization import TurnAuthorization

        recovery_authorization = (
            TurnAuthorization.blocked()
            if marker.get("personal_authorization_blocked")
            else TurnAuthorization.from_raw(None)
        )
        with session["history_lock"]:
            if (
                session.get("running")
                or session.get("_turn_interrupt_claim") is not None
                or session.get("_finalized")
            ):
                session["_auto_continue_scheduled"] = False  # a real user prompt beat us; it clears the marker
                return
            session["running"] = True
            auto_turn_id = _activate_turn_identity(session)
            session["_turn_cancel_requested"] = False
            session["_active_turn_route"] = "inline"
            session["_active_turn_authorization"] = recovery_authorization
            session["last_active"] = time.time()
        # Ownership admission BEFORE message.start: a sibling backend sharing this HERMES_HOME may have written the
        # marker and still be mid-turn. Leave the marker so a later resume retries.
        # Running the continuation anyway would be the double-writer this fence exists to prevent. See
        # #94778.
        if _ensure_active_session_slot(sid, session) is not None:
            logger.info("auto-continue for %s refused: session has another live owner", session_key)
            with session["history_lock"]:
                if session.get("_active_turn_id") == auto_turn_id:
                    session["running"] = False
                    _clear_active_turn_state(
                        session,
                        recovery_authorization,
                        expected_turn_id=auto_turn_id,
                    )
                session["_auto_continue_scheduled"] = False
            return
        with session["history_lock"]:
            # Marker inputs read back by _run_prompt_submit: attempt count (crash breaker) and the ORIGINAL prompt (no
            # nested notes). Set here, not at schedule time, so a bail above leaves nothing for a racing user turn.
            session["_auto_continue_attempt"], session["_auto_continue_prompt"] = attempt, marker["prompt"]
        try:
            _emit("status.update", sid, {"kind": "process", "text": "Resuming interrupted turn…"})
            _emit("message.start", sid)
            submit_kwargs = {
                "display_kind": "auto_continue",
                "turn_authorization": recovery_authorization,
                "expected_turn_id": auto_turn_id,
            }
            _run_prompt_submit(rid, sid, session, text, **submit_kwargs)
        except Exception as exc:
            try:
                with contextlib.suppress(Exception):
                    _notif_log_failure("auto-continue dispatch failed", exc)
            finally:
                _notif_release_turn(
                    session, auto_turn_id
                )  # rebound from session_notifications
    # Refused before kickoff runs: no turn identity was minted yet, so only the
    # scheduling flag has to come back off.
    if _start_session_work(kickoff, name=f"auto-continue-{sid}") is None:
        with session["history_lock"]:
            session["_auto_continue_scheduled"] = False
        return None
    logger.info("auto-continue scheduled for session %s (attempt %d, interrupted %.0fs ago)", session_key, attempt, age)
    return {"attempt": attempt, "interrupted_at": marker["started_at"]}


def _ac_inflight_original(session: dict) -> str:
    turn = session.get("inflight_turn")
    return str(turn.get("user") or "").strip() if isinstance(turn, dict) else ""


def _same_as_active_turn_authorization(session: dict, authorization) -> bool:
    """Treat legacy/no-token holders alike; compare personal credentials in constant time."""
    active = session.get("_active_turn_authorization")
    if authorization is None:
        return active is None or not active.is_personal
    if active is None:
        return not authorization.is_personal
    return authorization.same_principal(active)


_ANY_ACTIVE_AUTHORIZATION = object()


def _activate_turn_identity(session: dict, turn_id: str | None = None) -> str:
    """Mint the immutable identity for one admitted turn. Caller holds ``history_lock``."""
    identity = str(turn_id or uuid.uuid4().hex)
    session["_active_turn_id"] = identity
    return identity


def _clear_active_turn_state(
    session: dict,
    expected=_ANY_ACTIVE_AUTHORIZATION,
    *,
    expected_turn_id: str | None = None,
) -> bool:
    """Clear the authorization + execution route owned by one admitted turn.

    Caller holds ``history_lock``.  An expected holder makes late cleanup unable
    to erase a newer turn that won admission in the meantime.
    """
    if (
        expected_turn_id is not None
        and session.get("_active_turn_id") != expected_turn_id
    ):
        return False
    current = session.get("_active_turn_authorization")
    if expected is not _ANY_ACTIVE_AUTHORIZATION and current is not expected:
        # The expected turn may already have been revoked by interrupt/reset;
        # its late cleanup may still release ``running`` while the slot remains
        # unclaimed. Never touch a newer admission's holder or route.
        if current is not None or "_active_turn_route" in session:
            return False
    session.pop("_active_turn_authorization", None)
    session.pop("_active_turn_route", None)
    session.pop("_active_turn_id", None)
    return True


def _emit_person_admission(
    sid: str, authorization: object, status: str, *, reason: str | None = None
) -> bool:
    """Emit one private exact-ID lifecycle event when the holder is personal."""
    # A turn with no holder at all (non-personal callers, and entry points whose
    # authorization argument defaults to None) has no admission to account for.
    if authorization is None:
        return False
    admission_id = authorization._fizko_admission_id()
    if not admission_id:
        return False
    payload = {"admission_id": admission_id, "status": status}
    if reason is not None:
        payload["reason"] = reason

    def deliver() -> bool:
        try:
            # Private, single-peer delivery: a session with several attached
            # viewers holds a FanoutTransport, and this payload carries the
            # sidecar's private admission id. The peer is pinned at admission
            # time because deferred work runs outside the request context.
            return _emit_private_live(
                "person.admission", sid, payload, authorization._fizko_peer()
            )
        except Exception:
            logger.debug(
                "person admission live delivery failed sid=%s status=%s",
                sid,
                status,
                exc_info=True,
            )
            return False

    return authorization._fizko_emit_admission_event(
        status, deliver
    )


def _emit_person_admissions(
    sid: str, authorizations: list[Any], *, reason: str
) -> None:
    """Best-effort every abandoned admission without aborting lifecycle cleanup."""
    for authorization in authorizations:
        try:
            _emit_person_admission(sid, authorization, "terminal", reason=reason)
        except Exception:
            logger.debug(
                "person admission terminal delivery failed sid=%s reason=%s",
                sid,
                reason,
                exc_info=True,
            )


def _collect_person_admissions(session: dict, *, include_active: bool = True) -> list[Any]:
    """Snapshot unique personal admission holders while ``history_lock`` is held."""
    candidates = []
    if include_active:
        candidates.append(session.get("_active_turn_authorization"))
    candidates.extend(
        entry.get("turn_authorization")
        for entry in [session.get("queued_prompt"), *(session.get("queued_prompts") or [])]
        if isinstance(entry, dict)
    )
    admissions = []
    seen = set()
    for authorization in candidates:
        identity = id(authorization)
        if (
            identity not in seen
            and bool(getattr(authorization, "is_personal", False))
        ):
            seen.add(identity)
            admissions.append(authorization)
    return admissions


def _enqueue_prompt(
    session: dict, text: Any, transport: Any, image_paths: list[str] | None = None,
    turn_author: dict | None = None, turn_authorization=None,
) -> None:
    """Queue a message for the next turn. Text-only arrivals share a slot and merge losslessly (like the
    consecutive-user merge in ``repair_message_sequence``); image-bearing and authored ones stay separate
    envelopes so attachment chronology and the sender survive. ``transport`` is pinned so the drained turn
    streams to its sender."""
    from agent.turn_authorization import TurnAuthorization

    image_paths = list(image_paths or [])
    # Missing authorization is an explicit no-token identity for queue isolation:
    # an internal follow-up must never merge into a personal caller's envelope.
    authorization = turn_authorization or TurnAuthorization.from_raw(None)
    # Scrub live-turn self-duplicates only for ordinary prompts. Every personal
    # admission is a distinct accounting unit, even for identical text/owner.
    if not authorization.is_personal:
        _drop_queued_duplicates_of_inflight_user(session)
    text_only = not image_paths and isinstance(text, str)
    # A text-only self-copy of the live prompt would restart it on drain; an authored copy is another sender's message.
    if (not authorization.is_personal and text_only and not turn_author
            and _same_as_active_turn_authorization(session, authorization)
            and text.strip() == _ac_inflight_original(session) != ""):
        return
    queued = {
        "text": text,
        "transport": transport,
        **({"image_paths": image_paths} if image_paths else {}),
        **({"turn_author": turn_author} if turn_author else {}),
        **({"turn_authorization": authorization} if authorization.is_personal else {}),
    }
    existing = session.get("queued_prompt")
    existing_authorization = existing.get("turn_authorization") if existing else None
    same_authorization = authorization.same_principal(
        existing_authorization or TurnAuthorization.from_raw(None)
    )
    if (existing and not authorization.is_personal and same_authorization and text_only
            and not turn_author and isinstance(existing.get("text"), str)
            and not existing.get("image_paths") and not existing.get("turn_author")
            and not session.get("queued_prompts")):
        prev = existing["text"]
        existing["text"] = f"{prev}\n\n{text}" if prev and text else (prev or text)
    elif existing:
        session.setdefault("queued_prompts", []).append(queued)
    else:
        session["queued_prompt"] = queued


def _sanitize_queued_entry_vs_inflight_user(
    entry: Any, original: str, active_authorization=None,
) -> dict | None:
    """Drop (``None``) a text-only self-duplicate of the live user text, or rewrite a merged slot
    ``"{original}\\n\\n{later}"`` to ``later`` so the correction survives without re-firing the original. Image-bearing
    and authored envelopes are left alone: chronology and the sender's own words are kept.

    Returns ``None`` to drop the envelope, or a (possibly rewritten) dict to keep. A merged slot
    ``"{original}\\n\\n{later}"`` (from ``_enqueue_prompt``'s consecutive text merge) is rewritten to just
    ``later`` so a later correction is not lost and the original is not re-fired (#84417).
    """
    if not isinstance(entry, dict):
        return None
    entry_authorization = entry.get("turn_authorization")
    if bool(getattr(entry_authorization, "is_personal", False)) or bool(
        getattr(active_authorization, "is_personal", False)
    ):
        return entry
    same_authorization = (
        (active_authorization is None and not entry_authorization.is_personal)
        or (
            active_authorization is not None
            and entry_authorization.same_principal(active_authorization)
        )
    ) if entry_authorization is not None else (
        active_authorization is None or not active_authorization.is_personal
    )
    if not same_authorization:
        return entry
    text = entry.get("text")
    if not original or entry.get("image_paths") or entry.get("turn_author") or not isinstance(text, str):
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
    active_authorization = session.get("_active_turn_authorization")
    cleaned = (_sanitize_queued_entry_vs_inflight_user(e, original, active_authorization)
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
    with session["history_lock"]:
        if session.get("_busy_interrupt_pending"):
            return
        expected_turn_id = str(session.get("_active_turn_id") or "")
        use_compute_host = session.get("_active_turn_route") == "compute"
        use_agent = bool(
            not use_compute_host
            and expected_turn_id
            and agent is not None
            and callable(getattr(agent, "bind_gateway_turn", None))
        )
        if not use_agent and not use_compute_host:
            return
        session["_busy_interrupt_pending"] = True
        session["_busy_interrupt_pending_turn_id"] = expected_turn_id

    def interrupt() -> None:
        try:
            with session["history_lock"]:
                if session.get("_active_turn_id") != expected_turn_id:
                    return
            with contextlib.suppress(Exception):
                if use_agent:
                    from agent.interrupt_compat import request_hard_interrupt

                    request_hard_interrupt(agent, require_turn_id=expected_turn_id)
                else:
                    _get_compute_host_supervisor().interrupt(
                        sid, expected_turn_id=expected_turn_id
                    )
        finally:
            with session["history_lock"]:
                if session.get("_busy_interrupt_pending_turn_id") == expected_turn_id:
                    session["_busy_interrupt_pending"] = False
                    session.pop("_busy_interrupt_pending_turn_id", None)
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


def _handle_busy_submit(
    rid, sid: str, session: dict, text: Any, transport: Any, queued: bool = False,
    turn_author: dict | None = None, turn_authorization=None,
) -> dict | None:
    """Apply ``display.busy_input_mode`` to a mid-turn prompt instead of rejecting it (rejection made clients busy-retry
    and drop sends): ``interrupt`` (default) → redirect, falling back to hard interrupt + queue; ``queue`` → queue only;
    ``steer`` → inject after the current atomic action. ``queued=True`` (client queue drain) forces queue mode: a "run
    after" message must NEVER become a live correction."""
    mode = "queue" if queued else _load_busy_input_mode()
    agent = session.get("agent")
    with session["history_lock"]:
        interrupt_claim = session.get("_turn_interrupt_claim")
        interrupt_event = (
            interrupt_claim.get("event") if isinstance(interrupt_claim, dict) else None
        )
        if interrupt_event is not None:
            image_paths = []
            active_authorization = None
        else:
            if not session.get("running"):
                return None  # turn ended since prompt.submit's busy check; caller retries on the idle session
            image_paths = list(session.get("attached_images", []))
            if image_paths:
                session["attached_images"] = []  # claim now so a later paste isn't consumed when the turn yields
            active_authorization = session.get("_active_turn_authorization")
    if interrupt_event is not None:
        interrupt_event.wait()
        return None
    if (
        bool(getattr(active_authorization, "is_personal", False))
        or bool(getattr(turn_authorization, "is_personal", False))
    ):
        # Personal authorization is scoped to one admitted turn. Even the same
        # credential waits for a fresh authenticated turn rather than touching
        # a live agent across the holder/agent race window.
        mode = "queue"
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
        interrupt_claim = session.get("_turn_interrupt_claim")
        interrupt_event = (
            interrupt_claim.get("event") if isinstance(interrupt_claim, dict) else None
        )
        if interrupt_event is not None:
            if image_paths:
                session["attached_images"] = image_paths + list(
                    session.get("attached_images", [])
                )
        elif not session.get("running"):
            if image_paths:
                session["attached_images"] = image_paths + list(session.get("attached_images", []))
            return None
        else:
            _enqueue_prompt(
                session, text, transport, image_paths=image_paths, turn_author=turn_author,
                turn_authorization=turn_authorization)
            session["last_active"] = time.time()
    if interrupt_event is not None:
        interrupt_event.wait()
        return None
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
    """Fire the first runnable queued prompt. Invalid personal heads are rejected and
    draining continues in the same cycle, bounded so a poisoned queue cannot monopolize a worker."""
    from agent.turn_authorization import TurnAuthorization

    rejected_messages = []
    rejected_admissions = []
    queued = None
    turn_authorization = None
    use_compute_host = False
    claimed_turn_id = ""
    queue_generation = 0
    rejected_batch_has_more = False
    with _session_turn_admission(session) as admitted:
        if (
            not admitted
            or session.get("_closing")
            or session.get("running")
            or session.get("_turn_interrupt_claim") is not None
        ):
            return False
        for _ in range(32):
            queued = session.get("queued_prompt")
            if not queued:
                break
            queue_generation = int(session.get("_queued_prompt_generation", 0))
            _ac_set_queue(session, session.get("queued_prompts") or [])
            turn_authorization = queued.get("turn_authorization") or TurnAuthorization.from_raw(None)
            compute_host_required = _session_uses_compute_host(session)
            if turn_authorization.has_token and turn_authorization.is_expired:
                rejected_admissions.append((turn_authorization, "expired"))
                rejected_messages.append(
                    "person authorization expired before the queued turn ran"
                )
                queued = None
                session["last_active"] = time.time()
                continue
            if turn_authorization.is_personal and compute_host_required:
                rejected_admissions.append((turn_authorization, "isolation_rejected"))
                rejected_messages.append(
                    "person-authorized turns are unavailable while turn isolation is enabled"
                )
                queued = None
                session["last_active"] = time.time()
                continue
            use_compute_host = compute_host_required
            session["running"] = True
            _activate_turn_identity(session)
            claimed_turn_id = str(session.get("_active_turn_id") or "")
            session["_turn_cancel_requested"] = False
            session["_active_turn_route"] = "compute" if use_compute_host else "inline"
            session["_active_turn_authorization"] = turn_authorization
            queued_transport = queued.get("transport")
            # The queuer's transport is pinned so the drained turn reaches the client that sent it — but
            # ATTACHED, not rebound: a mid-turn prompt from a second client used to silence the first for the
            # whole drained turn. A peer that disconnected while its prompt sat in the queue is skipped: the
            # prompt still runs, only the dead pin is dropped.
            if (
                queued_transport is not None
                and not _transport_is_dead(queued_transport)
            ):
                _attach_session_transport(session, queued_transport)
            break
        else:
            rejected_batch_has_more = bool(session.get("queued_prompt"))
    if queued is None:
        for authorization, reason in rejected_admissions:
            with contextlib.suppress(Exception):
                _emit_person_admission(
                    sid, authorization, "terminal", reason=reason
                )
        for message in rejected_messages:
            with contextlib.suppress(Exception):
                _emit("error", sid, {"message": message})
        if rejected_batch_has_more:
            threading.Thread(
                target=_drain_queued_prompt,
                args=(rid, sid, session),
                daemon=True,
            ).start()
        return bool(rejected_messages)
    cancelled_before_dispatch = None
    generation_cancelled = False
    with session["history_lock"]:
        if int(session.get("_queued_prompt_generation", 0)) != queue_generation:
            generation_cancelled = True
            # A generation bump cancelled this claimed head. Ordinary prompts retain
            # their historical retry behavior; a personal admission is a one-shot
            # accounting unit and cannot be restored after cancellation.
            if turn_authorization.is_personal:
                cancelled_before_dispatch = turn_authorization
            else:
                advanced = session.get("queued_prompt")
                _ac_set_queue(
                    session,
                    [queued, *([advanced] if advanced else []), *(session.get("queued_prompts") or [])],
                )
            # Fenced on the claimed id: a Stop plus a new prompt.submit can land in
            # the window between the claim and this re-check, and publishing idle over
            # that newer turn silently drops the user's prompt (its worker then refuses
            # on the `not running` gate with no error frame).
            if session.get("_active_turn_id") == claimed_turn_id:
                session["running"] = False
                _clear_active_turn_state(
                    session, turn_authorization, expected_turn_id=claimed_turn_id
                )
    if cancelled_before_dispatch is not None:
        _emit_person_admission(
            sid, cancelled_before_dispatch, "terminal", reason="cancelled_before_dispatch"
        )
        return True
    if generation_cancelled:
        return True
    kwargs: dict = {
        "queued_prompt_generation": queue_generation,
        "expected_turn_id": claimed_turn_id,
    }
    if queued.get("image_paths"):
        kwargs["image_paths"] = queued["image_paths"]
    # The compute-host frame has no author field, so only the inline runner receives it.
    author_kwargs = {"turn_author": queued["turn_author"]} if queued.get("turn_author") else {}
    dispatch_failed = False
    restore_claim = False
    try:
        restore_claim = not bool(
            getattr(turn_authorization, "is_personal", False)
        )
        for authorization, reason in rejected_admissions:
            _emit_person_admission(
                sid, authorization, "terminal", reason=reason
            )
        for message in rejected_messages:
            _emit("error", sid, {"message": message})
        if not use_compute_host:
            kwargs["turn_authorization"] = turn_authorization
            if _run_prompt_submit(
                rid, sid, session, queued["text"], **kwargs, **author_kwargs
            ) is False:
                dispatch_failed = True
                # Static prompts preserve the historical retry behavior. A personal
                # admission was already terminated by _run_prompt_submit and must
                # not be replayed with the retired identifier.
                restore_claim = not bool(
                    getattr(turn_authorization, "is_personal", False)
                )

        if use_compute_host and (
            resp := _submit_prompt_to_compute_host(
                rid, sid, session, queued["text"],
                expected_turn_authorization=turn_authorization,
                **kwargs,
            )
        ).get("error"):
            cleaned_failed_turn = False
            with session["history_lock"]:
                if (
                    session.get("_active_turn_id") == claimed_turn_id
                    and session.get("_active_turn_authorization")
                    is turn_authorization
                    and _clear_active_turn_state(
                        session,
                        turn_authorization,
                        expected_turn_id=claimed_turn_id,
                    )
                ):
                    session["running"] = False
                    _clear_inflight_turn(session)
                    cleaned_failed_turn = True
            if cleaned_failed_turn:
                _emit("error", sid, {"message": str((resp.get("error") or {}).get("message") or "queued prompt failed")})
            dispatch_failed = True
    except Exception as exc:
        dispatch_failed = True
        with contextlib.suppress(Exception):
            _notif_log_failure("queued prompt dispatch failed", exc)
        if bool(getattr(turn_authorization, "is_personal", False)):
            with contextlib.suppress(Exception):
                _emit_person_admission(
                    sid, turn_authorization, "terminal", reason="dispatch_failed"
                )
        with contextlib.suppress(Exception):
            _notif_release_turn(session, claimed_turn_id)
    if dispatch_failed:
        with session["history_lock"]:
            if (restore_claim and not session.get("_closing") and not session.get("_turn_cancel_requested")
                    and int(session.get("_queued_prompt_generation", 0)) == queue_generation):
                advanced = session.get("queued_prompt")
                _ac_set_queue(
                    session,
                    [queued, *([advanced] if advanced else []), *(session.get("queued_prompts") or [])],
                )
            if (
                session.get("_active_turn_id") == claimed_turn_id
                and session.get("_active_turn_authorization") is turn_authorization
            ):
                session["running"] = False
                _clear_active_turn_state(
                    session,
                    turn_authorization,
                    expected_turn_id=claimed_turn_id,
                )
            drain_next = bool(session.get("queued_prompt")) and not session.get("_turn_cancel_requested")
        if drain_next and not restore_claim:
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
    if isinstance(display_kind := turn.get("display_kind"), str) and display_kind:
        snapshot["display_kind"] = display_kind
    if isinstance(display_metadata := turn.get("display_metadata"), dict):
        snapshot["display_metadata"] = dict(display_metadata)
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
    sid: str,
    session: dict,
    error: Any,
    error_surface: Optional[dict] = None,
    *,
    retire_marker: bool = True,
    expected_turn_id: str | None = None,
) -> None:
    """Close a failed turn with the same ``status: "error"`` ``message.complete`` frame as the returned-error path,
    retaining the turn so a client that missed the frame recovers it from ``session.resume``'s ``inflight``.
    ``error_surface`` ({layer, code, retryable}) is classified from an exception if absent."""
    agent = session.get("agent")
    if error_surface is None and isinstance(error, BaseException):
        with contextlib.suppress(Exception):
            from agent.error_surface import build_error_surface_from_exception
            error_surface = build_error_surface_from_exception(
                error, provider=str(getattr(agent, "provider", "") or ""), model=str(getattr(agent, "model", "") or ""),
                api_key=getattr(agent, "api_key", None))
    with session["history_lock"]:
        if (
            expected_turn_id
            and session.get("_active_turn_id") != expected_turn_id
        ):
            return
        _fail_inflight_turn(session, error, error_surface=error_surface)
        turn = session.get("inflight_turn") or {}
        message, partial = str(turn.get("error") or "turn failed"), str(turn.get("assistant") or "")
        cols = int(session.get("cols", 80))
    text = partial or turn_error_text(message, error_surface)
    rendered = ""
    with contextlib.suppress(Exception):
        rendered = render_message(text, cols)
    payload = {"text": text, "usage": _get_usage(agent) if agent is not None else {}, "status": "error",
               "error": message, "recoverable": True, **({"error_surface": error_surface} if error_surface else {}),
               **({"partial": True} if partial else {}), **({"rendered": rendered} if rendered else {})}
    if retire_marker and expected_turn_id:
        _retire_turn_marker(session, expected_turn_id=expected_turn_id)
    with session["history_lock"]:
        if (
            expected_turn_id
            and session.get("_active_turn_id") != expected_turn_id
        ):
            return
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
