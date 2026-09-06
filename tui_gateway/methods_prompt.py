"""Prompt / attachment / respond JSON-RPC handlers.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


_STALE_TARGET_MSG = "target user message is no longer in session history"
_GROUP_PROBE_FAILED_MSG = "Could not verify this group. Try again after the gateway recovers."


def _history_user_indices(history: list) -> list:
    """Indices of canonical live-user turns, including composite carriers."""
    from agent.context_compressor import user_originated_turn_view
    return [i for i, m in enumerate(history) if user_originated_turn_view(m) is not None]


def _message_row_id(msg: dict):
    """Durable SQLite row id from a history entry (``_row_id`` else ``row_id``), or None."""
    raw = msg.get("_row_id")
    if raw is None:
        raw = msg.get("row_id")
    with contextlib.suppress(TypeError, ValueError):
        return None if raw is None else int(raw)
    return None


def _mem_db_pair_agrees(mem, db_msg) -> bool:
    """True when a live-memory entry plausibly corresponds to a durable row: roles and
    display-marker status must match (a marker shifts every later position) and an
    addressable user turn must show the same text (multimodal: role/marker suffice)."""
    if not isinstance(mem, dict) or not isinstance(db_msg, dict):
        return False
    if mem.get("role") != db_msg.get("role"):
        return False
    if mem.get("role") == "user":
        from agent.context_compressor import user_originated_turn_view
        from agent.memory_manager import sanitize_context
        mem_view = user_originated_turn_view(mem)
        db_view = user_originated_turn_view(db_msg)
        if (mem_view is None) != (db_view is None):
            return False
        if mem_view is None:
            return bool(mem.get("display_kind")) == bool(db_msg.get("display_kind"))
        mem_content = mem_view.get("content")
        db_content = db_view.get("content")
        if isinstance(mem_content, str) and isinstance(db_content, str):
            return sanitize_context(mem_content).strip() == sanitize_context(db_content).strip()
        return True
    return bool(mem.get("display_kind")) == bool(db_msg.get("display_kind"))


def _find_user_turn_by_row_id(history: list, target_row_id: int):
    """``(user_ordinal, history_index)`` for ``target_row_id``, or None."""
    return next(
        ((u_ord, h_idx) for u_ord, h_idx in enumerate(_history_user_indices(history))
         if _message_row_id(history[h_idx]) == target_row_id), None)


def _load_durable_truncation_history(
    session: dict, fallback_sid: str = "", repair_alternation: bool = True):
    """Load the durable live-replay transcript, or None when it cannot be proven safe."""
    session_key = str(session.get("session_key") or fallback_sid or "")
    if not session_key:
        return []
    try:
        with _session_db(session) as db:
            get_conv = getattr(db, "get_messages_as_conversation", None)
            if not callable(get_conv):
                return None
            history = get_conv(
                session_key, repair_alternation=repair_alternation, include_row_ids=True)
    except Exception:
        logger.debug(
            "prompt.submit: failed loading durable history for session %s", session_key,
            exc_info=True)
        return None
    return history if isinstance(history, list) else None


def _resolve_truncate_row_id(session: dict, history: list, target_row_id: int):
    """Resolve ``truncate_before_row_id`` to ``(user_ordinal, history_index)``: in-memory
    stamps first, else the durable transcript mapped onto the live list by user ordinal.
    Never falls back to a client-supplied ordinal — unknown row ids refuse.

    Prefer in-memory ``_row_id`` / ``row_id`` stamps. When a live turn rewrote ``session["history"]``
    without stamps (provider-format messages), load the session's durable transcript with
    ``include_row_ids=True`` and map the matched user-turn ordinal onto the live list. See #82959.
    """
    if (hit := _find_user_turn_by_row_id(history, target_row_id)) is not None:
        return hit
    db_history = _load_durable_truncation_history(session)
    if db_history is None:
        return None
    # Heal missing stamps only when EVERY pair agrees: the durable copy is alternation-
    # repaired while the live list can carry optimistic/marker rows, and a stamp on a
    # misaligned pair is sticky (re-aims every later rewind at the wrong durable row).
    if len(db_history) == len(history) and all(
            _mem_db_pair_agrees(mem, db_msg) for mem, db_msg in zip(history, db_history)):
        for mem, db_msg in zip(history, db_history):
            if (db_rid := _message_row_id(db_msg)) is not None and _message_row_id(mem) is None:
                mem["_row_id"] = db_rid
        if (hit := _find_user_turn_by_row_id(history, target_row_id)) is not None:
            return hit
    if (db_hit := _find_user_turn_by_row_id(db_history, target_row_id)) is None:
        return None
    db_ord, db_idx = db_hit
    mem_user_indices = _history_user_indices(history)
    # Same-ordinal mapping across lists that can diverge (repair may merge a user;user
    # pair): trust it only when the mapped live turn shows the durable target's content.
    if db_ord >= len(mem_user_indices) or not _mem_db_pair_agrees(
            history[mem_user_indices[db_ord]], db_history[db_idx]):
        return None
    return db_ord, mem_user_indices[db_ord]


def _coerce_truncate_int(rid, value, param_name="truncate_before_user_ordinal"):
    """``(int_value, error_response)`` for a client integer param.  bool is refused like
    any non-integer: JSON ``true`` would int() to 1 and aim at the wrong turn."""
    if not isinstance(value, bool):
        with contextlib.suppress(TypeError, ValueError):
            return int(value), None
    return None, _err(rid, 4004, f"{param_name} must be an integer")


def _pending_reaction_notes(session: dict) -> str:
    """Note block for reactions since the last turn (model input only, announced once —
    rows are stamped ``seen`` on read), or "".  Gated on display.message_reactions."""
    session_key = str(session.get("session_key") or "")
    if not session_key:
        return ""
    try:
        display = _load_cfg().get("display")
        if not (isinstance(display, dict) and bool(display.get("message_reactions", False))):
            return ""
    except Exception:
        return ""
    try:
        with _session_db(session) as db:
            pending = None if db is None else db.take_unseen_reactions(session_key, author="user")
    except Exception:
        logger.debug("Failed to read pending reactions", exc_info=True)
        return ""
    notes = []
    for entry in pending or ():
        snippet = (entry.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "…"
        emoji = entry.get("emoji") or ""
        whose = "their own" if entry.get("role") == "user" else "your"
        if snippet:
            notes.append(f'[The user reacted {emoji} to {whose} message: "{snippet}"]')
        else:
            # Attachment-only / tool-call-only rows: no quote beats an empty quote.
            notes.append(f"[The user reacted {emoji} to {whose} earlier message]")
    return "\n".join(notes)


# ── prompt.submit pieces ────────────────────────────────────────────────────

def _typed_stop_phrase_response(rid, text):
    """RPC reply ending the voice chat when a bare stop phrase is TYPED while backend voice
    mode is on (typed twin of the spoken stop phrase), or None for a normal message."""
    if not (isinstance(text, str) and _voice_mode_enabled()):
        return None
    try:
        # Typed bare stop phrase while backend voice mode is active ends the voice chat instead of sending
        # "stop" to the agent — the typed twin of the spoken stop phrase (PR #73106), applied at the ONE
        # server-side choke point every TUI submit passes through. (The desktop's voice conversation is
        # renderer-owned and never flips the backend flag, so it handles its own typed stop client-side.)
        from tools.voice_mode import is_voice_stop_phrase
        if not is_voice_stop_phrase(text):
            return None
    except Exception:
        return None
    _end_voice_chat(stop_loop=True, stop_tts=True)
    _voice_emit("voice.transcript", {"stop_phrase": True, "typed": True})
    logger.info("prompt.submit: typed stop phrase — voice chat ended")
    return _ok(rid, {"voice_stopped": True})


_HOSTED_TASK_FIELDS = {"room_id", "task_id", "thread_id", "turn_id", "execution_generation"}


def _hosted_submit_error(rid, session, hosted_task, hosted_terminal_callback):
    """Validate the hosted-room turn proof carried by an internal submit."""
    if session.get("source") != "bot_room":
        return _err(rid, 4120, "hosted room turns require a bot_room session")
    valid = (
        isinstance(hosted_task, dict) and callable(hosted_terminal_callback)
        and set(hosted_task) == _HOSTED_TASK_FIELDS
        and all(isinstance(hosted_task.get(f), str) and hosted_task[f]
                for f in _HOSTED_TASK_FIELDS - {"execution_generation"})
        and isinstance(hosted_task.get("execution_generation"), int))
    return None if valid else _err(rid, 4120, "invalid hosted room turn proof")


def _legacy_group_fence_error(rid, session, params):
    """Fence direct prompts into a hosted room from older Desktop builds (they know the
    ``Group: <room-id>`` title but not the authority marker; a direct prompt would start a
    second renderer driver)."""
    title = str(session.get("title") or "")
    room_id = title.removeprefix("Group: ").strip() if title.startswith("Group: ") else ""
    if not room_id:
        return None
    try:
        from gateway.hosted_rooms import (
            HostedRoomError, RoomProbeUnavailableError, default_db_path, probe_hosted_room,
            probe_peer_room_reservation)
        hosted = probe_hosted_room(default_db_path(), room_id=room_id)
        peer = False
        if not hosted:
            from hermes_constants import named_profile_home
            session_profile_home = named_profile_home(str(session.get("profile_home") or ""))
            peer = probe_peer_room_reservation(
                default_db_path(), room_id=room_id, target_profile=(
                    (session_profile_home.name if session_profile_home is not None else "")
                    or str(params.get("profile") or "").strip()
                    or str(_current_profile_name() or "default").strip()))
    except RoomProbeUnavailableError:
        return _err(rid, 5122, _GROUP_PROBE_FAILED_MSG)
    except HostedRoomError:
        # Legacy Desktop sessions used the display name after "Group: " — not a room id.
        return None
    except Exception:
        return _err(rid, 5122, _GROUP_PROBE_FAILED_MSG)
    if hosted or peer:
        owner = "its gateway" if hosted else "its home host"
        return _err(
            rid, 4122, f"This room is managed by {owner}. Update Hermes Desktop to continue it.")
    return None


def _parse_truncation_params(rid, sid, session, params, history):
    """Coerce + admit the truncation params; ``(target_row_id, client_ordinal, err)``.
    Malformed (4004) -> unconfirmed (4029; checked BEFORE target resolution so a
    leaked-state request never pays the durable read or heal-stamps live dicts).  An
    ordinal/id alone is not consent: a leftover ordinal on an ORDINARY submit is
    indistinguishable from a real rewind, and the cut is a destructive replace."""
    target_row_id = client_ordinal = None
    if (truncate_row_id := params.get("truncate_before_row_id")) is not None:
        target_row_id, err = _coerce_truncate_int(rid, truncate_row_id, "truncate_before_row_id")
        if err is not None:
            return None, None, err
    if (truncate_user_ordinal := params.get("truncate_before_user_ordinal")) is not None:
        client_ordinal, err = _coerce_truncate_int(rid, truncate_user_ordinal)
        if err is not None:
            return None, None, err
    if is_truthy_value(params.get("confirm_truncate")):
        return target_row_id, client_ordinal, None
    logger.warning(
        "prompt.submit: REFUSED unconfirmed truncation of session %s (%d messages held; "
        "ordinal=%s, row_id=%s, message_id=%s). The client attached truncation parameters without "
        "confirm_truncate — likely stale truncation parameters on an ordinary submit.",
        sid, len(history), client_ordinal, target_row_id, params.get("truncate_before_message_id"))
    return None, None, _err(
        rid, 4029,
        "truncation parameters require confirm_truncate=true; "
        "an ordinary prompt.submit must not drop session history "
        "(update your Hermes client if a rewind was intended)")


def _resolve_truncation_ordinal(rid, sid, session, params, history):
    """Resolve the truncation target to ``(ordinal, cut_index, err)``: unresolvable target
    (4018, fail closed — never degrade a missing row_id/message_id into an ordinal cut) ->
    ordinal drift (4030) -> ordinal-only on a durable session (4004)."""
    target_row_id, client_ordinal, err = _parse_truncation_params(
        rid, sid, session, params, history)
    if err is not None:
        return None, None, err
    truncate_message_id = params.get("truncate_before_message_id")
    # Client ordinals count the full displayed lineage; after compression ancestors live in
    # display_history_prefix, so count their user turns once to translate ordinals.
    prefix_user_count = len(_history_user_indices(session.get("display_history_prefix") or []))
    user_indices = _history_user_indices(history)

    def _stale(resolved_ordinal=None):
        # Recovery fields: Desktop resyncs + retries, "compressed away" when segment < 0.
        segment = (
            client_ordinal - prefix_user_count if client_ordinal is not None else resolved_ordinal)
        return None, None, _err(rid, 4018, _STALE_TARGET_MSG, data={
            "user_turn_count": len(user_indices), "ordinal": client_ordinal,
            "segment_ordinal": segment, "prefix_user_count": prefix_user_count})

    if target_row_id is not None or truncate_message_id is not None:
        if target_row_id is not None:
            param_name, target_repr = "truncate_before_row_id", target_row_id
            found_match = _resolve_truncate_row_id(session, history, target_row_id)
            not_found = "target row_id %d not found for session %s (in-memory + durable)"
        else:
            param_name = "truncate_before_message_id"
            target_repr = msg_id_str = str(truncate_message_id)
            found_match = next(
                ((u_ord, h_idx) for u_ord, h_idx in enumerate(user_indices)
                 if history[h_idx].get("id") == msg_id_str
                 or history[h_idx].get("message_id") == msg_id_str), None)
            not_found = "target message_id %s not found in history for session %s"
        if found_match is None:
            logger.warning(
                "prompt.submit: " + not_found + "; refusing truncation without fallback",
                target_repr, sid)
            return _stale()
        ordinal = found_match[0]
        # A stale client ordinal beside a *resolved* durable id is drift — never guess
        # which the user meant.  Client ordinals count the full displayed lineage, so
        # after compression ``ordinal + prefix_user_count`` is the SAME turn.  The cut is
        # always aimed by the durable target, so this can never re-aim a truncation.
        if client_ordinal is not None and client_ordinal != ordinal and not (
                prefix_user_count > 0 and client_ordinal == ordinal + prefix_user_count):
            logger.warning(
                "prompt.submit: REFUSED truncation due to ordinal mismatch for session %s "
                "(ordinal=%d, %s_ordinal=%d, %s=%s, prefix_user_count=%d). "
                "Stale truncate_before_user_ordinal detected.",
                sid, client_ordinal, param_name, ordinal, param_name, target_repr,
                prefix_user_count)
            return None, None, _err(
                rid, 4030,
                f"truncate_before_user_ordinal ({client_ordinal}) does not match "
                f"{param_name} target turn ({ordinal})")
    else:
        ordinal = client_ordinal - prefix_user_count
        if ordinal < 0 or ordinal >= len(user_indices):
            return _stale()
        # Ordinal-only cut on a durable session: durability is a state.db property, not a
        # live-copy annotation (resume paths omitted _row_id stamps); an unreadable
        # durable state fails closed too.
        has_stamped_user = any(
            _message_row_id(history[h_idx]) is not None for h_idx in user_indices)
        durable = [] if has_stamped_user else _load_durable_truncation_history(session, sid)
        if has_stamped_user or durable is None or durable:
            logger.warning(
                "prompt.submit: REFUSED ordinal-only truncation of durable "
                "session %s (ordinal=%d); truncate_before_row_id required",
                sid, client_ordinal)
            return None, None, _err(
                rid, 4004,
                "ordinal-only truncation is unsafe for durable session history; "
                "include truncate_before_row_id")
    # BOTH ends: a negative ordinal would index user_indices[-1] and persist the loss.
    if ordinal < 0 or ordinal >= len(user_indices):
        return _stale(resolved_ordinal=ordinal)
    return ordinal, user_indices[ordinal], None


def _row_ids_of(messages) -> set:
    return {row_id for message in messages if isinstance((row_id := _message_row_id(message)), int)}


def _truncate_history_for_submit(rid, sid, session, params, requested_rebind_ids):
    """Rewind/regenerate cut under ``history_lock``: ``(err, survivor_fields)``; the fields
    are the client rowId-rebind payload."""
    history = _history_without_ephemeral_scaffolding(session.get("history", []))
    ordinal, cut_index, err = _resolve_truncation_ordinal(rid, sid, session, params, history)
    if err is not None:
        return err, {}
    from agent.context_compressor import history_before_user_originated_turn
    truncated, _live_view = history_before_user_originated_turn(history, cut_index)
    # Second gate: ordinal 0 would DELETE every durable row; wiping needs its own opt-in.
    if not truncated and history and not is_truthy_value(params.get("confirm_empty_truncate")):
        logger.warning(
            "prompt.submit: REFUSED empty truncation of session %s "
            "(%d messages would be wiped; ordinal=%d).",
            sid, len(history), ordinal)
        return _err(
            rid, 4028,
            "truncation would erase the entire session transcript; "
            "resubmit with confirm_empty_truncate=true if this is intended"), {}
    log_fn = logger.warning if not truncated else logger.info
    log_fn(
        "prompt.submit: truncating session %s history %d -> %d messages (ordinal=%d)",
        sid, len(history), len(truncated), ordinal)
    # Write the truncated transcript BEFORE touching memory (fail closed: a failed write
    # after the in-memory rewrite would stack the new exchange on the "undone" turns).
    # Writes through _session_db (profile sessions own their state.db).
    fields = {}
    with _session_db(session) as db:
        if db is not None:
            try:
                # NULL session_key (old CLI-origin sessions) would trip an FK violation.
                # active_only=True: replace only the live (active=1) rows. In-place compaction (#38763)
                # keeps the pre-compaction transcript as active=0/compacted=1 rows under this same session
                # key; a bare replace_messages() would DELETE that durable archive on every edit/regenerate
                # — the same bug class #80216 fixed for /retry. On an uncompacted session all rows are
                # active=1, so this is behaviorally identical to the full replace. archive_dropped: a rewind
                # overwrites turns the user may not have meant to drop, and this write is the last step
                # before they are gone — three reported incidents ended here with nothing to restore from
                # (#70516, #80763, #82756). Soft-archiving keeps them on disk (active=0) and in the FTS
                # index, so a mis-aimed cut is recoverable instead of terminal. The live transcript is
                # unchanged. Fall back to session id when session_key is NULL — CLI-origin sessions created
                # before the session_key default fix have no key, and replace_messages(None) triggers an FK
                # violation.
                truncation_key = session.get("session_key") or sid
                old_active_row_ids = _row_ids_of(history)
                if requested_rebind_ids is not None:
                    # Un-repaired pre-write active-id set: a rewritten row must never be
                    # mistaken for an untouched archived/ancestor row.
                    durable_rebind_history = _load_durable_truncation_history(
                        session, truncation_key, repair_alternation=False)
                    if durable_rebind_history is None:
                        raise RuntimeError("could not load durable row identities for truncation")
                    old_active_row_ids.update(_row_ids_of(durable_rebind_history))
                old_survivor_row_ids = [_message_row_id(message) for message in truncated]
                # active_only: a bare replace would DELETE the compaction archive (active=0
                # rows) on every edit.  archive_dropped: a mis-aimed cut stays recoverable.
                db.replace_messages(
                    truncation_key, truncated, active_only=True, archive_dropped=True,
                    reject_active_turn_lease=True)
            except Exception as exc:
                logger.error(
                    "prompt.submit: replace_messages failed for session %s (ordinal=%d); refusing "
                    "turn so memory and DB stay aligned: %s",
                    sid, ordinal, exc, exc_info=True)
                return _err(rid, 5008, f"failed to persist history truncation: {exc}"), {}
            # Survivors were re-inserted as NEW rows: surface the fresh ids so the client
            # rebinds its cached rowIds (else a second rewind refuses with 4018).  None
            # entries: the client must drop its cached id for that turn.
            if requested_rebind_ids is None:
                fields["survivor_user_row_ids"] = [
                    _message_row_id(truncated[i]) for i in _history_user_indices(truncated)]
            else:
                fields["survivor_row_id_map"] = row_id_map = {
                    str(old_row_id): new_row_id
                    for old_row_id, new_row_id in zip(
                        old_survivor_row_ids, (_message_row_id(message) for message in truncated))
                    if isinstance(old_row_id, int) and isinstance(new_row_id, int)
                    and old_row_id in requested_rebind_ids}
                for dropped_row_id in requested_rebind_ids.intersection(old_active_row_ids):
                    row_id_map.setdefault(str(dropped_row_id), None)
    session["history"] = truncated
    session["history_version"] = int(session.get("history_version", 0)) + 1
    return None, fields


def _persist_session_row_for_submit(rid, session):
    """Lazily persist the DB row now that the user sent a message (a branch becomes real
    here); the error reply is the only user-visible signal (desktop maps it to a toast)."""
    try:
        if _ensure_session_db_row(session) is False:
            return _err(
                rid, 5072,
                "session storage unavailable: "
                f"{_db_error or 'state.db could not be opened'} — the message "
                "was not saved; repair state.db and try again")
        _persist_branch_seed(session)
    except Exception as exc:
        from hermes_state_errors import is_disk_full_error
        with session["history_lock"]:
            session["running"] = False
            session["last_active"] = time.time()
            _clear_inflight_turn(session)
        if is_disk_full_error(exc):
            return _err(
                rid, 5070,
                "disk full: session storage could not be written — free some disk space and try again")
        logger.warning("prompt.submit: session persist failed: %s", exc, exc_info=True)
        return _err(rid, 5071, f"session storage could not be written: {exc}")
    return None


def _run_after_agent_ready(rid, sid, session, text, display_kind, hosted_terminal_callback):
    """Turn thread body: patient wait for a deferred build (a slow build must not eat the
    accepted in-flight message), then run."""
    # The wait delivers the prompt when the still-running build completes, honors a cancel promptly, notices
    # the user once past the slow threshold, and only errors when the build itself fails or the bounded cap
    # expires. See #63078.
    err = _wait_agent_for_prompt(session, rid, sid)
    if err:
        # Terminal frame + retained snapshot (not a bare "error" event): the snapshot is
        # the only way resume shows this to a disconnected client.
        _emit_terminal_turn_error(
            sid, session, (err.get("error") or {}).get("message", "agent initialization failed"),
            error_surface={"layer": "runtime", "code": "agent_init_failed", "retryable": True})
        with session["history_lock"]:
            session["running"] = False
            session["last_active"] = time.time()
        _emit("session.info", sid, _session_info(session.get("agent"), session))
        return
    with session["history_lock"]:
        if session.get("_turn_cancel_requested") or not session.get("running"):
            session["running"] = False
            _clear_inflight_turn(session)
            # Without this emit the turn vanishes silently after {"status": "streaming"}.
            _emit("error", sid, {"message": (
                "Turn cancelled before the agent was ready"
                if session.get("_turn_cancel_requested")
                else "Session no longer running before the agent was ready")})
            return
    # Bind the route after deferred construction, before the first provider turn.
    _persist_live_session_runtime(session)
    _run_prompt_submit(
        rid, sid, session, text, display_kind=display_kind,
        terminal_callback=hosted_terminal_callback)


_TRUNCATION_PARAMS = (
    "truncate_before_user_ordinal", "truncate_before_row_id", "truncate_before_message_id")


def _lock_in_submit_turn(
    rid, sid, session, text, params, has_truncation, requested_rebind_ids, hosted_task):
    """Under ``history_lock``: refuse watch-child races / malformed truncation, apply the
    cut, mark the turn running + in flight.  Returns ``(err, survivor_fields)``."""
    fields = {}
    with session["history_lock"]:
        # A watch session's run lives in the PARENT turn (own running flag False); typing
        # mid-run would build a second agent racing the child on the same stored session.
        if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
            return _err(rid, 4009, "subagent still running — wait for it to finish"), fields
        if is_truthy_value(params.get("confirm_truncate")) and not has_truncation:
            return _err(
                rid, 4004,
                "confirm_truncate requires truncate_before_user_ordinal, truncate_before_message_id, or truncate_before_row_id",
            ), fields
        if has_truncation:
            err, fields = _truncate_history_for_submit(
                rid, sid, session, params, requested_rebind_ids)
            if err is not None:
                return err, {}
        session["running"] = True
        session["_turn_cancel_requested"] = False
        session["last_active"] = time.time()
        if hosted_task is not None:
            session["_hosted_room_task"] = dict(hosted_task)
        _start_inflight_turn(session, text)
    return None, fields


@method("prompt.submit")
def _(rid, params: dict) -> dict:
    from hermes_cli.input_sanitize import sanitize_user_prompt_text
    sid = params.get("session_id", "")
    raw_text = params.get("text", "")
    text = sanitize_user_prompt_text(raw_text) if isinstance(raw_text, str) else raw_text
    # Off-screen sends (widget intents) type the row so no client renders a bubble;
    # whitelisted to "hidden" — this RPC must not mint kinds.
    display_kind = "hidden" if params.get("display_kind") == "hidden" else None
    if (stopped := _typed_stop_phrase_response(rid, text)) is not None:
        return stopped
    if params.get("interrupted"):
        # Client-side barge-in: latch so this turn's model message carries the note.
        from tools.tts_streaming import mark_speech_interrupted
        mark_speech_interrupted()
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    hosted_task = params.get("_hosted_task")
    hosted_terminal_callback = params.get("_hosted_terminal_callback")
    internal_hosted_submit = hosted_task is not None or hosted_terminal_callback is not None
    err = (
        _hosted_submit_error(rid, session, hosted_task, hosted_terminal_callback)
        if internal_hosted_submit else _legacy_group_fence_error(rid, session, params))
    if err is not None:
        return err
    if (limit_message := _ensure_active_session_slot(sid, session)) is not None:
        # Refused HERE — before the busy queue, db row and agent build — so a refusal
        # leaves the session untouched.  The reason travels as machine-readable data.
        reason = getattr(limit_message, "reason", None)
        return _err(rid, 4090, str(limit_message), {"reason": reason} if reason else None)
    # Rewritten every submit: a session alternates app window / HUD; stale "hud" misinforms.
    session["client_surface"] = "hud" if params.get("surface") == "hud" else ""
    has_truncation = any(params.get(k) is not None for k in _TRUNCATION_PARAMS)
    if has_truncation and isinstance(text, str):
        # A rewind replays what the transcript shows: re-expand a skill invocation or
        # `/work fix it` sends nine literal chars.
        text = _expand_skill_invocation_for_replay(text, str(session.get("session_key") or ""))
    turn_isolation = _session_uses_compute_host(session, _load_dashboard_process_isolation_config())
    if internal_hosted_submit and turn_isolation:
        return _err(rid, 4121, "hosted room turns do not support isolated compute workers yet")
    # Re-bind to the current transport: streaming must stay on the active websocket even
    # if a disconnect/fallback moved the session to stdio.
    if (t := current_transport()) is not None:
        session["transport"] = t
    # Claim the turn against a possibly-running session (busy/queued reply, else fall
    # through once ``running`` is observed False).  The provider interrupt happens after
    # history_lock is released (a non-interruptible tool may hold it); if the old turn
    # finished between the two acquisitions, retry the claim rather than strand this
    # prompt in a queue whose drain already ran.
    while True:
        with session["history_lock"]:
            if not session.get("running"):
                break
            if internal_hosted_submit:
                return _err(rid, 4091, "hosted room member session is busy")
            busy_transport = t or session.get("transport")
        busy_response = _handle_busy_submit(
            rid, sid, session, text, busy_transport, queued=bool(params.get("queued")))
        if busy_response is not None:
            return busy_response
    raw_rebind_ids = params.get("rebind_survivor_row_ids")
    requested_rebind_ids = (
        {r for r in raw_rebind_ids if isinstance(r, int) and not isinstance(r, bool)}
        if isinstance(raw_rebind_ids, list) else None)
    err, survivor_fields = _lock_in_submit_turn(
        rid, sid, session, text, params, has_truncation, requested_rebind_ids, hosted_task)
    if err is not None:
        return err
    if turn_isolation:
        isolated_response = _submit_prompt_to_compute_host(
            rid, sid, session, text, display_kind=display_kind)
        if not isolated_response.get("error"):
            # The truncation already happened inline above (memory + DB).
            isolated_response["result"].update(survivor_fields)
            return isolated_response
        # An ordinal/id alone is not consent. A client that carries a leftover ordinal into an ORDINARY
        # submit sends a request that is indistinguishable, field by field, from a real rewind — same
        # method, same shape, an in-range target — and the cut it asks for is a destructive
        # replace_messages() the user never requested (#80763: 296 -> 52 messages, 244 durable rows gone).
        # Only the client knows whether this submit is a rewind/edit/regenerate, so it has to say so; refuse
        # the cut when it doesn't. Consent is checked BEFORE target resolution: an unconfirmed
        # (leaked-state) request must refuse with 4029 without paying the durable transcript read or
        # heal-stamping live history dicts that row-id resolution performs.
        logger.warning(
            "compute-host dispatch failed for session %s; falling back inline: %s", sid,
            isolated_response["error"].get("message", "unknown error"))
    if (err := _persist_session_row_for_submit(rid, session)) is not None:
        return err
    # A completed FAILED build must not wedge the session: rebuild, don't replay it.
    if not _restart_completed_failed_agent_build(sid, session, session.get("agent_ready")):
        _start_agent_build(sid, session)
    run_thread = threading.Thread(
        target=lambda: _run_after_agent_ready(
            rid, sid, session, text, display_kind, hosted_terminal_callback),
        daemon=True)
    # Handle lets session.interrupt tell a live turn from a stuck `running` flag.
    session["_run_thread"] = run_thread
    run_thread.start()
    return _ok(rid, {"status": "streaming", **survivor_fields})


# ── attachments ─────────────────────────────────────────────────────────────

def _attached_image_result(session, image_path, **extra) -> dict:
    """Common ``{attached, path, count, ...meta}`` reply after queuing an image."""
    return {
        "attached": True, "path": str(image_path), "count": len(session["attached_images"]),
        **extra, **_image_meta(image_path)}


@method("clipboard.paste")
def _(rid, params: dict) -> dict:
    session, err = _sess_building(params, rid)
    if err:
        return err
    try:
        from hermes_cli.clipboard import has_clipboard_image, save_clipboard_image
    except Exception as e:
        return _err(rid, 5027, f"clipboard unavailable: {e}")
    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _session_images_dir(session)
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = (
        img_dir / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['image_counter']}.png")
    # Save-first (CLI keybinding parity): more robust than a has_image() precheck.
    if not save_clipboard_image(img_path):
        session["image_counter"] = max(0, session["image_counter"] - 1)
        return _ok(rid, {"attached": False, "message": (
            "Clipboard has image but extraction failed" if has_clipboard_image()
            else "No image found in clipboard")})
    session.setdefault("attached_images", []).append(str(img_path))
    return _ok(rid, _attached_image_result(session, img_path))


@method("image.attach")
def _(rid, params: dict) -> dict:
    session, err = _sess_building(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    try:
        from cli import (
            _IMAGE_EXTENSIONS, _detect_file_drop, _resolve_attachment_path, _split_path_input)
        if dropped := _detect_file_drop(raw):
            image_path, remainder = dropped["path"], dropped["remainder"]
        else:
            path_token, remainder = _split_path_input(raw)
            image_path = _resolve_attachment_path(path_token)
            if image_path is None:
                return _err(rid, 4016, f"image not found: {path_token}")
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _err(rid, 4016, f"unsupported image: {image_path.name}")
        session.setdefault("attached_images", []).append(str(image_path))
        return _ok(rid, _attached_image_result(
            session, image_path,
            remainder=remainder, text=remainder or f"[User attached image: {image_path.name}]"))
    except Exception as e:
        return _err(rid, 5027, str(e))


@method("image.attach_bytes")
def _(rid, params: dict) -> dict:
    """Attach an image from base64 bytes (remote client); reply mirrors ``image.attach``.
    ``filename``/``ext`` hint the extension, else magic bytes decide (fallback ``.png``)."""
    session, err = _sess_building(params, rid)
    if err:
        return err
    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_b64:
        return _err(rid, 4015, "content_base64 required")
    img_bytes, err = _decode_attach_payload(
        rid, raw_b64, mime_prefix="image/", max_bytes=_ATTACH_BYTES_MAX_BYTES,
        label="image", empty_msg="image is empty")
    if err is not None:
        return err
    filename = str(params.get("filename", "") or "")
    ext_hint = str(params.get("ext", "") or "").strip().lower()
    if ext_hint and not ext_hint.startswith("."):
        ext_hint = "." + ext_hint
    ext = _sniff_image_ext(img_bytes, filename or (f"x{ext_hint}" if ext_hint else ""))
    if ext not in _allowed_image_extensions():
        return _err(rid, 4016, f"unsupported image extension: {ext}")
    try:
        img_path = _queue_attached_image(session, img_bytes, ext, prefix="upload")
    except Exception as e:
        return _err(rid, 5027, f"write failed: {e}")
    return _ok(rid, _attached_image_result(
        session, img_path,
        remainder="", text=f"[User attached image: {img_path.name}]", bytes=len(img_bytes)))


def _pdf_attach_source(rid, params, td_path, raw_path, raw_b64):
    """Materialize the PDF to render: ``(pdf_path, display_name, err)``."""
    if raw_b64:
        pdf_bytes, err = _decode_attach_payload(
            rid, raw_b64, mime_prefix="application/pdf", max_bytes=_PDF_ATTACH_MAX_BYTES,
            label="PDF", empty_msg="decoded PDF is empty")
        if err is not None:
            return None, None, err
        if pdf_bytes[:5] != b"%PDF-":
            return None, None, _err(rid, 4017, "payload is not a PDF (missing %PDF- magic bytes)")
        pdf_path = td_path / "input.pdf"
        pdf_path.write_bytes(pdf_bytes)
        return pdf_path, str(params.get("filename", "") or "uploaded.pdf"), None
    try:
        from cli import _resolve_attachment_path
        resolved = _resolve_attachment_path(raw_path)
    except Exception:
        resolved = None
    if resolved is None or not (pdf := Path(resolved)).is_file():
        return None, None, _err(rid, 4016, f"PDF not found: {raw_path}")
    if pdf.suffix.lower() != ".pdf":
        return None, None, _err(rid, 4016, f"not a PDF: {pdf.name}")
    if pdf.stat().st_size > _PDF_ATTACH_MAX_BYTES:
        mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
        return None, None, _err(rid, 4018, f"PDF too large; cap is {mb} MB")
    return pdf, pdf.name, None


def _pdf_page_range(rid, params):
    """Validate first/last page against the per-call cap: ``(first, last, err)``."""
    try:
        first_page = int(params.get("first_page") or 1)
        last_page = None if params.get("last_page") is None else int(params.get("last_page"))
    except (TypeError, ValueError):
        return None, None, _err(rid, 4015, "first_page/last_page must be integers")
    if first_page < 1:
        return None, None, _err(rid, 4015, "first_page must be >= 1")
    if last_page is None:
        last_page = first_page + _PDF_ATTACH_MAX_PAGES - 1
    if last_page < first_page:
        return None, None, _err(rid, 4015, "last_page must be >= first_page")
    if last_page - first_page + 1 > _PDF_ATTACH_MAX_PAGES:
        return None, None, _err(
            rid, 4019, f"page range exceeds cap of {_PDF_ATTACH_MAX_PAGES} pages per attach call")
    return first_page, last_page, None


@method("pdf.attach")
def _(rid, params: dict) -> dict:
    """Attach a PDF by rendering each page to PNG (``pdftoppm``; 5028 if missing) and
    queuing the pages as images.  Host ``path`` or base64 ``content_base64``."""
    import shutil
    import subprocess
    import tempfile
    session, err = _sess_building(params, rid)
    if err:
        return err
    if shutil.which("pdftoppm") is None:
        return _err(rid, 5028, "pdftoppm not installed (poppler-utils package required)")
    raw_path = str(params.get("path", "") or "").strip()
    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_path and not raw_b64:
        return _err(rid, 4015, "path or content_base64 required")
    with tempfile.TemporaryDirectory(prefix="pdf_attach_") as td:
        td_path = Path(td)
        pdf_path, display_name, err = _pdf_attach_source(rid, params, td_path, raw_path, raw_b64)
        if err is not None:
            return err
        first_page, last_page, err = _pdf_page_range(rid, params)
        if err is not None:
            return err
        argv = [
            "pdftoppm", "-png", "-r", "150", "-f", str(first_page), "-l", str(last_page),
            str(pdf_path), str(td_path / "page")]
        from hermes_cli._subprocess_compat import windows_hide_flags
        try:
            # UTF-8 + lossy decode: non-UTF-8 child output must not crash the gateway
            # thread on locale-mismatched Windows.
            res = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                encoding="utf-8", errors="replace", creationflags=windows_hide_flags())
        except subprocess.TimeoutExpired:
            return _err(rid, 5028, "pdftoppm timed out (>120s)")
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
            return _err(rid, 5028, "pdftoppm failed: " + " | ".join(tail))
        rendered = sorted(td_path.glob("page-*.png"))
        if not rendered:
            return _err(rid, 5028, "pdftoppm produced no pages (corrupt PDF?)")
        attached_pages = []
        for src in rendered:
            page_num = src.stem.split("-", 1)[-1]
            try:
                page_int = int(page_num)
            except ValueError:
                page_int = first_page + len(attached_pages)
            dst = _queue_attached_image(
                session, src.read_bytes(), ".png", prefix=f"pdf_p{page_num}")
            attached_pages.append({"path": str(dst), "page": page_int, **_image_meta(dst)})
        return _ok(rid, {
            "attached": True, "filename": display_name, "pages_attached": len(attached_pages),
            "pages": attached_pages, "count": len(session["attached_images"]),
            "text": f"[User attached PDF: {display_name} ({len(attached_pages)} page(s))]"})


@method("file.attach")
def _(rid, params: dict) -> dict:
    """Stage a non-image file into the session workspace; returns a workspace-relative
    ``@file:`` ref.  ``data_url`` carries the bytes when ``path`` isn't gateway-visible."""
    session, err = _sess_building(params, rid)
    if err:
        return err
    raw, data_url, name = (
        str(params.get(k, "") or "").strip() for k in ("path", "data_url", "name"))
    if not raw and not data_url:
        return _err(rid, 4015, "path or data_url required")
    try:
        stored_path, uploaded = _stage_session_file_attachment(
            session, raw_path=raw, data_url=data_url, name=name)
        ref_path = _attachment_ref_path(session, stored_path)
        return _ok(rid, {
            "attached": True, "name": stored_path.name, "path": str(stored_path),
            "ref_path": ref_path, "ref_text": f"@file:{_format_ref_value(ref_path)}",
            "uploaded": uploaded})
    except Exception as e:
        return _err(rid, 5028, str(e))


@method("image.detach")
def _(rid, params: dict) -> dict:
    session, err = _sess_building(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    before = len(images := session.setdefault("attached_images", []))
    session["attached_images"] = [path for path in images if path != raw]
    return _ok(rid, {
        "detached": len(session["attached_images"]) != before,
        "count": len(session["attached_images"])})


@method("input.detect_drop")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    try:
        from cli import _detect_file_drop
        dropped = _detect_file_drop(str(params.get("text", "") or ""))
        if not dropped:
            return _ok(rid, {"matched": False})
        drop_path, remainder = dropped["path"], dropped["remainder"]
        if dropped["is_image"]:
            session.setdefault("attached_images", []).append(str(drop_path))
            return _ok(rid, {
                "matched": True, "is_image": True, "path": str(drop_path),
                "count": len(session["attached_images"]),
                "text": remainder or f"[User attached image: {drop_path.name}]",
                **_image_meta(drop_path)})
        text = f"[User attached file: {drop_path}]" + (f"\n{remainder}" if remainder else "")
        return _ok(rid, {
            "matched": True, "is_image": False, "path": str(drop_path), "name": drop_path.name,
            "text": text})
    except Exception as e:
        return _err(rid, 5027, str(e))


# ── side agents (background / btw / preview.restart) ────────────────────────

def _final_response_text(result) -> str:
    return (result.get("final_response", str(result)) if isinstance(result, dict) else str(result))


def _spawn_side_agent(
    rid, session, task_id, parent, event, body, *, cwd="", extra=None, cleanup=None):
    """Run ``body()`` on a daemon thread under the session's profile home (the ContextVar
    doesn't propagate across threads) and cwd; its text — or ``error: <exc>`` — lands on
    ``parent`` as ``event`` with ``task_id`` (+ ``extra``).  Replies ``{task_id}``."""
    extra = extra or {}

    def run():
        session_tokens = _set_session_context(task_id, cwd=(cwd or _session_cwd(session)))
        # Bug #50233: ephemeral agent threads don't inherit the session's HERMES_HOME override (the
        # ContextVar set on the session-create thread doesn't propagate here), so a background turn under a
        # non-default profile would run against the wrong home. Re-bind the override for the duration of
        # this turn, exactly as the normal prompt turn does, and restore it afterward.
        # Bug #50233: ephemeral preview-restart agent threads don't inherit the session's HERMES_HOME
        # override (the ContextVar set on the session-create thread doesn't propagate here). Re-bind it for
        # the duration of the turn, mirroring the normal prompt turn, then restore it. NOTE: we deliberately
        # do NOT close this agent through task-wide process cleanup — the whole point of preview.restart is
        # to leave a background server running under this task_id, and AIAgent.close() would kill every
        # process for the task_id and tear down the very server the restart just started.
        profile_home = session.get("profile_home")
        home_token = set_hermes_home_override(profile_home) if profile_home else None
        try:
            try:
                text = body()
            finally:
                if home_token is not None:
                    reset_hermes_home_override(home_token)
            _emit(event, parent, {"task_id": task_id, **extra, "text": text})
        except Exception as e:
            _emit(event, parent, {"task_id": task_id, **extra, "text": f"error: {e}"})
        finally:
            if cleanup is not None:
                cleanup()
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


def _side_agent_args(rid, params, prefix):
    """Shared admission for the side-agent RPCs: ``(session, text, parent, task_id, err)``."""
    session, err = _sess(params, rid)
    if err:
        return None, None, None, None, err
    text, parent = params.get("text", ""), params.get("session_id", "")
    if not text:
        return None, None, None, None, _err(rid, 4012, "text required")
    return session, text, parent, f"{prefix}_{uuid.uuid4().hex[:6]}", None


@method("prompt.background")
def _(rid, params: dict) -> dict:
    session, text, parent, task_id, err = _side_agent_args(rid, params, "bg")
    if err:
        return err

    def body():
        from run_agent import AIAgent
        result = AIAgent(**_background_agent_kwargs(session["agent"], task_id)).run_conversation(
            user_message=text, task_id=task_id)
        return _final_response_text(result)

    return _spawn_side_agent(rid, session, task_id, parent, "background.complete", body)


@method("prompt.btw")
def _(rid, params: dict) -> dict:
    """Side question over a snapshot of the live conversation (``agent/side_question.py``);
    history, alternation and prompt cache stay untouched.  Answer: ``btw.complete``."""
    session, text, parent, task_id, err = _side_agent_args(rid, params, "btw")
    if err:
        return err
    agent = session.get("agent")
    snapshot = list(getattr(agent, "_session_messages", None) or session.get("history") or [])
    main_runtime = {
        k: getattr(agent, k, None)
        for k in ("model", "provider", "base_url", "api_key", "api_mode")}

    def body():
        from agent.side_question import answer_side_question
        return answer_side_question(
            text, snapshot, parent_agent=agent, main_runtime=main_runtime) or ""

    return _spawn_side_agent(
        rid, session, task_id, parent, "btw.complete", body, extra={"question": text})


_PREVIEW_RESTART_RULES = (
    "Restart exactly the app intended for the Preview URL, not Hermes Desktop itself.",
    "The Preview URL and port are the target. Preserve that target unless you conclude it is impossible.",
    "If the prior conversation shows a specific command that bound this URL/port, prefer re-running THAT exact command (in the same cwd) over guessing a new one.",
    "First inspect what process, if any, owns the Preview URL port. If a stale server exists, inspect its cwd and prefer that cwd over the Hermes/Desktop process cwd.",
    "The Current working directory is only a hint. Do not assume it is the preview app root when the port owner or files indicate another root.",
    "If the console shows a module-script MIME error for src/main.tsx or similar, a static server is serving source files. Do not restart python -m http.server or any dumb static server for that app.",
    "For module-script MIME failures, inspect package.json/vite config in the candidate app root and start the real dev server/bundler (for example npm/pnpm/yarn dev) so module transforms happen.",
    "Before declaring success, verify the Preview URL responds with the intended app, not Hermes Desktop. If it serves Hermes/Desktop UI or another unrelated app, stop that process and report failure.",
    "Do not modify files. Do not ask the user unless blocked.",
    "Prefer existing project scripts or commands when they are clear.",
    "If a stale process owns the needed port, handle it safely.",
    "Start long-running servers detached/in the background, then return immediately.",
    "Do not run a foreground dev server command that blocks this background task.",
    "Keep the final response short: what command/server was started, or why it could not be restarted.",
)

_PREVIEW_RESTART_HISTORY_NOTE = (
    "The conversation history above is from the user's main session — including the commands you (the assistant) previously ran to start servers, edit files, or check ports. Use it to figure out exactly which server should be running at this Preview URL. The user did not start a brand new task; recover what they had working."
)


@method("preview.restart")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    url, cwd, context = (str(params.get(k) or "").strip() for k in ("url", "cwd", "context"))
    if not url:
        return _err(rid, 4012, "url required")
    task_id = f"preview_{uuid.uuid4().hex[:6]}"
    parent = params.get("session_id", "")
    parent_history = _preview_restart_history(session)
    prompt = "\n".join(
        line
        for line in [
            "The desktop preview pane cannot load a local server URL.",
            f"Preview URL: {url}",
            f"Current working directory: {cwd or '(unknown)'}",
            f"Preview console:\n{context}" if context else "",
            _PREVIEW_RESTART_HISTORY_NOTE if parent_history else None,
            *_PREVIEW_RESTART_RULES]
        if line)
    # A malformed client path (embedded NUL, etc.) is "no validated cwd".
    try:
        preview_cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else ""
        if preview_cwd and not os.path.isdir(preview_cwd):
            preview_cwd = ""
    except Exception:
        preview_cwd = ""

    def body():
        from run_agent import AIAgent
        from tools.terminal_tool import register_task_env_overrides
        if preview_cwd:
            register_task_env_overrides(task_id, {"cwd": preview_cwd})
        history_note = (
            f" (with {len(parent_history)} parent-session messages of context)"
            if parent_history else "")
        _emit(
            "preview.restart.progress", parent,
            {"task_id": task_id, "text": f"Starting hidden restart agent{history_note}"})
        # Deliberately NOT closed via AIAgent.close(): it would kill the background
        # server this task exists to leave running.
        result = AIAgent(
            **_ephemeral_preview_agent_kwargs(session["agent"], task_id),
            **_preview_restart_callbacks(parent, task_id),
        ).run_conversation(
            user_message=prompt, task_id=task_id, conversation_history=parent_history or None)
        return _final_response_text(result)

    def cleanup():
        with contextlib.suppress(Exception):
            from tools.terminal_tool import clear_task_env_overrides
            clear_task_env_overrides(task_id)

    # Pin the validated preview cwd, else the parent workspace — never an invalid path.
    return _spawn_side_agent(
        rid, session, task_id, parent, "preview.restart.complete", body,
        cwd=preview_cwd, cleanup=cleanup)


# ── late-answer RPCs for tool-driven UI cards ───────────────────────────────
# allow_expired=True everywhere: a tool's bounded wait can expire (its _pending entry
# popped) while the card is still visible; a late answer must not surface the raw 4009.


@method("clarify.respond")
def _(rid, params: dict) -> dict:
    if proxied := _respond_compute_host_clarify(rid, params):
        return proxied
    return _respond(rid, params, "answer", allow_expired=True)


_LATE_RESPOND_KEYS = {
    "terminal.read.respond": "text", "preview.read.respond": "text", "preview.act.respond": "text",
    "window.read.respond": "text", "tour.respond": "text", "mcp.setup.respond": "result",
    "sudo.respond": "password", "secret.respond": "value"}
for _name, _key in _LATE_RESPOND_KEYS.items():
    method(_name)(lambda rid, params, _k=_key: _respond(rid, params, _k, allow_expired=True))
del _name, _key


# ── approvals ───────────────────────────────────────────────────────────────

def _approval_reply(rid, result_key, call):
    """``_ok({result_key: call(tools.approval)})``, 5004 on any failure."""
    try:
        import tools.approval as approval
        return _ok(rid, {result_key: call(approval)})
    except Exception as e:
        return _err(rid, 5004, str(e))


@method("approval.pending")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    return _approval_reply(
        rid, "approvals", lambda a: a.list_gateway_approvals(session["session_key"]))


@method("approval.received")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    if not isinstance(request_id := params.get("request_id"), str) or not request_id:
        return _err(rid, 4006, "request_id required")
    return _approval_reply(
        rid, "acknowledged", lambda a: a.ack_gateway_approval(session["session_key"], request_id))


def _approval_respond_session_fallback(params: dict):
    """Durable-identity fallback for a stale live sid (re-minted after a reconnect while
    the prompt stayed on screen): (1) the ``request_id`` against every live session's
    pending approvals, then (2) ``session_id`` as a STORED id.  Live session or None.

    See #91684.
    """
    request_id = str(params.get("request_id") or "")
    if request_id:
        try:
            from tools.approval import list_gateway_approvals
            with _sessions_lock:
                live = list(_sessions.items())
            for sid, session in live:
                key = str(session.get("session_key") or "")
                if key and any(
                    str(pending.get("request_id") or "") == request_id
                    for pending in list_gateway_approvals(key)):
                    return session
        except Exception:
            logger.debug("approval.respond request_id fallback failed", exc_info=True)
    if target := str(params.get("session_id") or ""):
        try:
            if (live := _find_live_session_by_key(target)) is not None:
                return live[1]
        except Exception:
            logger.debug("approval.respond stored-id fallback failed", exc_info=True)
    return None


@method("approval.respond")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        # Session-not-found (4001) only: resolve by durable identity before failing.
        if (err.get("error") or {}).get("code") != 4001:
            return err
        session = _approval_respond_session_fallback(params)
        if session is None:
            return err
    return _approval_reply(
        rid, "resolved",
        lambda a: a.resolve_gateway_approval(
            session["session_key"], params.get("choice", "deny"),
            resolve_all=params.get("all", False), request_id=params.get("request_id")))


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import types  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
