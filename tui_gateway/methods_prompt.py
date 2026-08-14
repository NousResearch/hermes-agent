"""Prompt / attachment / respond JSON-RPC handlers (moved verbatim from server.py).

Handler bodies are byte-identical to their pre-split server.py form; they
are rebound onto server.py's globals at install time — see method_ctx.py.
"""

from .method_ctx import HandlerRegistry

import types

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _history_user_indices(history: list) -> list:
    """Indices of model-visible user turns (excludes display_kind timeline markers)."""
    return [
        i
        for i, m in enumerate(history)
        if m.get("role") == "user" and not m.get("display_kind")
    ]


def _message_row_id(msg: dict):
    """Parse durable SQLite row id from a history entry, or None."""
    raw = msg.get("_row_id")
    if raw is None:
        raw = msg.get("row_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _mem_db_pair_agrees(mem, db_msg) -> bool:
    """True when a live-memory entry plausibly corresponds to a durable row.

    Positional trust across the live and durable lists needs evidence, not
    just equal lengths/ordinals: roles must match, display-marker status must
    match (a marker living only on one side shifts every later position), and
    an addressable user turn must show the same text. Non-string (multimodal)
    content can't be compared cheaply — role/marker agreement suffices there.
    Self-contained on builtins: register() rebinds callers onto server
    globals, so any helper this calls must be in that namespace too.
    """
    if not isinstance(mem, dict) or not isinstance(db_msg, dict):
        return False
    if mem.get("role") != db_msg.get("role"):
        return False
    if bool(mem.get("display_kind")) != bool(db_msg.get("display_kind")):
        return False
    if mem.get("role") == "user" and not mem.get("display_kind"):
        mem_content = mem.get("content")
        db_content = db_msg.get("content")
        if (
            isinstance(mem_content, str)
            and isinstance(db_content, str)
            and mem_content.strip() != db_content.strip()
        ):
            return False
    return True


def _find_user_turn_by_row_id(history: list, target_row_id: int):
    """Return ``(user_ordinal, history_index)`` for ``target_row_id``, or None."""
    for u_ord, h_idx in enumerate(_history_user_indices(history)):
        if _message_row_id(history[h_idx]) == target_row_id:
            return u_ord, h_idx
    return None


def _resolve_truncate_row_id(session: dict, history: list, target_row_id: int):
    """Resolve ``truncate_before_row_id`` to ``(user_ordinal, history_index)``.

    Prefer in-memory ``_row_id`` / ``row_id`` stamps. When a live turn rewrote
    ``session["history"]`` without stamps (provider-format messages), load the
    session's durable transcript with ``include_row_ids=True`` and map the
    matched user-turn ordinal onto the live list. Does **not** fall back to a
    client-supplied ordinal — unknown row ids must refuse (#82959).
    """
    hit = _find_user_turn_by_row_id(history, target_row_id)
    if hit is not None:
        return hit

    session_key = str(session.get("session_key") or "")
    if not session_key:
        return None

    try:
        db = _get_db()
    except Exception:
        db = None
    if db is None:
        return None

    get_conv = getattr(db, "get_messages_as_conversation", None)
    if not callable(get_conv):
        return None

    try:
        db_history = get_conv(
            session_key, repair_alternation=True, include_row_ids=True
        )
    except Exception:
        logger.debug(
            "prompt.submit: failed loading DB history for row_id %s session %s",
            target_row_id,
            session_key,
            exc_info=True,
        )
        return None

    if not isinstance(db_history, list):
        return None

    # Heal missing in-memory stamps when the live list still lines up 1:1 with
    # the durable transcript (common after turn-completion rewrites). Equal
    # length alone is NOT proof of alignment: the durable copy above is loaded
    # with repair_alternation=True (which can merge/drop rows) while the live
    # list is unrepaired, and memory can carry optimistic/marker rows — so the
    # two can coincide in length while position-shifted. A positional stamp on
    # a misaligned pair is sticky and re-aims every later rewind at the wrong
    # durable row. Stamp only when EVERY pair agrees (all-or-nothing): roles
    # must match on every pair, and addressable user turns must match content.
    if len(db_history) == len(history) and all(
        _mem_db_pair_agrees(mem, db_msg)
        for mem, db_msg in zip(history, db_history)
    ):
        for mem, db_msg in zip(history, db_history):
            db_rid = _message_row_id(db_msg) if isinstance(db_msg, dict) else None
            if db_rid is not None and _message_row_id(mem) is None:
                mem["_row_id"] = db_rid
        hit = _find_user_turn_by_row_id(history, target_row_id)
        if hit is not None:
            return hit

    db_hit = _find_user_turn_by_row_id(db_history, target_row_id)
    if db_hit is None:
        return None
    db_ord, db_idx = db_hit
    mem_user_indices = _history_user_indices(history)
    if db_ord < 0 or db_ord >= len(mem_user_indices):
        return None
    mem_idx = mem_user_indices[db_ord]
    # Same-ordinal mapping across two lists that can diverge (the repaired
    # durable copy may have merged a user;user pair, shifting every later
    # user ordinal). Trust the mapping only when the mapped live turn shows
    # the same content as the durable target — otherwise refuse (the caller
    # returns fail-closed 4018) rather than cut the wrong turn (#82959).
    if not _mem_db_pair_agrees(history[mem_idx], db_history[db_idx]):
        return None
    return db_ord, mem_idx


def _coerce_truncate_int(rid, value, param_name="truncate_before_user_ordinal"):
    """Return ``(int_value, error_response)`` for a client-supplied integer param.

    bool is an int subclass: a JSON ``true`` would coerce via int() to
    1 and aim a confirmed rewind at the wrong turn — refuse it like any
    other non-integer.
    """
    if isinstance(value, bool):
        return None, _err(rid, 4004, f"{param_name} must be an integer")
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, _err(rid, 4004, f"{param_name} must be an integer")


def _reconcile_client_ordinal(rid, sid, client_ordinal, msg_ordinal, param_name, target_repr):
    """Cross-check a client ordinal against a resolved durable target.

    Returns ``(ordinal, error_response)``: the target's ordinal when the
    client sent none or agreed, else the 4004/4030 refusal. A stale ordinal
    alongside a *resolved* durable id is the #82756 drift class — refuse
    rather than guess which address the user meant.
    """
    if client_ordinal is None:
        return msg_ordinal, None
    ordinal, err = _coerce_truncate_int(rid, client_ordinal)
    if err is not None:
        return None, err
    if ordinal != msg_ordinal:
        logger.warning(
            "prompt.submit: REFUSED truncation due to ordinal mismatch for session %s "
            "(ordinal=%d, %s_ordinal=%d, %s=%s). "
            "Stale truncate_before_user_ordinal detected.",
            sid,
            ordinal,
            param_name,
            msg_ordinal,
            param_name,
            target_repr,
        )
        return None, _err(
            rid,
            4030,
            f"truncate_before_user_ordinal ({ordinal}) does not match "
            f"{param_name} target turn ({msg_ordinal})",
        )
    return ordinal, None


def _pending_reaction_notes(session: dict) -> str:
    """Note block describing reactions the user added since the last turn, or "".

    Applied to the MODEL INPUT only (``run_message``, beside the
    speech-interrupted note) — never to the text that gets persisted. Prefixing
    the persisted prompt bakes scaffolding into the transcript. Each reaction is
    announced once — the row is stamped ``seen`` on read.
    """
    session_key = str(session.get("session_key") or "")
    if not session_key:
        return ""

    # Feature-gated (off by default, Settings → Appearance): when disabled the
    # model hears nothing, even about reactions set while it was on.
    try:
        display = _load_cfg().get("display")
        if not (isinstance(display, dict) and bool(display.get("message_reactions", False))):
            return ""
    except Exception:
        return ""

    try:
        with _session_db(session) as db:
            if db is None:
                return ""
            pending = db.take_unseen_reactions(session_key, author="user")
    except Exception:
        logger.debug("Failed to read pending reactions", exc_info=True)
        return ""

    if not pending:
        return ""

    notes = []
    for entry in pending:
        snippet = (entry.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "…"
        emoji = entry.get("emoji") or ""
        whose = "their own" if entry.get("role") == "user" else "your"
        if snippet:
            notes.append(f'[The user reacted {emoji} to {whose} message: "{snippet}"]')
        else:
            # A row with no plain text (attachment-only, or a tool-call-only
            # assistant turn) — an empty quote reads worse than no quote.
            notes.append(f"[The user reacted {emoji} to {whose} earlier message]")

    return "\n".join(notes)


@method("prompt.submit")
def _(rid, params: dict) -> dict:
    from hermes_cli.input_sanitize import sanitize_user_prompt_text

    sid = params.get("session_id", "")
    raw_text = params.get("text", "")
    text = sanitize_user_prompt_text(raw_text) if isinstance(raw_text, str) else raw_text
    # Typed bare stop phrase while backend voice mode is active ends the
    # voice chat instead of sending "stop" to the agent — the typed twin of
    # the spoken stop phrase (PR #73106), applied at the ONE server-side
    # choke point every TUI submit passes through. Guarded on voice mode
    # being ON: typed "stop" outside a voice chat is a normal message.
    # (The desktop's voice conversation is renderer-owned and never flips
    # the backend flag, so it handles its own typed stop client-side.)
    if isinstance(text, str) and _voice_mode_enabled():
        try:
            from tools.voice_mode import is_voice_stop_phrase

            typed_stop = is_voice_stop_phrase(text)
        except Exception:
            typed_stop = False
        if typed_stop:
            os.environ["HERMES_VOICE"] = "0"
            os.environ["HERMES_VOICE_TTS"] = "0"
            try:
                from hermes_cli.voice import stop_continuous

                stop_continuous()
            except Exception:
                pass
            try:
                _tts_stream_stop(user_barge=False)
            except Exception:
                pass
            _voice_emit("voice.transcript", {"stop_phrase": True, "typed": True})
            logger.info("prompt.submit: typed stop phrase — voice chat ended")
            return _ok(rid, {"voice_stopped": True})
    truncate_user_ordinal = params.get("truncate_before_user_ordinal")
    if params.get("interrupted"):
        # Client-side barge-in (desktop VAD / typing over playback) — latch it
        # so this turn's model message carries the interruption note.
        from tools.tts_streaming import mark_speech_interrupted

        mark_speech_interrupted()
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if (limit_message := _ensure_active_session_slot(sid, session)) is not None:
        return _err(rid, 4090, limit_message)
    # Which desktop window this message was typed into. Rewritten on every
    # submit, because one session can be driven from the app window and the HUD
    # in turn: a stale "hud" would tell the model the user is still floating
    # over another app when they are back in Hermes.
    session["client_surface"] = "hud" if params.get("surface") == "hud" else ""
    has_truncation = (
        truncate_user_ordinal is not None
        or params.get("truncate_before_row_id") is not None
        or params.get("truncate_before_message_id") is not None
    )
    if has_truncation and isinstance(text, str):
        # A rewind/regenerate replays a turn from what the transcript shows. A
        # skill turn shows its invocation, so re-expand it here — otherwise
        # re-running `/work fix it` sends the agent nine literal characters
        # instead of the skill it originally loaded.
        text = _expand_skill_invocation_for_replay(
            text, str(session.get("session_key") or "")
        )
    isolation_cfg = _load_dashboard_process_isolation_config()
    turn_isolation = _session_uses_compute_host(session, isolation_cfg)
    # Re-bind to the current client transport for this request. This keeps
    # streaming events on the active websocket even if an earlier disconnect
    # or fallback moved the session transport to stdio.
    if (t := current_transport()) is not None:
        session["transport"] = t
    while True:
        busy_transport = None
        with session["history_lock"]:
            if session.get("running"):
                # Don't reject a mid-turn prompt — queue it (and, by default,
                # interrupt the live turn) so it runs as the next turn. The
                # provider interrupt itself must happen after this lock is
                # released: a non-interruptible tool may keep it waiting.
                busy_transport = t or session.get("transport")
            else:
                break
        busy_response = _handle_busy_submit(
            rid, sid, session, text, busy_transport,
            queued=bool(params.get("queued")),
        )
        if busy_response is not None:
            return busy_response
        # The old turn finished between the two lock acquisitions. Retry the
        # claim so this prompt starts normally instead of being stranded in a
        # queue whose drain already ran.

    # Filled when this submit performed a truncation against a durable session:
    # the fresh post-rewrite row ids of the surviving user turns, for client
    # rowId rebinding (see comment at the assignment site).
    survivor_user_row_ids = None
    with session["history_lock"]:
        # A watch session's run lives in the PARENT turn, so its own running
        # flag is False — without this, typing mid-run builds a second agent
        # racing the in-flight child on the same stored session (interleaved
        # transcript, stale fork). After the run completes, submitting is fine:
        # the upgrade resumes the child's transcript as a normal conversation.
        if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
            return _err(rid, 4009, "subagent still running — wait for it to finish")
        truncate_message_id = params.get("truncate_before_message_id")
        truncate_row_id = params.get("truncate_before_row_id")
        if (
            is_truthy_value(params.get("confirm_truncate"))
            and truncate_user_ordinal is None
            and truncate_message_id is None
            and truncate_row_id is None
        ):
            return _err(
                rid,
                4004,
                "confirm_truncate requires truncate_before_user_ordinal, truncate_before_message_id, or truncate_before_row_id",
            )
        if (
            truncate_user_ordinal is not None
            or truncate_message_id is not None
            or truncate_row_id is not None
        ):
            history = session.get("history", [])

            # Malformed params refuse first (4004), regardless of consent —
            # the historical ordinal-path precedence.
            target_row_id = None
            if truncate_row_id is not None:
                target_row_id, err = _coerce_truncate_int(
                    rid, truncate_row_id, "truncate_before_row_id"
                )
                if err is not None:
                    return err
            client_ordinal = None
            if truncate_user_ordinal is not None:
                client_ordinal, err = _coerce_truncate_int(rid, truncate_user_ordinal)
                if err is not None:
                    return err

            # An ordinal/id alone is not consent. A client that carries a leftover
            # ordinal into an ORDINARY submit sends a request that is
            # indistinguishable, field by field, from a real rewind — same
            # method, same shape, an in-range target — and the cut it asks for
            # is a destructive replace_messages() the user never requested
            # (#80763: 296 -> 52 messages, 244 durable rows gone). Only the
            # client knows whether this submit is a rewind/edit/regenerate, so
            # it has to say so; refuse the cut when it doesn't. Consent is
            # checked BEFORE target resolution: an unconfirmed (leaked-state)
            # request must refuse with 4029 without paying the durable
            # transcript read or heal-stamping live history dicts that
            # row-id resolution performs.
            if not is_truthy_value(params.get("confirm_truncate")):
                logger.warning(
                    "prompt.submit: REFUSED unconfirmed truncation of session %s "
                    "(%d messages held; ordinal=%s, row_id=%s, message_id=%s). "
                    "The client attached truncation parameters without "
                    "confirm_truncate — likely stale truncation parameters on "
                    "an ordinary submit.",
                    sid,
                    len(history),
                    client_ordinal,
                    target_row_id,
                    truncate_message_id,
                )
                return _err(
                    rid,
                    4029,
                    "truncation parameters require confirm_truncate=true; "
                    "an ordinary prompt.submit must not drop session history "
                    "(update your Hermes client if a rewind was intended)",
                )

            user_indices = _history_user_indices(history)

            ordinal = None

            if target_row_id is not None:
                # Durable address first — never degrade a missing row_id into a
                # client ordinal cut (#82959 / #82766 review). Unknown id refuses
                # without touching data; stale ordinal with a *resolved* row_id
                # is a separate 4030 mismatch below.
                found_match = _resolve_truncate_row_id(
                    session, history, target_row_id
                )

                if found_match is None:
                    logger.warning(
                        "prompt.submit: target row_id %d not found for session %s "
                        "(in-memory + durable); refusing truncation without fallback",
                        target_row_id,
                        sid,
                    )
                    return _err(
                        rid,
                        4018,
                        "target user message is no longer in session history",
                    )

                msg_ordinal, _ = found_match
                ordinal, err = _reconcile_client_ordinal(
                    rid, sid, client_ordinal, msg_ordinal,
                    "truncate_before_row_id", target_row_id,
                )
                if err is not None:
                    return err
            elif truncate_message_id is not None:
                msg_id_str = str(truncate_message_id)
                found_match = None
                for u_ord, h_idx in enumerate(user_indices):
                    msg = history[h_idx]
                    if msg.get("id") == msg_id_str or msg.get("message_id") == msg_id_str:
                        found_match = (u_ord, h_idx)
                        break

                if found_match is None:
                    # Fail closed: a supplied message_id that does not resolve
                    # must not fall back to a (possibly stale) ordinal. Desktop
                    # clients should send truncate_before_row_id instead.
                    logger.warning(
                        "prompt.submit: target message_id %s not found in history "
                        "for session %s; refusing truncation without fallback",
                        msg_id_str,
                        sid,
                    )
                    return _err(
                        rid,
                        4018,
                        "target user message is no longer in session history",
                    )

                msg_ordinal, _ = found_match
                ordinal, err = _reconcile_client_ordinal(
                    rid, sid, client_ordinal, msg_ordinal,
                    "truncate_before_message_id", msg_id_str,
                )
                if err is not None:
                    return err
            else:
                ordinal = client_ordinal

            # Reject out-of-range ordinals on BOTH ends. A negative value would
            # otherwise sail past the upper-bound check and hit Python's negative
            # indexing below (user_indices[-1] -> the LAST user turn), silently
            # truncating history to everything before it and persisting that loss
            # via replace_messages — an unrecoverable overwrite of the session DB.
            if ordinal < 0 or ordinal >= len(user_indices):
                return _err(rid, 4018, "target user message is no longer in session history")
            truncated = history[: user_indices[ordinal]]
            # Second gate, on top of confirm_truncate: ordinal 0 resolves to
            # history[:0] == [] and replace_messages() DELETEs every durable
            # row. A confirmed rewind that happens to erase the whole
            # transcript still needs its own opt-in (legitimate restore/
            # regenerate of the first user turn).
            if (
                not truncated
                and history
                and not is_truthy_value(params.get("confirm_empty_truncate"))
            ):
                logger.warning(
                    "prompt.submit: REFUSED empty truncation of session %s "
                    "(%d messages would be wiped; ordinal=%d).",
                    sid,
                    len(history),
                    ordinal,
                )
                return _err(
                    rid,
                    4028,
                    "truncation would erase the entire session transcript; "
                    "resubmit with confirm_empty_truncate=true if this is intended",
                )
            # Info for routine rewind/edit cuts; warning only when the client
            # explicitly opts into wiping the whole transcript.
            log_fn = logger.warning if not truncated else logger.info
            log_fn(
                "prompt.submit: truncating session %s history %d -> %d messages "
                "(ordinal=%d)",
                sid,
                len(history),
                len(truncated),
                ordinal,
            )
            # Write-before-memory (mirrors gateway hygiene / manual /compress):
            # persist the truncated transcript first. If replace_messages fails
            # after we already rewrote session["history"], the turn still runs
            # against the short list while state.db keeps the old tail. The
            # agent flush is append-only for history-dict identities, so the
            # new exchange is appended on top of the "undone" turns — durable
            # zombie history on resume, and the edit/regenerate never sticks.
            # Fail closed: refuse the turn and leave memory/DB unchanged.
            if (db := _get_db()) is not None:
                try:
                    # active_only=True: replace only the live (active=1) rows.
                    # In-place compaction (#38763) keeps the pre-compaction
                    # transcript as active=0/compacted=1 rows under this same
                    # session key; a bare replace_messages() would DELETE that
                    # durable archive on every edit/regenerate — the same bug
                    # class #80216 fixed for /retry. On an uncompacted session
                    # all rows are active=1, so this is behaviorally identical
                    # to the full replace.
                    # archive_dropped: a rewind overwrites turns the user may
                    # not have meant to drop, and this write is the last step
                    # before they are gone — three reported incidents ended
                    # here with nothing to restore from (#70516, #80763,
                    # #82756). Soft-archiving keeps them on disk (active=0) and
                    # in the FTS index, so a mis-aimed cut is recoverable
                    # instead of terminal. The live transcript is unchanged.
                    db.replace_messages(
                        session["session_key"],
                        truncated,
                        active_only=True,
                        archive_dropped=True,
                    )
                except Exception as exc:
                    logger.error(
                        "prompt.submit: replace_messages failed for session %s "
                        "(ordinal=%d); refusing turn so memory and DB stay "
                        "aligned: %s",
                        sid,
                        ordinal,
                        exc,
                        exc_info=True,
                    )
                    return _err(
                        rid,
                        5008,
                        f"failed to persist history truncation: {exc}",
                    )
            session["history"] = truncated
            session["history_version"] = int(session.get("history_version", 0)) + 1
            if db is not None:
                # replace_messages re-inserted the surviving prefix as NEW rows
                # and stamped fresh _row_id values onto these same dicts.
                # Surface the surviving user-turn ids (in visible-user-ordinal
                # order) so the client can rebind its cached rowId stamps —
                # otherwise a second rewind targeting an older surviving turn
                # sends the pre-rewind id and the fail-closed resolver refuses
                # it with 4018 (#83202 review: consecutive-rewind staleness).
                # Ordinal order matches the client's visible-user filter the
                # same way truncate ordinals already do. Entries are None when
                # a row somehow has no stamp — the client must drop its cached
                # id for that turn rather than keep a stale one.
                survivor_user_row_ids = [
                    _message_row_id(truncated[i])
                    for i in _history_user_indices(truncated)
                ]
        session["running"] = True
        session["_turn_cancel_requested"] = False
        session["last_active"] = time.time()
        _start_inflight_turn(session, text)

    if turn_isolation:
        isolated_response = _submit_prompt_to_compute_host(rid, sid, session, text)
        if not isolated_response.get("error"):
            if survivor_user_row_ids is not None:
                # The truncation already happened inline above (memory + DB),
                # before compute-host dispatch — the rebind payload applies to
                # this path exactly as it does to the inline one.
                isolated_response["result"][
                    "survivor_user_row_ids"
                ] = survivor_user_row_ids
            return isolated_response
        logger.warning(
            "compute-host dispatch failed for session %s; falling back inline: %s",
            sid,
            isolated_response["error"].get("message", "unknown error"),
        )

    # Persist the DB row lazily, now that the user has actually sent a message.
    # Disk-full must fail the RPC (not stream silently): desktop maps the error
    # string to a "disk full" toast so the user knows why the send vanished.
    try:
        _ensure_session_db_row(session)
        # A branch becomes real here: copy its parent's transcript into the row so it
        # resumes with full context (the agent won't persist the seed itself).
        _persist_branch_seed(session)
    except Exception as exc:
        from hermes_state import is_disk_full_error

        with session["history_lock"]:
            session["running"] = False
            session["last_active"] = time.time()
            _clear_inflight_turn(session)
        if is_disk_full_error(exc):
            return _err(
                rid,
                5070,
                "disk full: session storage could not be written — free some disk space and try again",
            )
        logger.warning("prompt.submit: session persist failed: %s", exc, exc_info=True)
        return _err(
            rid,
            5071,
            f"session storage could not be written: {exc}",
        )
    _start_agent_build(sid, session)

    def run_after_agent_ready() -> None:
        # Patient wait (#63078): the user's message is already the accepted
        # in-flight turn, so a slow deferred build must not eat it. The wait
        # delivers the prompt when the still-running build completes, honors a
        # cancel promptly, notices the user once past the slow threshold, and
        # only errors when the build itself fails or the bounded cap expires.
        err = _wait_agent_for_prompt(session, rid, sid)
        if err:
            # Terminal frame + retained snapshot (not a bare "error" event +
            # cleared inflight): if the client is disconnected right now, the
            # retained snapshot is the only way resume can show this failure.
            _emit_terminal_turn_error(
                sid,
                session,
                (err.get("error") or {}).get("message", "agent initialization failed"),
            )
            with session["history_lock"]:
                session["running"] = False
                session["last_active"] = time.time()
            _emit("session.info", sid, _session_info(session.get("agent"), session))
            return
        with session["history_lock"]:
            if session.get("_turn_cancel_requested") or not session.get("running"):
                session["running"] = False
                _clear_inflight_turn(session)
                # Surface the cancellation to the client. Without this emit the
                # turn vanishes silently — the Desktop sees `prompt.submit`
                # return `{"status": "streaming"}` but never receives a
                # `message.start` or `error` event, so the composer shows no
                # feedback (issue #63078 server-side half). Match the
                # `_wait_agent` error branch above: emit, then bail.
                _emit(
                    "error",
                    sid,
                    {
                        "message": "Turn cancelled before the agent was ready"
                        if session.get("_turn_cancel_requested")
                        else "Session no longer running before the agent was ready"
                    },
                )
                return
        _run_prompt_submit(rid, sid, session, text)

    run_thread = threading.Thread(target=run_after_agent_ready, daemon=True)
    # Keep a handle so session.interrupt can tell a live turn from a stuck
    # `running` flag (a turn that died without clearing it) and recover the latter.
    session["_run_thread"] = run_thread
    run_thread.start()
    return _ok(
        rid,
        {
            "status": "streaming",
            **(
                {"survivor_user_row_ids": survivor_user_row_ids}
                if survivor_user_row_ids is not None
                else {}
            ),
        },
    )


@method("clipboard.paste")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
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
        img_dir
        / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['image_counter']}.png"
    )

    # Save-first: mirrors CLI keybinding path; more robust than has_image() precheck
    if not save_clipboard_image(img_path):
        session["image_counter"] = max(0, session["image_counter"] - 1)
        msg = (
            "Clipboard has image but extraction failed"
            if has_clipboard_image()
            else "No image found in clipboard"
        )
        return _ok(rid, {"attached": False, "message": msg})

    session.setdefault("attached_images", []).append(str(img_path))
    return _ok(
        rid,
        {
            "attached": True,
            "path": str(img_path),
            "count": len(session["attached_images"]),
            **_image_meta(img_path),
        },
    )


@method("image.attach")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    try:
        from cli import (
            _IMAGE_EXTENSIONS,
            _detect_file_drop,
            _resolve_attachment_path,
            _split_path_input,
        )

        dropped = _detect_file_drop(raw)
        if dropped:
            image_path = dropped["path"]
            remainder = dropped["remainder"]
        else:
            path_token, remainder = _split_path_input(raw)
            image_path = _resolve_attachment_path(path_token)
            if image_path is None:
                return _err(rid, 4016, f"image not found: {path_token}")
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _err(rid, 4016, f"unsupported image: {image_path.name}")
        session.setdefault("attached_images", []).append(str(image_path))
        return _ok(
            rid,
            {
                "attached": True,
                "path": str(image_path),
                "count": len(session["attached_images"]),
                "remainder": remainder,
                "text": remainder or f"[User attached image: {image_path.name}]",
                **_image_meta(image_path),
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


@method("image.attach_bytes")
def _(rid, params: dict) -> dict:
    """Attach an image to the session from base64 bytes (remote-client path).

    A desktop app or web dashboard running on a DIFFERENT machine than the
    gateway can't hand us a local path — that file only exists on the client's
    disk. So it uploads the raw image bytes (base64) and we write them into the
    gateway's own images dir. The response shape mirrors ``image.attach`` so the
    client treats both identically.

    Params:
      content_base64 / data (str, required): base64 image bytes. Accepts a
        ``data:image/...;base64,`` prefix and embedded whitespace. ``data`` is
        an accepted alias for older desktop builds.
      filename / ext (str, optional): extension hint. Without it, magic bytes
        identify PNG/JPEG/GIF/WebP/BMP, falling back to ``.png``.
    """
    session, err = _sess(params, rid)
    if err:
        return err

    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_b64:
        return _err(rid, 4015, "content_base64 required")

    img_bytes = _decode_attach_base64(raw_b64, mime_prefix="image/")
    if img_bytes is None:
        return _err(rid, 4017, "data is not valid base64")
    if not img_bytes:
        return _err(rid, 4017, "image is empty")
    if len(img_bytes) > _ATTACH_BYTES_MAX_BYTES:
        mb = _ATTACH_BYTES_MAX_BYTES // (1024 * 1024)
        return _err(rid, 4018, f"image too large ({len(img_bytes)} bytes; cap is {mb} MB)")

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

    return _ok(
        rid,
        {
            "attached": True,
            "path": str(img_path),
            "count": len(session["attached_images"]),
            "remainder": "",
            "text": f"[User attached image: {img_path.name}]",
            "bytes": len(img_bytes),
            **_image_meta(img_path),
        },
    )


@method("pdf.attach")
def _(rid, params: dict) -> dict:
    """Attach a PDF by rendering each page to PNG and queuing the pages.

    Anthropic's vision pipeline accepts images, not PDFs, so this runs
    ``pdftoppm`` (poppler-utils) at 150 DPI per page and queues each rendered
    page as an attached image. Accepts either a host ``path`` (local mode) or
    base64 ``content_base64`` (remote upload). Caps at 50 MB / 25 pages per call.

    Requires ``pdftoppm`` on $PATH (``apt install poppler-utils``); returns 5028
    if missing.
    """
    import shutil
    import subprocess
    import tempfile

    session, err = _sess(params, rid)
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
        if raw_b64:
            pdf_bytes = _decode_attach_base64(raw_b64, mime_prefix="application/pdf")
            if pdf_bytes is None:
                return _err(rid, 4017, "data is not valid base64")
            if not pdf_bytes:
                return _err(rid, 4017, "decoded PDF is empty")
            if len(pdf_bytes) > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(rid, 4018, f"PDF too large ({len(pdf_bytes)} bytes; cap is {mb} MB)")
            if pdf_bytes[:5] != b"%PDF-":
                return _err(rid, 4017, "payload is not a PDF (missing %PDF- magic bytes)")
            pdf_path = td_path / "input.pdf"
            pdf_path.write_bytes(pdf_bytes)
            display_name = str(params.get("filename", "") or "uploaded.pdf")
        else:
            try:
                from cli import _resolve_attachment_path

                resolved = _resolve_attachment_path(raw_path)
            except Exception:
                resolved = None
            if resolved is None or not Path(resolved).is_file():
                return _err(rid, 4016, f"PDF not found: {raw_path}")
            if Path(resolved).suffix.lower() != ".pdf":
                return _err(rid, 4016, f"not a PDF: {Path(resolved).name}")
            if Path(resolved).stat().st_size > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(rid, 4018, f"PDF too large; cap is {mb} MB")
            pdf_path = Path(resolved)
            display_name = pdf_path.name

        try:
            first_page = int(params.get("first_page") or 1)
            last_page_param = params.get("last_page")
            last_page = int(last_page_param) if last_page_param is not None else None
        except (TypeError, ValueError):
            return _err(rid, 4015, "first_page/last_page must be integers")

        if first_page < 1:
            return _err(rid, 4015, "first_page must be >= 1")
        if last_page is None:
            last_page = first_page + _PDF_ATTACH_MAX_PAGES - 1
        if last_page < first_page:
            return _err(rid, 4015, "last_page must be >= first_page")
        if last_page - first_page + 1 > _PDF_ATTACH_MAX_PAGES:
            return _err(rid, 4019, f"page range exceeds cap of {_PDF_ATTACH_MAX_PAGES} pages per attach call")

        out_prefix = td_path / "page"
        argv = [
            "pdftoppm", "-png", "-r", "150",
            "-f", str(first_page), "-l", str(last_page),
            str(pdf_path), str(out_prefix),
        ]
        from hermes_cli._subprocess_compat import windows_hide_flags

        try:
            res = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                # Force UTF-8 + lossy decode so non-UTF-8 child output can't
                # crash the gateway thread on locale-mismatched Windows (#53137).
                encoding="utf-8", errors="replace",
                creationflags=windows_hide_flags(),
            )
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
            dst = _queue_attached_image(session, src.read_bytes(), ".png", prefix=f"pdf_p{page_num}")
            attached_pages.append({"path": str(dst), "page": page_int, **_image_meta(dst)})

        return _ok(
            rid,
            {
                "attached": True,
                "filename": display_name,
                "pages_attached": len(attached_pages),
                "pages": attached_pages,
                "count": len(session["attached_images"]),
                "text": f"[User attached PDF: {display_name} ({len(attached_pages)} page(s))]",
            },
        )


@method("file.attach")
def _(rid, params: dict) -> dict:
    """Stage a non-image file attachment into the session workspace.

    The image/PDF path renders to vision tiles; this one keeps the file as a
    readable artifact and returns a workspace-relative ``@file:`` ref so the
    agent's file tools (and ``agent.context_references``) can read it. Solves the
    remote-gateway case where the desktop passes a path that only exists on the
    CLIENT's disk: the client uploads ``data_url`` bytes and we materialize the
    file on the gateway.

    Params:
      session_id (str, required)
      path (str): client/host path of the file (used for naming + local-mode
        gateway-visible resolution).
      data_url (str): ``data:<mime>;base64,<b64>`` upload of the file bytes,
        required when the path isn't visible to the gateway.
      name (str, optional): preferred filename.
    """
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    data_url = str(params.get("data_url", "") or "").strip()
    name = str(params.get("name", "") or "").strip()
    if not raw and not data_url:
        return _err(rid, 4015, "path or data_url required")
    try:
        stored_path, uploaded = _stage_session_file_attachment(
            session, raw_path=raw, data_url=data_url, name=name
        )
        ref_path = _attachment_ref_path(session, stored_path)
        return _ok(
            rid,
            {
                "attached": True,
                "name": stored_path.name,
                "path": str(stored_path),
                "ref_path": ref_path,
                "ref_text": f"@file:{_format_ref_value(ref_path)}",
                "uploaded": uploaded,
            },
        )
    except Exception as e:
        return _err(rid, 5028, str(e))


@method("image.detach")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    images = session.setdefault("attached_images", [])
    before = len(images)
    session["attached_images"] = [path for path in images if path != raw]
    return _ok(
        rid,
        {
            "detached": len(session["attached_images"]) != before,
            "count": len(session["attached_images"]),
        },
    )


@method("input.detect_drop")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    try:
        from cli import _detect_file_drop

        raw = str(params.get("text", "") or "")
        dropped = _detect_file_drop(raw)
        if not dropped:
            return _ok(rid, {"matched": False})

        drop_path = dropped["path"]
        remainder = dropped["remainder"]
        if dropped["is_image"]:
            session.setdefault("attached_images", []).append(str(drop_path))
            text = remainder or f"[User attached image: {drop_path.name}]"
            return _ok(
                rid,
                {
                    "matched": True,
                    "is_image": True,
                    "path": str(drop_path),
                    "count": len(session["attached_images"]),
                    "text": text,
                    **_image_meta(drop_path),
                },
            )

        text = f"[User attached file: {drop_path}]" + (
            f"\n{remainder}" if remainder else ""
        )
        return _ok(
            rid,
            {
                "matched": True,
                "is_image": False,
                "path": str(drop_path),
                "name": drop_path.name,
                "text": text,
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


@method("prompt.background")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    text, parent = params.get("text", ""), params.get("session_id", "")
    if not text:
        return _err(rid, 4012, "text required")
    task_id = f"bg_{uuid.uuid4().hex[:6]}"

    def run():
        session_tokens = _set_session_context(task_id, cwd=_session_cwd(session))
        try:
            from run_agent import AIAgent

            result = AIAgent(
                **_background_agent_kwargs(session["agent"], task_id)
            ).run_conversation(
                user_message=text,
                task_id=task_id,
            )
            _emit(
                "background.complete",
                parent,
                {
                    "task_id": task_id,
                    "text": (
                        result.get("final_response", str(result))
                        if isinstance(result, dict)
                        else str(result)
                    ),
                },
            )
        except Exception as e:
            _emit(
                "background.complete",
                parent,
                {"task_id": task_id, "text": f"error: {e}"},
            )
        finally:
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


@method("preview.restart")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    url = str(params.get("url") or "").strip()
    cwd = str(params.get("cwd") or "").strip()
    context = str(params.get("context") or "").strip()

    if not url:
        return _err(rid, 4012, "url required")

    task_id = f"preview_{uuid.uuid4().hex[:6]}"
    parent = params.get("session_id", "")
    parent_history = _preview_restart_history(session)
    has_history = bool(parent_history)
    prompt = "\n".join(
        line
        for line in [
            "The desktop preview pane cannot load a local server URL.",
            "",
            f"Preview URL: {url}",
            f"Current working directory: {cwd or '(unknown)'}",
            "",
            f"Preview console:\n{context}" if context else "",
            "" if context else "",
            (
                "The conversation history above is from the user's main session — including the commands you (the assistant) previously ran to start servers, edit files, or check ports. Use it to figure out exactly which server should be running at this Preview URL. The user did not start a brand new task; recover what they had working."
                if has_history
                else None
            ),
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
        ]
        if line
    )

    # Normalize defensively: a malformed client path (embedded NUL, etc.) must
    # not blow up the whole restart — treat it as "no validated cwd".
    try:
        preview_cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else ""
        if preview_cwd and not os.path.isdir(preview_cwd):
            preview_cwd = ""
    except Exception:
        preview_cwd = ""

    def run():
        # Pin the validated preview cwd, else the parent workspace — never an
        # invalid client path, which would silently fall back to the launch dir.
        session_tokens = _set_session_context(task_id, cwd=(preview_cwd or _session_cwd(session)))
        try:
            from run_agent import AIAgent
            from tools.terminal_tool import register_task_env_overrides

            if preview_cwd:
                register_task_env_overrides(task_id, {"cwd": preview_cwd})

            history_note = (
                f" (with {len(parent_history)} parent-session messages of context)"
                if parent_history
                else ""
            )
            _emit(
                "preview.restart.progress",
                parent,
                {"task_id": task_id, "text": f"Starting hidden restart agent{history_note}"},
            )
            result = AIAgent(
                **_ephemeral_preview_agent_kwargs(session["agent"], task_id),
                **_preview_restart_callbacks(parent, task_id),
            ).run_conversation(
                user_message=prompt,
                task_id=task_id,
                conversation_history=parent_history or None,
            )
            text = (
                result.get("final_response", str(result))
                if isinstance(result, dict)
                else str(result)
            )
            _emit("preview.restart.complete", parent, {"task_id": task_id, "text": text})
        except Exception as e:
            _emit(
                "preview.restart.complete",
                parent,
                {"task_id": task_id, "text": f"error: {e}"},
            )
        finally:
            try:
                from tools.terminal_tool import clear_task_env_overrides

                clear_task_env_overrides(task_id)
            except Exception:
                pass
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


@method("clarify.respond")
def _(rid, params: dict) -> dict:
    # allow_expired=True: a clarify can time out server-side (its entry is popped
    # from _pending) while the card is still visible — common when a WebSocket
    # reconnect during the wait drops tool.complete. A late answer must resolve
    # gracefully instead of hitting the raw 4009 "no pending answer request".
    return _respond(rid, params, "answer", allow_expired=True)


@method("terminal.read.respond")
def _(rid, params: dict) -> dict:
    # `text` is a JSON string of the serialized terminal buffer + line metadata.
    # allow_expired=True: the read_terminal tool's _block() uses a short 30s
    # timeout, so a slow renderer losing the race is the common case — a late
    # response must not error after the tool already returned empty.
    return _respond(rid, params, "text", allow_expired=True)


@method("preview.read.respond")
def _(rid, params: dict) -> dict:
    # `text` is a JSON string of the active preview tab's serialized contents.
    # allow_expired=True for the same reason as terminal.read: the tool's
    # bounded wait can expire while a slow page extraction is still running.
    return _respond(rid, params, "text", allow_expired=True)


@method("window.read.respond")
def _(rid, params: dict) -> dict:
    # `text` is a JSON string describing the OS window underneath the Hermes
    # window (read_window_below tool). allow_expired=True for the same reason
    # as terminal.read: the tool's bounded wait can expire while the renderer's
    # round-trip to the main process is still in flight.
    return _respond(rid, params, "text", allow_expired=True)


@method("mcp.setup.respond")
def _(rid, params: dict) -> dict:
    # `result` is a JSON string of the setup card's outcome ({status, server,
    # detail?, tools?}). allow_expired=True: the setup_mcp tool waits 10
    # minutes, but an OAuth round-trip or a slow install can outlive that —
    # a late answer must resolve gracefully, not surface a raw 4009.
    return _respond(rid, params, "result", allow_expired=True)


@method("sudo.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "password", allow_expired=True)


@method("secret.respond")
def _(rid, params: dict) -> dict:
    return _respond(rid, params, "value", allow_expired=True)


@method("approval.respond")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        from tools.approval import resolve_gateway_approval

        return _ok(
            rid,
            {
                "resolved": resolve_gateway_approval(
                    session["session_key"],
                    params.get("choice", "deny"),
                    resolve_all=params.get("all", False),
                )
            },
        )
    except Exception as e:
        return _err(rid, 5004, str(e))


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
    # Module-level helpers aren't @method handlers, so install() doesn't see
    # them. Rebind onto server globals so handler bodies (and server.py call
    # sites) resolve the same free names after the split.
    g = vars(server)
    for helper in (
        _history_user_indices,
        _message_row_id,
        _mem_db_pair_agrees,
        _find_user_turn_by_row_id,
        _resolve_truncate_row_id,
        _coerce_truncate_int,
        _reconcile_client_ordinal,
        _pending_reaction_notes,
    ):
        setattr(
            server,
            helper.__name__,
            types.FunctionType(
                helper.__code__,
                g,
                helper.__name__,
                helper.__defaults__,
                helper.__closure__,
            ),
        )
