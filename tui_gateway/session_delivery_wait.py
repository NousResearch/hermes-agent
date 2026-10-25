"""Desktop follow-ups waiting on a local Bot Chat delivery's single-writer lease.

Bodies are rebound onto server.py's globals (method_ctx.bind_module).
"""

from __future__ import annotations

from .method_ctx import bind_module


_BOT_DELIVERY_WAIT_SECONDS = 1.0


def _queue_behind_bot_delivery(rid, sid: str, session: dict, text: str, display_kind: str | None):
    # Do not write an accept-time transcript row: the delivery still owns that store.
    # Keep each send separate, including repeated text and its attachment snapshot.
    with session["history_lock"]:
        if not session.get("_bot_delivery_wait") and session.get("active_session_lease") is not None:
            return None  # a settled wait raced this arrival; use the normal busy/idle path
        images = list(session.get("attached_images") or [])
        session["attached_images"] = []
        envelope = {"text": text, "transport": current_transport() or session.get("transport"),
                    **({"image_paths": images} if images else {}),
                    **({"_queued_display_kind": display_kind} if display_kind else {})}
        entries = ([session["queued_prompt"]] if session.get("queued_prompt") else []) + list(session.get("queued_prompts") or [])
        _ac_set_queue(session, [*entries, envelope])
        waiting = session.get("_bot_delivery_wait")
        if waiting is None:
            waiting = object()
            session["_bot_delivery_wait"] = waiting
            generation = int(session.get("_queued_prompt_generation", 0))
        else:
            generation = None
    message = "This bot is finishing a task; your message will send next."
    _emit("notice", sid, {"message": message})
    if generation is not None:
        worker = _start_session_work(
            lambda: _wait_for_bot_delivery(rid, sid, session, waiting, generation),
            name=f"bot-delivery-wait-{sid}")
        if worker is None:
            with session["history_lock"]:
                session.pop("_bot_delivery_wait", None)
            return _err(rid, 4009, "backend retiring; reconnect to send")
        session["_run_thread"] = worker
    return _ok(rid, {"status": "queued"})


def _wait_for_bot_delivery(rid, sid: str, session: dict, waiting, generation: int) -> None:
    from hermes_cli.active_sessions import SESSION_NOT_OWNED

    dispatched = False
    try:
        while True:
            with session["history_lock"]:
                if (session.get("_closing") or session.get("_bot_delivery_wait") is not waiting
                        or int(session.get("_queued_prompt_generation", 0)) != generation
                        or not session.get("queued_prompt")):
                    return
            # The delivery can compress Bot Chat while holding it. Always claim the current
            # continuation, never the now-unowned parent. Recheck after acquisition as well.
            with _session_db(session) as db:
                key = str(session.get("session_key") or "")
                tip = (db.resolve_resume_session_id(key) if db is not None else key) or key
            session["session_key"] = tip
            refusal = _ensure_active_session_slot(sid, session)
            if refusal is not None:
                if getattr(refusal, "reason", None) != SESSION_NOT_OWNED:
                    raise RuntimeError(str(refusal))
                time.sleep(_BOT_DELIVERY_WAIT_SECONDS)
                continue
            with _session_db(session) as db:
                latest = (db.resolve_resume_session_id(tip) if db is not None else tip) or tip
            if latest != tip:
                _release_active_session_slot(session)
                session["session_key"] = latest
                continue
            break

        with session["history_lock"]:
            session["running"] = True
            session["_turn_cancel_requested"] = False
        if session.get("agent") is None:
            _start_agent_build(sid, session)
        if (failure := _wait_agent_for_prompt(session, rid, sid)) is not None:
            raise RuntimeError(failure["error"]["message"])
        with session["history_lock"]:
            if (session.get("_closing") or int(session.get("_queued_prompt_generation", 0)) != generation):
                return
            # Delivery history is now stable and owned by this runtime. Preserve metadata
            # and row identities so the next user turn sees the completed task.
            with _session_db(session) as db:
                if db is not None:
                    session["history"] = canonicalize_replay_history(
                        db.get_messages_as_conversation(tip, include_row_ids=True))
                    db.reopen_session(tip)
            session["history_version"] = int(session.get("history_version", 0)) + 1
            agent = session["agent"]
            if agent.session_id != tip:
                agent.session_id = tip
                _wire_session_agent(sid, tip, agent)
            session["running"] = False
        _emit_session_info_for_session(sid, session)
        dispatched = True
        _drain_queued_prompt(rid, sid, session)
    except Exception as exc:
        logger.exception("Bot delivery wait failed for %s", sid)
        _emit_terminal_turn_error(sid, session, exc, retire_marker=False)
    finally:
        with session["history_lock"]:
            if session.get("_bot_delivery_wait") is waiting:
                session.pop("_bot_delivery_wait", None)
            if not dispatched:
                session["running"] = False
        if not dispatched:
            _release_active_session_slot(session)


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
