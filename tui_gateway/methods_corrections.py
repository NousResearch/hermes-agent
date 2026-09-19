"""Direct steer/redirect RPCs and their shared-session observations."""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def _apply_correction(rid, session: dict, verb: str, text: str, accepted_status: str,
                      *, sid="", input_batch=None, visible=False, queue_fallback=False) -> dict:
    """Call the captured agent without history/publication locks; observe only its target."""
    from tui_gateway.input_observation import (state, new_input, reply_submission, project_inputs,
                                               record_outcome, merge_inputs, publish_state, MAX_RECORDS)
    from tui_gateway.turn_observation import turn_scope
    from contextlib import nullcontext
    from copy import deepcopy
    owner = state(session)
    with owner.control:
        with session["history_lock"]:
            input_batch = input_batch or new_input(session, text)
            reply_batch = deepcopy(input_batch)
            target = session.get("_turn_observation")
            snapshot = session.get("inflight_turn")
            agent = session.get("agent")
        try:
            accepted = getattr(agent, verb)(text)
        except Exception as exc:
            if not queue_fallback:
                with session["history_lock"]:
                    record_outcome(session, input_batch, "failed_before_start", reason="correction_failed")
                if sid:
                    publish_state(sid, session)
            return reply_submission(_err(rid, 5000, f"{verb} failed: {exc}"), reply_batch, "unresolved")
        disposition = "steered" if verb == "steer" else "redirected"
        gate = target.gate if target is not None else nullcontext()
        with gate:
            with session["history_lock"]:
                same_target = target.is_current() if target is not None else session.get("inflight_turn") is snapshot
                if accepted and same_target:
                    offset = len(str((snapshot or {}).get("assistant") or ""))
                    _record_inflight_correction(session, text)
                    _drop_queued_duplicates_of_inflight_user(session)
                    session["last_active"] = time.time()
                    payload = {"kind": verb, "input": _display_turn_input(text, "steer"), "offset": offset}
                    if target is not None:
                        target.inputs = merge_inputs(target.inputs or {"parts": [], "complete": False}, input_batch, 0)
                    if visible and target is not None:
                        current = session["inflight_turn"]
                        observations = [*current.get("input_observations", []),
                                        {"payload": payload, "input_batch": input_batch}]
                        current["input_observations"] = observations[-MAX_RECORDS:]
                        current["input_observations_complete"] = (
                            current.get("input_observations_complete", True) and len(observations) <= MAX_RECORDS)
                        payload = {**payload, **project_inputs(session, input_batch)}
                else:
                    disposition = "unresolved"
                    if accepted or not queue_fallback:
                        record_outcome(session, input_batch, "unresolved" if accepted else "failed_before_start",
                                       reason="target_ended" if accepted else "correction_rejected")
            if accepted and same_target and visible and target is not None:
                with turn_scope(target):
                    _emit("message.input", sid or target.sid, payload)
    if sid:
        publish_state(sid, session)
    response = _ok(rid, {"status": accepted_status if accepted else "rejected", "text": text})
    return reply_submission(response, reply_batch, disposition,
                            target.wire() if accepted and same_target and target is not None else None)


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
        if params.get("input_visibility") == "hidden":
            # Reuse the hidden prompt envelope: a hidden correction must never be
            # persisted as a visible steer row or interrupt an unrelated live turn.
            return _methods["prompt.submit"](rid, {
                "session_id": params.get("session_id"), "text": text,
                "display_kind": "hidden", "queued": True,
                "submission_ref": params.get("submission_ref")})
        from tui_gateway.input_observation import new_input, reply_submission, publish_state, record_outcome
        from tui_gateway.turn_observation import connection_source
        sid = str(params.get("session_id") or "")
        with session["history_lock"]:
            input_batch = new_input(session, text, params.get("submission_ref"))
            from copy import deepcopy
            reply_batch = deepcopy(input_batch)
        agent = session.get("agent")
        # Redirect during the turn-build window (running=True, agent None): queue for the next turn instead of
        # a misleading 4010 the client swallows into a lost follow-up.
        if (verb == "redirect" and agent is None and session.get("running")) or _session_compression_in_flight(session):
            transport = current_transport() or _stdio_transport
            with session["history_lock"]:
                if not session.get("running"):
                    record_outcome(session, input_batch, "failed_before_start", reason="build_ended")
                    return reply_submission(_err(rid, 4010, unsupported), reply_batch, "unresolved")
                envelope = _enqueue_prompt(session, text, transport, turn_source=connection_source(transport),
                                               input_batch=input_batch)
                disposition = envelope["_input_disposition"] if envelope is not None else "absorbed"
                session["last_active"] = time.time()
            publish_state(sid, session)
            return reply_submission(_ok(rid, {"status": "queued", "text": text}), reply_batch, disposition)
        if not supported(agent):
            with session["history_lock"]:
                record_outcome(session, input_batch, "failed_before_start", reason="unsupported_correction")
            publish_state(sid, session)
            return reply_submission(_err(rid, 4010, unsupported), reply_batch, "unresolved")
        # An idle steer would be spliced after an old tool row in the next turn.
        if verb == "steer" and not session.get("running"):
            with session["history_lock"]:
                record_outcome(session, input_batch, "failed_before_start", reason="idle_steer")
            publish_state(sid, session)
            return reply_submission(_ok(rid, {"status": "rejected", "text": text}), reply_batch, "unresolved")
        return _apply_correction(rid, session, verb, text, accepted_status, sid=sid,
                                 input_batch=input_batch, visible=params.get("input_visibility") == "visible")


# Inject text into the next tool result without interrupting (AIAgent.steer(): no new user turn, no role
# alternation violation).
_correction_method("session.steer", "steer", "queued", lambda agent: hasattr(agent, "steer"),
                   "agent does not support steer")
# Redirect the active model turn while preserving valid work/context.
_correction_method("session.redirect", "redirect", "redirected",
                   lambda agent: getattr(agent, "_supports_active_turn_redirect", False) is True
                   and hasattr(agent, "redirect"), "agent does not support active-turn redirect")


def register(server):
    bind_module(globals(), server)
