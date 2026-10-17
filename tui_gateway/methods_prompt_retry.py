"""Continue a proven failed provider request without rewinding completed work (#137277).

Checkpoints are process-local and replayable on reconnect. Missing checkpoints (including
backend restart), changed transcripts and unknown tool outcomes fail closed; Retry never
falls back to regenerate. This cannot close the side-effect-before-result-commit window.
"""
from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

_RETRY_FIELDS = (
    "_row_id", "role", "content", "api_content", "tool_calls", "tool_call_id",
    "reasoning", "reasoning_content", "reasoning_details", "reasoning_opaque",
    "reasoning_text", "effect_disposition", "display_metadata", "extra_content", "_anthropic_content_blocks", "bedrock_content_blocks",
)


def _retry_history_signature(history):
    import hashlib
    import json
    projected = [{key: row[key] for key in _RETRY_FIELDS if row.get(key) is not None}
                 for row in history]
    return hashlib.sha256(json.dumps(projected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _retry_context_is_durable(history):
    """Replayed screenshots depend on external bytes/config, beyond the durable row signature."""
    return not any(row.get("role") == "user" and isinstance(row.get(key), str)
                   and "[screenshot" in row[key]
                   for row in history for key in ("content", "api_content"))


def _retry_tools_complete(history):
    """Only observed, paired outcomes can prove work completed; repair is never evidence."""
    pending = set()
    for row in history:
        if row.get("role") == "assistant":
            if pending:
                return False
            ids = [call.get("id") for call in row.get("tool_calls") or []]
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                return False
            pending.update(ids)
        elif row.get("role") == "tool":
            call_id = row.get("tool_call_id")
            if call_id not in pending or not (row.get("display_metadata") or {}).get("tool_result_received") or row.get("effect_disposition") == "unknown":
                return False
            pending.remove(call_id)
        elif pending:
            return False
    return not pending


def _record_retry_checkpoint(session, st, surface):
    """Expose a token only when the exact failed model context is already durable."""
    if not surface or surface.get("layer") != "provider" or surface.get("code") != "rate_limit":
        return None
    if (getattr(st.agent, "api_mode", "") == "codex_app_server" or session.get("source") == "bot_room"
            or _session_uses_compute_host(session, _load_dashboard_process_isolation_config())):
        return None
    history = _load_durable_truncation_history(session, repair_alternation=False)
    live = st.result.get("messages") or []
    if (not history or not live or not _retry_context_is_durable(history)
            or _retry_history_signature(history) != _retry_history_signature(live)):
        return None
    user_idx = next((i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "user"), None)
    if user_idx is None or not _retry_tools_complete(history[user_idx + 1:]):
        return None
    user = history[user_idx]
    if not isinstance(user.get("_row_id"), int):
        return None
    from uuid import uuid4
    token = uuid4().hex
    session["inflight_turn"]["retry_checkpoint"] = {
        "token": token, "signature": _retry_history_signature(history),
        "user_row_id": user["_row_id"], "session_key": _submit_row_target_key(session),
    }
    return token


def _claim_prompt_retry(rid, session, token):
    """Validate and consume once, under admission's history lock; duplicate clicks never queue."""
    with _session_turn_admission(session) as admitted:
        if not admitted:
            return _err(rid, 5035, "backend is retiring; reconnect to continue"), None
        if session.get("running"):
            return _err(rid, 4009, "session busy"), None
        turn = session.get("inflight_turn") or {}
        checkpoint = turn.get("retry_checkpoint") or {}
        if turn.get("status") != "error" or checkpoint.get("token") != token:
            return _err(rid, 4034, "Retry checkpoint is no longer available. Review completed work before continuing."), None
        history = _load_durable_truncation_history(session, repair_alternation=False)
        if (not history or checkpoint["session_key"] != _submit_row_target_key(session)
                or checkpoint["signature"] != _retry_history_signature(history)):
            return _err(rid, 4034, "Session history changed. Review completed work before continuing."), None
        user = next((row for row in history if row.get("_row_id") == checkpoint["user_row_id"]), None)
        if user is None:
            return _err(rid, 4034, "Retry's original user row is no longer active."), None
        session["history"] = history
        session["history_version"] = int(session.get("history_version", 0)) + 1
        session["running"] = True
        session["_turn_cancel_requested"] = False
        _start_inflight_turn(session, user["content"])
        session["_retry_user_row_id"] = checkpoint["user_row_id"]
        return None, user["content"]


@method("prompt.retry")
@_profile_scoped
def _prompt_retry(rid, params):
    sid = params["session_id"]
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if _session_uses_compute_host(session, _load_dashboard_process_isolation_config()):
        return _err(rid, 4034, "This isolated worker cannot continue a failed turn. Review completed work before continuing.")
    if (refusal := _ensure_active_session_slot(sid, session)) is not None:
        return _err(rid, 4090, str(refusal))
    with _session_resume_lock:
        if (refusal := _reattach_refusal(rid, sid, session)) is not None:
            return refusal
        if (transport := current_transport()) is not None:
            _rebind_live_transport(sid, session, transport)
    err, text = _claim_prompt_retry(rid, session, params["retry_token"])
    if err:
        return err
    if not _restart_completed_failed_agent_build(sid, session, session.get("agent_ready")):
        _start_agent_build(sid, session)
    run_thread = _start_session_work(
        lambda: _run_after_agent_ready(rid, sid, session, text, None, None, None, None),
        name=f"retry-{sid}", session=session)
    if run_thread is None:
        with session["history_lock"]:
            session["running"] = False
            _clear_inflight_turn(session)
        return _err(rid, 5035, "backend is retiring; reconnect to continue")
    return _ok(rid, {"status": "streaming"})


def register(server):
    bind_module(globals(), server)
