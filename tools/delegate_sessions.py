"""Durable conversation controls layered on the existing delegation runner."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from tools.registry import tool_error


def _store(parent):
    from tools.delegate_session_store import DelegateSessionStore
    db = getattr(parent, "_session_db", None)
    if db is None:
        raise ValueError("Resumable delegation requires a persisted parent session.")
    return DelegateSessionStore(db)


def bind_child(parent, child, task, resume=None, *, extension, options, routing_cfg, retain=False):
    """Reserve a persisted conversation before dispatch, including queued children."""
    if not (resume is not None or retain or options.get("retain", False)):
        return
    if getattr(child, "_delegate_profile_home", None) is not None:
        raise ValueError("Retained delegation cannot yet resume a profile route; choose a model/provider route.")
    child._delegate_review_skills = options.get("review_skills", False)
    child._delegate_assignment_formatter = extension.format_assignment if extension is not None else None
    from tools.delegate_session_store import DelegateSessionStore
    if resume is None:
        child._ensure_db_session()
    else:
        child._session_db_created = True
    db = child._session_db
    store = DelegateSessionStore(db)
    if resume is None:
        from tools.delegate_session_routes import capture_route
        record = store.register(
            parent.session_id, child.session_id,
            model=child.model, provider=getattr(child, "requested_provider", None) or child.provider,
            assignment=task["goal"], context=task.get("context") or "", title=options.get("title", task.get("title")),
            role=child._delegate_role, depth=child._delegate_depth,
            workspace=child._delegate_workspace, forked_context=child._delegate_forked_context,
            route_descriptor=capture_route(child, routing_cfg=routing_cfg),
        )
        record, token = store.acquire(parent.session_id, record["session_id"], task["goal"])
    else:
        record, token = resume["record"], resume["token"]
        db.reopen_session(child.session_id)
        child._delegate_resume_history = db.get_resume_conversations(child.session_id)[0]
        resume["transferred"] = True
    child._delegate_resumable = True
    from agent.prompt_builder import clear_skills_system_prompt_cache
    clear_skills_system_prompt_cache(clear_snapshot=True)
    child._delegate_session_id = record["session_id"]
    child._delegate_session_token = token
    child._delegate_store_path = str(Path(db.db_path).resolve())
    child._delegate_assignment_at = record["last_activity_at"]
    progress = child.tool_progress_callback
    last_touch = time.monotonic()

    def touch_progress(*args, **kwargs):
        nonlocal last_touch
        now = time.monotonic()
        if now - last_touch >= 1:
            store.touch(record["session_id"], token)
            last_touch = now
        if progress:
            return progress(*args, **kwargs)

    if hasattr(progress, "_flush"):
        touch_progress._flush = progress._flush
    child.tool_progress_callback = touch_progress


def release_child(child):
    """Release only after the worker and resource cleanup finish; fence old callbacks."""
    token = vars(child).pop("_delegate_session_token", None)
    if not token:
        return
    from hermes_state_registry import acquire, release
    from tools.delegate_session_store import DelegateSessionStore
    db = acquire(Path(child._delegate_store_path))
    try:
        DelegateSessionStore(db).release(
            child._delegate_session_id, token, current_session_id=child.session_id,
        )
    finally:
        release(db)


def retire_abandoned_child(child):
    """Close admission without closing resources still owned by the live worker."""
    token = vars(child).get("_delegate_session_token")
    if not token:
        return False
    child._delegate_review_skills = False
    from hermes_state_registry import acquire, release
    from tools.delegate_session_store import DelegateSessionStore
    db = acquire(Path(child._delegate_store_path))
    try:
        return DelegateSessionStore(db).retire_abandoned(child._delegate_session_id, token)
    finally:
        release(db)


def assignment_message(child, message):
    """Only temporal context is added; the parent owns the assignment's instructions."""
    if getattr(child, "_delegate_resumable", False) is not True:
        return message
    formatter = getattr(child, "_delegate_assignment_formatter", None)
    if formatter is None:
        return message
    return formatter(message, session_id=child._delegate_session_id,
                     timestamp=datetime.now(UTC).isoformat(timespec="seconds"))


def control(action, parent, *, session_id=None, message=None, model=None, query=None, limit=20, background=False):
    from tools.delegate_session_store import DelegateSessionError
    try:
        store = _store(parent)
        if action == "list":
            records = store.list(parent.session_id, query=query, limit=limit)
            return json.dumps({"action": action, "count": len(records), "subagents": records})
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"action='{action}' requires session_id from spawn or list.")
        session_id = session_id.strip()
        record = store.get(parent.session_id, session_id)
        if action == "observe":
            observed = store.observe(parent.session_id, session_id, limit=limit)
            return json.dumps(observed, ensure_ascii=False)
        if action == "close":
            record = store.close(parent.session_id, session_id)
            _cancel_review(parent, session_id)
            return json.dumps({"action": action, **record})
        if action == "resume":
            return _resume(parent, store, record, message, model, background)
        return _live_control(action, parent, record, message)
    except (ValueError, DelegateSessionError) as exc:
        return tool_error(str(exc))


def _cancel_review(parent, session_id):
    from agent.delegated_review import cancel_delegated_review
    cancel_delegated_review(session_id, db_path=parent._session_db.db_path)


def _resume(parent, store, record, message, model, background):
    from tools.delegate_tool import (
        delegate_task, is_spawn_paused, _resolve_delegation_credentials,
    )
    from tools.delegate_session_routes import resume_route
    if not isinstance(message, str) or not message.strip():
        raise ValueError("action='resume' requires a non-empty message describing the next assignment.")
    if is_spawn_paused():
        raise ValueError("Delegation spawning is paused; resume starts another run and is paused too.")
    task = {"goal": message, "context": record["context"], "fork_context": False}
    # An invalid route must not replace the conversation's saved model.
    route = resume_route(parent, record, model)
    _resolve_delegation_credentials(route, parent)
    record, token = store.acquire(parent.session_id, record["session_id"], message, model=model)
    reservation = {"record": record, "token": token, "transferred": False}
    try:
        _cancel_review(parent, record["session_id"])
        return delegate_task(tasks=[task], parent_agent=parent, background=background,
                             credentials_cfg=route, _resume=reservation, _apply_extension=True)
    finally:
        if not reservation["transferred"]:
            store.release(record["session_id"], token)


def _live_control(action, parent, record, message):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _handle_control_action
    profile_db = str(Path(parent._session_db.db_path).resolve())
    with _active_subagents_lock:
        live_id = next((sid for sid, live in _active_subagents.items()
                        if getattr(live.get("agent"), "_delegate_session_id", None) == record["session_id"]
                        and getattr(live.get("agent"), "_delegate_store_path", None) == profile_db), None)
    if live_id is None:
        raise ValueError("This child is not accepting live control. Use resume for an idle child.")
    result = json.loads(_handle_control_action(action, live_id, message, parent))
    result.pop("subagent_id", None)
    result["session_id"] = record["session_id"]
    return json.dumps(result)
