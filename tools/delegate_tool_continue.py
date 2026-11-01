"""delegate_task(action='continue'): run a FINISHED child again on its own conversation.

The child is rebuilt from what its registry record retained when it finished (``_resume_spec``: route incl. profile
home, toolsets, the goal/context its system prompt came from, model, session id, and the provider and endpoint it ran
on, which the rebuild must resolve to again), its history is loaded from its saved transcript, and ``message`` is its
next user turn. It keeps its subagent_id AND its session id (the transcript
appends, like ``--resume``), and runs as a new one-task call through the normal delegate_task dispatch, so delivery,
the stale monitor and restart recovery are the spawn path's. Process-local like the registry: a child evicted from
the bounded record, or from before a restart, cannot be continued."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from tools.registry import tool_error

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

_NOT_FOUND = (
    "No finished subagent '{sid}' in this conversation's spawn tree can be continued (unknown id, not yours, or no "
    "longer retained). Spawn a new task instead."
)


def _claim_resume(sid: str, parent_agent: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Take the finished child's resume spec out of its record, so a second continue can never run the same
    conversation twice at once (the attribution stub stays). ``(spec, None)`` or ``(None, error)``."""
    from tools.delegate_tool_registry import (
        _active_subagents, _active_subagents_lock, _owns_subagent_record, _recent_subagents,
    )
    with _active_subagents_lock:
        live, record = _active_subagents.get(sid), _recent_subagents.get(sid)
    if live is not None and _owns_subagent_record(live, parent_agent):
        return None, f"Subagent '{sid}' is still running: use action='steer' to send it this message."
    if record is None or not _owns_subagent_record(record, parent_agent):
        return None, _NOT_FOUND.format(sid=sid)
    with _active_subagents_lock:
        spec = record.pop("resume", None)
    return (spec, None) if spec else (None, _NOT_FOUND.format(sid=sid))


def _release_resume(sid: str, spec: dict[str, Any]) -> None:
    """Hand an unused spec back (the continue was refused before anything ran)."""
    from tools.delegate_tool_registry import _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        record = _recent_subagents.get(sid)
        if record is not None:
            record.setdefault("resume", spec)


@contextlib.contextmanager
def _transcript_store(parent_agent: Any) -> Iterator[Any]:
    """A handle on the db FILE the parent's children write to (``_open_child_session_db``), or None. Never the
    parent's own object: a gateway parent holds an async wrapper whose methods are coroutines."""
    db_path = getattr(getattr(parent_agent, "_session_db", None), "db_path", None)
    if not isinstance(db_path, (str, Path)):
        yield None
        return
    from hermes_state_registry import acquire, release_or_close
    db = acquire(db_path)
    try:
        yield db
    finally:
        release_or_close(db)


def _refusal(parent_agent: Any) -> Optional[str]:
    from agent.oneshot_footprint import agent_is_oneshot
    from agent.runtime_session_store import RuntimeSessionStore, is_worker_process
    from tools.delegate_tool_registry import is_spawn_paused
    if agent_is_oneshot(parent_agent):
        # The same frozen fact that strips live control from its schema: its children are joined inside the call.
        return "action='continue' is unavailable in one-shot sessions. Spawn a new task instead."
    if is_worker_process() or isinstance(getattr(parent_agent, "_session_db", None), RuntimeSessionStore):
        # A managed worker reserves every child's row through its owner; resuming one is not supported there.
        return "action='continue' is unavailable in managed worker sessions. Spawn a new task instead."
    if is_spawn_paused():
        return ("Delegation spawning is paused. Clear the pause via the TUI (`p` in /agents) or the "
                "`delegation.pause` RPC before retrying.")
    return None


def _start(
    sid: str, message: str, spec: dict[str, Any], parent_agent: Any, background: bool,
) -> tuple[Optional[str], Optional[str]]:
    """Rebuild, hydrate and dispatch the continued child: ``(tool result JSON, None)`` or ``(None, error)`` when
    it was refused before anything ran."""
    from tools.delegate_tool import (
        DEFAULT_MAX_ITERATIONS, _build_children, _get_max_concurrent_children, _load_config, _oneshot_spawn_budget,
        _parent_live_home,
    )
    from tools.delegate_tool_dispatch import _Batch, _capture_origin, _run_batch
    from tools.delegation_live_log import create_live_transcripts

    route, session_id = spec["route"], spec["session_id"]
    with _transcript_store(parent_agent) as db:
        history = db.get_messages_as_conversation(session_id, repair_alternation=True) if db is not None else []
        if not history:
            return None, f"Subagent '{sid}' has no saved transcript to continue from. Spawn a new task instead."
        err = _oneshot_spawn_budget(parent_agent, 1)
        if err:
            return None, err
        task_list = [{"goal": message}]  # the run's user turn; the system prompt is rebuilt from the original goal
        live_home = _parent_live_home(parent_agent)
        live_id, writers, paths = create_live_transcripts(
            task_list, None, model=spec.get("model"), provider=route.creds.get("provider"), home=live_home,
        )
        origin = _capture_origin()  # before the build: child construction clobbers the session-id binding
        children, err = _build_children(
            [{"goal": spec["goal"], "context": spec.get("context")}], [None], [route], top_role="leaf",
            max_iterations=_load_config().get("max_iterations", DEFAULT_MAX_ITERATIONS), parent_agent=parent_agent,
            live_deleg_id=live_id, live_writers=writers,
            rebuild=[{"toolsets": spec.get("toolsets") or None, "model": spec.get("model") or route.creds["model"],
                      "subagent_id": sid, "session_id": session_id, "expect_endpoint": spec.get("endpoint")}],
        )
        if err:
            return None, err
        db.reopen_session(session_id)  # its close() ended it; the continued child's close() ends it again
        children[0][2]._delegate_continue_history = history
    batch = _Batch(
        task_list, [(0, task_list[0], children[0][2])], parent_agent, route.creds, None, "leaf",
        _get_max_concurrent_children(), live_id, writers, paths, *origin, time.monotonic(), live_home=live_home,
    )
    return _run_batch(batch, background), None


def continue_subagent(subagent_id: Optional[str], message: Optional[str], parent_agent: Any, *, background: bool) -> str:
    """delegate_task(action='continue'): validate ownership and finished state, then run ``message`` as the
    child's next turn. Returns the usual dispatch handle (new delegation_id, same subagent_id) or, where the
    session joins children inline, the joined result."""
    sid, text = (subagent_id or "").strip(), (message or "").strip()
    if not sid or not text:
        return tool_error("action='continue' requires subagent_id (a finished child of yours) and a non-empty "
                          "'message' (its next instruction).")
    refusal = _refusal(parent_agent)
    if refusal:
        return tool_error(refusal)
    spec, err = _claim_resume(sid, parent_agent)
    if spec is None:
        return tool_error(err)
    try:
        result, err = _start(sid, text, spec, parent_agent, background)
    except Exception as exc:
        logger.warning("delegate_task continue of %s failed before dispatch", sid, exc_info=True)
        result, err = None, f"Could not continue subagent '{sid}': {exc}"
    if result is None:
        _release_resume(sid, spec)
        return tool_error(err)
    return result
