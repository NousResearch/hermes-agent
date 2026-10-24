"""Validated transfer of child-owned background work to the parent conversation."""

from typing import Optional


def handoff_process(session_id: str, args: dict, task_id: Optional[str]) -> dict:
    """Subagent-only: transfer a running background process to the parent agent so its completion is delivered THERE
    (child-owned process notices are suppressed and child teardown kills what it owns). Validated against the live spawn
    tree: the caller must be a registered child and must own the process; anything else is an error, never a silent
    no-op, so a PID mentioned in prose can't masquerade as a transfer."""
    from tools import process_registry as registry
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
    from tools.terminal_tool import _resolve_container_task_id
    with _active_subagents_lock:
        record = _active_subagents.get(str(task_id or ""))
    child = record.get("agent") if record else None
    parent_ref = getattr(child, "_delegate_parent_ref", None)
    parent = parent_ref() if callable(parent_ref) else None
    if parent is None:
        return {"error": "handoff is only available to a running subagent with a live parent; you are not one."}
    parent_owner = str(getattr(parent, "_current_task_id", "") or getattr(parent, "session_id", "") or "")
    if not parent_owner:
        return {"error": "parent has no process owner id yet; retry after the parent's turn has started."}
    handed = getattr(child, "_handed_off_processes", None)
    if handed is None:
        handed = child._handed_off_processes = []
    if len(handed) >= registry._MAX_HANDOFFS_PER_CHILD:
        return {"error": f"handoff cap reached ({registry._MAX_HANDOFFS_PER_CHILD} per subagent); wait on or kill the rest yourself."}
    note = str(args.get("data") or "").strip()
    if not note:
        return {"error": "handoff requires `data`: one sentence saying what the process is for and what the parent should do with its result."}
    session = registry.process_registry.transfer_ownership(
        session_id, from_owner=str(task_id or ""), to_owner=parent_owner,
        to_task_id=_resolve_container_task_id(parent_owner),
        to_session_key=str(getattr(parent, "session_id", "") or ""), note=note,
        to_parent_session_id=str(getattr(parent, "session_id", "") or ""))
    if session is None:
        return {"error": f"cannot hand off {session_id}: not a running process you own (already exited? read its result "
                         "with poll/log and report it instead)."}
    handed.append({"session_id": session.id, "command": session.command, "note": note})
    return {"status": "handed_off", "session_id": session.id, "command": session.command,
            "note": "Your parent now owns this process and will receive its completion; you will not. Mention the handoff "
                    "in your final answer."}
