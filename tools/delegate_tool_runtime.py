"""Delegation lifecycle finalization and compatibility helpers."""

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from tools.delegation_outcome import delegation_stop_evidence
from tools.delegate_tool_control import _subagent_stop_tool_call_history
from tools.delegate_tool_setup import _apply_summary_budget

logger = logging.getLogger("tools.delegate_tool")


def _facade():
    from tools import delegate_tool as facade

    return facade


def _build_child_agent(**kwargs):
    return _facade()._build_child_agent(**kwargs)


def _run_single_child(*args, **kwargs):
    return _facade()._run_single_child(*args, **kwargs)


_PARENT_FINALIZATION_LOCK_GUARD = threading.Lock()
_PARENT_FINALIZATION_FALLBACK_LOCK = threading.RLock()
_CHILD_CONSTRUCTION_LOCK = threading.RLock()


def _build_child_preserving_parent_tools(**kwargs):
    """Build a child without leaking its resolved toolset into the parent."""
    import model_tools

    with _CHILD_CONSTRUCTION_LOCK:
        parent_tool_names = list(model_tools._last_resolved_tool_names)
        try:
            child = _build_child_agent(**kwargs)
        finally:
            model_tools._last_resolved_tool_names = parent_tool_names
    child._delegate_saved_tool_names = parent_tool_names
    return child


def _parent_finalization_lock(parent_agent) -> threading.RLock:
    """Return the per-parent lock that serializes lifecycle side effects."""
    if parent_agent is None:
        return _PARENT_FINALIZATION_FALLBACK_LOCK
    lock = getattr(parent_agent, "_subagent_finalization_lock", None)
    if lock is not None:
        return lock
    with _PARENT_FINALIZATION_LOCK_GUARD:
        lock = getattr(parent_agent, "_subagent_finalization_lock", None)
        if lock is None:
            lock = threading.RLock()
            try:
                setattr(parent_agent, "_subagent_finalization_lock", lock)
            except Exception:
                return _PARENT_FINALIZATION_FALLBACK_LOCK
    return lock


def _finalize_child_results(
    results: List[Dict[str, Any]],
    task_list: List[Dict[str, Any]],
    children: List[tuple[int, Dict[str, Any], Any]],
    parent_agent,
) -> None:
    """Apply host-owned summary, memory, hook, and cost contracts once."""
    with _parent_finalization_lock(parent_agent):
        _apply_summary_budget(results, parent_agent)
        child_by_index = {index: child for index, _task, child in children}

        if parent_agent and getattr(parent_agent, "_memory_manager", None):
            for entry in results:
                try:
                    task_index = entry.get("task_index", -1)
                    task_goal = (
                        task_list[task_index]["goal"]
                        if isinstance(task_index, int)
                        and 0 <= task_index < len(task_list)
                        else ""
                    )
                    child = child_by_index.get(task_index)
                    parent_agent._memory_manager.on_delegation(
                        task=task_goal,
                        result=entry.get("summary", "") or "",
                        child_session_id=getattr(child, "session_id", ""),
                    )
                except Exception:
                    pass

        parent_session_id = getattr(parent_agent, "session_id", None)
        try:
            from hermes_cli.plugins import invoke_hook as invoke_hook
        except Exception:
            invoke_hook = None

        children_cost_total = 0.0
        for entry in results:
            child_role = entry.pop("_child_role", None)
            child_cost = entry.pop("_child_cost_usd", 0.0)
            try:
                if child_cost:
                    children_cost_total += float(child_cost)
            except (TypeError, ValueError):
                pass
            if invoke_hook is None:
                continue
            try:
                child_index = entry.get("task_index", -1)
                child = child_by_index.get(child_index)
                invoke_hook(
                    "subagent_stop",
                    parent_session_id=parent_session_id,
                    parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "",
                    child_session_id=getattr(child, "session_id", None),
                    child_role=child_role,
                    child_summary=entry.get("summary"),
                    child_status=entry.get("status"),
                    **delegation_stop_evidence(entry),
                    tool_call_history=_subagent_stop_tool_call_history(
                        entry.get("tool_trace")
                    ),
                    duration_ms=int((entry.get("duration_seconds") or 0) * 1000),
                )
            except Exception:
                logger.debug("subagent_stop hook invocation failed", exc_info=True)

        if children_cost_total > 0.0:
            try:
                current = float(
                    getattr(parent_agent, "session_estimated_cost_usd", 0.0) or 0.0
                )
                parent_agent.session_estimated_cost_usd = current + children_cost_total
                if getattr(parent_agent, "session_cost_source", "none") in {
                    None,
                    "",
                    "none",
                }:
                    parent_agent.session_cost_source = "subagent"
                if getattr(parent_agent, "session_cost_status", "unknown") in {
                    None,
                    "",
                    "unknown",
                }:
                    parent_agent.session_cost_status = "estimated"
            except Exception:
                logger.debug("Subagent cost rollup failed", exc_info=True)


def _run_child_lifecycle(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
) -> Dict[str, Any]:
    """Run one child and apply the same host lifecycle used by delegate_task."""
    result = _run_single_child(task_index, goal, child, parent_agent)
    result.setdefault("task_index", task_index)
    task = {"goal": goal}
    _finalize_child_results(
        [result],
        [{"goal": ""} for _ in range(task_index)] + [task],
        [(task_index, task, child)],
        parent_agent,
    )
    return result


def _recover_tasks_from_json_string(
    tasks: Any,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if not isinstance(tasks, str):
        return None, None
    raw = tasks.strip()
    if not raw:
        return None, "Provide either 'goal' (single task) or 'tasks' (batch)."
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, (
            "tasks must be a JSON array of task objects; received a string "
            f"that could not be parsed as JSON ({exc.msg})."
        )
    if not isinstance(parsed, list):
        return None, (
            f"tasks must be a JSON array of task objects; parsed "
            f"{type(parsed).__name__} instead."
        )
    return parsed, None
