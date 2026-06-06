"""Goal management tools for autonomous AI goal-setting.

Allows the AI agent to set, check, and clear standing goals that the
Judge loop (the Ralph loop) then automatically drives to completion.

Usage (from the AI's perspective):
  - ``set_goal(goal=\"...\", max_turns=20)`` — start working toward a goal.
  - ``get_goal_status()`` — check whether a goal is active / done / paused.
  - ``clear_goal()`` — abandon the current goal.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


def _get_goal_manager(task_id: str, max_turns: int = 20):
    """Create a ``GoalManager`` bound to the current session.

    ``task_id`` is the agent's ``session_id`` — the same value the gateway
    and CLI use when constructing GoalManagers for the /goal command.
    """
    try:
        from hermes_cli.goals import GoalManager
    except ImportError:
        return None
    if not task_id:
        return None
    return GoalManager(session_id=task_id, default_max_turns=max_turns)


# ──────────────────────────────────────────────────────────────────────
# Tool: set_goal
# ──────────────────────────────────────────────────────────────────────


def _set_goal_available() -> bool:
    """Available when the goals module can be imported."""
    try:
        from hermes_cli import goals  # noqa: F401
        return True
    except ImportError:
        return False


def set_goal(goal: str, max_turns: int = 20, task_id: Optional[str] = None) -> str:
    """Set a standing goal the agent works on across turns.

    The Judge loop evaluates progress after each turn and automatically
    continues if the goal is not yet achieved.

    Args:
        goal: The goal text describing what to achieve.
        max_turns: Maximum number of turns before auto-pause (default: 20).

    Returns:
        A JSON string describing the result.
    """
    if not goal or not goal.strip():
        return json.dumps({
            "success": False,
            "error": "Goal text cannot be empty.",
        })

    mgr = _get_goal_manager(task_id or "")
    if mgr is None:
        return json.dumps({
            "success": False,
            "error": "Goal system unavailable (SessionDB not reachable).",
        })

    try:
        state = mgr.set(goal, max_turns=max_turns)
        return json.dumps({
            "success": True,
            "goal": state.goal,
            "max_turns": state.max_turns,
            "message": f"Goal set: {state.goal}. Working toward it across turns.",
        }, ensure_ascii=False)
    except ValueError as exc:
        return json.dumps({
            "success": False,
            "error": str(exc),
        })


registry.register(
    name="set_goal",
    toolset="core",
    schema={
        "name": "set_goal",
        "description": (
            "Set a standing goal that the agent will work on across multiple turns. "
            "Once set, the system automatically evaluates progress after each turn "
            "and continues working until the goal is achieved, the turn budget runs "
            "out, or the user interrupts. Use this when a task requires multiple "
            "steps that span several conversation turns — e.g. refactoring a large "
            "codebase, running a multi-phase analysis, or implementing a feature "
            "with multiple sub-tasks. After setting the goal, continue working "
            "normally; the system handles continuation automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal to work toward. Be specific about what constitutes completion.",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum number of turns before auto-pausing (default: 20).",
                    "default": 20,
                },
            },
            "required": ["goal"],
        },
    },
    handler=lambda args, **kw: set_goal(
        goal=args.get("goal", ""),
        max_turns=int(args.get("max_turns", 20)),
        task_id=kw.get("task_id"),
    ),
    check_fn=_set_goal_available,
)


# ──────────────────────────────────────────────────────────────────────
# Tool: get_goal_status
# ──────────────────────────────────────────────────────────────────────


def get_goal_status(task_id: Optional[str] = None) -> str:
    """Check the current goal status.

    Returns:
        A JSON string with status information.
    """
    mgr = _get_goal_manager(task_id or "")
    if mgr is None:
        return json.dumps({
            "success": False,
            "error": "Goal system unavailable.",
        })

    state = mgr.state
    if state is None:
        return json.dumps({
            "success": True,
            "has_goal": False,
            "message": "No active goal. Use set_goal to create one.",
        })

    sub = f", {len(state.subgoals)} subgoal(s)" if state.subgoals else ""
    return json.dumps({
        "success": True,
        "has_goal": True,
        "goal": state.goal,
        "status": state.status,
        "turns_used": state.turns_used,
        "max_turns": state.max_turns,
        "last_verdict": state.last_verdict,
        "last_reason": state.last_reason,
        "summary": f"Goal ({state.status}, {state.turns_used}/{state.max_turns} turns{sub}): {state.goal}",
    }, ensure_ascii=False)


registry.register(
    name="get_goal_status",
    toolset="core",
    schema={
        "name": "get_goal_status",
        "description": (
            "Check the current goal status: whether a standing goal is active, "
            "paused, done, or cleared. Also shows turn usage and the last judge "
            "verdict. Useful to understand progress before deciding next steps."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    handler=lambda args, **kw: get_goal_status(task_id=kw.get("task_id")),
    check_fn=_set_goal_available,
)


# ──────────────────────────────────────────────────────────────────────
# Tool: clear_goal
# ──────────────────────────────────────────────────────────────────────


def clear_goal(task_id: Optional[str] = None) -> str:
    """Clear (abandon) the current goal.

    The Judge loop stops, and the agent returns to normal turn-by-turn
    conversation.

    Returns:
        A JSON string describing the result.
    """
    mgr = _get_goal_manager(task_id or "")
    if mgr is None:
        return json.dumps({
            "success": False,
            "error": "Goal system unavailable.",
        })

    had_goal = mgr.has_goal()
    mgr.clear()
    return json.dumps({
        "success": True,
        "had_goal": had_goal,
        "message": "Goal cleared." if had_goal else "No active goal to clear.",
    })


registry.register(
    name="clear_goal",
    toolset="core",
    schema={
        "name": "clear_goal",
        "description": (
            "Clear (abandon) the current standing goal. The Judge loop stops "
            "and the agent returns to normal turn-by-turn conversation. "
            "Use this when the goal has been superseded by a new priority or "
            "when the current work is no longer needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    handler=lambda args, **kw: clear_goal(task_id=kw.get("task_id")),
    check_fn=_set_goal_available,
)
