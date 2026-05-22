"""Agent-callable bridge for Hermes standing goals.

The `/goal` slash command is handled before normal agent turns, so a skill
cannot invoke it directly.  This tool gives the agent a first-class, session-
scoped way to activate the same GoalManager state after it has drafted a goal
and the user approves running it.
"""

from __future__ import annotations

from typing import List, Optional

from tools.registry import registry, tool_error, tool_result


def _current_session_id(task_id: Optional[str] = None) -> str:
    """Resolve the session id for the current tool call.

    `run_conversation(..., task_id=...)` passes the active session id to tool
    handlers in both CLI and gateway paths.  Gateway sessions also expose a
    contextvar-backed `HERMES_SESSION_ID` for tools that need to recover it
    without relying on handler kwargs.
    """
    if task_id:
        return str(task_id).strip()
    try:
        from gateway.session_context import get_session_env

        return (get_session_env("HERMES_SESSION_ID", "") or "").strip()
    except Exception:
        return ""


def set_goal(
    goal: str,
    subgoals: Optional[List[str]] = None,
    max_turns: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    """Set a standing goal for the active Hermes session.

    Returns JSON.  This intentionally mirrors `/goal <text>` + optional
    `/subgoal <text>` state mutation, but does not enqueue a separate kickoff
    message.  The current turn's normal post-turn goal hook will evaluate and
    continue the loop if the newly-set goal is not already satisfied.
    """
    sid = _current_session_id(task_id)
    if not sid:
        return tool_error(
            "Cannot set a standing goal because no active Hermes session id was available.",
            success=False,
        )

    try:
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id=sid)
        state = mgr.set(goal, max_turns=max_turns)
        added_subgoals: List[str] = []
        for raw in subgoals or []:
            text = str(raw or "").strip()
            if text:
                added_subgoals.append(mgr.add_subgoal(text))
        state = mgr.state or state
        return tool_result(
            success=True,
            session_id=sid,
            goal=state.goal,
            status=state.status,
            max_turns=state.max_turns,
            subgoals=list(state.subgoals),
            message=(
                "Standing goal activated. The session goal loop will continue "
                "after this turn if the goal is not already satisfied."
            ),
        )
    except ValueError as exc:
        return tool_error(str(exc), success=False)
    except Exception as exc:
        return tool_error(f"Failed to set standing goal: {exc}", success=False)


_GOAL_SCHEMA = {
    "name": "set_goal",
    "description": (
        "Activate or replace the standing /goal for the current Hermes session. "
        "Use after drafting a high-quality goal and the user approves running it. "
        "Optional subgoals become /subgoal criteria. Does not require the user to paste a slash command."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The polished standing goal text to activate for this session.",
            },
            "subgoals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional additional success criteria, equivalent to /subgoal entries.",
            },
            "max_turns": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional turn budget override for this goal. Defaults to configured goals.max_turns.",
            },
        },
        "required": ["goal"],
    },
}


registry.register(
    name="set_goal",
    toolset="goal",
    schema=_GOAL_SCHEMA,
    handler=lambda args, **kw: set_goal(
        goal=args.get("goal", ""),
        subgoals=args.get("subgoals"),
        max_turns=args.get("max_turns"),
        task_id=kw.get("task_id"),
    ),
    check_fn=lambda: True,
    description="Activate the current session's standing /goal from agent-generated text.",
    emoji="⊙",
)
