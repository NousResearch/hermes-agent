#!/usr/bin/env python3
"""
Goal Tool Module - model-callable standing goals.

This tool lets the model set a persistent per-session goal without asking the
user to type the /goal slash command. The continuation loop itself remains
owned by the CLI/gateway goal hooks: after the current turn ends, those hooks
judge the active goal and enqueue continuation prompts as needed.
"""

from __future__ import annotations

from typing import Optional

from tools.registry import registry, tool_error, tool_result


def _normalize_max_turns(max_turns: Optional[int]) -> Optional[int]:
    """Return a positive integer max_turns value, or None for the default."""
    if max_turns is None:
        return None
    if isinstance(max_turns, bool):
        raise ValueError("max_turns must be a positive integer")
    if isinstance(max_turns, float) and not max_turns.is_integer():
        raise ValueError("max_turns must be a positive integer")
    try:
        value = int(max_turns)
    except (TypeError, ValueError):
        raise ValueError("max_turns must be a positive integer")
    if value <= 0:
        raise ValueError("max_turns must be a positive integer")
    return value


def _resolve_default_max_turns(default_max_turns: Optional[int] = None) -> int:
    """Resolve the configured goal turn budget, falling back to DEFAULT_MAX_TURNS."""
    from hermes_cli.goals import DEFAULT_MAX_TURNS

    if default_max_turns is not None:
        value = _normalize_max_turns(default_max_turns)
        return int(value or DEFAULT_MAX_TURNS)

    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        goals_cfg = cfg.get("goals") or {}
        return int(goals_cfg.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS)
    except Exception:
        return DEFAULT_MAX_TURNS


def set_goal_tool(
    goal: str,
    *,
    session_id: str,
    max_turns: Optional[int] = None,
    default_max_turns: Optional[int] = None,
) -> str:
    """Set or replace the standing goal for the current Hermes session."""
    sid = (session_id or "").strip()
    if not sid:
        return tool_error(
            "set_goal requires an active session_id; this tool must be handled by the agent loop",
            success=False,
        )

    if not isinstance(goal, str):
        return tool_error("goal must be a string", success=False)
    goal_text = goal.strip()
    if not goal_text:
        return tool_error("goal text is empty", success=False)

    try:
        default_turns = _resolve_default_max_turns(default_max_turns)
        turns = _normalize_max_turns(max_turns)
    except ValueError as exc:
        return tool_error(str(exc), success=False)

    if turns is not None and turns > default_turns:
        return tool_error(
            f"max_turns ({turns}) exceeds configured goal budget ({default_turns})",
            success=False,
        )

    try:
        from hermes_cli.goals import GoalManager, load_goal

        manager = GoalManager(session_id=sid, default_max_turns=default_turns)
        state = manager.set(goal_text, max_turns=turns)
        persisted = load_goal(sid)
        if persisted is None or persisted.to_json() != state.to_json():
            return tool_error("failed to persist goal state", success=False)
    except Exception as exc:
        return tool_error(f"failed to set goal: {type(exc).__name__}: {exc}", success=False)

    return tool_result(
        success=True,
        action="set",
        goal=state.goal,
        status=state.status,
        turns_used=state.turns_used,
        max_turns=state.max_turns,
        message=(
            "Standing goal set. Hermes will keep working toward it after this "
            "turn until the goal is judged complete, paused, cleared, or the "
            "turn budget is exhausted."
        ),
    )


def check_goal_requirements() -> bool:
    """Goal tool has no external requirements -- always available."""
    return True


SET_GOAL_SCHEMA = {
    "name": "set_goal",
    "description": (
        "Set or replace the standing goal for the current Hermes session. "
        "Use this when the user gives an objective that should persist across "
        "turns and Hermes should keep taking concrete steps until it is done. "
        "Do not use this for ordinary short task planning; use todo for that. "
        "After this tool succeeds, the CLI/gateway /goal loop will judge the "
        "goal after each turn and continue automatically when appropriate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The persistent objective Hermes should work toward.",
            },
            "max_turns": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional turn budget before the goal auto-pauses. Omit to "
                    "use the configured default."
                ),
            },
        },
        "required": ["goal"],
    },
}


registry.register(
    name="set_goal",
    toolset="goal",
    schema=SET_GOAL_SCHEMA,
    handler=lambda args, **kw: set_goal_tool(
        goal=args.get("goal", ""),
        max_turns=args.get("max_turns"),
        session_id=kw.get("session_id", ""),
        default_max_turns=kw.get("default_max_turns"),
    ),
    check_fn=check_goal_requirements,
    emoji="⊙",
)
