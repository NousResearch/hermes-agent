"""Opt-in model control for Hermes' persistent Ralph goal loop.

The slash command remains the default human entry point. This service-gated
tool lets the model activate the same controller when natural language
explicitly requests durable autonomous continuation.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from tools.registry import registry, tool_error, tool_result


def _available() -> bool:
    if not os.environ.get("HERMES_SESSION_ID", "").strip():
        return False
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        goals = config.get("goals") if isinstance(config.get("goals"), dict) else {}
        return bool((goals or {}).get("model_tool_enabled", False))
    except Exception:
        return False


def goal_control(args: Dict[str, Any], **_: Any) -> str:
    session_id = os.environ.get("HERMES_SESSION_ID", "").strip()
    if not session_id:
        return tool_error("persistent goal control requires an active Hermes session")

    action = str(args.get("action", "status") or "status").strip().lower()
    try:
        from hermes_cli.config import load_config
        from hermes_cli.goals import GoalContract, GoalManager

        config = load_config() or {}
        goals = config.get("goals") if isinstance(config.get("goals"), dict) else {}
        max_turns = int((goals or {}).get("max_turns", 20) or 20)
        manager = GoalManager(session_id=session_id, default_max_turns=max_turns)

        if action == "status":
            return tool_result(
                success=True,
                active=manager.is_active(),
                status=manager.status_line(),
                contract=manager.render_contract(),
            )

        if action != "start":
            return tool_error("action must be start or status")

        objective = str(args.get("objective", "") or "").strip()
        if not objective:
            return tool_error("objective is required when action is start")

        if manager.has_goal():
            return tool_result(
                success=True,
                started=False,
                reason="an active or paused goal already exists; preserve it unless the user explicitly replaces it",
                status=manager.status_line(),
            )

        contract = GoalContract.from_dict(args.get("contract"))
        state = manager.set(objective, max_turns=max_turns, contract=contract)
        return tool_result(
            success=True,
            started=True,
            status=manager.status_line(),
            max_turns=state.max_turns,
            contract_attached=not contract.is_empty(),
            contract=manager.render_contract(),
            resume="/goal resume",
            pause="/goal pause",
            clear="/goal clear",
        )
    except Exception as exc:
        return tool_error(f"goal control failed: {exc}")


GOAL_CONTROL_SCHEMA = {
    "name": "goal_control",
    "description": (
        "Activate or inspect Hermes' persistent autonomous goal loop. Use start exactly once when the user explicitly "
        "asks to keep working until done, perfect/improve something autonomously, or continue for hours or days. "
        "Do not use for ordinary one-turn tasks, and never replace an existing goal without explicit user intent. "
        "Preserve the user's intent in objective. For start, add a completion contract whenever the request defines "
        "evidence, constraints, boundaries, or user/account/legal/cost gates; the contract is the authoritative "
        "definition of done."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "status"]},
            "objective": {
                "type": "string",
                "description": "The user's exact durable objective. Required for start; omit for status.",
            },
            "contract": {
                "type": "object",
                "description": (
                    "Optional structured definition of done. Keep factual acceptance evidence in verification, "
                    "scope limits in constraints/boundaries, and human/account/legal/cost gates in stop_when."
                ),
                "properties": {
                    "outcome": {"type": "string"},
                    "verification": {"type": "string"},
                    "constraints": {"type": "string"},
                    "boundaries": {"type": "string"},
                    "stop_when": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


registry.register(
    name="goal_control",
    toolset="goal",
    schema=GOAL_CONTROL_SCHEMA,
    handler=goal_control,
    check_fn=_available,
    emoji="🎯",
)
