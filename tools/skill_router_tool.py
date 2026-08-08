"""skill_route tool: deterministic local skill-router proof output."""

from __future__ import annotations

import json
from typing import Any

from agent.skill_router import route_skills
from tools.registry import registry


SKILL_ROUTE_SCHEMA = {
    "name": "skill_route",
    "description": (
        "Rank available skills for a task and return proof: candidate skills, "
        "selected skills, why selected, and skipped matches. Use before "
        "complex technical/release/security/sandbox/memory tasks, then load "
        "selected skills with skill_view."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Short description of the user's task to route skills for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum selected skills to return (1-25, default 8).",
            },
            "available_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional explicit tool names available in the current agent session.",
            },
            "available_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional explicit toolsets available in the current agent session.",
            },
        },
        "required": ["task"],
    },
}


def _set_from_args(args: dict[str, Any], key: str) -> set[str] | None:
    raw = args.get(key)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw if str(item).strip()}
    if isinstance(raw, str):
        return {part.strip() for part in raw.split(",") if part.strip()}
    return None


def skill_route(args: dict[str, Any], **_kw) -> str:
    """Registry handler for the deterministic local skill router."""
    task = str(args.get("task") or "").strip()
    if not task:
        return json.dumps(
            {"success": False, "error": "task is required"},
            ensure_ascii=False,
        )
    try:
        top_k = int(args.get("top_k") or 8)
    except (TypeError, ValueError):
        top_k = 8
    result = route_skills(
        task,
        available_tools=_set_from_args(args, "available_tools"),
        available_toolsets=_set_from_args(args, "available_toolsets"),
        top_k=top_k,
    )
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="skill_route",
    toolset="skills",
    schema=SKILL_ROUTE_SCHEMA,
    handler=skill_route,
    emoji="🧭",
)
