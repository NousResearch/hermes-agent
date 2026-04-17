#!/usr/bin/env python3
"""Reasoning effort tool — adjust the agent's reasoning effort at runtime."""

import json

from hermes_constants import VALID_REASONING_EFFORTS, parse_reasoning_effort
from tools.registry import registry


def check_reasoning_effort_requirements() -> bool:
    """Reasoning effort has no external requirements."""
    return True


def reasoning_effort_tool(level: str, persist: bool = False, callback=None) -> str:
    """Validate and dispatch a reasoning-effort update through a runtime callback."""
    if not level or not str(level).strip():
        return json.dumps({"error": "level is required"}, ensure_ascii=False)

    normalized = str(level).strip().lower()
    parsed = parse_reasoning_effort(normalized)
    if parsed is None:
        return json.dumps(
            {
                "error": f"invalid level '{normalized}'",
                "valid_levels": ["none", *VALID_REASONING_EFFORTS],
            },
            ensure_ascii=False,
        )

    if callback is None:
        return json.dumps(
            {"error": "reasoning_effort tool is not available in this execution context"},
            ensure_ascii=False,
        )

    try:
        result = callback(parsed, level=normalized, persist=bool(persist))
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to set reasoning effort: {exc}"},
            ensure_ascii=False,
        )

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


REASONING_EFFORT_SCHEMA = {
    "name": "reasoning_effort",
    "description": (
        "Adjust the model's reasoning effort for the current session. "
        "Use this when the task becomes more complex or simpler and you need "
        "to increase or decrease thinking depth. Valid levels: none, minimal, "
        "low, medium, high, xhigh."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh"],
                "description": "Desired reasoning effort level.",
            },
            "persist": {
                "type": "boolean",
                "description": "If true, also save the setting through the platform layer.",
                "default": False,
            },
        },
        "required": ["level"],
    },
}


registry.register(
    name="reasoning_effort",
    toolset="reasoning",
    schema=REASONING_EFFORT_SCHEMA,
    handler=lambda args, **kw: reasoning_effort_tool(
        level=args.get("level", ""),
        persist=args.get("persist", False),
        callback=kw.get("callback"),
    ),
    check_fn=check_reasoning_effort_requirements,
    emoji="🧠",
)