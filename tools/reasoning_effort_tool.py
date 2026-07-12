#!/usr/bin/env python3
"""Reasoning effort tool — adjust the agent's reasoning effort at runtime.

The tool itself is a thin validation + dispatch layer: the actual state
change lives in the agent loop (``AIAgent._apply_reasoning_effort``), which
mutates the live agent's ``reasoning_config`` and, when the platform provides
one, routes scope handling through ``reasoning_update_callback`` (session
override on the gateway, config persistence on request). Request construction
reads ``agent.reasoning_config`` on every API call, so a change takes effect
on the next request without touching the lifetime-stable system prompt.
"""

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
        "Adjust your own reasoning effort for the current session. "
        "Use it when the task at hand needs materially more or less thinking "
        "depth. The change applies from the next model request onward. "
        "Levels from lowest to highest: none, "
        + ", ".join(VALID_REASONING_EFFORTS)
        + "."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": ["none", *VALID_REASONING_EFFORTS],
                "description": "Desired reasoning effort level.",
            },
            "persist": {
                "type": "boolean",
                "description": (
                    "If true, also persist the level as the global default "
                    "via the platform layer. Only use when the user asked "
                    "for a permanent change."
                ),
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
