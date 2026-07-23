#!/usr/bin/env python3
"""
describe_tool Tool - Lazy-load tool schema definitions

Returns the full JSON schema for any named tool.  The system prompt carries
only a compact tool index (name + one-line description); calling
``describe_tool`` retrieves the complete OpenAI-format function-calling schema
(parameters, types, enums, etc.) for a specific tool on demand.

Use this when you need to inspect a tool's parameters before calling it,
especially for tools you haven't used before or whose API surface is large
enough that carrying the full schema for every tool in every turn is
wasteful.
"""

import json

from tools.registry import registry


def describe_tool(tool_name: str) -> str:
    """Return the full OpenAI-format schema for *tool_name*.

    Args:
        tool_name: The exact tool name to look up (e.g. ``"read_file"``,
                   ``"terminal"``, ``"delegate_task"``).

    Returns:
        JSON string containing the tool's full schema, or an error dict if
        the tool is not found or the registry is unavailable.
    """
    if not tool_name or not isinstance(tool_name, str):
        return json.dumps(
            {"error": "tool_name must be a non-empty string."},
            ensure_ascii=False,
        )

    try:
        schema = registry.get_schema(tool_name.strip())
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to look up tool schema: {exc}"},
            ensure_ascii=False,
        )

    if schema is None:
        return json.dumps(
            {"error": f"Tool '{tool_name.strip()}' not found in registry."},
            ensure_ascii=False,
        )

    return json.dumps(schema, indent=2, ensure_ascii=False)


def check_describe_tool_requirements() -> bool:
    """No external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

DESCRIBE_TOOL_SCHEMA = {
    "name": "describe_tool",
    "description": (
        "Retrieve the full JSON schema (parameters, types, required fields, "
        "descriptions, enums) for any named tool.  The system prompt lists "
        "tools compactly (name + description); call this to get the complete "
        "OpenAI-format function-calling definition when you need to inspect "
        "a tool's parameters before calling it.  Returns the full schema as "
        "a JSON object."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": (
                    "The exact name of the tool whose schema you want. "
                    "Examples: ``read_file``, ``terminal``, ``delegate_task``, "
                    "``memory``.  Case-sensitive, must match the registered "
                    "tool name exactly."
                ),
            },
        },
        "required": ["tool_name"],
    },
}


# --- Registry ---

registry.register(
    name="describe_tool",
    toolset="memory",
    schema=DESCRIBE_TOOL_SCHEMA,
    handler=lambda args, **kw: describe_tool(
        tool_name=args.get("tool_name", ""),
    ),
    check_fn=check_describe_tool_requirements,
    emoji="🔍",
)
