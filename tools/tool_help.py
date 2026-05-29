"""On-demand full tool schema/help for compact-schema sessions."""

import json
from typing import Any

from tools.registry import discover_builtin_tools, registry


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe copy for schema/help data."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)


def tool_help(args: dict, **kwargs) -> str:
    """Return full registry help/schema for one tool."""
    name = str((args or {}).get("name") or "").strip()
    if not name:
        return json.dumps(
            {
                "success": False,
                "error": "Missing required parameter: name",
                "hint": "Pass an exact tool name, e.g. read_file or terminal.",
            },
            ensure_ascii=False,
        )

    entry = registry.get_entry(name)
    if entry is None:
        # In normal agent runs model_tools has already discovered all tools.
        # This fallback makes tool_help robust in direct tests / scripts too.
        try:
            discover_builtin_tools()
            entry = registry.get_entry(name)
        except Exception:
            entry = None
    if entry is None:
        candidates = registry.get_all_tool_names()
        close = [tool for tool in candidates if name.lower() in tool.lower()][:20]
        return json.dumps(
            {
                "success": False,
                "error": f"Unknown tool: {name}",
                "matches": close,
                "hint": "Use an exact tool name from the current session.",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": True,
            "name": entry.name,
            "toolset": entry.toolset,
            "description": entry.description,
            "schema": _jsonable(entry.schema),
            "hint": "This is the full registry schema/help. The active model may see a compact schema in the initial prompt, but the handler and full schema remain unchanged.",
        },
        ensure_ascii=False,
        indent=2,
    )


TOOL_HELP_SCHEMA = {
    "name": "tool_help",
    "description": "Show full help/schema for a tool when compact schemas omit details.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact tool name."},
        },
        "required": ["name"],
    },
}


registry.register(
    name="tool_help",
    toolset="skills",
    schema=TOOL_HELP_SCHEMA,
    handler=tool_help,
    description=TOOL_HELP_SCHEMA["description"],
)
