"""Lazy tool schema loading for Hermes Agent (issue #6839).

When enabled (``tools.loading: lazy``), the model-visible tools array
contains compact summaries — ``{name, description}`` only, no parameter
schemas — plus a ``request_tool_schema`` bridge tool.  The model calls
``request_tool_schema`` to load a tool's full JSON schema on demand, then
invokes the real tool directly.

Design constraints (see AGENTS.md):

* Eager remains the default.  Lazy is opt-in via ``tools.loading: lazy``.
* Core tools are NOT excluded — unlike ``tool_search`` (which only defers
  MCP/plugin tools), lazy mode compacts *every* tool, including core.
  The model still sees every tool name and description; it just doesn't
  get the parameter schema until it asks for it.
* No premature execution.  ``request_tool_schema`` returns metadata only
  — it does NOT invoke the underlying tool.
* Prompt caching is preserved.  Compact summaries are byte-stable for the
  life of a conversation (just like eager schemas), so the cached prefix
  is not invalidated by switching to lazy mode.
* Backward compatible.  Default is eager (unchanged behaviour).  The lazy
  path only activates when the user explicitly opts in.

Integration with ``tool_search``:
  When both lazy loading and tool_search are active, lazy runs first.
  Since lazy already compacts everything, tool_search's deferral of
  MCP/plugin tools becomes redundant (they're already compact).  This is
  correct — the two features compose cleanly.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("tools.lazy_tool_loading")

# Keep a patchable wrapper while resolving the live config function at call time.
def load_config():
    from hermes_cli.config import load_config as _load_config
    return _load_config()


# ---------------------------------------------------------------------------
# Bridge tool name
# ---------------------------------------------------------------------------

REQUEST_TOOL_SCHEMA_NAME = "request_tool_schema"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_loading_mode() -> str:
    """Return the resolved ``tools.loading`` value: ``"eager"`` or ``"lazy"``.

    Reads from the user config file.  Falls back to ``"eager"`` on any
    error (missing config, import failure, etc.) so the default path is
    always safe.
    """
    try:
        cfg = load_config() or {}
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        if not isinstance(tools_cfg, dict):
            return "eager"
        raw = str(tools_cfg.get("loading", "eager")).strip().lower()
        if raw in ("lazy",):
            return "lazy"
        return "eager"
    except Exception:
        return "eager"


# ---------------------------------------------------------------------------
# Bridge tool schema
# ---------------------------------------------------------------------------

def request_tool_schema_tool_def() -> Dict[str, Any]:
    """Build the ``request_tool_schema`` bridge tool schema.

    This is injected into the model-visible tools array when lazy mode is
    active.  The model calls it to retrieve the full JSON schema for a
    tool it wants to use.
    """
    return {
        "type": "function",
        "function": {
            "name": REQUEST_TOOL_SCHEMA_NAME,
            "description": (
                "Load the full JSON schema for one tool.  Call this before "
                "invoking a tool whose parameters you need to see.  Returns "
                "the tool's name, description, and complete parameter schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact tool name to retrieve the schema for.",
                    },
                },
                "required": ["name"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------


def dispatch_request_tool_schema(
    args: Dict[str, Any],
    *,
    tool_names_in_scope: Optional[Set[str]] = None,
) -> str:
    """Execute the ``request_tool_schema`` bridge tool.

    Returns the full OpenAI-format tool definition for the requested tool,
    or an error JSON string if the tool is not found or not in scope.

    Args:
        args: The bridge tool call arguments (must contain ``name``).
        tool_names_in_scope: If provided, restrict to these tool names.
            Prevents a session with a restricted toolset from loading
            schemas for out-of-scope tools.
    """
    name = str(args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name is required"}, ensure_ascii=False)

    if tool_names_in_scope is not None and name not in tool_names_in_scope:
        return json.dumps({
            "error": f"Tool '{name}' is not available in this session.",
        }, ensure_ascii=False)

    from tools.registry import registry
    entry = registry.get_entry(name)
    if entry is None:
        return json.dumps({
            "error": f"Tool '{name}' not found. Check the spelling against the tools list.",
        }, ensure_ascii=False)

    # Build the full OpenAI-format definition, mirroring
    # registry.get_definitions() output shape.
    schema_with_name = {**entry.schema, "name": entry.name}
    if entry.dynamic_schema_overrides is not None:
        try:
            overrides = entry.dynamic_schema_overrides()
            if isinstance(overrides, dict):
                schema_with_name.update(overrides)
        except Exception as exc:
            logger.warning(
                "dynamic_schema_overrides for tool %s raised %s; using static schema",
                name, exc,
            )

    result = {
        "tool_schema": {
            "type": "function",
            "function": schema_with_name,
        },
    }
    return json.dumps(result, ensure_ascii=False)
