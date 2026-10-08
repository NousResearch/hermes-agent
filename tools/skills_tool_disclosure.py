"""Disclose a skill's deferred tools when the skill is read.

A skill that declares ``metadata.hermes.requires_tools`` (or ``requires_toolsets``) names the tools
its procedure calls. When Tool Search has deferred those tools behind the ``tool_call`` bridge, the
skill body still reads "call X" while the model-facing tool list has no X: it tries the bare name,
gets an unknown-tool error, and loops through ``tool_search`` before it reaches ``tool_call``
(#130153). Reading the skill IS the moment the model needs those schemas, so ``skill_view`` attaches
them here — the deepagents ``include_tools`` pattern (langchain-ai/deepagents#6552), without a
second frontmatter key: ``requires_*`` already names the tools.

Scope is the session's own catalog (``enabled_toolsets``/``disabled_toolsets`` forwarded by the
dispatcher), so a restricted session is never shown a schema its ``tool_call`` would refuse.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Per-view cap: a skill naming a whole MCP server's toolset must not paste 50 schemas into one
# tool result; the rest stay one `tool_describe` away and the note says so.
MAX_DISCLOSED_TOOLS = 8


def _required_tool_names(frontmatter: Dict[str, Any], tool_defs: List[Dict[str, Any]]) -> List[str]:
    """Tool names a skill's ``requires_tools``/``requires_toolsets`` resolve to inside *tool_defs*."""
    from agent.skill_utils import extract_skill_conditions
    from tools.registry import registry

    conditions = extract_skill_conditions(frontmatter)
    wanted = {str(t) for t in conditions.get("requires_tools", []) if str(t).strip()}
    toolsets = {str(ts) for ts in conditions.get("requires_toolsets", []) if str(ts).strip()}
    names: List[str] = []
    for td in tool_defs:
        name = (td.get("function") or {}).get("name", "")
        if not name:
            continue
        if name in wanted or (toolsets and registry.get_toolset_for_tool(name) in toolsets):
            names.append(name)
    return names


def disclosed_tools_for_skill(
    frontmatter: Dict[str, Any], *, enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """``{"tools": {name: {description, parameters}}, "note": ...}`` for the skill's required tools
    that this session reaches only through the ``tool_call`` bridge; None when nothing is deferred.
    Fail-open: a catalog read error never blocks the skill content."""
    try:
        import model_tools
        from tools import tool_search as ts

        raw_defs = model_tools.get_tool_definitions(
            enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
            quiet_mode=True, skip_tool_search_assembly=True) or []
        required = _required_tool_names(frontmatter, raw_defs)
        if not required:
            return None
        deferred = ts.scoped_deferrable_names(raw_defs)
        bridged = [name for name in required if name in deferred]
        if not bridged:
            return None
        # The bridge must actually be active for this session: with tool_search off every
        # deferrable tool is direct and already in the model's tool list.
        assembled = model_tools.get_tool_definitions(
            enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets, quiet_mode=True) or []
        assembled_names = {(td.get("function") or {}).get("name", "") for td in assembled}
        if ts.TOOL_CALL_NAME not in assembled_names:
            return None
        bridged = [name for name in bridged if name not in assembled_names]
        if not bridged:
            return None
        by_name = {(td.get("function") or {}).get("name", ""): td.get("function") or {} for td in raw_defs}
        shown, omitted = bridged[:MAX_DISCLOSED_TOOLS], bridged[MAX_DISCLOSED_TOOLS:]
        tools = {name: {"description": by_name[name].get("description", ""),
                        "parameters": by_name[name].get("parameters", {})} for name in shown}
        note = (
            f"This skill's tools ({', '.join(shown)}) are deferred behind tool search in this session: "
            f"invoke each through tool_call {{\"calls\":[{{\"name\": \"<tool>\", \"arguments\": {{...}}}}]}} "
            "using the schemas above — a direct call to the bare name is rejected as unknown."
        )
        if omitted:
            note += f" {len(omitted)} more ({', '.join(omitted)}) load via tool_describe."
        return {"tools": tools, "note": note}
    except Exception:  # catalog boundary: a tool-defs fault must never block a skill read
        logger.debug("skill tool disclosure skipped", exc_info=True)
        return None
