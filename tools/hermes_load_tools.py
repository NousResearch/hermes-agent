#!/usr/bin/env python3
"""
hermes_load_tools — Client-side lazy tool loading.

Background — the Anthropic ``tool_search_tool_*_20251119`` server tool is the
default Anthropic-provided mechanism for deferred tool loading.  It does the
right thing semantically (deferred MCP tool schemas, model discovers on demand)
but bills the entire prompt context once per server-side iteration within a
single API call.  In the wild we've seen 2x / 3x / 4x prompt-token multipliers
on turns that stack tool_search calls — one such turn on 2026-05-13 pushed a
405K-token prompt to 1.22M and forced compaction in mid-debug.

This module provides the client-side equivalent.  The model still sees
schema stubs for deferred MCP tools (preserves the lazy-loading UX) but
the discovery call is a regular client-side tool, so each round-trip is
a single normal API request billed once.

The handler itself is trivial: validate names, mutate the agent's
``_promoted_tools`` set, return a small confirmation.  On the next
``build_kwargs`` call the anthropic adapter expands the promoted names
into full schemas instead of stubs (see ``_apply_tool_search`` ->
``mode="client_side"`` branch in ``agent/anthropic_adapter.py``).

Like ``todo`` / ``memory`` / ``session_search`` / ``delegate_task``, this is
an agent-loop tool — it's intercepted in run_agent.py BEFORE
``handle_function_call`` so the handler has access to mutable agent state.
The registry entry below is what get_definitions() returns; the
safety-net handler is unreachable in normal operation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def load_tools(
    names: List[str],
    *,
    promoted: Set[str],
    available_names: Set[str],
    deferred_names: Optional[Set[str]] = None,
) -> str:
    """Promote MCP tool names into the active tools array.

    Args:
        names: tool names the model wants the full schema for.
        promoted: the agent's session-scoped set of already-promoted tool
            names.  Mutated in place.
        available_names: full set of registered tool names (the universe).
            Used to reject typos / hallucinated names.
        deferred_names: optional pre-computed set of names that are currently
            deferred (i.e. the model only sees stubs for these).  When
            provided, names that aren't in this set get classified as
            ``already_eager`` for a clearer return value (helps the model
            stop asking for tools it already has).

    Returns:
        JSON string with four buckets + a hint:
            loaded: names newly promoted this call
            already_loaded: names that were already promoted
            already_eager: names that ship with full schema by default
            unknown: names that aren't registered tools (typo / hallucination)
    """
    loaded: List[str] = []
    already_loaded: List[str] = []
    already_eager: List[str] = []
    unknown: List[str] = []

    for raw in names or []:
        n = (raw or "").strip()
        if not n:
            continue
        if n not in available_names:
            unknown.append(n)
            continue
        if n in promoted:
            already_loaded.append(n)
            continue
        if deferred_names is not None and n not in deferred_names:
            already_eager.append(n)
            continue
        promoted.add(n)
        loaded.append(n)

    result: Dict[str, Any] = {
        "loaded": sorted(loaded),
        "already_loaded": sorted(already_loaded),
        "already_eager": sorted(already_eager),
        "unknown": sorted(unknown),
        "total_promoted": len(promoted),
    }
    # Make the model's next move obvious: tell it to call the tool now.
    if loaded:
        result["hint"] = (
            "Schemas are now available. Call the loaded tool(s) directly on your "
            "next turn — no further hermes_load_tools call needed."
        )
    elif unknown and not (loaded or already_loaded or already_eager):
        result["hint"] = (
            "None of the requested names matched a registered tool. "
            "If you're not sure what's available, the deferred tools list "
            "includes every MCP-prefixed tool registered for this session."
        )
    return json.dumps(result, ensure_ascii=False)


def check_load_tools_requirements() -> bool:
    """No external requirements — always available."""
    return True


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

LOAD_TOOLS_SCHEMA: Dict[str, Any] = {
    "name": "hermes_load_tools",
    "description": (
        "Load full schemas for one or more MCP tools that are currently shown "
        "as name-only stubs. This is Hermes' client-side lazy-loading mechanism: "
        "MCP-prefixed tools (slack_*, salesforce_*, tanium_gateway_*, notion_*, etc.) "
        "default to stub schemas to keep the prompt small. Before calling such a "
        "tool, call this with the names you need — the full schemas become "
        "available on your next turn.\n\n"
        "**Batch your loads.** If you know you need tools A and B and C in this "
        "session, pass them all in one call: `names=[\"a\",\"b\",\"c\"]`. Don't "
        "stack multiple hermes_load_tools calls in one turn or across consecutive "
        "turns — each turn is a normal API round-trip, so loading 3 tools across "
        "3 separate turns costs 3 round-trips while loading them in one call "
        "costs 1.\n\n"
        "Tools loaded earlier in the session stay loaded — you don't need to "
        "re-call this every turn. The response tells you what was newly loaded "
        "vs already loaded vs unknown."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tool names to load full schemas for. Use the canonical "
                    "name as shown in the stub (e.g. 'slack_slack_send_message')."
                ),
                "minItems": 1,
            }
        },
        "required": ["names"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# The handler here is a safety net.  The real dispatch happens in
# run_agent.py — hermes_load_tools needs access to the agent's mutable
# ``_promoted_tools`` set, which only the agent loop has a handle to.
# If for some reason this handler IS invoked (e.g. a test calling
# registry.dispatch directly), we return a structured error so the
# failure mode is obvious instead of silently mutating nothing.

from tools.registry import registry  # noqa: E402  (registration at import time)


def _safety_net_handler(args: Dict[str, Any], **kwargs: Any) -> str:
    return json.dumps(
        {
            "error": (
                "hermes_load_tools must be handled by the agent loop. This "
                "fallback handler indicates the agent-loop interception in "
                "run_agent.py did not fire — likely a wiring bug."
            ),
            "received_args": args,
        },
        ensure_ascii=False,
    )


registry.register(
    name="hermes_load_tools",
    toolset="hermes_load_tools",
    schema=LOAD_TOOLS_SCHEMA,
    handler=_safety_net_handler,
    check_fn=check_load_tools_requirements,
    emoji="🧰",
)
