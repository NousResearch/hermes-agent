"""ToolSearch — deferred-tool fetch mechanism for MCP context-bloat.

When ``mcp.tool_search.enabled`` is true, MCP tool schemas are stripped
from the per-call ``self.tools`` array and registered into the deferred
pool (``tools/deferred_pool.py``). The model sees only this single
``tool_search`` tool plus the deferred names in a system message. To
invoke a deferred tool, the model calls ``tool_search`` with either a
keyword query or an explicit ``select:<name>[,<name>...]`` directive;
the response is a ``<functions>...</functions>`` block carrying the full
JSON schema(s). A side effect of select is that the chosen schemas get
promoted back into ``self.tools`` / ``valid_tool_names`` so the model
can invoke them on the next turn exactly like a normal registered tool.

Mirrors Anthropic Claude Code's MCP Tool Search (rolled out in v2.1.7,
2026-02). See NousResearch/hermes-agent issue #6839 for background.
"""
from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from tools.registry import registry
from tools.deferred_pool import get_pool

logger = logging.getLogger(__name__)


TOOL_SEARCH_SCHEMA = {
    "name": "tool_search",
    "description": (
        "Fetch full schema definitions for deferred MCP tools so they can be called. "
        "Deferred tools appear by name in the system message; their full input schemas "
        "are not loaded by default to keep the context window small. Use this tool to "
        "fetch the schema for the tool(s) you need before calling them. Once a tool "
        "appears in the returned <functions> block, it is callable exactly like any "
        "tool listed at the top of the prompt.\n\n"
        "Query forms:\n"
        "  • 'select:Read,Edit'  — fetch these exact tools by name\n"
        "  • 'notebook jupyter'  — keyword search, up to max_results best matches\n"
        "  • '+slack send'       — require 'slack' in the name, rank by remaining terms"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Query for selecting deferred tools. Use 'select:<name>[,<name>...]' "
                    "for direct selection, or keywords (optionally prefixed with '+' for "
                    "must-match) to search."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def _score(name: str, summary: str, terms: List[str], must: List[str]) -> float:
    """Rank a candidate. Higher is better. Negative for must-mismatch."""
    haystack = f"{name} {summary}".lower()
    for tok in must:
        if tok not in haystack:
            return -1.0
    if not terms:
        # must-only query: rank by name length (prefer shorter, more specific)
        return 1000.0 - len(name)
    score = 0.0
    for tok in terms:
        if tok in name.lower():
            score += 3.0
        if tok in haystack:
            score += 1.0
        score += SequenceMatcher(None, tok, name.lower()).ratio()
    return score


def _parse_query(raw: str) -> Tuple[List[str], List[str], List[str]]:
    """Return (explicit_names, must_terms, optional_terms).

    'select:foo,bar' → (['foo','bar'], [], [])
    '+x y z'         → ([], ['x'], ['y','z'])
    'x y'            → ([], [], ['x','y'])
    """
    raw = (raw or "").strip()
    if raw.lower().startswith("select:"):
        rest = raw.split(":", 1)[1]
        names = [n.strip() for n in rest.split(",") if n.strip()]
        return (names, [], [])
    must: List[str] = []
    opt: List[str] = []
    for tok in raw.split():
        if tok.startswith("+") and len(tok) > 1:
            must.append(tok[1:].lower())
        else:
            opt.append(tok.lower())
    return ([], must, opt)


def _format_functions_block(entries: List[Dict[str, Any]]) -> str:
    """Render entries as a <functions>...</functions> block.

    Each line is one ``<function>{...}</function>`` carrying the full JSON
    schema (name + description + parameters). Mirrors the Anthropic Claude
    Code wire format so models trained on that data recognise it natively.
    """
    if not entries:
        return "<functions></functions>\n(no matches)"
    lines = ["<functions>"]
    for entry in entries:
        schema = entry["schema"]
        payload = {
            "description": schema.get("description", ""),
            "name": schema["name"],
            "parameters": schema.get("parameters", schema.get("input_schema", {})),
        }
        lines.append(f"<function>{json.dumps(payload, ensure_ascii=False)}</function>")
    lines.append("</functions>")
    return "\n".join(lines)


def _promote_to_session(entries: List[Dict[str, Any]]) -> None:
    """Register selected deferred tools back into the registry's per-session
    visibility (via the deferred pool's promotion hook in run_agent)."""
    # The schemas are already in tools/registry.py (MCP tools register at
    # discover-time regardless of tool_search). The agent loop reads the
    # deferred pool's "promoted" set on each get_tool_definitions() refresh
    # — see model_tools._compute_tool_definitions. We just mark them as
    # promoted by removing them from the pool; subsequent self.tools
    # rebuilds will pick them up from the registry normally.
    pool = get_pool()
    for entry in entries:
        # Mark as promoted: remove from deferred so model_tools includes the
        # full schema in self.tools on next refresh. We intentionally keep
        # registry registration untouched.
        pool.remove(entry["name"])


def _handle_tool_search(args: Dict[str, Any], **_kwargs: Any) -> str:
    query = str(args.get("query", "") or "")
    try:
        max_results = int(args.get("max_results", 5) or 5)
    except (TypeError, ValueError):
        max_results = 5
    pool = get_pool()
    if len(pool) == 0:
        return "<functions></functions>\n(no deferred tools registered)"

    explicit, must, opt = _parse_query(query)
    selected: List[Dict[str, Any]] = []

    if explicit:
        for name in explicit:
            entry = pool.get(name)
            if entry is None:
                # Try case-insensitive fallback
                lname = name.lower()
                for n in pool.names():
                    if n.lower() == lname:
                        entry = pool.get(n)
                        break
            if entry is not None:
                selected.append(entry)
            else:
                logger.info("tool_search select: name not found in deferred pool: %s", name)
    else:
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for name, entry in pool.items():
            s = _score(name, entry["summary"], opt, must)
            if s >= 0:
                ranked.append((s, entry))
        ranked.sort(key=lambda x: x[0], reverse=True)
        max_n = max(1, min(int(max_results or 5), 20))
        selected = [e for _, e in ranked[:max_n]]

    block = _format_functions_block(selected)
    if selected:
        _promote_to_session(selected)
        names = ", ".join(e["name"] for e in selected)
        logger.info("tool_search: promoted %d tool(s): %s", len(selected), names)
    return block


def _check_tool_search() -> bool:
    """tool_search is available iff the deferred pool is non-empty."""
    return len(get_pool()) > 0


registry.register(
    name="tool_search",
    toolset="hermes-cli",
    schema=TOOL_SEARCH_SCHEMA,
    handler=_handle_tool_search,
    check_fn=_check_tool_search,
    is_async=False,
    description=TOOL_SEARCH_SCHEMA["description"],
    emoji="🔍",
)
