#!/usr/bin/env python3
"""Tool discovery helper for deferred tool loading."""

import difflib
import re
from typing import List

from tools.registry import registry, tool_error, tool_result

TOOL_SEARCH_SCHEMA = {
    "name": "tool_search",
    "description": "Search available tools by keyword. Matches against tool name, description, toolset, and discovery hints.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword or phrase to search for across registered tools.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matches to return. Defaults to 10 and caps at 50.",
                "minimum": 1,
            },
        },
        "required": ["query"],
    },
}

_FIELD_WEIGHTS = {
    "name": 6.0,
    "search_hint": 5.0,
    "description": 3.0,
    "toolset": 2.0,
}


def _normalize(text: str) -> str:
    """Normalize text for substring and fuzzy matching."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase alphanumeric tokens."""
    return [token for token in re.split(r"[^a-z0-9_]+", _normalize(text)) if token]


def _score_field(query: str, value: str, weight: float) -> float:
    """Score a single field against the query."""
    normalized_value = _normalize(value)
    if not query or not normalized_value:
        return 0.0

    score = 0.0
    if query == normalized_value:
        score = max(score, 100.0 * weight)
    if query in normalized_value:
        score = max(score, 70.0 * weight)

    query_tokens = _tokenize(query)
    if query_tokens:
        matched_tokens = sum(1 for token in query_tokens if token in normalized_value)
        if matched_tokens:
            score += matched_tokens * 12.0 * weight

    candidates = {normalized_value, *(_tokenize(normalized_value))}
    best_ratio = max(
        difflib.SequenceMatcher(None, query, candidate).ratio()
        for candidate in candidates
        if candidate
    )
    if best_ratio >= 0.6:
        score += best_ratio * 10.0 * weight

    return score


def _search_registry_tools(query: str, limit: int = 10) -> List[dict]:
    """Return ranked registry matches for *query*."""
    normalized_query = _normalize(query)
    if not normalized_query:
        return []

    matches = []
    for name in registry.get_all_tool_names():
        schema = registry.get_schema(name) or {}
        meta = registry.get_metadata(name)
        record = {
            "name": name,
            "description": schema.get("description", ""),
            "toolset": registry.get_toolset_for_tool(name) or "",
            "search_hint": meta.get("search_hint", ""),
            "deferred": meta.get("deferred", False),
        }
        score = sum(
            _score_field(normalized_query, record[field], weight)
            for field, weight in _FIELD_WEIGHTS.items()
        )
        if score <= 0:
            continue
        matches.append((score, record))

    matches.sort(key=lambda item: (-item[0], item[1]["name"]))
    return [record for _, record in matches[:limit]]


def tool_search_handler(args: dict, **kwargs) -> str:
    """Search the tool registry by keyword."""
    query = str(args.get("query", "")).strip()
    if not query:
        return tool_error("query is required")

    raw_limit = args.get("limit", 10)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return tool_error("limit must be an integer")
    limit = max(1, min(limit, 50))

    results = _search_registry_tools(query, limit=limit)
    return tool_result(
        query=query,
        count=len(results),
        results=results,
    )


registry.register(
    name="tool_search",
    toolset="core",
    schema=TOOL_SEARCH_SCHEMA,
    handler=tool_search_handler,
    description="Search the registry for tools by keyword",
    emoji="🧭",
    allowed_in_plan_mode_default=True,
    always_load=True,
    search_hint="Find tools by capability, keyword, or deferred-tool discovery hint",
)
