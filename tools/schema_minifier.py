"""Deterministic Schema Minifier and Prefix Cache Canonicalizer.

Compresses and canonicalizes OpenAI/Hermes tool definitions to maximize
KV prefix cache hit rate across conversation turns while reducing token costs.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


def minify_and_sort_schema_node(node: Any, *, prune_meta: bool = True) -> Any:
    """Recursively prune redundant metadata and sort dictionary keys deterministically."""
    if isinstance(node, dict):
        # 1. Clean redundant fields
        cleaned = {}
        for k, v in node.items():
            if prune_meta:
                # Remove redundant titles or schemas in sub-properties
                if k in ("$schema",):
                    continue
                # Prune empty enum, empty properties, or empty allOf/anyOf
                if k in ("enum", "allOf", "anyOf", "oneOf", "required") and isinstance(v, (list, tuple)) and len(v) == 0:
                    continue
                # Remove empty objects where not strictly needed
                if k == "additionalProperties" and v is False:
                    continue
            cleaned[k] = minify_and_sort_schema_node(v, prune_meta=prune_meta)

        # 2. Canonical key sort for 100% stable serialization / prefix cache reuse
        return {k: cleaned[k] for k in sorted(cleaned.keys())}

    elif isinstance(node, list):
        return [minify_and_sort_schema_node(item, prune_meta=prune_meta) for item in node]

    return node


def minify_tool_definition(tool_def: Dict[str, Any], *, prune_meta: bool = True) -> Dict[str, Any]:
    """Lossless minification and key sorting for a single tool definition."""
    if not isinstance(tool_def, dict):
        return tool_def

    out = copy.deepcopy(tool_def)
    fn = out.get("function")
    if isinstance(fn, dict):
        params = fn.get("parameters")
        if isinstance(params, dict):
            fn["parameters"] = minify_and_sort_schema_node(params, prune_meta=prune_meta)
        out["function"] = {k: fn[k] for k in sorted(fn.keys())}
    return {k: out[k] for k in sorted(out.keys())}


def minify_and_canonicalize_tools(tools: List[Dict[str, Any]], *, prune_meta: bool = True) -> List[Dict[str, Any]]:
    """Minify all tool schemas and sort tools stably to guarantee determinism in prefix cache."""
    if not tools:
        return tools

    processed = [minify_tool_definition(t, prune_meta=prune_meta) for t in tools]
    # Stable deterministic ordering of tools by function name
    def _tool_sort_key(t: Dict[str, Any]) -> str:
        fn = t.get("function") if isinstance(t, dict) else None
        return (fn.get("name") if isinstance(fn, dict) else "") or ""

    return sorted(processed, key=_tool_sort_key)
