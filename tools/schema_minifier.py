"""Deterministic Schema Minifier and Prefix Cache Canonicalizer.

Compresses and canonicalizes OpenAI/Hermes tool definitions to maximize
KV prefix cache hit rate across conversation turns while reducing token costs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Optional


def minify_and_sort_schema_node(
    node: Any,
    *,
    prune_meta: bool = True,
    _depth: int = 0,
    _visited: Optional[Set[int]] = None,
) -> Any:
    """Recursively prune redundant metadata and sort dictionary keys deterministically."""
    if _depth > 30:
        return node

    if isinstance(node, dict):
        node_id = id(node)
        if _visited is None:
            _visited = set()
        if node_id in _visited:
            return node
        _visited.add(node_id)

        try:
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
                    # Note: additionalProperties: false is preserved as required for Structured Outputs / strict mode
                cleaned[k] = minify_and_sort_schema_node(
                    v, prune_meta=prune_meta, _depth=_depth + 1, _visited=_visited
                )

            # 2. Canonical key sort for 100% stable serialization / prefix cache reuse
            return {k: cleaned[k] for k in sorted(cleaned.keys())}
        finally:
            _visited.remove(node_id)

    elif isinstance(node, list):
        node_id = id(node)
        if _visited is None:
            _visited = set()
        if node_id in _visited:
            return node
        _visited.add(node_id)

        try:
            return [
                minify_and_sort_schema_node(
                    item, prune_meta=prune_meta, _depth=_depth + 1, _visited=_visited
                )
                for item in node
            ]
        finally:
            _visited.remove(node_id)

    return node


def minify_tool_definition(tool_def: Dict[str, Any], *, prune_meta: bool = True) -> Dict[str, Any]:
    """Lossless minification and key sorting for a single tool definition."""
    if not isinstance(tool_def, dict):
        return tool_def

    # Recursive rebuild without deepcopy
    out = {}
    for k, v in tool_def.items():
        if k == "function" and isinstance(v, dict):
            fn_dict = {}
            for fn_k, fn_v in v.items():
                if fn_k == "parameters" and isinstance(fn_v, dict):
                    fn_dict["parameters"] = minify_and_sort_schema_node(fn_v, prune_meta=prune_meta)
                else:
                    fn_dict[fn_k] = minify_and_sort_schema_node(fn_v, prune_meta=prune_meta)
            out["function"] = {fn_k: fn_dict[fn_k] for fn_k in sorted(fn_dict.keys())}
        else:
            out[k] = minify_and_sort_schema_node(v, prune_meta=prune_meta)

    return {k: out[k] for k in sorted(out.keys())}


def minify_and_canonicalize_tools(tools: List[Dict[str, Any]], *, prune_meta: bool = True) -> List[Dict[str, Any]]:
    """Minify all tool schemas while preserving tool ordering to avoid behavioral shifts."""
    if not tools:
        return tools

    return [minify_tool_definition(t, prune_meta=prune_meta) for t in tools]
