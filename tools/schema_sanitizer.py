"""Sanitize tool JSON schemas for broad LLM-backend compatibility.

Some local inference backends (notably llama.cpp's ``json-schema-to-grammar``
converter used to build GBNF tool-call parsers) are strict about what JSON
Schema shapes they accept. Schemas that OpenAI / Anthropic / most cloud
providers silently accept can make llama.cpp fail the entire request with:

    HTTP 400: Unable to generate parser for this template.
    Automatic parser generation failed: JSON schema conversion failed:
    Unrecognized schema: "object"

The failure modes we've seen in the wild:

* ``{"type": "object"}`` with no ``properties`` — rejected as a node the
  grammar generator can't constrain.
* A schema value that is the bare string ``"object"`` instead of a dict
  (malformed MCP server output, e.g. ``additionalProperties: "object"``).
* ``"type": ["string", "null"]`` array types — many converters only accept
  single-string ``type``.
* Unconstrained ``additionalProperties`` on objects with empty properties.

This module walks the final tool schema tree (after MCP-level normalization
and any per-tool dynamic rebuilds) and fixes the known-hostile constructs
in-place on a deep copy. It is intentionally conservative: it only modifies
shapes the LLM backend couldn't use anyway.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def sanitize_tool_schemas(tools: list[dict]) -> list[dict]:
    """Return a copy of ``tools`` with each tool's parameter schema sanitized.

    Input is an OpenAI-format tool list:
    ``[{"type": "function", "function": {"name": ..., "parameters": {...}}}]``

    The returned list is a deep copy — callers can safely mutate it without
    affecting the original registry entries.
    """
    if not tools:
        return tools

    sanitized: list[dict] = []
    for tool in tools:
        sanitized.append(_sanitize_single_tool(tool))
    return sanitized


def _sanitize_single_tool(tool: dict) -> dict:
    """Deep-copy and sanitize a single OpenAI-format tool entry."""
    out = copy.deepcopy(tool)
    fn = out.get("function") if isinstance(out, dict) else None
    if not isinstance(fn, dict):
        return out

    params = fn.get("parameters")
    # Missing / non-dict parameters → substitute the minimal valid shape.
    if not isinstance(params, dict):
        fn["parameters"] = {"type": "object", "properties": {}}
        return out

    fn["parameters"] = _sanitize_node(params, path=fn.get("name", "<tool>"))
    # After recursion, guarantee the top-level is an object with properties.
    top = fn["parameters"]
    if not isinstance(top, dict):
        fn["parameters"] = {"type": "object", "properties": {}}
    else:
        if top.get("type") != "object":
            top["type"] = "object"
        if "properties" not in top or not isinstance(top.get("properties"), dict):
            top["properties"] = {}
    return out


def _sanitize_node(node: Any, path: str) -> Any:
    """Recursively sanitize a JSON-Schema fragment.

    - Replaces bare-string schema values ("object", "string", ...) with
      ``{"type": <value>}`` so downstream consumers see a dict.
    - Injects ``properties: {}`` into object-typed nodes missing it.
    - Normalizes ``type: [X, "null"]`` arrays to single ``type: X`` (keeping
      ``nullable: true`` as a hint).
    - Recurses into ``properties``, ``items``, ``additionalProperties``,
      ``anyOf``, ``oneOf``, ``allOf``, and ``$defs`` / ``definitions``.
    """
    # Malformed: the schema position holds a bare string like "object".
    if isinstance(node, str):
        if node in {"object", "string", "number", "integer", "boolean", "array", "null"}:
            logger.debug(
                "schema_sanitizer[%s]: replacing bare-string schema %r "
                "with {'type': %r}",
                path, node, node,
            )
            return {"type": node} if node != "object" else {
                "type": "object",
                "properties": {},
            }
        # Any other stray string is not a schema — drop it by replacing with
        # a permissive object schema rather than propagate something the
        # backend will reject.
        logger.debug(
            "schema_sanitizer[%s]: replacing non-schema string %r "
            "with empty object schema", path, node,
        )
        return {"type": "object", "properties": {}}

    if isinstance(node, list):
        return [_sanitize_node(item, f"{path}[{i}]") for i, item in enumerate(node)]

    if not isinstance(node, dict):
        return node

    out: dict = {}
    for key, value in node.items():
        # type: [X, "null"] → type: X (the backend's tool-call parser only
        # accepts singular string types; nullable is lost but the call still
        # succeeds, and the model can still pass null on its own.)
        if key == "type" and isinstance(value, list):
            non_null = [t for t in value if t != "null"]
            if len(non_null) == 1 and isinstance(non_null[0], str):
                out["type"] = non_null[0]
                if "null" in value:
                    out.setdefault("nullable", True)
                continue
            # Fallback: pick the first string type, drop the rest.
            first_str = next((t for t in value if isinstance(t, str) and t != "null"), None)
            if first_str:
                out["type"] = first_str
                continue
            # All-null or empty list → treat as object.
            out["type"] = "object"
            continue

        if key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            out[key] = {
                sub_k: _sanitize_node(sub_v, f"{path}.{key}.{sub_k}")
                for sub_k, sub_v in value.items()
            }
        elif key in {"items", "additionalProperties"}:
            if isinstance(value, bool):
                # Keep bool ``additionalProperties`` as-is — it's a valid form
                # and widely accepted. ``items: true/false`` is non-standard
                # but we preserve rather than drop.
                out[key] = value
            else:
                out[key] = _sanitize_node(value, f"{path}.{key}")
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(value, list):
            out[key] = [
                _sanitize_node(item, f"{path}.{key}[{i}]")
                for i, item in enumerate(value)
            ]
        elif key in {"required", "enum", "examples"}:
            # Schema "sibling" keywords whose values are NOT schemas:
            #  - ``required``: list of property-name strings
            #  - ``enum``: list of literal values (any JSON type)
            #  - ``examples``: list of example values (any JSON type)
            # Recursing into these with _sanitize_node() would mis-interpret
            # literal strings like "path" as bare-string schemas and replace
            # them with {"type": "object"} dicts. Pass through unchanged.
            out[key] = copy.deepcopy(value) if isinstance(value, (list, dict)) else value
        else:
            out[key] = _sanitize_node(value, f"{path}.{key}") if isinstance(value, (dict, list)) else value

    # Object nodes without properties: inject empty properties dict.
    # llama.cpp's grammar generator can't constrain a free-form object.
    if out.get("type") == "object" and not isinstance(out.get("properties"), dict):
        out["properties"] = {}

    # Prune ``required`` entries that don't exist in properties (defense
    # against malformed MCP schemas; also caught upstream for MCP tools, but
    # built-in tools or plugin tools may not have been through that path).
    if out.get("type") == "object" and isinstance(out.get("required"), list):
        props = out.get("properties") or {}
        valid = [r for r in out["required"] if isinstance(r, str) and r in props]
        if not valid:
            out.pop("required", None)
        elif len(valid) != len(out["required"]):
            out["required"] = valid

    return out
