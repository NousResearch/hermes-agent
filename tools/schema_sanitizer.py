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
* ``anyOf`` / ``oneOf`` unions whose only purpose is to permit ``null`` for
  optional fields (common Pydantic/MCP shape). Anthropic rejects these at
  the top of ``input_schema``; collapse them to the non-null branch.
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
    # Final pass: collapse nullable anyOf/oneOf unions that the recursive
    # sanitizer above leaves intact (it only handles the array-form
    # ``type: [X, "null"]``). Keep the ``nullable: true`` hint so runtime
    # argument coercion (``model_tools._schema_allows_null``) can still
    # map a model-emitted ``"null"`` string to Python ``None``.
    fn["parameters"] = strip_nullable_unions(fn["parameters"], keep_nullable_hint=True)
    # Strip top-level combinators that strict backends (OpenAI's Codex
    # endpoint at chatgpt.com/backend-api/codex) reject outright. Nested
    # combinators inside properties are preserved.
    fn["parameters"] = _strip_top_level_combinators(
        fn["parameters"], path=fn.get("name", "<tool>")
    )
    return out


_TOP_LEVEL_FORBIDDEN_KEYS = ("allOf", "anyOf", "oneOf", "enum", "not")


def _strip_top_level_combinators(params: dict, *, path: str = "<tool>") -> dict:
    """Drop combinator keywords from the top-level of a function parameters schema.

    OpenAI's Codex backend (``chatgpt.com/backend-api/codex``) is stricter
    than the public Functions API and rejects requests with::

        Invalid schema for function 'X': schema must have type 'object' and
        not have 'oneOf'/'anyOf'/'allOf'/'enum'/'not' at the top level.

    These keywords are typically used for conditional required-fields hints
    (``allOf: [{if: ..., then: {required: [...]}}]``). Removing them at the
    top level discards the hint but does not change which argument *values*
    are valid — the tool handler always re-validates required fields.

    Only the *top* level is stripped; combinators nested inside a property's
    schema are preserved (the strict rule only applies to the outermost
    parameters object).
    """
    if not isinstance(params, dict):
        return params
    out = dict(params)
    for key in _TOP_LEVEL_FORBIDDEN_KEYS:
        if key in out:
            logger.debug(
                "schema_sanitizer[%s]: stripped top-level %r combinator "
                "from tool parameters (strict-backend compat)",
                path, key,
            )
            out.pop(key, None)
    return out


def strip_nullable_unions(
    schema: Any,
    *,
    keep_nullable_hint: bool = True,
) -> Any:
    """Collapse ``anyOf`` / ``oneOf`` nullable unions to the non-null branch.

    MCP / Pydantic optional fields commonly arrive as::

        {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}

    Anthropic's tool input-schema validator rejects the null branch. Tool
    optionality is already represented by the parent object's ``required``
    array, so we collapse the union to the single non-null variant.

    Metadata (``title``, ``description``, ``default``, ``examples``) on the
    outer union node is carried over to the replacement variant.

    Args:
        schema: JSON-Schema fragment (dict, list, or scalar).
        keep_nullable_hint: If True, set ``nullable: true`` on the replacement
            to preserve the "this field may be None" signal for downstream
            consumers that care (e.g. runtime argument coercion that maps the
            literal string ``"null"`` to Python ``None``). Anthropic's
            validator accepts ``nullable: true`` but strict producers may
            prefer False.

    Returns:
        The schema with nullable unions collapsed. Non-union nodes are
        returned unchanged.
    """
    if isinstance(schema, list):
        return [strip_nullable_unions(item, keep_nullable_hint=keep_nullable_hint) for item in schema]
    if not isinstance(schema, dict):
        return schema

    stripped = {
        k: strip_nullable_unions(v, keep_nullable_hint=keep_nullable_hint)
        for k, v in schema.items()
    }
    for key in ("anyOf", "oneOf"):
        variants = stripped.get(key)
        if not isinstance(variants, list):
            continue
        non_null = [
            item for item in variants
            if not (isinstance(item, dict) and item.get("type") == "null")
        ]
        # Only collapse when we actually dropped a null branch AND exactly
        # one non-null branch survives (otherwise the union is meaningful
        # and we leave it alone).
        if len(non_null) == 1 and len(non_null) != len(variants):
            replacement = dict(non_null[0]) if isinstance(non_null[0], dict) else {}
            if keep_nullable_hint:
                replacement.setdefault("nullable", True)
            for meta_key in ("title", "description", "default", "examples"):
                if meta_key in stripped and meta_key not in replacement:
                    replacement[meta_key] = stripped[meta_key]
            return strip_nullable_unions(replacement, keep_nullable_hint=keep_nullable_hint)
    return stripped


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


# =============================================================================
# Reactive strip — only invoked when llama.cpp rejects a schema
# =============================================================================

_STRIP_ON_RECOVERY_KEYS = frozenset({"pattern", "format"})


def strip_pattern_and_format(tools: list[dict]) -> tuple[list[dict], int]:
    """Strip ``pattern`` and ``format`` JSON Schema keywords from tool schemas.

    This is a *reactive* sanitizer invoked only when llama.cpp's
    ``json-schema-to-grammar`` converter has rejected a tool schema with an
    HTTP 400 grammar-parse error.  llama.cpp's regex engine supports only a
    small subset of ECMAScript regex (literals, ``.``, ``[...]``, ``|``,
    ``*``, ``+``, ``?``, ``{n,m}``) — it rejects escape classes like ``\\d``,
    ``\\w``, ``\\s`` and most ``format`` values.  Cloud providers (OpenAI,
    Anthropic, OpenRouter, Gemini) accept these keywords fine and rely on
    them as prompting hints, so we keep them in the default schema and only
    strip on demand.

    The strip operates on a sibling of ``type`` (so schema keywords are
    removed) — a property literally *named* ``pattern`` (e.g. the first arg
    of the built-in ``search_files`` tool) is not affected because property
    names live in the ``properties`` dict, not as siblings of ``type``.

    Args:
        tools: OpenAI-format tool list, mutated in place for efficiency.
            Callers that need to preserve the original should deep-copy first.

    Returns:
        ``(tools, stripped_count)`` — the same list reference plus a count of
        how many ``pattern``/``format`` keywords were removed across all tools.
    """
    if not tools:
        return tools, 0

    stripped = 0

    def _walk(node: Any) -> None:
        nonlocal stripped
        if isinstance(node, dict):
            # Only strip as a sibling of ``type`` — i.e. when this node is
            # itself a schema.  This avoids stripping literal property keys
            # named "pattern" (search_files.pattern, etc.) because those live
            # inside a ``properties`` dict, not as siblings of ``type``.
            is_schema_node = "type" in node or "anyOf" in node or "oneOf" in node or "allOf" in node
            for key in list(node.keys()):
                if is_schema_node and key in _STRIP_ON_RECOVERY_KEYS:
                    node.pop(key, None)
                    stripped += 1
                    continue
                _walk(node[key])
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict):
            params = fn.get("parameters")
            if isinstance(params, dict):
                _walk(params)

    if stripped:
        logger.info(
            "schema_sanitizer: stripped %d pattern/format keyword(s) from "
            "tool schemas (llama.cpp grammar-parse recovery)",
            stripped,
        )
    return tools, stripped
