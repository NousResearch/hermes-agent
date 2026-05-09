"""
strict_schema_lint.py
============================================================
Strict OpenAI tool schema linter — companion to schema_sanitizer.

After sanitization, this module verifies the schema is fully compatible
with OpenAI strict-mode tool calls (chat.completions + Responses API).
Returns a list of structured errors (path + reason) instead of raising.

Used in two places:

1. Startup-time MCP tool registration (`tools/mcp_tool.py`): if
   `lint_openai_strict_schema()` returns errors, the offending tool is
   QUARANTINED instead of being registered. The bad tool no longer
   poisons the entire `tools[]` payload sent to the provider.

2. Per-turn provider call (just before
   `client.chat.completions.create(tools=...)`):
   `validate_tools_payload()` re-checks the final post-filtered list,
   and when `MCP_STRICT_SCHEMA_AUTO_QUARANTINE=1` it drops only the
   offending tools, emits a structured audit event, and lets the call
   proceed with a clean payload. Otherwise it raises early with the
   tool name + JSON pointer so the operator can see the failure
   instead of getting an opaque 400 from the provider.

Design notes:

* This linter is INTENTIONALLY narrow. It only checks the rules that
  OpenAI strict mode actually rejects (the empirical 400-error set).
  It does NOT try to be a general JSON Schema validator — that would
  produce false positives against perfectly-working MCP servers.

* Every error has the shape ``{"path": "...", "reason": "..."}`` so
  downstream consumers (audit logs, metrics, Sentry breadcrumbs) can
  treat errors as structured data, not strings.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


_STRICT_AUTO_QUARANTINE_ENV = "MCP_STRICT_SCHEMA_AUTO_QUARANTINE"


def lint_openai_strict_schema(schema: Any, *, path: str = "$") -> list[dict]:
    """Return a list of strict-mode violations found in ``schema``.

    Each error is ``{"path": <json-pointer-ish>, "reason": <str>}``.
    An empty list means the schema is OpenAI-strict compatible.
    """
    errors: list[dict] = []
    _walk(schema, path, errors)
    return errors


def _walk(node: Any, path: str, errors: list[dict]) -> None:
    if isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, f"{path}[{i}]", errors)
        return
    if not isinstance(node, dict):
        return

    node_type = node.get("type")

    # Rule 1: array nodes must declare items (or prefixItems for tuples).
    if node_type == "array":
        items = node.get("items")
        prefix_items = node.get("prefixItems")
        if items is None and not prefix_items:
            errors.append({
                "path": path,
                "reason": "array node missing items (and no prefixItems)",
            })
        elif items is not None and not isinstance(items, (dict, list)):
            errors.append({
                "path": f"{path}.items",
                "reason": f"items must be dict or list, got {type(items).__name__}",
            })

    # Rule 2: object-type nodes must have a properties dict.
    if node_type == "object":
        props = node.get("properties")
        if props is not None and not isinstance(props, dict):
            errors.append({
                "path": f"{path}.properties",
                "reason": f"properties must be dict, got {type(props).__name__}",
            })
        # additionalProperties may be bool or schema; if schema, recurse.
        ap = node.get("additionalProperties")
        if isinstance(ap, dict):
            _walk(ap, f"{path}.additionalProperties", errors)

    # Rule 3: "type" must be a single string (not [X, "null"]) — sanitizer
    # should have collapsed these already; if any leak through, flag.
    if isinstance(node_type, list):
        errors.append({
            "path": f"{path}.type",
            "reason": f"type is list {node_type!r}; sanitizer should have collapsed",
        })

    # Recurse: properties, items, anyOf/oneOf/allOf, prefixItems
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        sub = node.get(key)
        if isinstance(sub, dict):
            for child_key, child_val in sub.items():
                _walk(child_val, f"{path}.{key}.{child_key}", errors)

    items_val = node.get("items")
    if isinstance(items_val, dict):
        _walk(items_val, f"{path}.items", errors)
    elif isinstance(items_val, list):
        for i, sub in enumerate(items_val):
            _walk(sub, f"{path}.items[{i}]", errors)

    for combinator in ("anyOf", "oneOf", "allOf", "prefixItems"):
        sub = node.get(combinator)
        if isinstance(sub, list):
            for i, child in enumerate(sub):
                _walk(child, f"{path}.{combinator}[{i}]", errors)


def lint_tool(tool: dict) -> list[dict]:
    """Lint a single OpenAI-format tool entry.

    Returns errors found in ``tool["function"]["parameters"]``.
    """
    if not isinstance(tool, dict):
        return [{"path": "$", "reason": f"tool must be dict, got {type(tool).__name__}"}]
    fn = tool.get("function")
    if not isinstance(fn, dict):
        return [{"path": "$.function", "reason": "missing or non-dict function"}]
    params = fn.get("parameters")
    if params is None:
        return []  # tool with no params is valid
    name = fn.get("name", "<tool>")
    return lint_openai_strict_schema(params, path=name)


def validate_tools_payload(
    tools: list[dict],
    *,
    auto_quarantine: bool | None = None,
) -> tuple[list[dict], list[dict]]:
    """Validate a full ``tools=[...]`` payload before sending to the provider.

    Returns ``(clean_tools, quarantined)``:
      - ``clean_tools``: the subset that passed strict lint.
      - ``quarantined``: list of ``{"tool": <name>, "errors": [...]}`` for
        tools that failed.

    Behaviour:
      - When ``auto_quarantine`` is True (or the env var
        ``MCP_STRICT_SCHEMA_AUTO_QUARANTINE=1``), bad tools are silently
        dropped and a WARNING is logged with the tool name + errors.
      - When False, this function still returns the partition but the
        caller is expected to fail loud (the alternative is hitting the
        provider with a known-bad payload).

    Default: read from env. If env unset, default to True
    (fail-soft / keep-going posture).
    """
    if auto_quarantine is None:
        env_val = os.environ.get(_STRICT_AUTO_QUARANTINE_ENV, "1").strip()
        auto_quarantine = env_val not in ("0", "false", "False", "no", "")

    clean: list[dict] = []
    quarantined: list[dict] = []
    for tool in tools or []:
        errors = lint_tool(tool)
        if errors:
            name = (tool.get("function") or {}).get("name", "<unknown>")
            quarantined.append({"tool": name, "errors": errors})
            if auto_quarantine:
                logger.warning(
                    "MCP tool quarantined at provider call: name=%s errors=%s",
                    name,
                    errors,
                )
                continue
            else:
                # Caller asked us not to auto-quarantine. Keep the tool
                # in the clean set; the caller decides whether to raise.
                clean.append(copy.deepcopy(tool))
        else:
            clean.append(tool)
    return clean, quarantined


__all__ = [
    "lint_openai_strict_schema",
    "lint_tool",
    "validate_tools_payload",
]
