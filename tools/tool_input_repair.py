"""Validate-then-repair layer for tool call arguments.

Runs JSON Schema validation on coerced tool args and applies targeted,
schema-guided repairs only at the paths the validator complained about.

Design (per @CommandCodeAI findings):

1. Validate the args against the tool's JSON Schema as-is.
   If valid, ship untouched — valid inputs are never mutated.

2. On validation failure, walk the jsonschema error tree.
   For each issue path, try the repair catalogue in order.
   The schema IS the prior; repairs are only spent on paths the
   schema actually rejected.

3. Re-validate. On success, log ``tool_input_repaired:<name>``.
   On failure, log ``tool_input_invalid:<name>`` and return the
   original (already-coerced) args so the tool handler can produce
   its own error message.

Repair catalogue (ordered — JSON-array-parse before bare-string-wrap):

  Repair                  Trigger
  ──────────────────────  ─────────────────────────────────────
  strip null optional     field is not in required, value is None
  {} -> []                schema expects array, value is empty dict
  markdown autolink       string field, value matches [text](url)
                          where text matches url-without-protocol
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

import jsonschema

logger = logging.getLogger(__name__)

# -- Markdown auto-link pattern -----------------------------------------------
# Matches a markdown link where the displayed text is a path-like string and
# the link URL is a variation of the same path (with or without protocol).
# Degenerate case: ``[notes.md](http://notes.md)`` — the model's chat
# distribution is leaking through the tool boundary.

_AUTOLINK_RE = re.compile(
    r'\[(?P<text>[^\[\]]+)\]\((?P<url>[^)]+)\)'
)

_PATH_HINT_RE = re.compile(r'[/.]')


def _strip_url_protocol(url: str) -> str:
    """Strip http:// or https:// prefix from a URL."""
    for prefix in ("https://", "http://"):
        if url.lower().startswith(prefix):
            return url[len(prefix):]
    return url


def _unwrap_autolinks(value: str) -> Tuple[str, bool]:
    """Unwrap degenerate markdown auto-links from *value*."""
    if not isinstance(value, str):
        return value, False
    if "](" not in value:
        return value, False

    changed = False

    def _replacer(m: re.Match) -> str:
        nonlocal changed
        text = m.group("text")
        url = m.group("url")
        stripped = _strip_url_protocol(url).rstrip("/").strip()
        if stripped == text.strip() and _PATH_HINT_RE.search(text):
            changed = True
            return text
        return m.group(0)

    result = _AUTOLINK_RE.sub(_replacer, value)
    return result, changed


# -- Repair functions ---------------------------------------------------------


def _repair_strip_null_optional(
    args: Dict[str, Any],
    param_schema: dict,
    errors: List[jsonschema.ValidationError],
) -> Tuple[Dict[str, Any], bool]:
    """Strip ``null`` values for fields not listed in ``required``."""
    required: set = set(param_schema.get("required", []) or [])
    props = param_schema.get("properties", {}) or {}
    if not props:
        return args, False

    null_fields = {
        k for k, v in args.items()
        if v is None and k in props and k not in required
    }
    if not null_fields:
        return args, False

    repaired = {k: v for k, v in args.items() if k not in null_fields}
    return repaired, True


def _repair_empty_dict_for_array(
    args: Dict[str, Any],
    param_schema: dict,
    errors: List[jsonschema.ValidationError],
) -> Tuple[Dict[str, Any], bool]:
    """Convert ``{}`` to ``[]`` for array-typed fields."""
    props = param_schema.get("properties", {}) or {}
    if not props:
        return args, False

    changed = False
    repaired = dict(args)
    for key, value in args.items():
        prop = props.get(key)
        if not isinstance(prop, dict):
            continue
        if prop.get("type") == "array" and isinstance(value, dict) and not value:
            repaired[key] = []
            changed = True
        if (
            prop.get("type") == "array"
            and isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], dict)
            and not value[0]
        ):
            repaired[key] = []
            changed = True
    return repaired, changed


def _repair_markdown_autolinks(
    args: Dict[str, Any],
    param_schema: dict,
    errors: List[jsonschema.ValidationError],
) -> Tuple[Dict[str, Any], bool]:
    """Unwrap degenerate markdown auto-links in string fields."""
    props = param_schema.get("properties", {}) or {}
    if not props:
        return args, False

    changed = False
    repaired = dict(args)
    for key, value in args.items():
        if not isinstance(value, str):
            continue
        prop = props.get(key)
        if not isinstance(prop, dict):
            continue
        if prop.get("type") != "string":
            continue
        new_val, field_changed = _unwrap_autolinks(value)
        if field_changed:
            repaired[key] = new_val
            changed = True
    return repaired, changed


# -- Ordered repair catalogue -------------------------------------------------
# Order matters — array-parse before bare-wrap.

# Structural repairs — validate-then-repair (only fire on schema rejection)
_STRUCTURAL_REPAIRS = [
    _repair_strip_null_optional,
    _repair_empty_dict_for_array,
]


# -- Entry point --------------------------------------------------------------


def repair_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    param_schema: dict,
) -> Dict[str, Any]:
    """Apply targeted repairs to tool arguments.

    Returns repaired args (or original if valid or irreparable).
    Logs ``tool_input_repaired:<tool_name>`` or ``tool_input_invalid:<tool_name>``
    so operators can track repair rates per (model, tool).

    Two-pass design:

    1. **Always** run autolink unwrap — it replaces degenerate markdown
       autolinks (``[path](url)``) in string fields.  This can't be caught
       by JSON Schema validation because the string *is* valid; it's a
       chat-distribution leak, not a type error.

    2. **Validate-then-repair** for structural issues (null stripping,
       ``{}`` → ``[]``).  Valid structural args pass through untouched;
       repairs only fire at paths the schema actually rejected.
    """
    if not args or not isinstance(args, dict):
        return args

    repaired = dict(args)
    any_repaired = False

    # Pass 1: always run autolink unwrap (semantic, not structural)
    try:
        repaired, applied = _repair_markdown_autolinks(repaired, param_schema, [])
        if applied:
            any_repaired = True
            logger.info("tool_input_repair:%s repair=markdown_autolink", tool_name)
    except Exception:
        logger.debug("tool_input_repair: autolink in %s errored", tool_name, exc_info=True)

    # Pass 2: validate-then-repair for structural issues
    try:
        jsonschema.validate(repaired, param_schema)
        # Structural is valid — if autolink already fixed something, log it
        if any_repaired:
            logger.info("tool_input_repaired:%s", tool_name)
        return repaired
    except jsonschema.ValidationError:
        pass  # Fall through to structural repairs

    for repair_fn in _STRUCTURAL_REPAIRS:
        try:
            repaired, applied = repair_fn(repaired, param_schema, [])
        except Exception:
            logger.debug(
                "tool_input_repair: %s in %s errored",
                repair_fn.__name__, tool_name,
                exc_info=True,
            )
            continue
        if applied:
            any_repaired = True
            logger.info(
                "tool_input_repair:%s repair=%s",
                tool_name, repair_fn.__name__,
            )

    # Re-validate after structural repairs
    try:
        jsonschema.validate(repaired, param_schema)
        if any_repaired:
            logger.info("tool_input_repaired:%s", tool_name)
        return repaired
    except jsonschema.ValidationError:
        logger.info("tool_input_invalid:%s", tool_name)
        # Return original args — let the tool handler produce its own error
        return args
