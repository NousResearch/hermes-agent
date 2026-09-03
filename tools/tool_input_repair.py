"""Shared repair helpers for model-emitted tool arguments.

LLMs occasionally emit an array-typed tool parameter as a *string* instead
of a real list: either a JSON string ("[\"a\", \"b\"]") or a bare string
("a") when the schema asked for ["a"]. Without a repair, each tool handler
either rejects the input (a wasted round-trip the model must recover from)
or, worse, silently misbehaves: a ``for x in urls`` loop over a bare string
iterates character-by-character and returns a misleading "content not found"
error instead of the real type mismatch.

Historically this repair was duplicated ad hoc across individual tools
(e.g. ``todo_tool`` and ``delegate_tool``). This module centralizes the
common case so every array-taking tool gets the same, consistent repair.

The repair is deliberately *narrow*: it only fires when the value is a
string, and it only ever upgrades a string into a list. Valid inputs (real
lists, dicts, None) are returned untouched, so no legitimate payload is ever
transformed or corrupted.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple


def recover_list_from_json_string(
    value: Any,
    *,
    param_name: str = "value",
    wrap_bare_string: bool = False,
) -> Tuple[Optional[list], Optional[str]]:
    """Repair a stringified-JSON array parameter into a real list.

    Returns ``(repaired, error)``:

    * ``value`` is not a string -> ``(None, None)``. The input is left for
      the caller's normal validation; nothing is transformed.
    * ``value`` is a string that parses to a JSON array -> ``(list, None)``.
    * ``value`` is a bare string that does not parse to JSON and
      ``wrap_bare_string`` is True -> ``([value], None)``. This is the
      ``"foo"`` -> ``["foo"]`` case, valid for params whose items are
      strings (e.g. ``urls``).
    * ``value`` is a string that does *not* parse to an array (unparseable,
      or parsed to a non-list) -> ``(None, error_message)``.

    Valid non-string inputs are never touched, so this is safe to call on any
    parameter without risking corruption of well-formed payloads.
    """
    if not isinstance(value, str):
        return None, None

    raw = value.strip()
    if not raw:
        return None, f"{param_name} must be a list; received an empty string."

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        if wrap_bare_string:
            return [value], None
        return None, (
            f"{param_name} must be a JSON array; received a string that could "
            f"not be parsed as JSON ({exc.msg})."
        )

    if not isinstance(parsed, list):
        return None, (
            f"{param_name} must be a JSON array; parsed {type(parsed).__name__} "
            f"instead."
        )

    return parsed, None