"""Dispatch-time argument checks for tools whose silent argument drops mislead the model.

``coerce_tool_args`` repairs types but ignores keys outside the schema and never checks
``required``; handlers then default missing values (``pattern=""`` matches every line). The
registry schema is not always what a handler accepts (hidden keys, dynamic overrides), so
checking is opt-in per tool in ``_RULES`` rather than applied to every tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from tools.registry import registry, tool_error


@dataclass(frozen=True)
class _Rule:
    # Keys the handler reads that the schema does not declare; never reported as unknown.
    extra_keys: frozenset = frozenset()
    # Unknown keys reject the call (destructive tools) instead of warning after it ran.
    reject_unknown: bool = False


_RULES: Dict[str, _Rule] = {
    "search_files": _Rule(),
    "web_search": _Rule(),
    "write_file": _Rule(extra_keys=frozenset({"cross_profile"}), reject_unknown=True),
    "execute_code": _Rule(),
}


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """``(error_json, warning)`` for a call about to be dispatched. ``error_json`` means do NOT
    run the tool; ``warning`` is advisory text for a call that runs with some keys ignored."""
    rule = _RULES.get(tool_name)
    schema = registry.get_schema(tool_name) if rule else None
    params = (schema or {}).get("parameters") or {}
    properties = params.get("properties")
    if not rule or not isinstance(args, dict) or not properties:
        return None, None

    valid = ", ".join(properties)
    missing = [
        key for key in params.get("required") or []
        if args.get(key) is None]
    if missing:
        return tool_error(
            f"{tool_name} is missing required argument(s): {', '.join(missing)}. "
            f"The tool was NOT invoked. Valid arguments: {valid}."), None

    unknown = [key for key in args if key not in properties and key not in rule.extra_keys]
    if not unknown:
        return None, None
    if rule.reject_unknown:
        return tool_error(
            f"{tool_name} does not accept argument(s): {', '.join(unknown)}. "
            f"The tool was NOT invoked and nothing was changed. Valid arguments: {valid}."), None
    return None, (f"Ignored unknown argument(s): {', '.join(unknown)}. The tool ran WITHOUT them, so the result "
                  f"may not be what you intended. Valid arguments: {valid}. See the tool description for what "
                  f"each argument does, and retry with the correct ones if you still need that behavior.")


def attach_arg_warning(result: Any, warning: Optional[str]) -> Any:
    """Put ``warning`` first in a JSON-object tool result; anything else passes through.
    A handler's own ``warning`` (e.g. execute_code truncation) is kept after ours."""
    if not warning or not isinstance(result, str):
        return result
    try:
        data = json.loads(result)
    except ValueError:
        return result
    if not isinstance(data, dict):
        return result
    existing = data.pop("warning", None)
    first = f"{warning}\n{existing}" if existing else warning
    return json.dumps({"warning": first, **data}, ensure_ascii=False)
