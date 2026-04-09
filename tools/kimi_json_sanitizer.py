"""Repair malformed tool-call JSON.

The immediate trigger here was Kimi returning truncated tool arguments during
chat-completions streaming, but the repair path is generic and only runs after
normal ``json.loads()`` has already failed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Historical context: this helper started as a Kimi-specific fix. Keep the
# model detector for callers and logging, even though the repair itself is now
# safe to try for any malformed tool JSON.
KIMI_MODELS = {
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k1.5-instruct",
    "kimi-k2-instruct",
    "kimi-k1.5-instruct",
}


def is_kimi_model(model: str) -> bool:
    """Check if the model is a Kimi variant."""
    if not model:
        return False
    model_lower = model.lower()
    return any(k.lower() in model_lower for k in KIMI_MODELS)


def sanitize_kimi_json(raw_args: Any, *, tool_name: str = "") -> tuple[Any | None, str | None]:
    """Backward-compatible wrapper around the generic repair helper."""
    return repair_tool_call_arguments(raw_args, tool_name=tool_name)


def repair_tool_call_arguments(raw_args: Any, *, tool_name: str = "") -> tuple[Any | None, str | None]:
    """Attempt to repair malformed tool-call arguments.

    Returns ``(parsed_args, None)`` on success or ``(None, error)`` when the
    payload is too damaged to recover.
    """
    if isinstance(raw_args, (dict, list)):
        return raw_args, None
    if raw_args is None:
        return {}, None

    text = raw_args if isinstance(raw_args, str) else str(raw_args)
    text = text.strip()
    if not text:
        return {}, None

    last_error = None
    for candidate in _repair_candidates(text):
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            last_error = str(exc)

    extracted = _regex_extract_params(text)
    if extracted is not None:
        if tool_name:
            logger.warning(
                "Recovered malformed tool JSON via regex extraction for %s",
                tool_name,
            )
        else:
            logger.warning("Recovered malformed tool JSON via regex extraction")
        return extracted, None

    return None, last_error or "Invalid JSON tool arguments"


def _repair_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    cleaned = text.lstrip("\ufeff").strip()
    add(cleaned)

    try:
        add(json.dumps(json.loads(cleaned, strict=False), ensure_ascii=False))
    except Exception:
        pass

    closed = _close_unterminated_string(cleaned)
    balanced = _balance_brackets(closed)
    add(balanced)
    add(_balance_brackets(_drop_trailing_partial_member(closed)))
    add(_balance_brackets(_drop_trailing_partial_member(cleaned)))

    return candidates


def _close_unterminated_string(text: str) -> str:
    quote_count = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quote_count += 1
    if quote_count % 2:
        return text + '"'
    return text


def _balance_brackets(text: str) -> str:
    opens = []
    in_string = False
    escaped = False

    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            opens.append(char)
        elif char in "}]":
            if opens:
                opens.pop()

    suffix = []
    for opener in reversed(opens):
        suffix.append("}" if opener == "{" else "]")
    return text + "".join(suffix)


def _drop_trailing_partial_member(text: str) -> str:
    candidate = text.rstrip()
    patterns = [
        r',\s*"[^"]+"\s*:\s*"[^"]*$',
        r',\s*"[^"]+"\s*:\s*[^,}\]]*$',
        r',\s*"[^"]+"\s*:\s*$',
        r',\s*"[^"]+\s*$',
        r',\s*"[^"]+"\s*$',
        r',\s*$',
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "", candidate)
        if updated != candidate:
            return updated.rstrip()
    return candidate


def _regex_extract_params(broken_json: str) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    for match in re.finditer(r'"([^"]+)":\s*"([^"]*)"', broken_json):
        result[match.group(1)] = match.group(2)

    for match in re.finditer(r'"([^"]+)":\s*(-?\d+(?:\.\d+)?)', broken_json):
        raw_value = match.group(2)
        result[match.group(1)] = float(raw_value) if "." in raw_value else int(raw_value)

    for match in re.finditer(r'"([^"]+)":\s*(true|false)', broken_json):
        result[match.group(1)] = match.group(2) == "true"

    return result or None
