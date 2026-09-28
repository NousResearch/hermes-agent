"""Opt-in JSON whitespace compaction at the tool-result append boundary.

Only insignificant whitespace is removed: string literals, number spellings,
key order and even duplicate keys survive byte-for-byte. No parse/reserialize
round trip, auxiliary model, recovery call or rewrite of past messages is needed.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
import json
import logging
import re

logger = logging.getLogger(__name__)

# Bound work on untrusted results (including tools exempt from spillover).
_MAX_INPUT_CHARS = 1_000_000
_JSON_STRING_OR_SPACE = re.compile(r'"(?:[^"\\]|\\.)*"|[ \t\r\n]+')


def _reject_constant(value: str):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _compact_json(content: str) -> str:
    """Validate first, then remove only whitespace outside JSON string literals."""
    if not content.lstrip(" \t\r\n").startswith(("{", "[")):
        return content
    try:
        # Validation must not round floats, normalize escapes or hit Python's
        # integer-digit conversion limit. The parsed object is never serialized.
        json.loads(content, parse_int=lambda _: None, parse_float=lambda _: None,
                   parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return content
    return _JSON_STRING_OR_SPACE.sub(
        lambda match: match[0] if match[0].startswith('"') else "", content,
    )


def compact_tool_result(content, tool_name: str):
    """Apply this profile's policy to a new text result; off/malformed = unchanged.

    Called after per-result spillover, before hints/wrapping/append. Spill files
    and UI results retain their original formatting. Multimodal results bypass it.
    ``observe`` measures characters only and never replaces the result; token and
    monetary savings depend on the model tokenizer and subsequent use.
    """
    if not isinstance(content, str) or not 1024 <= len(content) <= _MAX_INPUT_CHARS:
        return content
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        policy = config.get("tool_output", {}).get("json_compaction", {})
    except (AttributeError, TypeError, ValueError, OSError):
        return content
    if not isinstance(policy, dict):
        return content
    mode = policy.get("mode", "off")
    if mode not in ("observe", "compact"):
        return content
    excluded = policy.get("exclude_tools", ["read_file"])
    if (not isinstance(excluded, list)
            or any(not isinstance(pattern, str) for pattern in excluded)
            or any(fnmatchcase(tool_name, pattern) for pattern in excluded)):
        return content
    minimum = policy.get("min_chars", 1024)
    ratio = policy.get("min_savings_ratio", 0.1)
    if (type(minimum) is not int or minimum < 1024
            or type(ratio) not in (int, float) or not 0 < ratio < 1
            or len(content) < minimum):
        return content
    compacted = _compact_json(content)
    if len(compacted) > len(content) * (1 - ratio):
        return content
    logger.info(
        "JSON tool-result compaction: mode=%s tool=%s original_chars=%d compact_chars=%d",
        mode, tool_name, len(content), len(compacted),
    )
    return compacted if mode == "compact" else content
