"""Pre-execution argument validation for tool calls.

Phase 1 of issue #522: validates tool arguments before execution to catch
placeholder values, missing required parameters, non-existent paths, and
basic type/enum mismatches.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PLACEHOLDER_PATTERNS = [
    r"your_\w+_here",
    r"<[A-Z_]+>",
    r"\bTODO\b",
    r"\bPLACEHOLDER\b",
    r"example\.com",
    r"/path/to/",
    r"\bINSERT_",
    r"\bCHANGE_ME\b",
]

_PATH_LIKE_KEYS = {"path", "file", "target", "location", "output", "dest", "destination"}

_PATH_READ_CHECK_TOOLS = {"read_file"}
_PATH_EXISTENCE_TOOLS = {"read_file", "write_file", "patch", "search_files"}  # write_file/patch: path logged but not blocked
_PATH_WRITE_TOOLS = {"write_file", "patch"}  # skip placeholder check for write operations

_BASIC_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def _looks_like_placeholder(value: str) -> bool:
    for pattern in _PLACEHOLDER_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            return True
    return False


def _is_path_like(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _PATH_LIKE_KEYS)


def _get_required_fields(schema: dict[str, Any]) -> list[str]:
    for key in ("parameters", "input"):
        block = schema.get(key)
        if isinstance(block, dict):
            required = block.get("required")
            if isinstance(required, list):
                return [str(f) for f in required]
    return []


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def _check_type_and_enum(value: Any, expected: Any) -> Optional[str]:
    if not isinstance(expected, dict):
        return None
    expected_type = expected.get("type")
    if expected_type and expected_type not in _BASIC_TYPES:
        return None
    if expected_type and not _type_matches(value, expected_type):
        return (
            f"Parameter type mismatch: expected {expected_type}, "
            f"got {type(value).__name__}."
        )
    enum = expected.get("enum")
    if enum is not None and value not in enum:
        return (
            f"Parameter value {value!r} is not one of the allowed values: "
            f"{enum}."
        )
    return None


def validate_tool_arguments(
    tool_name: str,
    args: dict[str, Any],
    registry: Any,
) -> tuple[bool, str]:
    """Validate tool arguments before execution.

    Returns (True, "") on success or (False, error_message) on failure.
    """
    if not isinstance(args, dict):
        return False, "Tool arguments must be a JSON object."

    entry = registry.get_entry(tool_name)
    schema: dict[str, Any] = getattr(entry, "schema", None) or {}

    required = _get_required_fields(schema)
    missing = [f for f in required if f not in args or args[f] is None]
    if missing:
        return False, f"Missing required parameter(s): {', '.join(missing)}."

    properties: Optional[dict[str, Any]] = None
    for key in ("parameters", "input"):
        block = schema.get(key)
        if isinstance(block, dict):
            props = block.get("properties")
            if isinstance(props, dict):
                properties = props
                break

    for key, value in args.items():
        if isinstance(value, str) and _is_path_like(key):
            if tool_name in _PATH_WRITE_TOOLS:
                continue  # write operations create files — skip placeholder check
            expanded = os.path.expanduser(value.strip())
            if os.path.exists(expanded):
                continue  # real path — skip both placeholder and type checks
            if _looks_like_placeholder(value):
                logger.debug(
                    "argument_validator: placeholder detected in %s.%s",
                    tool_name,
                    key,
                )
                return (
                    False,
                    f"Parameter '{key}' looks like a placeholder value. "
                    "Provide a concrete value.",
                )

        if properties and key in properties:
            type_error = _check_type_and_enum(value, properties[key])
            if type_error:
                return False, type_error

    if tool_name in _PATH_EXISTENCE_TOOLS:
        path = args.get("path")
        if isinstance(path, str) and path.strip():
            if tool_name in _PATH_READ_CHECK_TOOLS and not os.path.exists(os.path.expanduser(path.strip())):
                return False, f"File not found: {path.strip()}"

    return True, ""
