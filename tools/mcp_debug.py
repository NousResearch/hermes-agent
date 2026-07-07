"""Debug logging helpers for MCP/tool-call argument plumbing.

Enabled only when ``HERMES_MCP_DEBUG_ARGS`` is truthy.  The helpers keep the
hot path cheap when disabled and redact secret-looking keys before anything is
sent to logs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any


logger = logging.getLogger(__name__)

_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "password",
    "secret",
    "token",
)
_MAX_STRING = 1200


def debug_args_enabled() -> bool:
    raw = os.getenv("HERMES_MCP_DEBUG_ARGS", "")
    return raw.strip().lower() in {"1", "true", "yes", "on", "debug"}


def _is_secret_key(key: Any) -> bool:
    lowered = str(key).lower()
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def redact_debug_value(value: Any, *, key: Any = None) -> Any:
    if _is_secret_key(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_debug_value(v, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_debug_value(v) for v in value]
    if isinstance(value, tuple):
        return [redact_debug_value(v) for v in value]
    if isinstance(value, str) and len(value) > _MAX_STRING:
        return value[:_MAX_STRING] + f"...[truncated {len(value) - _MAX_STRING} chars]"
    return value


def type_shape(value: Any, *, depth: int = 4) -> Any:
    """Return a compact tree of JSON-ish type names for diagnostics."""
    if depth <= 0:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(k): type_shape(v, depth=depth - 1) for k, v in value.items()}
    if isinstance(value, list):
        if not value:
            return []
        return [type_shape(value[0], depth=depth - 1)]
    if isinstance(value, tuple):
        if not value:
            return []
        return [type_shape(value[0], depth=depth - 1)]
    return type(value).__name__


def changed_type_paths(before: Any, after: Any, *, path: str = "") -> list[dict[str, str]]:
    """Return paths whose Python value type changed between two JSON-ish trees."""
    changes: list[dict[str, str]] = []
    before_type = type(before).__name__
    after_type = type(after).__name__
    if before_type != after_type:
        changes.append({
            "path": path or "$",
            "from_type": before_type,
            "to_type": after_type,
        })
        return changes
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after), key=lambda v: str(v)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in before or key not in after:
                changes.append({
                    "path": child_path,
                    "from_type": type(before.get(key)).__name__ if key in before else "missing",
                    "to_type": type(after.get(key)).__name__ if key in after else "missing",
                })
                continue
            changes.extend(changed_type_paths(before[key], after[key], path=child_path))
    elif isinstance(before, list) and isinstance(after, list):
        for idx, (left, right) in enumerate(zip(before, after)):
            changes.extend(changed_type_paths(left, right, path=f"{path}[{idx}]"))
    return changes


def debug_args_log(label: str, **fields: Any) -> None:
    if not debug_args_enabled():
        return
    payload = {key: redact_debug_value(value, key=key) for key, value in fields.items()}
    try:
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        rendered = str(payload)
    logger.info("%s:\n%s", label, rendered)
