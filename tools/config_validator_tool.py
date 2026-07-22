"""Hermes config/schema validator tool.

Registered as ``config_validator``. Reads ``<HERMES_HOME>/config.yaml`` via
``get_hermes_home()`` and returns a validation envelope.  Uses only stdlib
plus the existing ``tools.validation_helper`` module.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml

from hermes_constants import get_hermes_home
from tools.validation_helper import Failure, Success, relay_success, relay_failure


# ---------------------------------------------------------------------------
# Compact known top-level allowlist derived from cli-config.yaml.example.
# If Hermes adds a new top-level key, this tool reports it as a warning
# rather than failing the strict pass.
# ---------------------------------------------------------------------------
_KNOWN_TOP_LEVEL_KEYS = {
    "model",
    "terminal",
    "browser",
    "tool_loop_guardrails",
    "compression",
    "prompt_caching",
    "memory",
    "session_reset",
    "max_concurrent_sessions",
    "group_sessions_per_user",
    "streaming",
    "skills",
    "agent",
    "platform_toolsets",
    "stt",
    "code_execution",
    "delegation",
    "display",
    "updates",
}


def _config_path() -> Optional[Any]:
    """Return the expected config path, or ``None`` if Hermes home is unavailable."""
    try:
        home = get_hermes_home()
        home_str = str(home)
    except Exception:
        return None
    if not home_str or not os.path.isdir(home_str):
        return None
    path = home / "config.yaml"
    return path if path.exists() else path


def _read_config_text(path: Any) -> Optional[str]:
    """Read config file text, returning ``None`` when unavailable."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _safe_snippet(raw: str, needle: str, context: int = 60) -> str:
    """Return a short context snippet around *needle* in *raw*, or ``""``."""
    lower = raw.lower()
    idx = lower.find(needle.lower())
    if idx < 0:
        return ""
    start = max(0, idx - context)
    end = min(len(raw), idx + len(needle) + context)
    snippet = raw[start:end]
    prefix = "..." if start else ""
    suffix = "..." if end < len(raw) else ""
    return f"{prefix}{snippet}{suffix}"


def _check_unknown_top_level(data: Dict[str, Any], raw: str) -> List[Dict[str, Any]]:
    """Report unknown top-level keys as warnings with context."""
    warnings: List[Dict[str, Any]] = []
    unknown = sorted(set(data.keys()) - _KNOWN_TOP_LEVEL_KEYS)
    for key in unknown:
        warnings.append(
            {
                "key": key,
                "message": (
                    f"Unknown top-level key {key!r}; expected one of "
                    f"{sorted(_KNOWN_TOP_LEVEL_KEYS)}"
                ),
                "snippet": _safe_snippet(raw, key),
            }
        )
    return warnings


def _check_duplicate_top_level(raw: str) -> List[Dict[str, Any]]:
    """Detect duplicate top-level keys in raw YAML text."""
    issues: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ": " not in line and line.endswith(":"):
            # could be mapping key without value on this line
            key = line.strip().rstrip(":").strip()
        else:
            key = line.split(":", 1)[0].strip()
        if not key or not any(ch.isalnum() for ch in key):
            continue
        depth = (len(line) - len(line.lstrip())) // 2
        if depth == 0:
            if key in seen:
                issues.append(
                    {
                        "key": key,
                        "message": f"Duplicate top-level key {key!r} at line {seen[key]} and again below",
                        "snippet": _safe_snippet(raw, key),
                    }
                )
            else:
                seen[key] = seen.get(key, 0)
                seen[key] = len(seen)  # rough line count; good enough for snippet

    # More accurate duplicate count by re-scanning lines
    counts: Dict[str, int] = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key = line.split(":", 1)[0].strip()
        depth = (len(line) - len(line.lstrip())) // 2
        if depth == 0 and key:
            counts[key] = counts.get(key, 0) + 1
    duplicated = [k for k, c in counts.items() if c > 1]
    issues = []
    for key in duplicated:
        issues.append(
            {
                "key": key,
                "message": f"Duplicate top-level key {key!r} found {counts[key]} times",
                "snippet": _safe_snippet(raw, key),
            }
        )
    return issues


def _check_basic_semantics(data: Dict[str, Any], raw: str) -> List[Dict[str, Any]]:
    """A few lightweight sanity checks for keys Hermes actually uses at runtime."""
    issues: List[Dict[str, Any]] = []
    model = data.get("model") if isinstance(data.get("model"), dict) else None
    if isinstance(model, dict):
        if "default" not in model and "model" not in model:
            issues.append(
                {
                    "key": "model",
                    "message": "model block is missing 'default' or 'model'",
                    "snippet": _safe_snippet(raw or "", "model"),
                }
            )
    return issues


def _envelope(
    valid: bool,
    warnings: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
    *,
    checked_path: Optional[Any],
    checked_exists: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "valid": valid,
        "warnings": warnings,
        "issues": issues,
        "config_path": str(checked_path) if checked_path is not None else None,
        "config_exists": checked_exists,
    }
    return relay_success(payload)


def check_fn() -> bool:
    """Return False when the profile home or config path is unavailable."""
    return _config_path() is not None


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # noqa: ARG001
    """Validate Hermes config and return a structured envelope."""
    path = _config_path()
    if path is None:
        return relay_failure(
            "Configuration is missing or Hermes profile home is unavailable",
            code="config_validator_error",
            meta={"path": str(path) if path is not None else None},
        )

    raw = _read_config_text(path)
    if raw is None:
        return relay_failure(
            f"Configuration file is unreadable: {path}",
            code="config_validator_error",
            meta={"path": str(path)},
        )

    data: Any = None
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return relay_failure(
            f"Malformed YAML in config: {exc}",
            code="config_validator_error",
            meta={"path": str(path)},
        )
    except Exception as exc:  # noqa: BLE001
        return relay_failure(
            f"Failed to parse config: {exc}",
            code="config_validator_error",
            meta={"path": str(path)},
        )

    if data is None:
        data = {}

    # Structural: root must be a mapping and file must not be empty.
    warnings: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    if not raw.strip() or not isinstance(data, dict):
        issues.append(
            {
                "key": "<root>",
                "message": (
                    "Config file is empty or does not parse as a YAML mapping"
                ),
                "snippet": _safe_snippet(raw, "root") or raw[:120],
            }
        )
        return _envelope(False, warnings, issues, checked_path=path, checked_exists=True)

    # Unknown top-level keys → warnings.
    warnings.extend(_check_unknown_top_level(data, raw))
    # Duplicate top-level keys → issues.
    issues.extend(_check_duplicate_top_level(raw))
    # Basic semantics.
    issues.extend(_check_basic_semantics(data, raw))

    valid = not issues
    return _envelope(valid, warnings, issues, checked_path=path, checked_exists=True)


# =============================================================================
# OpenAI Function-Calling Schema / Registry
# =============================================================================

CONFIG_VALIDATOR_SCHEMA: Dict[str, Any] = {
    "name": "config_validator",
    "description": (
        "Validate ~/.hermes/config.yaml for structure, duplicates, unknown keys, "
        "and basic schema sanity. Returns a structured envelope with valid / warnings / issues."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def config_validator_tool(
    args: Optional[Dict[str, Any]] = None,
    **_kwargs: Any,
) -> str:
    payload = run(args)
    kind = payload.get("kind", "unknown")
    if kind == "success":
        return f"{kind}: {payload['data']['valid']} - {len(payload['data']['issues'])} issue(s), {len(payload['data']['warnings'])} warning(s)"
    return f"{kind}: {payload.get('error')} (code={payload.get('code')})"


def _check_config_validator_requirements() -> bool:
    return True


from tools.registry import registry  # noqa: E402

registry.register(
    name="config_validator",
    toolset="config",
    schema=CONFIG_VALIDATOR_SCHEMA,
    handler=config_validator_tool,
    check_fn=_check_config_validator_requirements,
    emoji="🛡️",
)

__all__ = ["config_validator_tool", "check_fn", "run", "CONFIG_VALIDATOR_SCHEMA"]
