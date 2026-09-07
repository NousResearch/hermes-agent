"""Configurable tool-output truncation limits (``tool_output`` in config.yaml):
``max_bytes`` (terminal output cap), ``max_lines`` (read_file pagination cap),
``max_line_length`` (per-line cap before '... [truncated]'). Defaults equal the
constants once hardcoded in terminal_tool / file_operations and the reader never
raises, so behaviour is unchanged when the section is absent or malformed."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_MAX_BYTES = 50_000       # terminal_tool.MAX_OUTPUT_CHARS
DEFAULT_MAX_LINES = 2000         # file_operations.MAX_LINES
DEFAULT_MAX_LINE_LENGTH = 2000   # file_operations.MAX_LINE_LENGTH
_cached_limits: dict | None = None  # process-lifetime: no config.yaml re-read per tool call


def _coerce_int(value: Any, default: int, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum``, or ``default`` on any issue."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    return default if iv < minimum else iv


def _coerce_positive_int(value: Any, default: int) -> int:
    return _coerce_int(value, default, 1)  # positive int, or ``default`` on any issue


def get_tool_output_limits() -> Dict[str, int]:
    """Resolved ``{max_bytes, max_lines, max_line_length}``; never raises. Cached for the
    process — ``_reset_tool_output_limits_cache()`` forces a fresh read."""
    global _cached_limits
    if _cached_limits is not None:
        return _cached_limits
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        section = cfg.get("tool_output") if isinstance(cfg, dict) else None
    except Exception:
        section = None
    if not isinstance(section, dict):
        section = {}
    _cached_limits = {
        "max_bytes": _coerce_positive_int(section.get("max_bytes"), DEFAULT_MAX_BYTES),
        "max_lines": _coerce_positive_int(section.get("max_lines"), DEFAULT_MAX_LINES),
        "max_line_length": _coerce_positive_int(
            section.get("max_line_length"), DEFAULT_MAX_LINE_LENGTH)}
    return _cached_limits


def _reset_tool_output_limits_cache() -> None:
    """Reset the cached limits — for tests or after config hot-reload."""
    global _cached_limits
    _cached_limits = None


def get_max_bytes() -> int: return get_tool_output_limits()["max_bytes"]
def get_max_lines() -> int: return get_tool_output_limits()["max_lines"]
def get_max_line_length() -> int: return get_tool_output_limits()["max_line_length"]
