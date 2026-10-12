"""Configurable tool-output truncation limits (``tool_output`` in config.yaml):
``max_bytes`` (terminal output cap), ``max_lines`` (read_file pagination cap),
``max_line_length`` (per-line cap before '... [truncated]'). Defaults equal the
constants once hardcoded in terminal_tool / file_operations and the reader never
raises, so behaviour is unchanged when the section is absent or malformed."""

from __future__ import annotations

from typing import Any, Dict

from hermes_constants import hermes_home_key

DEFAULT_MAX_BYTES = 50_000       # terminal_tool.MAX_OUTPUT_CHARS
DEFAULT_MAX_LINES = 2000         # file_operations.MAX_LINES
DEFAULT_MAX_LINE_LENGTH = 2000   # file_operations.MAX_LINE_LENGTH
# Keyed by profile home: the multiplexed gateway serves every profile from one process, so a
# single slot would hand the launch profile's limits to every other profile.
_cached_limits: dict[str, dict[str, int]] = {}


def _coerce_int(value: Any, default: int, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum``, or ``default`` on any issue."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    return default if iv < minimum else iv


def _coerce_positive_int(value: Any, default: int) -> int:
    return _coerce_int(value, default, 1)  # positive int, or ``default`` on any issue


def get_tool_output_limits() -> dict[str, int]:
    """Resolved ``{max_bytes, max_lines, max_line_length}``; never raises. Cached per profile
    home for the process — ``_reset_tool_output_limits_cache()`` forces a fresh read."""
    key = hermes_home_key()
    cached = _cached_limits.get(key)
    if cached is not None:
        return cached
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        section = cfg.get("tool_output") if isinstance(cfg, dict) else None
    except Exception:
        section = None
    if not isinstance(section, dict):
        section = {}
    _cached_limits[key] = limits = {
        "max_bytes": _coerce_positive_int(section.get("max_bytes"), DEFAULT_MAX_BYTES),
        "max_lines": _coerce_positive_int(section.get("max_lines"), DEFAULT_MAX_LINES),
        "max_line_length": _coerce_positive_int(
            section.get("max_line_length"), DEFAULT_MAX_LINE_LENGTH)}
    return limits


def _reset_tool_output_limits_cache() -> None:
    """Reset the cached limits — for tests or after config hot-reload."""
    _cached_limits.clear()


def get_max_bytes() -> int: return get_tool_output_limits()["max_bytes"]
def get_max_lines() -> int: return get_tool_output_limits()["max_lines"]
def get_max_line_length() -> int: return get_tool_output_limits()["max_line_length"]

def cap_json_output(
    payload: str,
    *,
    max_chars: int | None = None,
    list_fields: tuple[str, ...] = ("results", "messages", "sessions", "jobs"),
    string_fields: tuple[str, ...] = (),
    truncation_message: str | None = None,
) -> str:
    """Bound a JSON tool response to ``tool_output.max_bytes``.

    Shared helper for tool families whose serialized responses can exceed the
    centralized ``tool_output`` cap (cron list, delegate results, skill
    dumps).  When the payload fits, it is returned unchanged (fast path is a
    length check).  When it exceeds the cap, entries are dropped from the tail
    of the largest list field named in ``list_fields`` (or the largest string
    field named in ``string_fields`` is tail-truncated) until the serialized
    response fits, and the truncation is marked in-band (``truncated`` +
    ``truncated_count``) so the model knows to narrow its request.

    Defensive: any parse or shrink failure returns the original payload
    unchanged, so a malformed or non-list-shaped response is never corrupted.
    """
    import json as _json

    try:
        cap = max_chars if max_chars is not None else get_max_bytes()
        if len(payload) <= cap:
            return payload
        resp = _json.loads(payload)
        if not isinstance(resp, dict):
            return payload

        # Prefer shrinking the largest list field (drop tail entries), then
        # fall back to tail-truncating the largest string field.
        biggest_key, biggest_len = None, 0
        for key in list_fields:
            val = resp.get(key)
            if isinstance(val, list) and len(val) > biggest_len:
                biggest_key, biggest_len = key, len(val)
        if biggest_key is not None:
            while resp[biggest_key] and len(_json.dumps(resp, ensure_ascii=False)) > cap:
                resp[biggest_key].pop()
            dropped = biggest_len - len(resp[biggest_key])
        else:
            biggest_key, biggest_len = None, 0
            for key in string_fields:
                val = resp.get(key)
                if isinstance(val, str) and len(val) > biggest_len:
                    biggest_key, biggest_len = key, len(val)
            if biggest_key is None:
                return payload
            original = resp[biggest_key]
            # Binary-search the largest prefix that fits under the cap.
            lo, hi = 0, len(original)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                resp[biggest_key] = original[:mid] + "\n… [truncated]"
                if len(_json.dumps(resp, ensure_ascii=False)) <= cap:
                    lo = mid
                else:
                    hi = mid - 1
            resp[biggest_key] = original[:lo] + "\n… [truncated]"
            dropped = biggest_len - lo

        resp["truncated"] = True
        resp["truncated_count"] = dropped
        if truncation_message is not None:
            resp["message"] = truncation_message.format(
                cap=cap, dropped=dropped, field=biggest_key
            )
        return _json.dumps(resp, ensure_ascii=False)
    except Exception:
        return payload
