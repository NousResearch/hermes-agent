"""Bounded, model-safe telemetry for ``delegate_task(action="inspect")``.

This module owns the privacy boundary for live subagent inspection. It stores
only bounded tool lifecycle metadata, never reasoning, assistant text, raw tool
arguments, raw tool results, or transcript contents.

URL-like targets fail closed to ``scheme://host[:port]``. Paths are dropped by
default because common APIs carry credentials in the path (for example Slack
incoming-webhook URLs and Telegram ``/bot<token>/`` endpoints). Any future
path-preserving exception must be explicitly allowlisted and backed by
adversarial regression coverage; the allowlist intentionally ships empty.

The live registry itself remains owned by ``tools.delegate_tool_registry`` and
is injected once with :func:`bind_registry`, avoiding a circular import.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("tools.delegate_tool")

# ---------------------------------------------------------------------------
# Registry binding
# ---------------------------------------------------------------------------

_registry_lock: Optional[threading.Lock] = None
_registry: Optional[Dict[str, Dict[str, Any]]] = None


def bind_registry(lock: threading.Lock, registry: Dict[str, Dict[str, Any]]) -> None:
    """Bind the canonical live-subagent registry owned by the control plane."""
    global _registry_lock, _registry
    _registry_lock = lock
    _registry = registry


# ---------------------------------------------------------------------------
# Privacy boundary: tool argument summaries
# ---------------------------------------------------------------------------

_TOOL_INPUT_TARGET_KEYS = frozenset(
    {
        "cwd",
        "destination_path",
        "directory",
        "dst",
        "endpoint",
        "file_path",
        "new_path",
        "old_path",
        "path",
        "source_path",
        "src",
        "target_path",
        "url",
        "urls",
    }
)
_TOOL_INPUT_URL_KEYS = frozenset({"endpoint", "url", "urls"})

# Fail closed. Adding an entry requires tool-specific adversarial evidence that
# the URL path cannot contain credentials.
_TOOL_INPUT_PATH_PRESERVING_HOSTS: frozenset[str] = frozenset()


def sanitize_url_target(value: str) -> Optional[str]:
    """Return only a validated URL origin, dropping credentials and location."""
    try:
        parsed = urlsplit(value)
        if not (parsed.scheme and parsed.netloc):
            return None
        hostname = parsed.hostname
        if not hostname:
            return None
        host = f"[{hostname}]" if ":" in hostname else hostname
        port = parsed.port
        netloc = f"{host}:{port}" if port is not None else host
        path = parsed.path if hostname.lower() in _TOOL_INPUT_PATH_PRESERVING_HOSTS else ""
        return urlunsplit((parsed.scheme, netloc, path, "", ""))
    except (TypeError, ValueError):
        return None


def sanitize_target(key: str, value: Any) -> Any:
    """Keep a bounded side-effect target while dropping URL secrets."""
    if isinstance(value, list):
        cleaned = [
            item
            for item in (sanitize_target(key, item) for item in value[:16])
            if item is not None
        ]
        return cleaned or None
    if not isinstance(value, str) or not value:
        return None
    bounded = value[:1024]
    if key in _TOOL_INPUT_URL_KEYS:
        return sanitize_url_target(bounded)
    return bounded


def _sanitize_targets(mapping: Dict[str, Any]) -> Dict[str, Any]:
    targets: Dict[str, Any] = {}
    for raw_key, value in mapping.items():
        key = str(raw_key).lower()
        if key not in _TOOL_INPUT_TARGET_KEYS:
            continue
        cleaned = sanitize_target(key, value)
        if cleaned is not None:
            targets[key] = cleaned
    return targets


def sanitize_input_summary(summary: Any) -> Dict[str, Any]:
    """Return a bounded, re-sanitized ``{argument_keys, targets}`` mapping."""
    if not isinstance(summary, dict):
        return {"argument_keys": [], "targets": {}}
    keys = summary.get("argument_keys")
    targets = summary.get("targets")
    return {
        "argument_keys": [str(key)[:128] for key in keys[:64]] if isinstance(keys, list) else [],
        "targets": _sanitize_targets(targets) if isinstance(targets, dict) else {},
    }


def summarize_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Summarize argument names and safe side-effect targets without payloads."""
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else None
    except (TypeError, ValueError):
        parsed = None
    if not isinstance(parsed, dict):
        return {"argument_keys": [], "targets": {}}
    return sanitize_input_summary(
        {"argument_keys": sorted(str(key)[:128] for key in parsed), "targets": parsed}
    )


# ---------------------------------------------------------------------------
# Bounded live event capture
# ---------------------------------------------------------------------------

SUBAGENT_INSPECT_EVENT_LIMIT = 12
_CAPTURE_ERROR_LIMIT = 2_147_483_647


def _live_record(subagent_id: str) -> Optional[Dict[str, Any]]:
    if _registry is None:
        return None
    return _registry.get(subagent_id)


def record_tool_started(subagent_id: str, tool_name: Any) -> None:
    """Update canonical live tool counters even when no display relay exists."""
    if _registry is None or _registry_lock is None:
        return
    safe_tool = str(tool_name or "unknown")[:256]
    with _registry_lock:
        record = _live_record(subagent_id)
        if record is None:
            return
        count = record.get("tool_count", 0)
        if isinstance(count, bool) or not isinstance(count, int):
            count = 0
        record["tool_count"] = max(0, count) + 1
        record["last_tool"] = safe_tool


def record_capture_error(subagent_id: str) -> None:
    """Mark supplementary capture degraded without retaining exception text."""
    if _registry is None or _registry_lock is None:
        return
    with _registry_lock:
        record = _live_record(subagent_id)
        if record is None:
            return
        value = record.get("_inspect_capture_errors", 0)
        if isinstance(value, bool) or not isinstance(value, int):
            value = 0
        record["_inspect_capture_errors"] = min(_CAPTURE_ERROR_LIMIT, max(0, value) + 1)


def append_inspect_event(subagent_id: str, event: Dict[str, Any]) -> None:
    """Append one sanitized event to the bounded per-child ring."""
    kind = str(event.get("type") or "")
    if kind not in {"tool_started", "tool_completed"}:
        return
    safe_event: Dict[str, Any] = {
        "type": kind,
        "tool": str(event.get("tool") or "unknown")[:256],
    }
    if kind == "tool_started":
        safe_event["tool_input"] = sanitize_input_summary(event.get("tool_input"))
    else:
        status = str(event.get("status") or "unknown").lower()
        safe_event["status"] = status if status in {"ok", "error"} else "unknown"
        duration = event.get("duration_seconds")
        if (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(float(duration))
        ):
            safe_event["duration_seconds"] = round(max(0.0, float(duration)), 3)

    if _registry is None or _registry_lock is None:
        return
    with _registry_lock:
        record = _live_record(subagent_id)
        if record is None:
            return
        events = record.setdefault("_inspect_events", [])
        if not isinstance(events, list):
            events = []
            record["_inspect_events"] = events
        events.append(safe_event)
        if len(events) > SUBAGENT_INSPECT_EVENT_LIMIT:
            del events[:-SUBAGENT_INSPECT_EVENT_LIMIT]


def _event_kind(event_type: Any) -> Optional[str]:
    """Normalize legacy strings or DelegateEvent values without importing relay code."""
    value = getattr(event_type, "value", event_type)
    if value in {"tool.started", "delegate.tool_started"}:
        return "tool_started"
    if value in {"tool.completed", "delegate.tool_completed"}:
        return "tool_completed"
    return None


def wrap_inspect_callback(inner_cb: Any, subagent_id: str) -> Any:
    """Tee tool lifecycle into inspect telemetry while preserving callback behavior."""

    def _cb(event_type, tool_name=None, preview=None, args=None, **kwargs):
        try:
            kind = _event_kind(event_type)
            if kind == "tool_started":
                # Count first so a sanitizer failure cannot make live activity disappear.
                record_tool_started(subagent_id, tool_name)
                if isinstance(args, str):
                    serialized_args = args
                elif args is None:
                    serialized_args = ""
                else:
                    serialized_args = json.dumps(args, ensure_ascii=False, default=str)
                append_inspect_event(
                    subagent_id,
                    {
                        "type": "tool_started",
                        "tool": tool_name,
                        "tool_input": summarize_tool_arguments(serialized_args),
                    },
                )
            elif kind == "tool_completed":
                append_inspect_event(
                    subagent_id,
                    {
                        "type": "tool_completed",
                        "tool": tool_name,
                        "status": "error" if kwargs.get("is_error") else "ok",
                        "duration_seconds": kwargs.get("duration"),
                    },
                )
        except Exception:
            record_capture_error(subagent_id)
            # Deliberately omit exception text: sanitizer exceptions can be
            # influenced by secret-bearing input and must not become a log leak.
            logger.debug("Subagent inspect capture degraded for %s", subagent_id)

        if inner_cb is not None:
            return inner_cb(event_type, tool_name, preview, args, **kwargs)
        return None

    def _flush():
        inner_flush = getattr(inner_cb, "_flush", None)
        if callable(inner_flush):
            return inner_flush()
        return None

    _cb._flush = _flush  # type: ignore[attr-defined]
    return _cb


# ---------------------------------------------------------------------------
# Strict response normalization
# ---------------------------------------------------------------------------


def nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(numeric) or numeric < 0:
        return 0
    return int(numeric)


def bounded_tool_label(value: Any) -> Optional[str]:
    return value[:256] if isinstance(value, str) and value else None


def finite_elapsed(started_at: Any, now: float) -> Optional[float]:
    if (
        isinstance(started_at, (int, float))
        and not isinstance(started_at, bool)
        and math.isfinite(float(started_at))
        and math.isfinite(float(now))
    ):
        return round(max(0.0, float(now) - float(started_at)), 1)
    return None


def serialize_inspect_response(
    base_snapshot: Dict[str, Any],
    agent: Any,
    activity_raw: Dict[str, Any],
    *,
    now: Optional[float] = None,
) -> str:
    """Serialize one strict-JSON, model-safe live snapshot."""
    now = time.time() if now is None else now
    cost_value = getattr(agent, "session_estimated_cost_usd", 0.0)
    try:
        cost_float = float(cost_value)
        estimated_cost_usd = max(0.0, cost_float) if math.isfinite(cost_float) else 0.0
    except (TypeError, ValueError, OverflowError):
        estimated_cost_usd = 0.0

    capture_errors = nonnegative_int(base_snapshot.get("capture_errors"))
    payload = {
        "action": "inspect",
        "subagent_id": base_snapshot.get("subagent_id"),
        "goal": base_snapshot.get("goal"),
        "model": base_snapshot.get("model"),
        "status": base_snapshot.get("status"),
        "running_seconds": finite_elapsed(base_snapshot.get("started_at"), now),
        "activity": {
            "current_tool": bounded_tool_label(activity_raw.get("current_tool")),
            "last_tool": bounded_tool_label(base_snapshot.get("last_tool")),
            "tool_count": nonnegative_int(base_snapshot.get("tool_count")),
            "api_calls": nonnegative_int(activity_raw.get("api_call_count", 0)),
            "max_iterations": nonnegative_int(activity_raw.get("max_iterations", 0)),
            "seconds_since_activity": finite_elapsed(activity_raw.get("last_activity_ts"), now),
        },
        "usage": {
            "input_tokens": nonnegative_int(getattr(agent, "session_prompt_tokens", 0)),
            "output_tokens": nonnegative_int(getattr(agent, "session_completion_tokens", 0)),
            "estimated_cost_usd": round(estimated_cost_usd, 6),
        },
        "recent_events": base_snapshot.get("recent_events", []),
        "telemetry": {
            "capture_degraded": capture_errors > 0,
            "capture_errors": capture_errors,
            "recent_event_limit": SUBAGENT_INSPECT_EVENT_LIMIT,
        },
        "accepting_steer": bool(base_snapshot.get("accepting_steer", False)),
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)
