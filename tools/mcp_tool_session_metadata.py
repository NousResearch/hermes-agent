"""Opt-in, host-authored MCP session metadata. Never read ambient environment identity."""

import logging

from tools.mcp_tool_common import _core

logger = logging.getLogger(__name__)
_PREFIX = "com.nousresearch.hermes/"
_FIELDS = ("platform", "session_id", "session_key", "chat_id", "thread_id", "user_id", "message_id")


def build_session_context_meta(server_name: str) -> dict[str, str] | None:
    with _core._lock:
        if server_name not in _core._session_context_forwarding_servers:
            return None
    from gateway.session_context import _UNSET, _VAR_MAP, _SESSION_REDACT_PII

    values = {}
    for field in _FIELDS:
        var = _VAR_MAP.get("HERMES_SESSION_" + field.removeprefix("session_").upper())
        if var is None:
            return None
        value = var.get()
        if value is _UNSET or (value is not None and not isinstance(value, str)):
            return None
        values[field] = value or ""
    if any(not values[field] for field in _FIELDS if field != "thread_id"):
        return None
    redact_pii = _SESSION_REDACT_PII.get()
    if redact_pii is not True and redact_pii is not False:
        return None
    if redact_pii:
        try:
            from gateway.session import _PII_SAFE_PLATFORMS, _hash_chat_id, _hash_id, _hash_sender_id
            from gateway.platform_registry import platform_registry

            entry = platform_registry.get(values["platform"])
            eligible = any(p.value == values["platform"] for p in _PII_SAFE_PLATFORMS) or bool(entry and entry.pii_safe)
            if eligible:
                transforms = {
                    "chat_id": _hash_chat_id, "user_id": _hash_sender_id,
                    "thread_id": lambda value: "thread_" + _hash_id(value),
                    "session_key": lambda value: "session_" + _hash_id(value),
                    "message_id": lambda value: "message_" + _hash_id(value),
                }
                values.update({key: transform(values[key]) for key, transform in transforms.items() if values[key]})
        except Exception:
            logger.warning("MCP session redaction unavailable; omitting session metadata")
            return None
    return {_PREFIX + field: values[field] for field in _FIELDS}
