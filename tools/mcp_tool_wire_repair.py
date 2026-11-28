"""Lenient JSON-RPC parsing for MCP server replies.

The MCP SDK parses every inbound message (stdio lines, SSE events, HTTP JSON bodies) with one
pydantic ``TypeAdapter``. pydantic's JSON parser rejects a lone UTF-16 surrogate escape
(``"\\ud83d"``, e.g. a tool description cut inside an emoji) and invalid UTF-8 bytes. The SDK
logs the error and drops the message, so the request it answered waits for the full timeout:
discovery stalls for ``connect_timeout`` and a tool call hangs for its 300 s budget.

``install()`` wraps the adapter's ``validate_json`` once: a reply pydantic rejects is re-read
with the stdlib parser (which accepts lone surrogates), each lone surrogate becomes U+FFFD,
and the result is validated as usual. A reply valid on the strict path never reaches the
fallback; one invalid on both paths raises the original error.
"""

import importlib
import json
import logging
import re

logger = logging.getLogger("tools.mcp_tool")

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
_INSTALLED_MARK = "_hermes_lenient_json"


def _scrub(value):
    if isinstance(value, str):
        return _LONE_SURROGATE.sub("\ufffd", value)
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {_scrub(k): _scrub(v) for k, v in value.items()}
    return value


def _lenient(strict, adapter):
    def validate_json(data, *args, **kwargs):
        try:
            return strict(data, *args, **kwargs)
        except ValueError as exc:  # pydantic ValidationError subclasses ValueError
            try:
                text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else data
                repaired = _scrub(json.loads(text))
            except (TypeError, ValueError):
                raise exc from None
            logger.debug("MCP reply had invalid Unicode; read it with replacement characters")
            return adapter.validate_python(repaired, **kwargs)
    setattr(validate_json, _INSTALLED_MARK, True)
    return validate_json


def install() -> bool:
    """Wrap the SDK's shared JSON-RPC adapter (idempotent). False when the SDK lacks one."""
    for module_name in ("mcp_types", "mcp.types"):
        try:
            adapter = getattr(importlib.import_module(module_name), "jsonrpc_message_adapter", None)
        except ImportError:
            continue
        if adapter is None:
            continue
        current = adapter.validate_json
        if not getattr(current, _INSTALLED_MARK, False):
            adapter.validate_json = _lenient(current, adapter)
        return True
    return False
