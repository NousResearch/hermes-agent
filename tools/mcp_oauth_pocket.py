"""Which MCP resource requests may recover from a spurious 401 (``HermesMCPOAuthProvider``).

Pocket intermittently answers 401 to a still-valid Bearer token. Recovery (one same-token retry,
then at most one fenced refresh) is limited to the exact Pocket endpoint and to read-only
JSON-RPC traffic, so a mutation is never replayed and no other server changes behaviour.
"""

from __future__ import annotations

import json
import math
from typing import Any

try:
    from mcp.client.auth import OAuthFlowError as _AuthFailure
except ImportError:  # pragma: no cover — SDK required in CI; module must still import
    _AuthFailure = RuntimeError

POCKET_SERVER_NAME = "pocket"
_POCKET_HOST = "public.heypocketai.com"
_POCKET_PATH = b"/mcp"
_READ_METHODS = frozenset({"initialize", "notifications/initialized", "tools/list"})
_READ_TOOLS = frozenset({
    "get_account_info", "list_pocket_folders", "search_pocket_actionitems", "search_pocket_conversations",
})
_JSONRPC_MEMBERS = frozenset({"jsonrpc", "id", "method", "params"})


class PocketReadAuthExhausted(_AuthFailure):
    """A read request is still rejected after the bounded recovery. Fails closed: the SDK's
    interactive authorization is never entered from here. The message is fixed (no token,
    body, session or identity)."""

    def __init__(self) -> None:
        super().__init__("MCP OAuth 'pocket': read request still unauthorized after one retry and at most "
                         "one token refresh; run `hermes mcp login pocket` if this persists")


class _AmbiguousJSON(ValueError):
    """Parses differently across JSON implementations (duplicate members, NaN/Infinity)."""


def _unique_members(pairs: list[tuple[str, Any]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise _AmbiguousJSON
    return result


def _reject_constant(_name: str) -> Any:
    raise _AmbiguousJSON


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # e.g. 1e999 overflows to inf
        raise _AmbiguousJSON
    return value


def _is_pocket_endpoint(url: Any) -> bool:
    return (url.scheme == "https" and url.host == _POCKET_HOST and url.port in (None, 443)
            and url.raw_path == _POCKET_PATH and not url.userinfo and not url.fragment)


def _is_read_rpc(message: Any) -> bool:
    if not isinstance(message, dict) or not set(message) <= _JSONRPC_MEMBERS or message.get("jsonrpc") != "2.0":
        return False
    method, params = message.get("method"), message.get("params", {})
    if not isinstance(method, str) or not isinstance(params, dict):
        return False
    if method == "notifications/initialized":
        return "id" not in message
    msg_id = message.get("id")
    if isinstance(msg_id, bool) or not isinstance(msg_id, (int, str)):
        return False
    if method in _READ_METHODS:
        return True
    if method != "tools/call":
        return False
    name, arguments = params.get("name"), params.get("arguments", {})
    return isinstance(name, str) and name in _READ_TOOLS and isinstance(arguments, dict)


def is_recoverable_read_request(server_name: str, request: Any) -> bool:
    """True only for the ``pocket`` server's read-only requests to its exact HTTPS endpoint."""
    if server_name != POCKET_SERVER_NAME or not _is_pocket_endpoint(request.url):
        return False
    try:
        body = request.content
    # health: allow BLE001 -- httpx raises RequestNotRead (class owned by the SDK's httpx) for a streaming body; any failure must deny replay
    except Exception:  # streaming body: cannot be inspected or safely replayed
        return False
    if request.method == "GET":
        return body == b""
    if request.method != "POST":
        return False
    try:
        message = json.loads(body, object_pairs_hook=_unique_members, parse_constant=_reject_constant,
                             parse_float=_finite_float)
    except (ValueError, UnicodeDecodeError, RecursionError):  # malformed or ambiguous: never replayed
        return False
    return _is_read_rpc(message)
