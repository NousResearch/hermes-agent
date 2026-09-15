"""SEP-2243 Mcp-Method / Mcp-Name headers on handshake-era Streamable HTTP.

mcp SDK 2.0's ``_make_handshake_stamp`` only stamps ``mcp-protocol-version``.
Hermes defaults to ``protocol=auto`` (initialize first), so most servers stay
in the handshake era and outbound POSTs never carry the routing headers
SEP-2243 requires. Hermes wraps ``session._stamp`` after negotiate so every
subsequent JSON-RPC request gets ``mcp-method`` (and ``mcp-name`` when the
method is name-bearing) without overriding a modern stamp that already set
them. Complements #88698; issue #106799.
"""

from __future__ import annotations

import asyncio

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_transport import _ensure_sep2243_headers_stamp


def _handshake_stamp(data, opts):
    """Mirrors mcp SDK ``_make_handshake_stamp``: protocol version only."""
    opts.setdefault("headers", {})["mcp-protocol-version"] = "2025-03-26"


def _modern_stamp(data, opts):
    """Mirrors a modern stamp that already wrote routing headers."""
    headers = opts.setdefault("headers", {})
    headers["mcp-protocol-version"] = "2026-07-28"
    headers["mcp-method"] = "already-stamped"
    headers["mcp-name"] = "already-named"


class _StampSession:
    def __init__(self, stamp):
        self._stamp = stamp

    async def initialize(self):
        return "INIT_RESULT"


def _header(opts, name):
    headers = opts.get("headers") or {}
    lower = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lower:
            return value
    return None


def _stamp(session, method, params=None):
    data = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        data["params"] = params
    opts = {}
    session._stamp(data, opts)
    return opts


class TestHandshakeStampWrap:
    def test_tools_call_stamps_method_and_name(self):
        session = _StampSession(_handshake_stamp)
        _ensure_sep2243_headers_stamp(session)
        opts = _stamp(session, "tools/call", {"name": "get_weather"})
        assert _header(opts, "mcp-method") == "tools/call"
        assert _header(opts, "mcp-name") == "get_weather"
        assert _header(opts, "mcp-protocol-version") == "2025-03-26"

    def test_tools_list_stamps_method_without_name(self):
        session = _StampSession(_handshake_stamp)
        _ensure_sep2243_headers_stamp(session)
        opts = _stamp(session, "tools/list")
        assert _header(opts, "mcp-method") == "tools/list"
        assert _header(opts, "mcp-name") is None
        assert _header(opts, "mcp-protocol-version") == "2025-03-26"

    def test_modern_stamp_headers_are_not_overwritten(self):
        session = _StampSession(_modern_stamp)
        _ensure_sep2243_headers_stamp(session)
        opts = _stamp(session, "tools/call", {"name": "get_weather"})
        assert _header(opts, "mcp-method") == "already-stamped"
        assert _header(opts, "mcp-name") == "already-named"

    def test_name_bearing_without_str_name_skips_mcp_name(self):
        session = _StampSession(_handshake_stamp)
        _ensure_sep2243_headers_stamp(session)
        opts = _stamp(session, "tools/call", {"arguments": {}})
        assert _header(opts, "mcp-method") == "tools/call"
        assert _header(opts, "mcp-name") is None

    def test_missing_method_does_not_invent_headers(self):
        session = _StampSession(_handshake_stamp)
        _ensure_sep2243_headers_stamp(session)
        opts = {}
        session._stamp({"jsonrpc": "2.0", "id": 1, "params": {}}, opts)
        assert _header(opts, "mcp-method") is None
        assert _header(opts, "mcp-name") is None


class TestServeSessionWiresWrap:
    def test_handshake_session_stamps_after_negotiate(self):
        session = _StampSession(_handshake_stamp)
        task = MCPServerTask("sep2243")

        async def _discover():
            return None

        async def _done():
            return "shutdown"

        task._discover_tools = _discover
        task._wait_for_lifecycle_event = _done

        asyncio.new_event_loop().run_until_complete(task._serve_session(session, 5, label="HTTP"))

        opts = _stamp(session, "tools/call", {"name": "get_weather"})
        assert _header(opts, "mcp-method") == "tools/call"
        assert _header(opts, "mcp-name") == "get_weather"
