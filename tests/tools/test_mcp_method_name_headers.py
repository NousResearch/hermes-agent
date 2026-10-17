"""SEP-2243 routing headers on handshake-era Streamable HTTP requests."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_transport import _ensure_sep2243_headers_stamp, _make_sep2243_request_header_stamper


def _handshake_stamp(data, opts):
    opts.setdefault("headers", {})["mcp-protocol-version"] = "2025-03-26"


def _modern_stamp(data, opts):
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
    for key, value in (opts.get("headers") or {}).items():
        if str(key).lower() == name.lower():
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

    def test_modern_stamp_headers_are_not_overwritten(self):
        session = _StampSession(_modern_stamp)
        _ensure_sep2243_headers_stamp(session)

        opts = _stamp(session, "tools/call", {"name": "get_weather"})

        assert _header(opts, "mcp-method") == "already-stamped"
        assert _header(opts, "mcp-name") == "already-named"

    def test_modern_stamp_with_http_equivalent_casing_is_not_overwritten(self):
        def mixed_case_stamp(data, opts):
            opts["headers"] = {"Mcp-Method": "already-stamped", "Mcp-Name": "already-named"}

        session = _StampSession(mixed_case_stamp)
        _ensure_sep2243_headers_stamp(session)

        opts = _stamp(session, "tools/call", {"name": "get_weather"})

        assert opts["headers"] == {"Mcp-Method": "already-stamped", "Mcp-Name": "already-named"}

    def test_non_string_name_and_missing_method_do_not_invent_name_headers(self):
        session = _StampSession(_handshake_stamp)
        _ensure_sep2243_headers_stamp(session)

        assert _header(_stamp(session, "tools/call", {"arguments": {}}), "mcp-name") is None
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


class _Request:
    def __init__(self, body, headers=None):
        self.method = "POST"
        self.content = body
        self.headers = {} if headers is None else headers


class TestStreamableHttpRequestHook:
    def test_stamps_v1_transport_post_without_session_stamp(self):
        request = _Request(b'{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"uri":"file:///a"}}')

        asyncio.run(_make_sep2243_request_header_stamper()(request))

        assert _header({"headers": request.headers}, "mcp-method") == "resources/read"
        assert _header({"headers": request.headers}, "mcp-name") == "file:///a"

    def test_preserves_mixed_case_existing_routing_headers(self):
        request = _Request(
            b'{"jsonrpc":"2.0","id":1,"method":"prompts/get","params":{"name":"help"}}',
            {"Mcp-Method": "modern", "Mcp-Name": "existing"},
        )

        asyncio.run(_make_sep2243_request_header_stamper()(request))

        assert request.headers == {"Mcp-Method": "modern", "Mcp-Name": "existing"}


def test_legacy_streamable_http_client_receives_owned_header_hook_factory():
    """SDK <1.24 owns its transport, so it needs the equivalent request hook factory."""
    task = MCPServerTask("legacy")

    with patch("tools.mcp_tool._MCP_NEW_HTTP", False), \
         patch("tools.mcp_tool.streamablehttp_client", create=True) as legacy_client:
        task._streamable_http_transport("https://example.com/mcp", {}, 5, True, None, None, False, set())

    assert callable(legacy_client.call_args.kwargs["httpx_client_factory"])


def test_legacy_header_hook_factory_preserves_redirect_following():
    task = MCPServerTask("legacy")
    captured = {}

    class _Httpx:
        class AsyncHTTPTransport:
            def __init__(self, **kwargs):
                pass

        class AsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

    with patch("tools.mcp_tool._MCP_NEW_HTTP", False), \
         patch("tools.mcp_tool.streamablehttp_client", create=True) as legacy_client, \
         patch("tools.mcp_tool.sdk_httpx", return_value=_Httpx), \
         patch("tools.mcp_tool_transport._mcp_proxy_mounts", return_value=None), \
         patch("tools.mcp_tool_transport._make_mcp_body_cap_transport", side_effect=lambda _, transport: transport):
        task._streamable_http_transport("https://example.com/mcp", {}, 5, True, None, None, False, set())
        legacy_client.call_args.kwargs["httpx_client_factory"](headers={}, timeout=5, auth=None)

    assert captured["follow_redirects"] is True
    assert len(captured["event_hooks"]["request"]) == 1
