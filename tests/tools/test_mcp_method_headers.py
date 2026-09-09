"""Tests for the SEP-2243 routing-header hook in ``tools/mcp_tool_transport.py``.

The SDK stamps ``Mcp-Method``/``Mcp-Name`` only in modern (stateless) mode; the hook
covers the default legacy sessions at the transport layer. Tests drive the hook with
real httpx requests — no network.
"""

import builtins
import json

import pytest

httpx = pytest.importorskip("httpx")

from tools.mcp_tool_transport import _make_method_header_hook


def _request(body, headers=None):
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.Request("POST", "https://mcp.example/mcp", content=content,
                         headers=headers or {})


def _run_hook(body, headers=None):
    hook = _make_method_header_hook()
    assert hook is not None  # pinned SDK ships the header table
    request = _request(body, headers)
    hook(request)
    return request


class TestMethodHeaderHook:
    def test_tools_call_stamps_method_and_name(self):
        request = _run_hook({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": "search", "arguments": {"q": "x"}}})
        assert request.headers["mcp-method"] == "tools/call"
        assert request.headers["mcp-name"] == "search"

    def test_resources_read_uses_uri_param(self):
        request = _run_hook({"jsonrpc": "2.0", "id": 2, "method": "resources/read",
                             "params": {"uri": "file:///tmp/a.txt"}})
        assert request.headers["mcp-method"] == "resources/read"
        assert request.headers["mcp-name"] == "file:///tmp/a.txt"

    def test_plain_method_has_no_name(self):
        request = _run_hook({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        assert request.headers["mcp-method"] == "tools/list"
        assert "mcp-name" not in request.headers

    def test_notification_without_id_still_routed(self):
        request = _run_hook({"jsonrpc": "2.0", "method": "notifications/cancelled",
                             "params": {"requestId": 1}})
        assert request.headers["mcp-method"] == "notifications/cancelled"

    def test_existing_headers_win(self):
        request = _run_hook(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "a"}},
            headers={"mcp-method": "tools/call", "mcp-name": "sdk-value"})
        assert request.headers["mcp-name"] == "sdk-value"

    def test_non_json_body_untouched(self):
        hook = _make_method_header_hook()
        assert hook is not None
        request = _request(b"not json{{{")
        hook(request)
        assert "mcp-method" not in request.headers

    def test_batch_body_untouched(self):
        hook = _make_method_header_hook()
        assert hook is not None
        request = _request([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
        hook(request)
        assert "mcp-method" not in request.headers

    def test_non_ascii_name_encoded(self):
        request = _run_hook({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": "recherche été"}})
        assert request.headers["mcp-name"].startswith("=?base64?")

    def test_missing_sdk_table_means_no_hook(self, monkeypatch):
        real_import = builtins.__import__

        def _raise_for_inbound(name, *args, **kwargs):
            if name == "mcp.shared.inbound":
                raise ImportError("no table")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raise_for_inbound)
        assert _make_method_header_hook() is None


class TestTransportWiring:
    def test_streamable_http_installs_request_hook(self):
        import asyncio
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        from tools import mcp_tool
        from tools.mcp_tool import MCPServerTask, sdk_httpx
        from tools.mcp_tool_common import _core

        _core._ensure_mcp_sdk()  # production order: _run_http ensures before transport
        captured: dict = {}

        class _DummyAsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        @asynccontextmanager
        async def _dummy_streams(url, http_client=None):
            yield (object(), object())

        server = MCPServerTask("hook-wiring")

        async def _enter_transport():
            with patch.object(sdk_httpx(), "AsyncClient", _DummyAsyncClient), \
                    patch.object(mcp_tool, "streamable_http_client", _dummy_streams):
                async with server._streamable_http_transport(
                        "https://mcp.example/mcp", {}, 5.0, True, None, None, False, set()):
                    pass

        asyncio.run(_enter_transport())
        hooks = captured.get("event_hooks", {})
        assert "response" in hooks and len(hooks.get("request", [])) == 1
        request = _request({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "search"}})
        for hook in hooks["request"]:
            hook(request)
        assert request.headers["mcp-method"] == "tools/call"
        assert request.headers["mcp-name"] == "search"
