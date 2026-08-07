"""Regression tests for the MCP ``token`` config field.

Covers #61562: when an HTTP/SSE/Streamable HTTP MCP server is configured with a
separate ``token`` value, the client must append ``?token=<token>`` to the
request URL, matching the Windmill convention. The token must also take
precedence over any token already embedded in the saved URL so duplicate tokens
are not sent.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _mcp_url_with_token


class TestMcpUrlWithToken:
    def test_no_token_returns_url_unchanged(self):
        url = "https://example.com/mcp"
        assert _mcp_url_with_token(url, None) is url
        assert _mcp_url_with_token(url, "") == url

    def test_appends_token_to_bare_url(self):
        assert _mcp_url_with_token("https://example.com/mcp", "abc123") == (
            "https://example.com/mcp?token=abc123"
        )

    def test_preserves_other_query_params(self):
        assert _mcp_url_with_token("https://example.com/mcp?foo=bar", "tok") == (
            "https://example.com/mcp?foo=bar&token=tok"
        )

    def test_replaces_existing_token_case_insensitive(self):
        assert _mcp_url_with_token(
            "https://example.com/mcp?TOKEN=old&x=1", "new"
        ) == "https://example.com/mcp?x=1&token=new"

    def test_non_string_token_is_coerced(self):
        assert _mcp_url_with_token("https://example.com/mcp", 12345) == (
            "https://example.com/mcp?token=12345"
        )


class _FakeClientSession:
    def __init__(self, *args, **kwargs):
        self.initialize = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeStreamableHttp:
    def __init__(self, captured, url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs

    async def __aenter__(self):
        return (MagicMock(), MagicMock(), lambda: None)

    async def __aexit__(self, *exc):
        return False


def _build_server() -> MCPServerTask:
    server = MCPServerTask("windmill-test")
    server._auth_type = ""
    server._sampling = None
    server._elicitation = None
    return server


class TestStreamableHttpTokenQueryParam:
    def test_new_http_client_receives_token_query_param(self, monkeypatch):
        """The mcp >= 1.24.0 streamable_http_client must see ``?token=...``."""
        import tools.mcp_tool as mod

        captured: dict = {}

        def fake_streamable_http_client(url, *, http_client, **kwargs):
            return _FakeStreamableHttp(captured, url, http_client=http_client, **kwargs)

        fake_httpx_client = MagicMock()
        fake_httpx_client.__aenter__ = AsyncMock(return_value=fake_httpx_client)
        fake_httpx_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(mod, "_MCP_HTTP_AVAILABLE", True)
        monkeypatch.setattr(mod, "_MCP_NEW_HTTP", True)
        monkeypatch.setattr(mod, "streamable_http_client", fake_streamable_http_client)

        server = _build_server()

        async def drive():
            with patch("httpx.AsyncClient", return_value=fake_httpx_client), \
                 patch.object(MCPServerTask, "_wait_for_lifecycle_event", new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()), \
                 patch.object(mod, "ClientSession", _FakeClientSession):
                await asyncio.wait_for(
                    server._run_http({
                        "url": "https://windmill.example.com/api/mcp",
                        "token": "windmill-secret",
                        "connect_timeout": 5,
                        "ssl_verify": True,
                    }),
                    timeout=2.0,
                )

        asyncio.run(drive())

        assert captured["url"] == "https://windmill.example.com/api/mcp?token=windmill-secret"

    def test_legacy_http_client_receives_token_query_param(self, monkeypatch):
        """The older streamablehttp_client must also see ``?token=...``."""
        import tools.mcp_tool as mod

        captured: dict = {}

        def fake_streamablehttp_client(url, **kwargs):
            return _FakeStreamableHttp(captured, url, **kwargs)

        monkeypatch.setattr(mod, "_MCP_HTTP_AVAILABLE", True)
        monkeypatch.setattr(mod, "_MCP_NEW_HTTP", False)
        monkeypatch.setattr(mod, "streamablehttp_client", fake_streamablehttp_client, raising=False)

        server = _build_server()

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event", new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()), \
                 patch.object(mod, "ClientSession", _FakeClientSession):
                await asyncio.wait_for(
                    server._run_http({
                        "url": "https://windmill.example.com/api/mcp",
                        "token": "windmill-secret",
                        "connect_timeout": 5,
                        "ssl_verify": True,
                    }),
                    timeout=2.0,
                )

        asyncio.run(drive())

        assert captured["url"] == "https://windmill.example.com/api/mcp?token=windmill-secret"


class TestSseTokenQueryParam:
    def test_sse_client_receives_token_query_param(self, monkeypatch):
        """The SSE transport must receive the URL with ``?token=...``."""
        import tools.mcp_tool as mod

        captured: dict = {}

        class _FakeSseStream:
            async def __aenter__(self):
                return (MagicMock(), MagicMock())

            async def __aexit__(self, *exc):
                return False

        def fake_sse_client(**kwargs):
            captured.clear()
            captured.update(kwargs)
            return _FakeSseStream()

        monkeypatch.setattr(mod, "_MCP_HTTP_AVAILABLE", True)
        monkeypatch.setattr(mod, "sse_client", fake_sse_client, raising=False)

        server = _build_server()

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event", new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()), \
                 patch.object(mod, "ClientSession", _FakeClientSession):
                await asyncio.wait_for(
                    server._run_http({
                        "url": "https://windmill.example.com/api/mcp/sse",
                        "token": "windmill-secret",
                        "transport": "sse",
                        "connect_timeout": 5,
                    }),
                    timeout=2.0,
                )

        asyncio.run(drive())

        assert captured.get("url") == "https://windmill.example.com/api/mcp/sse?token=windmill-secret"
