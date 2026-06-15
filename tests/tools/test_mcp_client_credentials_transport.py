"""Transport wiring tests for MCP client_credentials auth."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def test_http_transport_passes_client_credentials_auth_to_httpx(monkeypatch):
    """auth: client_credentials is a first-class HTTP auth provider, not a static header."""
    import httpx
    from tools.mcp_tool import MCPServerTask

    captured_client_kwargs: dict = {}
    fake_auth_provider = object()
    fake_manager = MagicMock()
    fake_manager.get_or_build_provider.return_value = fake_auth_provider

    class _FakeHTTPClient:
        def __init__(self, **kwargs):
            captured_client_kwargs.clear()
            captured_client_kwargs.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _FakeStreamableHTTP:
        async def __aenter__(self):
            return (AsyncMock(), AsyncMock(), lambda: None)

        async def __aexit__(self, *args):
            return False

    def _fake_streamable_http_client(url, http_client):
        return _FakeStreamableHTTP()

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            session = MagicMock()
            session.initialize = AsyncMock()
            return session

        async def __aexit__(self, *args):
            return False

    server = MCPServerTask("linear")
    server._auth_type = "client_credentials"
    server._sampling = None

    async def drive():
        with patch.object(MCPServerTask, "_wait_for_lifecycle_event", new=AsyncMock(return_value="shutdown")), \
             patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()), \
             patch("tools.mcp_client_credentials.get_client_credentials_manager", return_value=fake_manager), \
             patch("tools.mcp_tool.streamable_http_client", new=_fake_streamable_http_client), \
             patch("tools.mcp_tool.ClientSession", new=_FakeSession):
            monkeypatch.setattr("tools.mcp_tool._MCP_HTTP_AVAILABLE", True)
            monkeypatch.setattr("tools.mcp_tool._MCP_NEW_HTTP", True)
            monkeypatch.setattr(httpx, "AsyncClient", _FakeHTTPClient)
            await server._run_http({
                "url": "https://mcp.linear.app/mcp",
                "auth": "client_credentials",
                "oauth": {
                    "client_id": "linear-client-id",
                    "client_secret": "linear-client-secret",
                    "scope": "read,comments:create",
                },
            })

    asyncio.run(drive())

    assert captured_client_kwargs["auth"] is fake_auth_provider
    fake_manager.get_or_build_provider.assert_called_once()
    called_name, called_url, called_cfg = fake_manager.get_or_build_provider.call_args.args
    assert called_name == "linear"
    assert called_url == "https://mcp.linear.app/mcp"
    assert called_cfg["client_id"] == "linear-client-id"
