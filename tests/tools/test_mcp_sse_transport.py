"""Regression tests for SSE transport in ``MCPServerTask._run_http``.

Covers fixes distilled from @amiller's PR #5981 that couldn't be cherry-picked
due to stale-branch divergence:

1. ``sse_read_timeout`` is set to 300s (not the tool timeout). SSE servers
   commonly hold the stream idle for minutes between events; a 60s read
   timeout drops the connection after the first slow stretch. Original
   observation: Router Teamwork / Supermemory on Cloudflare Workers dropping
   at ~60s idle.

2. OAuth auth is forwarded to ``sse_client`` when configured. Previously the
   code built ``_oauth_auth`` but never passed it to the SSE path, so SSE MCP
   servers behind OAuth 2.1 PKCE would silently fail with 401s.

3. Streamable HTTP uses a bounded 16 MiB SSE event limit. ``mcp==2.0.0`` does
   not expose httpx2's 1 MiB default, which rejects large one-event tools/list
   responses before Hermes can discover the server's tools (#138329).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

async def _noop_initialize():
    return None

def _build_server_with_sse(oauth: bool = False):
    """Stand up an MCPServerTask configured for SSE transport, with mocks
    threaded through so ``_run_http`` can enter the SSE branch without a
    real network call."""
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("sse-test")
    server._auth_type = "oauth" if oauth else ""
    server._sampling = None
    return server

@pytest.fixture
def patch_sse_client():
    """Replace ``sse_client`` with a MagicMock that records its kwargs.

    Returns the mock so tests can assert how ``_run_http`` called it.
    """
    captured_kwargs: dict = {}

    class _FakeStream:
        def __init__(self):
            self._read = AsyncMock()
            self._write = AsyncMock()

        async def __aenter__(self):
            return (self._read, self._write)

        async def __aexit__(self, *a):
            return False

    def fake_sse_client(**kwargs):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return _FakeStream()

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            mock_session = MagicMock()
            mock_session.initialize = AsyncMock()
            return mock_session

        async def __aexit__(self, *a):
            return False

    with patch("tools.mcp_tool.sse_client", new=fake_sse_client), \
         patch("tools.mcp_tool.ClientSession", new=_FakeSession):
        yield captured_kwargs

class TestSSEReadTimeout:
    def test_sse_read_timeout_is_300s_not_tool_timeout(self, patch_sse_client):
        """``sse_read_timeout`` must be 300s regardless of the configured
        ``timeout``. Using the tool timeout (60s default) causes Cloudflare-
        Workers-style SSE MCP servers to drop the connection at ~60s idle."""
        from tools.mcp_tool import MCPServerTask

        server = _build_server_with_sse()

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                            "timeout": 60,
                        }),
                        timeout=2.0,
                    )
                except (TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        assert patch_sse_client.get("sse_read_timeout") == 300.0, (
            f"sse_read_timeout = {patch_sse_client.get('sse_read_timeout')} "
            f"(expected 300.0) — SSE idle disconnect regression"
        )

class TestSSEOAuthForwarding:
    def test_sse_client_receives_oauth_auth_when_configured(self, patch_sse_client):
        """If ``_auth_type == 'oauth'``, ``sse_client`` must receive the
        constructed OAuth provider via ``auth=``. Previously the provider
        was built but never forwarded to the SSE path."""
        from tools.mcp_tool import MCPServerTask

        server = _build_server_with_sse(oauth=True)
        fake_oauth_provider = MagicMock(name="fake_oauth_provider")
        fake_manager = MagicMock()
        fake_manager.get_or_build_provider.return_value = fake_oauth_provider

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()), \
                 patch("tools.mcp_oauth_manager.get_manager", return_value=fake_manager):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                            "auth": "oauth",
                            "timeout": 60,
                        }),
                        timeout=2.0,
                    )
                except (TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        assert "auth" in patch_sse_client, (
            "sse_client was NOT called with auth= — SSE OAuth forwarding regressed"
        )
        assert patch_sse_client["auth"] is fake_oauth_provider


def test_streamable_http_accepts_a_tools_list_event_over_httpx2_default_limit():
    """Regression for #138329: MCP's direct EventSource construction is bounded at 16 MiB."""
    import httpx2
    from mcp.client import streamable_http
    from tools import mcp_tool_transport as transport

    event = b"data: " + (b"x" * (1024 * 1024 + 1)) + b"\n\n"
    request = httpx2.Request("POST", "https://example.com/mcp")
    response = httpx2.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=event)
    with pytest.raises(httpx2.SSEError, match="1048576"):
        list(httpx2.EventSource(response))

    transport._raise_streamable_http_event_limit()
    response = httpx2.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=event)
    messages = list(streamable_http.EventSource(response))

    assert len(messages) == 1
    assert len(messages[0].data) == 1024 * 1024 + 1
