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
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        assert patch_sse_client.get("sse_read_timeout") == 300.0, (
            f"sse_read_timeout = {patch_sse_client.get('sse_read_timeout')} "
            f"(expected 300.0) — SSE idle disconnect regression"
        )

    def test_sse_read_timeout_still_300s_when_tool_timeout_is_large(self, patch_sse_client):
        """Even if user sets a large ``timeout``, ``sse_read_timeout`` stays
        decoupled — it's a transport-level budget for inter-event silence,
        not a per-call budget."""
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
                            "timeout": 600,
                        }),
                        timeout=2.0,
                    )
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        assert patch_sse_client.get("sse_read_timeout") == 300.0


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
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        assert "auth" in patch_sse_client, (
            "sse_client was NOT called with auth= — SSE OAuth forwarding regressed"
        )
        assert patch_sse_client["auth"] is fake_oauth_provider

    def test_sse_client_omits_auth_when_no_oauth_configured(self, patch_sse_client):
        """Without OAuth, ``sse_client`` should not receive an ``auth=`` kwarg.
        Passing ``None`` would be equally fine but the current code path only
        sets it when configured — lock that in."""
        from tools.mcp_tool import MCPServerTask

        server = _build_server_with_sse(oauth=False)

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
                except (asyncio.TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        assert "auth" not in patch_sse_client, (
            f"sse_client was called with auth= when no OAuth was configured: "
            f"{patch_sse_client!r}"
        )
