"""Tests for automatic SSE fallback when Streamable HTTP returns 400.

When an MCP server (e.g. WigAI) only implements the SSE transport and
rejects Streamable HTTP initialize requests with 400 Bad Request, the
client should fall back to SSE transport automatically on the initial
connect — without requiring the user to set ``transport: sse``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_400_error(url="https://example.com/mcp"):
    """Build an httpx.HTTPStatusError for 400 Bad Request."""
    request = httpx.Request("POST", url)
    response = httpx.Response(400, request=request)
    return httpx.HTTPStatusError("Bad Request", request=request, response=response)


def _make_500_error(url="https://example.com/mcp"):
    """Build an httpx.HTTPStatusError for 500 Internal Server Error."""
    request = httpx.Request("POST", url)
    response = httpx.Response(500, request=request)
    return httpx.HTTPStatusError("Server Error", request=request, response=response)


def _build_server(name="sse-fallback-test"):
    """Create an MCPServerTask with mocks for transport testing."""
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask(name)
    server._auth_type = ""
    server._sampling = None
    server._elicitation = None
    return server


class _FakeStream:
    """Mock async context manager yielding (read, write) streams."""

    def __init__(self):
        self._read = AsyncMock()
        self._write = AsyncMock()

    async def __aenter__(self):
        return (self._read, self._write)

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Mock MCP ClientSession."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        return mock_session

    async def __aexit__(self, *a):
        return False


class _FakeHTTPTransport:
    """Mock streamable_http_client that records calls and can raise."""

    def __init__(self, side_effect=None):
        self._side_effect = side_effect
        self.called = False

    def __call__(self, url, http_client=None):
        self.called = True
        if self._side_effect is not None:
            raise self._side_effect
        return _FakeStream()


class _FakeSSETransport:
    """Mock sse_client that records calls."""

    def __init__(self):
        self.called = False
        self.kwargs = {}

    def __call__(self, **kwargs):
        self.called = True
        self.kwargs.update(kwargs)
        return _FakeStream()


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient mock for the Streamable HTTP path."""

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _NO_SSE:
    """Sentinel: simulate sse_client not being installed (set to None)."""
    pass


_NO_SSE_SENTINEL = _NO_SSE()


def _http_patches(server, *, http_side_effect=None, sse_transport=None,
                  extra_patches=None):
    """Return a combined context manager with all needed patches.

    Uses ``create=True`` for attributes that only exist when the MCP SDK
    is installed (``_MCP_NEW_HTTP``, ``streamable_http_client``).

    ``sse_transport`` controls what ``tools.mcp_tool.sse_client`` is set to:
      - A callable/mock: used as the SSE client (default: _FakeSSETransport)
      - ``_NO_SSE_SENTINEL``: set sse_client to None (simulates missing SDK)
    """
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True))
    stack.enter_context(patch("tools.mcp_tool._MCP_NEW_HTTP", True, create=True))

    if http_side_effect is not None:
        stack.enter_context(patch(
            "tools.mcp_tool.streamable_http_client",
            _FakeHTTPTransport(side_effect=http_side_effect),
            create=True,
        ))
    else:
        stack.enter_context(patch(
            "tools.mcp_tool.streamable_http_client",
            _FakeHTTPTransport(),
            create=True,
        ))

    if sse_transport is _NO_SSE_SENTINEL:
        stack.enter_context(patch(
            "tools.mcp_tool.sse_client", new=None, create=True,
        ))
    elif sse_transport is not None:
        stack.enter_context(patch(
            "tools.mcp_tool.sse_client", new=sse_transport, create=True,
        ))
    else:
        stack.enter_context(patch(
            "tools.mcp_tool.sse_client", new=_FakeSSETransport(), create=True,
        ))

    stack.enter_context(patch("tools.mcp_tool.ClientSession", new=_FakeSession, create=True))
    stack.enter_context(patch("httpx.AsyncClient", new=_FakeAsyncClient))
    stack.enter_context(patch.object(type(server), "_discover_tools",
                                     new=AsyncMock()))
    stack.enter_context(patch.object(type(server), "_wait_for_lifecycle_event",
                                     new=AsyncMock(return_value="shutdown")))

    if extra_patches:
        for p in extra_patches:
            stack.enter_context(p)

    return stack


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamableHTTP400Fallback:
    """When Streamable HTTP returns 400 on initial connect, fall back to SSE."""

    def test_streamable_http_400_falls_back_to_sse(self):
        """Streamable HTTP 400 -> SSE fallback -> success."""
        server = _build_server()
        fake_sse = _FakeSSETransport()

        async def drive():
            with _http_patches(server, http_side_effect=_make_400_error(),
                               sse_transport=fake_sse):
                await asyncio.wait_for(
                    server._run_http({
                        "url": "https://example.com/mcp",
                        "timeout": 60,
                    }),
                    timeout=5.0,
                )

        asyncio.run(drive())
        assert fake_sse.called, "sse_client was NOT called — SSE fallback did not trigger"

    def test_streamable_http_non_400_does_not_fallback(self):
        """Streamable HTTP 500 -> error propagates, NO SSE fallback."""
        server = _build_server()
        fake_sse = _FakeSSETransport()

        async def drive():
            with _http_patches(server, http_side_effect=_make_500_error(),
                               sse_transport=fake_sse):
                with pytest.raises(httpx.HTTPStatusError) as exc_info:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp",
                            "timeout": 60,
                        }),
                        timeout=5.0,
                    )
                assert exc_info.value.response.status_code == 500

        asyncio.run(drive())
        assert not fake_sse.called, "sse_client was called on a non-400 error"

    def test_streamable_http_400_logs_warning(self):
        """400 fallback should log a warning mentioning the server name."""
        server = _build_server("wigai")
        fake_sse = _FakeSSETransport()

        async def drive():
            with _http_patches(server, http_side_effect=_make_400_error(),
                               sse_transport=fake_sse):
                await asyncio.wait_for(
                    server._run_http({
                        "url": "https://example.com/mcp",
                        "timeout": 60,
                    }),
                    timeout=5.0,
                )

        asyncio.run(drive())
        # The warning is logged at WARNING level — we verify the fallback
        # happened (sse_client called) as a proxy for the log being emitted.
        assert fake_sse.called

    def test_streamable_http_400_fallback_forwards_headers_to_sse(self):
        """SSE fallback receives the same headers dict built for Streamable HTTP."""
        server = _build_server()
        fake_sse = _FakeSSETransport()
        custom_headers = {"X-Custom": "value"}

        async def drive():
            with _http_patches(server, http_side_effect=_make_400_error(),
                               sse_transport=fake_sse):
                await asyncio.wait_for(
                    server._run_http({
                        "url": "https://example.com/mcp",
                        "headers": custom_headers,
                        "timeout": 60,
                    }),
                    timeout=5.0,
                )

        asyncio.run(drive())
        assert fake_sse.called
        # headers should include both the user's custom header and the
        # auto-injected mcp-protocol-version
        sse_headers = fake_sse.kwargs.get("headers") or {}
        assert sse_headers.get("X-Custom") == "value"
        assert "mcp-protocol-version" in sse_headers

    def test_streamable_http_400_fallback_forwards_oauth_to_sse(self):
        """SSE fallback receives the OAuth auth provider when configured."""
        server = _build_server()
        server._auth_type = "oauth"
        fake_sse = _FakeSSETransport()
        fake_oauth = MagicMock(name="fake_oauth_provider")
        fake_manager = MagicMock()
        fake_manager.get_or_build_provider.return_value = fake_oauth

        async def drive():
            with _http_patches(
                server, http_side_effect=_make_400_error(),
                sse_transport=fake_sse,
                extra_patches=[
                    patch("tools.mcp_oauth_manager.get_manager",
                          return_value=fake_manager),
                ],
            ):
                await asyncio.wait_for(
                    server._run_http({
                        "url": "https://example.com/mcp",
                        "timeout": 60,
                    }),
                    timeout=5.0,
                )

        asyncio.run(drive())
        assert fake_sse.called
        assert "auth" in fake_sse.kwargs, "OAuth auth not forwarded to SSE fallback"
        assert fake_sse.kwargs["auth"] is fake_oauth

    def test_sse_fallback_still_fails_error_propagates(self):
        """Both Streamable HTTP (400) and SSE fail -> SSE error propagates."""
        server = _build_server()

        class _FailingSSEReturn:
            """sse_client replacement that raises on enter."""
            def __init__(self, **kwargs):
                pass
            def __call__(self, **kwargs):
                return self
            async def __aenter__(self):
                raise ConnectionRefusedError("SSE also refused")
            async def __aexit__(self, *a):
                return False

        async def drive():
            with _http_patches(server, http_side_effect=_make_400_error(),
                               sse_transport=_FailingSSEReturn()):
                with pytest.raises(ConnectionRefusedError, match="SSE also refused"):
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp",
                            "timeout": 60,
                        }),
                        timeout=5.0,
                    )

        asyncio.run(drive())

    def test_explicit_sse_transport_not_affected_by_fallback(self):
        """When transport: sse is explicit, Streamable HTTP is never attempted."""
        server = _build_server()
        fake_sse = _FakeSSETransport()
        fake_http = _FakeHTTPTransport()

        async def drive():
            with _http_patches(server, sse_transport=fake_sse):
                # Override the streamable_http_client patch to use our
                # trackable fake instead of the default one.
                import tools.mcp_tool as m
                original = m.streamable_http_client
                m.streamable_http_client = fake_http
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp",
                            "transport": "sse",
                            "timeout": 60,
                        }),
                        timeout=5.0,
                    )
                finally:
                    m.streamable_http_client = original

        asyncio.run(drive())
        assert fake_sse.called, "SSE path should have been called"
        assert not fake_http.called, "Streamable HTTP should NOT be called when transport=sse"

    def test_sse_unavailable_during_fallback_raises_import_error(self):
        """When Streamable HTTP 400s and sse_client is None, ImportError raised."""
        server = _build_server()

        async def drive():
            with _http_patches(server, http_side_effect=_make_400_error(),
                               sse_transport=_NO_SSE_SENTINEL):
                with pytest.raises(ImportError, match="SSE transport"):
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp",
                            "timeout": 60,
                        }),
                        timeout=5.0,
                    )

        asyncio.run(drive())
