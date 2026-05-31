"""Tests for _MCPServer._preflight_content_type early-fail behaviour."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask


@pytest.fixture()
def server():
    """Return a minimal MCPServerTask instance (bypasses __init__ complexity)."""
    s = MCPServerTask.__new__(MCPServerTask)
    s.name = "test-server"
    return s


# ---------------------------------------------------------------------------
# HTML response → ConnectionError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_rejects_html(server):
    """A text/html response must raise ConnectionError immediately."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html; charset=utf-8"}

    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ConnectionError, match="text/html"):
            await server._preflight_content_type("https://example.com")


@pytest.mark.asyncio
async def test_preflight_rejects_html_on_get_fallback(server):
    """When HEAD returns 405, fall back to GET — still reject HTML."""
    head_response = MagicMock()
    head_response.status_code = 405

    get_response = MagicMock()
    get_response.status_code = 200
    get_response.headers = {"content-type": "text/html"}

    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=head_response)
    mock_client.get = AsyncMock(return_value=get_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ConnectionError, match="text/html"):
            await server._preflight_content_type("https://example.com")


# ---------------------------------------------------------------------------
# Non-HTML responses → silent pass-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight_accepts_json(server):
    """application/json must NOT raise."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Should not raise
        await server._preflight_content_type("https://mcp-server.example.com/mcp")


@pytest.mark.asyncio
async def test_preflight_accepts_no_content_type(server):
    """Missing Content-Type header must NOT raise."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}

    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await server._preflight_content_type("https://mcp-server.example.com/mcp")


@pytest.mark.asyncio
async def test_preflight_swallows_network_errors(server):
    """Network errors / timeouts must silently pass through."""

    mock_client = AsyncMock()
    mock_client.head = AsyncMock(side_effect=TimeoutError("connect timed out"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Should not raise — let the real MCP handshake deal with it
        await server._preflight_content_type("https://unreachable.example.com")


@pytest.mark.asyncio
async def test_preflight_passes_headers_and_verify(server):
    """Custom headers and ssl_verify are forwarded to the probe client."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.head = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client) as client_cls:
        await server._preflight_content_type(
            "https://mcp.example.com/mcp",
            headers={"Authorization": "Bearer tok"},
            ssl_verify=False,
        )
        # Verify the client was created with ssl_verify=False
        client_cls.assert_called_once()
        call_kwargs = client_cls.call_args
        assert call_kwargs.kwargs.get("verify") is False
