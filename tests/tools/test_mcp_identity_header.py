"""Tests for the per-server MCP identity header (``identity_header``).

An optional per-server config key in ``mcp_servers`` attaches a static or
profile-derived identity header to that server's HTTP/SSE transport
requests:

    mcp_servers:
      remote_api:
        url: "https://my-mcp-server.example.com/mcp"
        identity_header:
          name: "X-User-Id"
          value_from: "static"      # or "profile"
          value: "alice"            # required for value_from: static

Covers:

1. ``_resolve_identity_header`` helper — static mode, profile mode,
   validation failures (warn + ignore, never break the server).

2. HTTP (new SDK ``streamable_http_client``) path attaches the header to
   the user-owned ``httpx.AsyncClient`` when configured, and not otherwise.

3. Explicit per-server ``headers`` with the same name win over the
   identity header (no silent override of user config).

4. stdio servers: ``identity_header`` is warn-and-ignore (headers don't
   exist on stdio transports).
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _resolve_identity_header helper
# ---------------------------------------------------------------------------


class TestResolveIdentityHeader:
    def test_returns_none_when_unset(self):
        from tools.mcp_tool import _resolve_identity_header

        assert _resolve_identity_header("srv", {}) is None
        assert _resolve_identity_header("srv", {"url": "https://x"}) is None

    def test_static_mode_returns_name_value(self):
        from tools.mcp_tool import _resolve_identity_header

        result = _resolve_identity_header("srv", {
            "identity_header": {
                "name": "X-User-Id",
                "value_from": "static",
                "value": "alice",
            },
        })
        assert result == ("X-User-Id", "alice")

    def test_static_is_default_value_from(self):
        from tools.mcp_tool import _resolve_identity_header

        result = _resolve_identity_header("srv", {
            "identity_header": {"name": "X-User-Id", "value": "bob"},
        })
        assert result == ("X-User-Id", "bob")

    def test_profile_mode_uses_active_profile_name(self):
        from tools.mcp_tool import _resolve_identity_header

        with patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="workbot",
        ):
            result = _resolve_identity_header("srv", {
                "identity_header": {
                    "name": "X-Hermes-Profile",
                    "value_from": "profile",
                },
            })
        assert result == ("X-Hermes-Profile", "workbot")

    def test_missing_name_warns_and_returns_none(self, caplog):
        from tools.mcp_tool import _resolve_identity_header

        with caplog.at_level(logging.WARNING):
            result = _resolve_identity_header("srv", {
                "identity_header": {"value": "alice"},
            })
        assert result is None
        assert any("identity_header" in r.message for r in caplog.records)

    def test_static_missing_value_warns_and_returns_none(self, caplog):
        from tools.mcp_tool import _resolve_identity_header

        with caplog.at_level(logging.WARNING):
            result = _resolve_identity_header("srv", {
                "identity_header": {"name": "X-User-Id"},
            })
        assert result is None
        assert any("identity_header" in r.message for r in caplog.records)

    def test_unknown_value_from_warns_and_returns_none(self, caplog):
        from tools.mcp_tool import _resolve_identity_header

        with caplog.at_level(logging.WARNING):
            result = _resolve_identity_header("srv", {
                "identity_header": {
                    "name": "X-User-Id",
                    "value_from": "per_call",
                    "value": "x",
                },
            })
        assert result is None
        assert any("identity_header" in r.message for r in caplog.records)

    def test_non_dict_config_warns_and_returns_none(self, caplog):
        from tools.mcp_tool import _resolve_identity_header

        with caplog.at_level(logging.WARNING):
            result = _resolve_identity_header("srv", {
                "identity_header": "X-User-Id: alice",
            })
        assert result is None
        assert any("identity_header" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# HTTP transport — header attached to httpx.AsyncClient
# ---------------------------------------------------------------------------


def _drive_http(server, config):
    """Run ``_run_http`` with the SDK boundary mocked out, capturing the
    kwargs passed to ``httpx.AsyncClient``. Mirrors the pattern in
    ``test_mcp_client_cert.py``.
    """
    from tools.mcp_tool import MCPServerTask

    captured: dict = {}

    class DummyAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class DummyTransportCtx:
        async def __aenter__(self):
            return MagicMock(), MagicMock(), (lambda: None)

        async def __aexit__(self, *a):
            return False

    class DummySession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def initialize(self):
            return None

    async def _discover_tools(self):
        self._shutdown_event.set()

    async def _drive():
        with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
             patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
             patch("httpx.AsyncClient", DummyAsyncClient), \
             patch("tools.mcp_tool.streamable_http_client",
                   return_value=DummyTransportCtx()), \
             patch("tools.mcp_tool.ClientSession", DummySession), \
             patch.object(MCPServerTask, "_discover_tools", _discover_tools):
            await server._run_http(config)

    asyncio.run(_drive())
    return captured


class TestHTTPIdentityHeader:
    def test_header_attached_when_configured(self):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        captured = _drive_http(server, {
            "url": "https://example.com/mcp",
            "identity_header": {
                "name": "X-User-Id",
                "value": "alice",
            },
        })
        headers = captured.get("headers") or {}
        assert headers.get("X-User-Id") == "alice"

    def test_header_absent_when_not_configured(self):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        captured = _drive_http(server, {
            "url": "https://example.com/mcp",
        })
        headers = captured.get("headers") or {}
        assert not any(k.lower() == "x-user-id" for k in headers)

    def test_explicit_header_with_same_name_wins(self):
        """A user-set per-server header of the same name (any casing) is
        not overridden by the identity header."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        captured = _drive_http(server, {
            "url": "https://example.com/mcp",
            "headers": {"x-user-id": "explicit-wins"},
            "identity_header": {
                "name": "X-User-Id",
                "value": "alice",
            },
        })
        headers = captured.get("headers") or {}
        assert headers.get("x-user-id") == "explicit-wins"
        assert "X-User-Id" not in headers

    def test_profile_mode_header_attached(self):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        with patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="workbot",
        ):
            captured = _drive_http(server, {
                "url": "https://example.com/mcp",
                "identity_header": {
                    "name": "X-Hermes-Profile",
                    "value_from": "profile",
                },
            })
        headers = captured.get("headers") or {}
        assert headers.get("X-Hermes-Profile") == "workbot"


# ---------------------------------------------------------------------------
# stdio transport — identity_header is warn-and-ignore
# ---------------------------------------------------------------------------


class TestStdioIdentityHeader:
    def test_stdio_warns_and_ignores(self, caplog):
        """identity_header on a stdio server logs a warning and does not
        break the transport path (headers don't exist on stdio)."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("local")

        async def _drive():
            # Force the SDK-unavailable fast path so no subprocess spawns;
            # the warning must fire before the availability check.
            with patch("tools.mcp_tool._MCP_AVAILABLE", False):
                await server._run_stdio({
                    "command": "echo",
                    "identity_header": {"name": "X-User-Id", "value": "a"},
                })

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ImportError):
                asyncio.run(_drive())

        assert any(
            "identity_header" in r.message and "stdio" in r.message
            for r in caplog.records
        )
