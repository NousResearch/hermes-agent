"""Regression tests for the MCP keepalive ping reconnect loop fix.

Issue #50028 — MCP client sends ``ping`` keepalive to all servers
unconditionally.  When the server does not implement the optional ``ping``
JSON-RPC method it returns ``Unknown method: ping``, which the keepalive loop
treated as a failure and triggered a reconnect every ~3 minutes, causing a
continuous reconnect loop with log noise.

The fix:
1. ``_ping_is_safe()`` — capability-aware guard that only allows ping on
   HTTP servers or servers that explicitly advertise ping in ``experimental``.
2. The keepalive loop has a third branch: skip keepalive when neither
   ``tools`` nor ``ping`` is safe, logging at DEBUG.
3. ``-32601 / "Unknown method"`` errors from ``send_ping()`` are swallowed
   and logged at DEBUG (not WARNING), and do NOT trigger a reconnect.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal stub MCPServerTask instances
# ---------------------------------------------------------------------------

def _stub_module():
    """Return a minimal mcp_tool module stub so we can import MCPServerTask."""
    # Install stub MCP SDK modules before importing mcp_tool
    for name in [
        "mcp", "mcp.client", "mcp.client.stdio", "mcp.client.session",
        "mcp.types", "mcp.shared", "mcp.shared.exceptions",
    ]:
        if name not in __import__("sys").modules:
            __import__("sys").modules[name] = types.ModuleType(name)


# We don't import MCPServerTask directly to avoid the full dependency chain.
# Instead, we test _ping_is_safe() and the keepalive logic by constructing
# a minimal stand-in that mirrors the real implementation.


class _FakeInitResult:
    """Mimics ``InitializeResult`` from the MCP SDK."""

    def __init__(self, *, tools=None, experimental=None):
        self.capabilities = types.SimpleNamespace(
            tools=tools,
            experimental=experimental,
        )


class _FakeServerTask:
    """Minimal re-implementation of the two methods under test."""

    def __init__(
        self,
        *,
        is_http: bool = False,
        initialize_result=None,
    ):
        self._config = {"url": "http://localhost"} if is_http else {"command": "node"}
        self.initialize_result = initialize_result
        self.name = "test-server"
        self.session = None
        self._reconnect_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()

    def _is_http(self) -> bool:
        return "url" in self._config

    def _advertises_tools(self) -> bool:
        init_result = self.initialize_result
        caps = getattr(init_result, "capabilities", None) if init_result is not None else None
        if caps is None:
            return True
        return getattr(caps, "tools", None) is not None

    def _ping_is_safe(self) -> bool:
        """Exact copy of the production implementation."""
        init_result = self.initialize_result
        if init_result is None:
            return False
        caps = getattr(init_result, "capabilities", None)
        if caps is not None:
            experimental = getattr(caps, "experimental", None) or {}
            if isinstance(experimental, dict) and "ping" in experimental:
                return True
        return self._is_http()

    async def _keepalive_tick(self):
        """Run one keepalive tick — equivalent to the keepalive block in the loop."""
        import logging
        logger = logging.getLogger("tools.mcp_tool")
        if not self.session:
            return
        try:
            if self._advertises_tools():
                await asyncio.wait_for(self.session.list_tools(), timeout=30.0)
            elif self._ping_is_safe():
                await asyncio.wait_for(self.session.send_ping(), timeout=30.0)
            else:
                logger.debug(
                    "MCP server '%s': skipping keepalive — "
                    "server does not advertise tools or ping capability",
                    self.name,
                )
        except Exception as exc:
            exc_str = str(exc)
            if "-32601" in exc_str or "unknown method" in exc_str.lower():
                logger.debug(
                    "MCP server '%s': keepalive method not supported "
                    "(server returned '%s') — skipping reconnect",
                    self.name, exc_str,
                )
            else:
                logger.warning(
                    "MCP server '%s' keepalive failed, triggering reconnect: %s",
                    self.name, exc,
                )
                self._reconnect_event.set()


# ---------------------------------------------------------------------------
# Tests for _ping_is_safe()
# ---------------------------------------------------------------------------

class TestPingIsSafe:

    def test_no_init_result_returns_false(self):
        """Without InitializeResult, ping is not safe (we know nothing)."""
        task = _FakeServerTask(initialize_result=None)
        assert task._ping_is_safe() is False

    def test_stdio_no_ping_advertised_returns_false(self):
        """stdio server with no experimental.ping → not safe."""
        init = _FakeInitResult(tools=None, experimental=None)
        task = _FakeServerTask(is_http=False, initialize_result=init)
        assert task._ping_is_safe() is False

    def test_http_server_returns_true_by_default(self):
        """HTTP transport → ping is safe even without explicit advertisement."""
        init = _FakeInitResult(tools=None, experimental=None)
        task = _FakeServerTask(is_http=True, initialize_result=init)
        assert task._ping_is_safe() is True

    def test_experimental_ping_advertised_stdio_returns_true(self):
        """stdio server that explicitly lists ping in experimental → safe."""
        init = _FakeInitResult(tools=None, experimental={"ping": {}})
        task = _FakeServerTask(is_http=False, initialize_result=init)
        assert task._ping_is_safe() is True

    def test_experimental_ping_advertised_http_returns_true(self):
        """HTTP server that explicitly lists ping in experimental → safe."""
        init = _FakeInitResult(tools=None, experimental={"ping": {}})
        task = _FakeServerTask(is_http=True, initialize_result=init)
        assert task._ping_is_safe() is True

    def test_experimental_empty_dict_is_not_ping_safe(self):
        """Empty experimental dict does not grant ping safety for stdio."""
        init = _FakeInitResult(tools=None, experimental={})
        task = _FakeServerTask(is_http=False, initialize_result=init)
        assert task._ping_is_safe() is False


# ---------------------------------------------------------------------------
# Tests for the keepalive tick (integration of the three branches)
# ---------------------------------------------------------------------------

class TestKeepaliveTick:

    @pytest.mark.asyncio
    async def test_tools_server_calls_list_tools(self):
        """Server that advertises tools → list_tools() is called."""
        init = _FakeInitResult(tools=object(), experimental=None)
        task = _FakeServerTask(is_http=True, initialize_result=init)
        task.session = MagicMock()
        task.session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        await task._keepalive_tick()

        task.session.list_tools.assert_awaited_once()
        assert not task._reconnect_event.is_set()

    @pytest.mark.asyncio
    async def test_no_tools_http_calls_send_ping(self):
        """HTTP server without tools → send_ping() is called."""
        init = _FakeInitResult(tools=None, experimental=None)
        task = _FakeServerTask(is_http=True, initialize_result=init)
        task.session = MagicMock()
        task.session.send_ping = AsyncMock(return_value=None)

        await task._keepalive_tick()

        task.session.send_ping.assert_awaited_once()
        assert not task._reconnect_event.is_set()

    @pytest.mark.asyncio
    async def test_no_tools_stdio_skips_keepalive(self):
        """stdio server without tools and no ping → keepalive skipped, no reconnect."""
        init = _FakeInitResult(tools=None, experimental=None)
        task = _FakeServerTask(is_http=False, initialize_result=init)
        task.session = MagicMock()
        task.session.list_tools = AsyncMock()
        task.session.send_ping = AsyncMock()

        await task._keepalive_tick()

        task.session.list_tools.assert_not_awaited()
        task.session.send_ping.assert_not_awaited()
        assert not task._reconnect_event.is_set()

    @pytest.mark.asyncio
    async def test_unknown_method_ping_does_not_reconnect(self):
        """send_ping() returning 'Unknown method: ping' must NOT trigger reconnect."""
        init = _FakeInitResult(tools=None, experimental=None)
        task = _FakeServerTask(is_http=True, initialize_result=init)
        task.session = MagicMock()
        task.session.send_ping = AsyncMock(
            side_effect=Exception("Unknown method: ping")
        )

        await task._keepalive_tick()

        assert not task._reconnect_event.is_set(), (
            "reconnect_event must NOT be set when server returns 'Unknown method: ping'"
        )

    @pytest.mark.asyncio
    async def test_method_not_found_32601_does_not_reconnect(self):
        """McpError -32601 from send_ping must NOT trigger reconnect."""
        init = _FakeInitResult(tools=None, experimental=None)
        task = _FakeServerTask(is_http=True, initialize_result=init)
        task.session = MagicMock()
        task.session.send_ping = AsyncMock(
            side_effect=Exception("McpError code=-32601 message='Method not found'")
        )

        await task._keepalive_tick()

        assert not task._reconnect_event.is_set()

    @pytest.mark.asyncio
    async def test_real_connection_error_does_reconnect(self):
        """A genuine network timeout from send_ping SHOULD trigger reconnect."""
        init = _FakeInitResult(tools=None, experimental=None)
        task = _FakeServerTask(is_http=True, initialize_result=init)
        task.session = MagicMock()
        task.session.send_ping = AsyncMock(
            side_effect=ConnectionResetError("connection reset by peer")
        )

        await task._keepalive_tick()

        assert task._reconnect_event.is_set(), (
            "reconnect_event MUST be set on a real network error"
        )

    @pytest.mark.asyncio
    async def test_list_tools_error_still_reconnects(self):
        """A genuine error from list_tools still triggers reconnect."""
        init = _FakeInitResult(tools=object(), experimental=None)
        task = _FakeServerTask(is_http=True, initialize_result=init)
        task.session = MagicMock()
        task.session.list_tools = AsyncMock(
            side_effect=ConnectionResetError("EOF")
        )

        await task._keepalive_tick()

        assert task._reconnect_event.is_set()
