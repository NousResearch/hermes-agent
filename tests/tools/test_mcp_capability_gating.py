"""Tests for capability-gated MCP tool discovery and keepalive.

Prompt-only / resource-only MCP servers do not implement the ``tools/*``
request family. Per the MCP spec, ``InitializeResult.capabilities.tools``
is non-None iff the server supports it. Before the capability gate, Hermes
always called ``tools/list`` during discovery, which raised
``McpError(-32601 Method not found)`` against such servers, so a prompt-only
server could never stay connected. Discovery/refresh remain capability-gated.

The keepalive probe uses ``ping`` (MCP base-protocol liveness) for every
server regardless of capability: it works uniformly and stays a few bytes
instead of pulling the full ``tools/list`` payload (which is ~1 MB on large
servers like Unreal Engine's editor MCP). Its cadence is configurable via
``keepalive_interval`` so servers with short session TTLs stay alive.

Discovery gating ported from anomalyco/opencode#31271.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools.mcp_tool import MCPServerTask


def _caps(tools=None, prompts=None, resources=None):
    """Build a fake InitializeResult with the given capability sub-objects."""
    return SimpleNamespace(
        capabilities=SimpleNamespace(tools=tools, prompts=prompts, resources=resources)
    )


class TestAdvertisesTools:
    def test_true_when_tools_capability_present(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace(listChanged=True))
        assert task._advertises_tools() is True

    def test_false_for_prompt_only_server(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(prompts=SimpleNamespace(listChanged=None))
        assert task._advertises_tools() is False

    def test_false_for_resource_only_server(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(resources=SimpleNamespace())
        assert task._advertises_tools() is False

    def test_legacy_fallback_no_initialize_result(self):
        """No captured capabilities → preserve old always-list_tools behavior."""
        task = MCPServerTask("test")
        assert task.initialize_result is None
        assert task._advertises_tools() is True

    def test_legacy_fallback_no_capabilities_attr(self):
        task = MCPServerTask("test")
        task.initialize_result = SimpleNamespace()  # no .capabilities
        assert task._advertises_tools() is True


@pytest.mark.asyncio
class TestDiscoverToolsGating:
    async def test_skips_list_tools_for_prompt_only_server(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(prompts=SimpleNamespace())
        task.session = SimpleNamespace(list_tools=AsyncMock())
        task._tools = ["stale"]

        await task._discover_tools()

        task.session.list_tools.assert_not_called()
        assert task._tools == []

    async def test_calls_list_tools_for_tool_capable_server(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        fake_tool = SimpleNamespace(name="echo")
        task.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[fake_tool]))
        )

        await task._discover_tools()

        task.session.list_tools.assert_awaited_once()
        assert task._tools == [fake_tool]

    async def test_legacy_fallback_still_calls_list_tools(self):
        task = MCPServerTask("test")
        task.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[]))
        )

        await task._discover_tools()

        task.session.list_tools.assert_awaited_once()


@pytest.mark.asyncio
class TestRefreshToolsGating:
    async def test_refresh_noop_for_prompt_only_server(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(prompts=SimpleNamespace())
        task.session = SimpleNamespace(list_tools=AsyncMock())

        await task._refresh_tools()

        task.session.list_tools.assert_not_called()


@pytest.mark.asyncio
class TestKeepaliveProbe:
    async def _run_one_keepalive_cycle(self, task):
        """Drive _wait_for_lifecycle_event through exactly one keepalive
        timeout, then fire shutdown so it returns."""
        real_wait = asyncio.wait
        cycles = {"n": 0}

        async def fake_wait(tasks, timeout=None, return_when=None):
            cycles["n"] += 1
            if cycles["n"] == 1:
                # Simulate keepalive timeout: nothing completed.
                return set(), set(tasks)
            # Second cycle: let shutdown win.
            task._shutdown_event.set()
            return await real_wait(
                tasks, timeout=0.5, return_when=return_when or asyncio.FIRST_COMPLETED
            )

        import tools.mcp_tool as mcp_mod
        orig = mcp_mod.asyncio.wait
        mcp_mod.asyncio.wait = fake_wait
        try:
            return await task._wait_for_lifecycle_event()
        finally:
            mcp_mod.asyncio.wait = orig

    async def test_keepalive_uses_ping_for_prompt_only_server(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(prompts=SimpleNamespace())
        task.session = SimpleNamespace(
            list_tools=AsyncMock(),
            send_ping=AsyncMock(),
        )

        reason = await self._run_one_keepalive_cycle(task)

        assert reason == "shutdown"
        task.session.send_ping.assert_awaited_once()
        task.session.list_tools.assert_not_called()

    async def test_keepalive_uses_ping_for_tool_capable_server(self):
        """Keepalive uses ``ping`` even for tool-capable servers, so the probe
        stays a few bytes regardless of tool count (no ``list_tools`` payload).
        Tool-list changes still arrive via tools/list_changed notifications."""
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        task.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])),
            send_ping=AsyncMock(),
        )

        reason = await self._run_one_keepalive_cycle(task)

        assert reason == "shutdown"
        task.session.send_ping.assert_awaited_once()
        task.session.list_tools.assert_not_called()

    async def test_keepalive_uses_ping_legacy_fallback(self):
        """No captured capabilities → still pings (no spurious list_tools)."""
        task = MCPServerTask("test")
        assert task.initialize_result is None
        task.session = SimpleNamespace(
            list_tools=AsyncMock(),
            send_ping=AsyncMock(),
        )

        reason = await self._run_one_keepalive_cycle(task)

        assert reason == "shutdown"
        task.session.send_ping.assert_awaited_once()
        task.session.list_tools.assert_not_called()


class TestKeepaliveInterval:
    """The keepalive cadence is configurable so servers with short session
    TTLs (e.g. Unreal Engine editor MCP, ~15s) can refresh fast enough to keep
    the session alive instead of hitting an expired session on every idle call.
    """

    async def _captured_interval(self, config):
        """Run one keepalive cycle and capture the ``asyncio.wait`` timeout."""
        task = MCPServerTask("test")
        task._config = config
        task.session = SimpleNamespace(send_ping=AsyncMock())
        captured = {}
        real_wait = asyncio.wait

        async def fake_wait(tasks, timeout=None, return_when=None):
            captured["timeout"] = timeout
            task._shutdown_event.set()
            return await real_wait(
                tasks, timeout=0.5, return_when=return_when or asyncio.FIRST_COMPLETED
            )

        import tools.mcp_tool as mcp_mod
        orig = mcp_mod.asyncio.wait
        mcp_mod.asyncio.wait = fake_wait
        try:
            await task._wait_for_lifecycle_event()
        finally:
            mcp_mod.asyncio.wait = orig
        return captured["timeout"]

    @pytest.mark.asyncio
    async def test_default_interval_when_unset(self):
        from tools.mcp_tool import _DEFAULT_KEEPALIVE_INTERVAL
        assert await self._captured_interval({}) == _DEFAULT_KEEPALIVE_INTERVAL

    @pytest.mark.asyncio
    async def test_configured_interval_honored(self):
        assert await self._captured_interval({"keepalive_interval": 10}) == 10

    @pytest.mark.asyncio
    async def test_interval_clamped_to_floor(self):
        from tools.mcp_tool import _MIN_KEEPALIVE_INTERVAL
        # A sub-floor value must clamp up, never busy-loop the keepalive.
        assert (
            await self._captured_interval({"keepalive_interval": 0.1})
            == _MIN_KEEPALIVE_INTERVAL
        )


def _mcp_error(code, message="boom"):
    """Build a real McpError carrying a JSON-RPC error code."""
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData
    return McpError(ErrorData(code=code, message=message))


class TestMethodNotFoundDetection:
    """``_is_method_not_found_error`` underpins the ping→list_tools fallback."""

    def test_structural_code_match(self):
        from tools.mcp_tool import _is_method_not_found_error
        assert _is_method_not_found_error(_mcp_error(-32601)) is True

    def test_other_mcp_error_code_is_not_match(self):
        from tools.mcp_tool import _is_method_not_found_error
        # Invalid params (-32602) is a real error, NOT "ping unsupported".
        assert _is_method_not_found_error(_mcp_error(-32602)) is False

    def test_substring_fallback(self):
        from tools.mcp_tool import _is_method_not_found_error
        assert _is_method_not_found_error(Exception("Method not found")) is True

    def test_unknown_method_phrasing_is_match(self):
        # agentmemory's MCP server surfaces method-not-found as a plain
        # "Unknown method: ping" string with no structural -32601 code (#50028).
        from tools.mcp_tool import _is_method_not_found_error
        assert _is_method_not_found_error(Exception("Unknown method: ping")) is True

    def test_unrelated_exception_is_not_match(self):
        from tools.mcp_tool import _is_method_not_found_error
        assert _is_method_not_found_error(TimeoutError()) is False
        assert _is_method_not_found_error(Exception("session terminated")) is False


@pytest.mark.asyncio
class TestKeepaliveProbeFallback:
    """The probe prefers ``ping`` but falls back to ``list_tools`` for servers
    that don't implement the optional ping utility — without reconnect-looping,
    and without regressing servers that DO support ping."""

    async def test_uses_ping_when_supported(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        task.session = SimpleNamespace(
            send_ping=AsyncMock(),
            list_tools=AsyncMock(),
        )

        await task._keepalive_probe()

        task.session.send_ping.assert_awaited_once()
        task.session.list_tools.assert_not_called()
        assert task._ping_unsupported is False

    async def test_falls_back_to_list_tools_on_method_not_found(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        task.session = SimpleNamespace(
            send_ping=AsyncMock(side_effect=_mcp_error(-32601)),
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])),
        )

        await task._keepalive_probe()

        # First cycle: ping tried, failed -32601, list_tools used as fallback.
        task.session.send_ping.assert_awaited_once()
        task.session.list_tools.assert_awaited_once()
        assert task._ping_unsupported is True

    async def test_falls_back_on_unknown_method_string(self):
        """Regression for #50028: a server that surfaces method-not-found as a
        plain "Unknown method: ping" string (no structural -32601 code) must
        still latch the fallback and use list_tools, NOT reconnect-loop."""
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        task.session = SimpleNamespace(
            send_ping=AsyncMock(side_effect=Exception("Unknown method: ping")),
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])),
        )

        await task._keepalive_probe()

        task.session.send_ping.assert_awaited_once()
        task.session.list_tools.assert_awaited_once()
        assert task._ping_unsupported is True

    async def test_latch_skips_ping_on_subsequent_cycles(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        task.session = SimpleNamespace(
            send_ping=AsyncMock(side_effect=_mcp_error(-32601)),
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])),
        )

        await task._keepalive_probe()  # latches _ping_unsupported
        await task._keepalive_probe()  # should NOT ping again

        task.session.send_ping.assert_awaited_once()  # only the first cycle
        assert task.session.list_tools.await_count == 2

    async def test_real_liveness_failure_propagates_not_swallowed(self):
        """A non-(-32601) ping error is a genuine connection failure: it must
        propagate so the caller reconnects, and must NOT latch the fallback."""
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        task.session = SimpleNamespace(
            send_ping=AsyncMock(side_effect=Exception("session terminated")),
            list_tools=AsyncMock(),
        )

        with pytest.raises(Exception, match="session terminated"):
            await task._keepalive_probe()

        task.session.list_tools.assert_not_called()
        assert task._ping_unsupported is False

    async def test_no_ping_no_tools_propagates_method_not_found(self):
        """A server advertising neither working ping nor tools has no cheaper
        probe — the -32601 must propagate rather than calling list_tools on a
        server that doesn't support it."""
        task = MCPServerTask("test")
        task.initialize_result = _caps(prompts=SimpleNamespace())  # not tool-capable
        task.session = SimpleNamespace(
            send_ping=AsyncMock(side_effect=_mcp_error(-32601)),
            list_tools=AsyncMock(),
        )

        with pytest.raises(Exception):
            await task._keepalive_probe()

        task.session.list_tools.assert_not_called()

    async def test_discover_resets_latch(self):
        """A fresh connection (_discover_tools) re-enables the cheap ping path."""
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        task._ping_unsupported = True
        task.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])),
        )

        await task._discover_tools()

        assert task._ping_unsupported is False


