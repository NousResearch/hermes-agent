"""Tests for capability-gated MCP tool discovery and keepalive.

Prompt-only / resource-only MCP servers do not implement the ``tools/*``
request family. Per the MCP spec, ``InitializeResult.capabilities.tools``
is non-None iff the server supports it. Before this fix, Hermes always
called ``tools/list`` during discovery and as the keepalive probe — both
raised ``McpError(-32601 Method not found)`` against such servers, so a
prompt-only server could never stay connected.

Ported from anomalyco/opencode#31271.
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

    async def test_keepalive_uses_list_tools_for_tool_capable_server(self):
        task = MCPServerTask("test")
        task.initialize_result = _caps(tools=SimpleNamespace())
        task.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])),
            send_ping=AsyncMock(),
        )

        reason = await self._run_one_keepalive_cycle(task)

        assert reason == "shutdown"
        task.session.list_tools.assert_awaited_once()
        task.session.send_ping.assert_not_called()
