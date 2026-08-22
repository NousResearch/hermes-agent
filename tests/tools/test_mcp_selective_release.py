"""Selective MCP release used by ACP session/close."""

import asyncio

from tools import mcp_tool
from tools.registry import registry


def test_release_mcp_servers_shuts_down_only_selected_live_server(monkeypatch):
    calls = []

    class FakeServer:
        def __init__(self, name):
            self.name = name

        async def shutdown(self):
            calls.append(self.name)

        def _deregister_tools(self):
            calls.append(f"deregister:{self.name}")

    selected = FakeServer("selected")
    preserved = FakeServer("preserved")
    monkeypatch.setitem(mcp_tool._servers, "selected", selected)
    monkeypatch.setitem(mcp_tool._servers, "preserved", preserved)
    monkeypatch.setattr(
        mcp_tool,
        "_run_on_mcp_loop",
        lambda factory, timeout=30: asyncio.run(factory()),
    )

    released = mcp_tool.release_mcp_servers(["selected"])

    assert released == ["selected"]
    assert calls == ["selected"]
    assert "selected" not in mcp_tool._servers
    assert mcp_tool._servers["preserved"] is preserved


def test_release_failure_keeps_live_server_tracked(monkeypatch):
    class FailingServer:
        name = "failing"

        async def shutdown(self):
            raise RuntimeError("shutdown failed")

    server = FailingServer()
    monkeypatch.setitem(mcp_tool._servers, "failing", server)
    monkeypatch.setattr(
        mcp_tool,
        "_run_on_mcp_loop",
        lambda factory, timeout=30: asyncio.run(factory()),
    )

    try:
        import pytest

        with pytest.raises(RuntimeError, match="shutdown failed"):
            mcp_tool.release_mcp_servers(["failing"])
        assert mcp_tool._servers["failing"] is server
        assert "failing" not in mcp_tool._server_releasing
    finally:
        mcp_tool._servers.pop("failing", None)
        mcp_tool._server_releasing.discard("failing")


def test_release_mcp_servers_deregisters_lazy_tools(monkeypatch):
    server_name = "acp-release-lazy"
    tool_name = "mcp_acp_release_lazy_demo"
    schema = {
        "name": tool_name,
        "description": "test",
        "parameters": {"type": "object", "properties": {}},
    }
    registry.register(
        name=tool_name,
        toolset=f"mcp-{server_name}",
        schema=schema,
        handler=lambda _args: "{}",
    )
    monkeypatch.setitem(mcp_tool._lazy_server_configs, server_name, {"lazy": True})
    monkeypatch.setitem(mcp_tool._lazy_server_tool_names, server_name, [tool_name])
    mcp_tool._track_mcp_tool_server(tool_name, server_name)

    try:
        released = mcp_tool.release_mcp_servers([server_name])

        assert released == [server_name]
        assert registry.get_toolset_for_tool(tool_name) is None
        assert tool_name not in mcp_tool._mcp_tool_server_names
        assert server_name not in mcp_tool._lazy_server_configs
    finally:
        registry.deregister(tool_name)
        mcp_tool._forget_mcp_tool_server(tool_name)
        mcp_tool._lazy_server_configs.pop(server_name, None)
        mcp_tool._lazy_server_tool_names.pop(server_name, None)
