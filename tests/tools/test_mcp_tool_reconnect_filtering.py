"""Regression coverage for MCP include filters across reconnects."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


def _mcp_tool(name: str):
    return SimpleNamespace(
        name=name,
        description=f"Tool {name}",
        inputSchema={"type": "object", "properties": {}},
    )


def _register_raw_tool(registry, server_name: str, tool_name: str) -> str:
    prefixed = f"mcp__{server_name}__{tool_name}"
    registry.register(
        name=prefixed,
        toolset=f"mcp-{server_name}",
        schema={
            "name": prefixed,
            "description": f"Tool {tool_name}",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda *_args, **_kwargs: "{}",
    )
    return prefixed


def test_59_tool_include_reconciles_initial_and_repeated_reconnects():
    """Every tools/list result must replace stale full-surface registrations."""
    from tools.mcp_tool import MCPServerTask, _register_server_tools
    from tools.registry import ToolRegistry

    server_name = "zernio_regression"
    raw_tools = [_mcp_tool(f"tool_{index:03d}") for index in range(445)]
    included_names = {tool.name for tool in raw_tools[:59]}
    config = {
        "url": "https://mcp.example.test/mcp",
        "tools": {
            "include": sorted(included_names),
            "resources": False,
            "prompts": False,
        },
    }
    mock_registry = ToolRegistry()
    server = MCPServerTask(server_name)
    server._config = config
    server.session = SimpleNamespace(
        list_tools=AsyncMock(
            side_effect=[
                SimpleNamespace(tools=list(raw_tools)),
                SimpleNamespace(tools=list(raw_tools)),
                SimpleNamespace(tools=list(raw_tools)),
                SimpleNamespace(tools=list(raw_tools)),
            ]
        )
    )

    async def scenario():
        # Initial handshake: discovery must retain only the configured include,
        # then the outer startup path publishes exactly those 59 tools.
        await server._discover_tools()
        assert {tool.name for tool in server._tools} == included_names
        server._registered_tool_names = _register_server_tools(
            server_name, server, config
        )
        assert len(server._registered_tool_names) == 59
        assert len(mock_registry.get_tool_names_for_toolset(f"mcp-{server_name}")) == 59

        # Reproduce the production failure state: an older/unfiltered path left
        # all 445 server tools in the registry. Reconnect must remove the 386
        # filtered-out entries, not merely overwrite ownership tracking with 59.
        stale_full_surface = [
            _register_raw_tool(mock_registry, server_name, tool.name)
            for tool in raw_tools
        ]
        # The previous refresh bug then overwrote ownership tracking with only
        # the included names, orphaning the other 386 live registry entries.
        server._registered_tool_names = [
            name
            for name in stale_full_surface
            if name.rsplit("__", 1)[-1] in included_names
        ]
        assert len(server._registered_tool_names) == 59
        assert len(mock_registry.get_tool_names_for_toolset(f"mcp-{server_name}")) == 445

        for _ in range(2):
            # MCPServerTask.run() clears readiness before rebuilding a transport.
            server._ready.clear()
            await server._discover_tools()

            registered = mock_registry.get_tool_names_for_toolset(
                f"mcp-{server_name}"
            )
            assert {tool.name for tool in server._tools} == included_names
            assert len(registered) == 59
            assert len(registered) == len(set(registered))
            assert set(server._registered_tool_names) == set(registered)

        # Retry exhaustion parks the task by deregistering every tool. The
        # subsequent self-probe still represents a reconnect even though both
        # readiness and current ownership tracking are empty.
        server._deregister_tools()
        server._ready.clear()
        assert mock_registry.get_tool_names_for_toolset(f"mcp-{server_name}") == []

        await server._discover_tools()
        revived = mock_registry.get_tool_names_for_toolset(f"mcp-{server_name}")
        assert len(revived) == 59
        assert len(revived) == len(set(revived))
        assert set(server._registered_tool_names) == set(revived)

    with patch("tools.registry.registry", mock_registry), patch.dict(
        "tools.mcp_tool._mcp_tool_server_names", {}, clear=True
    ):
        asyncio.run(scenario())
