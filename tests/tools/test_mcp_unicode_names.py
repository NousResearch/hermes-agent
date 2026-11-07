"""Unicode MCP names must remain distinct through registration, caching and dispatch."""

import asyncio
import re
from unittest.mock import AsyncMock

import pytest

from tools.mcp_tool_schema import mcp_prefixed_tool_name


@pytest.mark.parametrize(
    "pairs",
    [
        [("weather", "天气实况"), ("weather", "生活指数"), ("weather", "____")],
        [("天气", "search"), ("地图", "search"), ("__", "search")],
        [("weather", "预报" * 50 + tail) for tail in ("甲", "乙")],
        [("服务" * 50 + tail, "search") for tail in ("甲", "乙")],
    ],
)
def test_unicode_names_keep_distinct_stable_provider_valid_identities(pairs):
    names = [mcp_prefixed_tool_name(*pair) for pair in pairs]
    assert len(set(names)) == len(pairs)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in names)
    assert names == [mcp_prefixed_tool_name(*pair) for pair in pairs]


@pytest.mark.parametrize("from_cache", [False, True], ids=["live", "cached"])
def test_unicode_tools_register_and_dispatch_original_names(monkeypatch, tmp_path, from_cache):
    from mcp.types import CallToolResult, TextContent, Tool

    import tools.mcp_tool as mt
    import tools.registry as registry_module
    from tools import mcp_schema_cache, mcp_tool_handlers, mcp_tool_registration

    for attr in (
        "_lazy_server_tool_names", "_lazy_server_configs", "_lazy_server_fingerprints",
        "_mcp_tool_server_names", "_server_trust_levels", "_tool_read_only_hints",
        "_server_tool_scopes",
    ):
        monkeypatch.setattr(mt, attr, {})
    monkeypatch.setattr(mcp_schema_cache, "_cache_path", lambda: tmp_path / "mcp-cache.json")
    registry = registry_module.ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", registry)
    server_name = "weather"
    raw_names = ("天气实况", "生活指数", "天气预警", "限行数据", "空气质量指数")
    server = mt.MCPServerTask(server_name)
    server._tools = [
        Tool(name=name, description=name, inputSchema={
            "type": "object", "properties": {"cityId": {"type": "string"}},
            "required": ["cityId"],
        })
        for name in raw_names
    ]
    config = {"tools": {"resources": False, "prompts": False}}
    names = mcp_tool_registration._register_server_tools(server_name, server, config)
    if from_cache:
        entry = mcp_schema_cache.get_cached_entry(
            server_name, mcp_schema_cache.config_fingerprint(config))
        assert entry is not None
        registry = registry_module.ToolRegistry()
        monkeypatch.setattr(registry_module, "registry", registry)
        cached_names = mcp_tool_registration._register_from_cache_sync(server_name, config, entry)
        assert cached_names == names
        names = cached_names
    assert set(names) == {mcp_prefixed_tool_name(server_name, name) for name in raw_names}
    assert len(names) == len(raw_names)

    # Keep the real schema registration, handler closure and registry dispatcher;
    # replace only connection acquisition, the loop bridge and the network RPC.
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda *_: (server, None))
    monkeypatch.setattr(mcp_tool_handlers, "_dispatch",
                        lambda _name, _server, _op, call, *_args, **_kwargs: asyncio.run(call()))
    rpc = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="weather data")]))
    monkeypatch.setattr(mcp_tool_handlers, "_call_tool_racing_stdio_death", rpc)
    for original in raw_names:
        name = mcp_prefixed_tool_name(server_name, original)
        assert registry.get_schema(name)["parameters"]["required"] == ["cityId"]
        result = registry.dispatch(name, {"cityId": "example-city"})
        assert "weather data" in result
        rpc.assert_awaited_with(server, server_name, original, {"cityId": "example-city"})
