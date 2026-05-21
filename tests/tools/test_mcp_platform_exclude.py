"""Tests for per-platform MCP tool exclusion at runtime.

Covers the read-path half of the fix: ``get_tool_definitions`` must drop MCP
tools that a platform has scoped out via ``mcp_excludes`` (resolved from
``platform_mcp_excludes`` in config), while keeping them for platforms that
have not. The write-path half is covered in
``tests/hermes_cli/test_tools_disable_enable.py``.

All tests use mocks -- no real MCP servers or subprocesses are started.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import model_tools
from tools.registry import ToolRegistry
from tools.mcp_tool import (
    _discover_and_register_server,
    _mcp_tool_origins,
    _servers,
    get_mcp_tool_origin,
)


def _make_mcp_tool(name):
    """Create a fake MCP Tool object matching the SDK interface."""
    return SimpleNamespace(
        name=name,
        description=name,
        inputSchema={"type": "object", "properties": {}},
    )


@pytest.fixture
def mcp_env():
    """Register fake MCP servers, exposing a builder, and tear them down.

    Server names are unique per test instance so concurrent xdist workers
    never collide on the module-global ``_servers`` / ``_mcp_tool_origins``.
    """
    registry = ToolRegistry()
    spawned = []  # (server_name, [registered_tool_names])

    def register(server_name, tool_names):
        server_tools = [_make_mcp_tool(n) for n in tool_names]

        async def fake_connect(_name, _config):
            from tools.mcp_tool import MCPServerTask
            server = MCPServerTask(_name)
            server.session = SimpleNamespace()
            server._tools = server_tools
            return server

        async def run():
            with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
                 patch("tools.registry.registry", registry):
                return await _discover_and_register_server(server_name, {"url": "https://x"})

        registered = asyncio.run(run())
        spawned.append((server_name, registered))
        return registered

    def tool_names(enabled_toolsets, mcp_excludes=None, clear_cache=True):
        if clear_cache:
            model_tools._tool_defs_cache.clear()
        with patch("model_tools.registry", registry), \
             patch("tools.registry.registry", registry):
            defs = model_tools.get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                quiet_mode=True,
                mcp_excludes=mcp_excludes,
            )
        return {d["function"]["name"] for d in defs}

    env = SimpleNamespace(registry=registry, register=register, tool_names=tool_names)
    try:
        yield env
    finally:
        model_tools._tool_defs_cache.clear()
        for server_name, registered in spawned:
            _servers.pop(server_name, None)
            for n in registered:
                _mcp_tool_origins.pop(n, None)


class TestMcpToolOriginTracking:
    """Registration must record the (server, raw_tool_name) provenance the
    runtime resolver needs for ``server:tool`` matching."""

    def test_origin_captured_on_registration(self, mcp_env):
        mcp_env.register("originsrv", ["get_quote", "get_news"])
        assert get_mcp_tool_origin("mcp_originsrv_get_quote") == ("originsrv", "get_quote")
        assert get_mcp_tool_origin("mcp_originsrv_get_news") == ("originsrv", "get_news")

    def test_origin_none_for_unknown_tool(self):
        assert get_mcp_tool_origin("totally_not_a_tool") is None


class TestRuntimeMcpExclusion:
    """get_tool_definitions must honor the resolved mcp_excludes map."""

    def test_no_excludes_keeps_all_mcp_tools(self, mcp_env):
        mcp_env.register("noex", ["get_quote", "get_news"])
        names = mcp_env.tool_names(["noex"], mcp_excludes=None)
        assert "mcp_noex_get_quote" in names
        assert "mcp_noex_get_news" in names

    def test_scoped_exclude_drops_only_listed_tool(self, mcp_env):
        mcp_env.register("scopedex", ["get_quote", "get_news"])
        names = mcp_env.tool_names(["scopedex"], mcp_excludes={"scopedex": {"get_quote"}})
        assert "mcp_scopedex_get_quote" not in names
        assert "mcp_scopedex_get_news" in names

    def test_wildcard_exclude_drops_entire_server(self, mcp_env):
        mcp_env.register("wildex", ["get_quote", "get_news", "get_chart"])
        names = mcp_env.tool_names(["wildex"], mcp_excludes={"wildex": {"*"}})
        assert not any(n.startswith("mcp_wildex_") for n in names)

    def test_exclude_for_one_server_leaves_others_intact(self, mcp_env):
        mcp_env.register("exca", ["get_quote"])
        mcp_env.register("excb", ["place_order"])
        names = mcp_env.tool_names(["exca", "excb"], mcp_excludes={"exca": {"*"}})
        assert "mcp_exca_get_quote" not in names
        assert "mcp_excb_place_order" in names

    def test_cron_excludes_server_but_cli_keeps_it(self, mcp_env):
        """End-to-end of the production fix: a server scoped out of cron is
        absent for cron but present for cli."""
        from hermes_cli.tools_config import resolve_mcp_excludes

        config = {
            "mcp_servers": {"croncli": {"url": "https://x"}},
            "platform_mcp_excludes": {"cron": {"croncli": ["*"]}},
        }
        mcp_env.register("croncli", ["get_quote", "get_news"])
        cron_names = mcp_env.tool_names(
            ["croncli"], mcp_excludes=resolve_mcp_excludes(config, "cron")
        )
        cli_names = mcp_env.tool_names(
            ["croncli"], mcp_excludes=resolve_mcp_excludes(config, "cli")
        )
        assert not any(n.startswith("mcp_croncli_") for n in cron_names)
        assert "mcp_croncli_get_quote" in cli_names
        assert "mcp_croncli_get_news" in cli_names

    def test_global_exclude_applies_to_every_platform(self, mcp_env):
        """Backward compat: a global mcp_servers.<name>.tools.exclude resolves
        for all platforms (cron AND cli)."""
        from hermes_cli.tools_config import resolve_mcp_excludes

        config = {
            "mcp_servers": {
                "globex": {"url": "https://x", "tools": {"exclude": ["get_quote"]}}
            },
        }
        mcp_env.register("globex", ["get_quote", "get_news"])
        for platform in ("cron", "cli", "telegram"):
            names = mcp_env.tool_names(
                ["globex"], mcp_excludes=resolve_mcp_excludes(config, platform)
            )
            assert "mcp_globex_get_quote" not in names, platform
            assert "mcp_globex_get_news" in names, platform

    def test_cache_key_distinguishes_excludes(self, mcp_env):
        """The quiet_mode memo must key on mcp_excludes — without it, a cron
        call would get a stale cli result cached under the same key."""
        mcp_env.register("cachekey", ["get_quote"])
        model_tools._tool_defs_cache.clear()
        # First call (no excludes) populates the cache.
        cli_names = mcp_env.tool_names(["cachekey"], mcp_excludes=None, clear_cache=False)
        # Second call with a different exclude map must NOT hit the cli entry.
        cron_names = mcp_env.tool_names(
            ["cachekey"], mcp_excludes={"cachekey": {"*"}}, clear_cache=False
        )
        assert "mcp_cachekey_get_quote" in cli_names
        assert "mcp_cachekey_get_quote" not in cron_names
