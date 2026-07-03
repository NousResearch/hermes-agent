"""Behavior-contract tests for lazy MCP server startup."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    import tools.mcp_tool as mcp

    old_servers = dict(mcp._servers)
    old_lazy = dict(mcp._lazy_server_configs)
    old_fps = dict(mcp._lazy_server_fingerprints)
    old_connecting = set(mcp._server_connecting)
    yield
    mcp._servers.clear()
    mcp._servers.update(old_servers)
    mcp._lazy_server_configs.clear()
    mcp._lazy_server_configs.update(old_lazy)
    mcp._lazy_server_fingerprints.clear()
    mcp._lazy_server_fingerprints.update(old_fps)
    mcp._server_connecting.clear()
    mcp._server_connecting.update(old_connecting)


def _fake_cache_entry():
    return {
        "fingerprint": "abc",
        "tools": [
            {
                "name": "browser_navigate",
                "description": "Navigate",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
        "utility_tools": [],
    }


class TestLazyMcpRegistration:
    def test_registers_from_cache_without_connect(self, monkeypatch):
        config = {
            "playwright": {
                "command": "npx",
                "args": ["-y", "@playwright/mcp"],
                "lazy": True,
            }
        }
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=config), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", return_value=_fake_cache_entry()), \
             patch(
                 "tools.mcp_tool._register_from_cache_sync",
                 return_value=["mcp_playwright_browser_navigate"],
             ) as mock_register, \
             patch("tools.mcp_tool._discover_and_register_server", new_callable=AsyncMock) as mock_discover, \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run:

            from tools.mcp_tool import register_mcp_servers

            register_mcp_servers(config)

        mock_register.assert_called_once()
        mock_discover.assert_not_called()
        mock_run.assert_not_called()

    def test_cache_fingerprint_mismatch_falls_back_to_connect(self, monkeypatch):
        config = {
            "playwright": {
                "command": "npx",
                "args": ["-y", "@playwright/mcp"],
                "lazy": True,
            }
        }
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=config), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", return_value=None), \
             patch("tools.mcp_tool._discover_and_register_server", new_callable=AsyncMock) as mock_discover, \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop") as mock_run:

            mock_discover.return_value = ["mcp_playwright_browser_navigate"]
            mock_run.return_value = {"playwright": ["mcp_playwright_browser_navigate"]}

            from tools.mcp_tool import register_mcp_servers

            register_mcp_servers(config)

        mock_run.assert_called_once()

    def test_handler_lazy_connects_on_first_call(self):
        import tools.mcp_tool as mcp

        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)
        mcp._lazy_server_fingerprints["playwright"] = "abc"
        mcp._servers["playwright"] = SimpleNamespace(session=None, _rpc_lock=MagicMock())

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(isError=False, content=[])
        )
        connected = SimpleNamespace(
            session=mock_session,
            _rpc_lock=MagicMock(),
        )
        connected._rpc_lock.__aenter__ = AsyncMock(return_value=None)
        connected._rpc_lock.__aexit__ = AsyncMock(return_value=None)

        def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        with patch.object(mcp, "_ensure_server_connected", side_effect=_connect) as mock_connect, \
             patch.object(mcp, "_run_on_mcp_loop") as mock_run, \
             patch.object(mcp, "_load_mcp_config", return_value={"playwright": config}), \
             patch.object(mcp, "_resolve_server_lazy", return_value=True):

            def _run_call(coro_or_factory, timeout=120):
                import asyncio

                coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            mock_run.side_effect = _run_call
            handler = mcp._make_tool_handler("playwright", "browser_navigate", 5)
            out = handler({}, task_id="t1")

        mock_connect.assert_called_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload
        assert payload.get("result") == ""

    def test_check_fn_passes_for_lazy_cached_server(self, monkeypatch):
        import tools.mcp_tool as mcp

        mcp._lazy_server_configs["playwright"] = {"lazy": True}
        mcp._lazy_server_fingerprints["playwright"] = "abc"
        with patch("tools.mcp_schema_cache.has_cached_entry", return_value=True):
            assert mcp._make_check_fn("playwright")() is True
