"""Live MCP servers follow ``mcp_servers`` as it is on disk: an entry removed (or disabled) after
boot is torn down instead of self-probing for the life of the process."""

import asyncio

import pytest


@pytest.mark.no_isolate
def test_reconcile_tears_down_server_dropped_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as _loop
    from tools.mcp_tool import MCPServerTask

    configured = {"linear": {"url": "https://mcp.example.test/mcp", "auth": "oauth"}}
    monkeypatch.setattr(_config, "_load_mcp_config", lambda: dict(configured))
    discovered: list = []
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda *a, **k: discovered.append(1) or [])

    _loop._ensure_mcp_loop()
    srv = MCPServerTask("linear")

    async def _park():
        srv._task = asyncio.ensure_future(srv._wait_for_reconnect_or_shutdown())

    asyncio.run_coroutine_threadsafe(_park(), mcp_tool._mcp_loop).result(5)
    with mcp_tool._lock:
        mcp_tool._servers["linear"] = srv
        mcp_tool._server_scope_keys["linear"] = None
    try:
        assert disc.reconcile_mcp_servers_with_config() == {"removed": [], "added": []}
        assert "linear" in mcp_tool._servers and not discovered

        configured.clear()  # user deletes the entry
        assert disc.reconcile_mcp_servers_with_config()["removed"] == ["linear"]
        assert "linear" not in mcp_tool._servers
        assert srv._shutdown_event.is_set()

        configured["notion"] = {"url": "https://mcp.notion.test/mcp"}
        assert disc.reconcile_mcp_servers_with_config()["added"] == ["notion"]
        assert discovered, "a newly configured server must go through discovery"
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop("linear", None)
            mcp_tool._server_scope_keys.pop("linear", None)
        _loop._stop_mcp_loop()


def test_disabled_entry_counts_as_dropped(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_lifecycle as _lifecycle

    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"linear": {"url": "https://x/mcp", "enabled": False}})
    torn_down: list = []
    monkeypatch.setattr(_lifecycle, "shutdown_mcp_servers", lambda **kw: torn_down.append(kw))
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda *a, **k: [])
    with mcp_tool._lock:
        mcp_tool._servers["linear"] = object()
        mcp_tool._server_scope_keys["linear"] = None
    try:
        assert disc.reconcile_mcp_servers_with_config()["removed"] == ["linear"]
        assert torn_down == [{"scope": None, "names": {"linear"}}]
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop("linear", None)
            mcp_tool._server_scope_keys.pop("linear", None)
