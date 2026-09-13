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
        assert disc.reconcile_mcp_servers_with_config() == {"removed": [], "added": [], "pending": []}
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


def test_disabled_lazy_and_connecting_entries(monkeypatch, tmp_path):
    """``enabled: false`` counts as dropped; a schema-cache (lazy) registration loses its cached
    tools; a server still mid-connect is reported ``pending`` (torn down on a later pass)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_lifecycle as _lifecycle
    from tools import mcp_tool_registration as _registration

    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"linear": {"url": "https://x/mcp", "enabled": False}})
    torn_down: list = []
    deregistered: list = []
    monkeypatch.setattr(_lifecycle, "shutdown_mcp_servers", lambda **kw: torn_down.append(kw))
    monkeypatch.setattr(_registration, "_deregister_mcp_tool_all_scopes", lambda key, name: deregistered.append(name))
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda *a, **k: [])
    with mcp_tool._lock:
        mcp_tool._servers["linear"] = object()
        mcp_tool._server_scope_keys["linear"] = None
        mcp_tool._server_scope_keys["notion"] = None
        mcp_tool._server_connecting.add("notion")
        mcp_tool._lazy_server_configs["asana"] = {"url": "https://a/mcp"}
        mcp_tool._lazy_server_tool_names["asana"] = ["mcp__asana__list"]
    try:
        result = disc.reconcile_mcp_servers_with_config()
        assert result == {"removed": ["linear", "asana"], "added": [], "pending": ["notion"]}
        assert torn_down == [{"scope": None, "names": {"linear"}}]
        assert deregistered == ["mcp__asana__list"] and "asana" not in mcp_tool._lazy_server_configs
    finally:
        with mcp_tool._lock:
            for name in ("linear", "notion"):
                mcp_tool._servers.pop(name, None)
                mcp_tool._server_scope_keys.pop(name, None)
            mcp_tool._server_connecting.discard("notion")
            mcp_tool._lazy_server_configs.pop("asana", None)
            mcp_tool._lazy_server_tool_names.pop("asana", None)
