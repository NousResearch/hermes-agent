"""Explicit reload confirms live manifests even for no-TTL lazy cache entries."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from mcp.types import Tool
from tools import mcp_schema_cache as cache
from tools import mcp_tool as core
from tools import mcp_tool_discovery as discovery
from tools import mcp_tool_registration as registration
from tools.registry import ToolRegistry


def _tool(name, field):
    return Tool(name=name, description=field, inputSchema={"type": "object", "properties": {field: {"type": "string"}}, "required": [field]})


def test_explicit_refresh_replaces_no_ttl_manifest_and_keeps_last_good_on_failure(tmp_path, monkeypatch):
    # Real cache files and registration, with only the external transport replaced.
    cfg = {"probe": {"command": "never-start-provider", "lazy": True}}
    fp = cache.config_fingerprint(cfg["probe"])
    monkeypatch.setattr(discovery, "_select_new_servers", lambda configs: dict(configs))
    monkeypatch.setattr(registration, "register_connected_into_current_scope", lambda configs: False)
    monkeypatch.setattr(discovery._loop, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(discovery, "_log_summary", lambda *a, **k: None)
    monkeypatch.setattr(registration, "_existing_tool_names", lambda: registry.get_all_tool_names())
    monkeypatch.setattr(core, "_lazy_server_configs", {})
    monkeypatch.setattr(core, "_lazy_server_fingerprints", {})
    monkeypatch.setattr(core, "_lazy_server_tool_names", {})
    monkeypatch.setattr(core, "_server_connecting", set())
    live_calls = []
    registry = ToolRegistry()
    monkeypatch.setattr("tools.registry.registry", registry)
    fail = False

    def connect(configs):
        live_calls.append(cache._cache_path())
        if fail:
            return  # failed discovery has no successful registration/write-through
        server = core.MCPServerTask("probe")
        server.session = MagicMock()
        server._tools = [_tool("same", "new_field"), _tool("added", "new_field")]
        registration._register_server_tools("probe", server, configs["probe"])

    monkeypatch.setattr(discovery, "_run_discovery_pass", connect)
    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        monkeypatch.setenv("HERMES_HOME", str(home))
        cache.write_cache_entry("probe", fp, tools=[{"name": "same", "description": "old", "inputSchema": {}}, {"name": "removed", "description": "old", "inputSchema": {}}])
    for home in (homes[0], homes[1], homes[0]):
        other = homes[1] if home == homes[0] else homes[0]
        other_path = other / "cache" / "mcp_schema_cache.json"
        other_before = other_path.read_bytes()
        monkeypatch.setenv("HERMES_HOME", str(home))
        count_before = len(live_calls)
        discovery._register_mcp_servers(cfg)  # lazy normal startup still avoids transport
        assert len(live_calls) == count_before
        # Existing reload teardown deregisters the old owned tool names first.
        for name in list(registry.get_all_tool_names()):
            registry.deregister(name)
        discovery._register_mcp_servers(cfg, force_refresh=True)
        assert "mcp__probe__removed" not in registry.get_all_tool_names()
        assert "mcp__probe__added" in registry.get_all_tool_names()
        assert registry.get_schema("mcp__probe__same")["parameters"]["required"] == ["new_field"]
        assert {t["name"] for t in cache.get_cached_entry("probe", fp)["tools"]} == {"same", "added"}
        assert other_path.read_bytes() == other_before
    assert len(live_calls) == 3
    before = cache._cache_path().read_bytes()
    fail = True
    discovery._register_mcp_servers(cfg, force_refresh=True)
    assert cache._cache_path().read_bytes() == before
    assert registry.get_schema("mcp__probe__same")["parameters"]["required"] == ["new_field"]


def test_public_discovery_passes_explicit_refresh_without_changing_config(monkeypatch):
    cfg = {"probe": {"command": "never-start-provider", "lazy": True}}
    monkeypatch.setattr(discovery, "_owner_secret_scope", MagicMock())
    monkeypatch.setattr(discovery._config, "_load_mcp_config", lambda: cfg)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(discovery, "_acquire_discovery_lock_with_retry", lambda: None)
    monkeypatch.setattr(discovery, "_log_summary", lambda *a, **k: None)
    monkeypatch.setattr(core, "_servers", {})
    monkeypatch.setattr(core, "_server_connecting", set())
    register = MagicMock(return_value=[])
    monkeypatch.setattr(discovery, "register_mcp_servers", register)
    discovery.discover_mcp_tools(force_refresh=True)
    register.assert_called_once_with(cfg, force_refresh=True)
    assert cfg["probe"]["lazy"] is True
