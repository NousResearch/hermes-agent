"""A scoped MCP refresh retains callable manifests for dormant and lazy transports."""
import sys
from contextlib import contextmanager

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    import tools.mcp_tool as core
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    homes = {name: tmp_path / name for name in ("a", "b")}
    for home in homes.values():
        home.mkdir()

    @contextmanager
    def enter(name):
        token = set_hermes_home_override(homes[name])
        try:
            yield hermes_home_key(homes[name])
        finally:
            reset_hermes_home_override(token)

    yield enter
    for home in homes.values():
        scope = hermes_home_key(home)
        shutdown_mcp_servers(scope=scope)
        registry.deregister("mcp__dormant_fixture__echo", scope=scope)


def seed_server(tmp_path):
    from tools.mcp_schema_cache import config_fingerprint, write_cache_entry

    peer = tmp_path / "peer.py"
    peer.write_text("""import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if 'id' not in request:
        continue
    method = request.get('method')
    if method == 'initialize':
        result = {'protocolVersion': request['params']['protocolVersion'], 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'fixture', 'version': '1'}}
    elif method == 'tools/list':
        result = {'tools': [{'name': 'echo', 'description': 'Fixture', 'inputSchema': {'type': 'object', 'properties': {}}}]}
    elif method == 'tools/call':
        result = {'content': [{'type': 'text', 'text': 'DORMANT_READBACK'}]}
    else:
        result = {}
    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
""")
    config = {"command": sys.executable, "args": [str(peer)], "lazy": True, "connect_timeout": 5,
              "tools": {"resources": False, "prompts": False}}
    write_cache_entry("dormant_fixture", config_fingerprint(config), tools=[
        {"name": "echo", "description": "Fixture", "inputSchema": {"type": "object", "properties": {}}}])
    return config


@pytest.mark.parametrize("state", ["cached", "idle-recycled", "lost-overlay"])
def test_refresh_retains_callable_tools_and_repairs_missing_overlay(profiles, tmp_path, state):
    import tools.mcp_tool as core
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.registry import registry
    from toolsets import resolve_toolset

    with profiles("a") as scope:
        config = seed_server(tmp_path)
        register_mcp_servers({"dormant_fixture": config})
        if state != "cached":
            assert "DORMANT_READBACK" in registry.dispatch("mcp__dormant_fixture__echo", {})
            server = core._servers[(scope, "dormant_fixture")]
            # Exercise the same lifecycle state as the idle timer without a wall-clock sleep.
            server._mark_stdio_recycled("idle_timeout_seconds")
            if state == "lost-overlay":
                from tools.mcp_tool_registration import _remove_server_scope
                _remove_server_scope((scope, "dormant_fixture"), scope)
        register_mcp_servers({"dormant_fixture": config})
        assert resolve_toolset("dormant_fixture") == ["mcp__dormant_fixture__echo"]
        assert registry.get_entry("mcp__dormant_fixture__echo") is not None
        assert "DORMANT_READBACK" in registry.dispatch("mcp__dormant_fixture__echo", {})


def test_sibling_lazy_teardown_keeps_other_profiles_alias(profiles, tmp_path):
    from tools.mcp_tool_discovery import _forget_lazy_server, register_mcp_servers
    from toolsets import resolve_toolset

    with profiles("a"):
        config = seed_server(tmp_path)
        register_mcp_servers({"dormant_fixture": config})
    with profiles("b") as scope:
        seed_server(tmp_path)
        register_mcp_servers({"dormant_fixture": config})
        _forget_lazy_server((scope, "dormant_fixture"))
    with profiles("a"):
        assert resolve_toolset("dormant_fixture") == ["mcp__dormant_fixture__echo"]
