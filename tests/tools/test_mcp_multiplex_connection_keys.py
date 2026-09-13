"""Two multiplexed profiles that name the same MCP server with different credentials are two
connections (#106005, #91654): the ledgers in ``tools.mcp_tool`` are keyed per owning profile
scope, and an owner's scoped reload re-registers the profiles that had adopted its connection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override


def _tool():
    return SimpleNamespace(name="t", description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _server(name, cfg):
    return SimpleNamespace(name=name, session=object(), _config=cfg, _tools=[_tool()], tool_timeout=30,
                           initialize_result=None, _registered_tool_names=[], _sampling=None)


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Multiplex on, clean MCP ledgers, a scope switcher for homes A and B; restores everything."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools.registry import registry

    homes = {k: tmp_path / "profiles" / k for k in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
               "_mcp_tool_server_names", "_orphaned_adopters")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    tokens = []

    def enter(which):
        tokens.append(set_hermes_home_override(homes[which]))
        return hermes_home_key(homes[which])

    yield enter
    for tool_name in list(registry.get_tool_names_for_toolset("mcp-x")):
        for home in homes.values():
            registry.deregister(tool_name, scope=hermes_home_key(home))
    for token in reversed(tokens):
        reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def test_same_named_server_with_other_credentials_is_a_separate_connection(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_handlers as handlers
    from tools import mcp_tool_registration as reg
    from tools.registry import registry
    import toolsets

    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    cfg_b = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer B"}}

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)
    assert toolsets.resolve_toolset("mcp-x") == ["mcp__x__t"]
    for _ in range(core._CIRCUIT_BREAKER_THRESHOLD):
        core._bump_server_error("x")
    disc._note_connect_failure("y", RuntimeError("boom"))

    two_profiles("b")
    # B's own view: no tools yet, its memo is not A's, and A's connection is not "connected" for B.
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert toolsets.resolve_toolset("mcp-x") == []
    assert disc.get_mcp_status({"x": cfg_b})[0]["status"] == "configured"
    # B's differently-authenticated 'x' is a connect candidate, not shadowed by A's ledger entries.
    assert "x" in disc._select_new_servers({"x": cfg_b})
    assert not disc._connect_cooldown_active("y")
    assert handlers._check_circuit_breaker("x") is None


def test_owner_reload_reregisters_profiles_that_adopted_its_connection(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    scope_a = two_profiles("a")
    srv_a = _server("x", cfg)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg)

    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg}) == 1
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    # Owner A: scoped shutdown (no MCP loop here, so emulate the task teardown), then rediscovery.
    two_profiles("a")
    with patch.object(lifecycle._loop, "_stop_mcp_loop", lambda **_kw: False):
        lifecycle.shutdown_mcp_servers(scope=scope_a)
    for tool_name in list(srv_a._registered_tool_names):
        reg._deregister_mcp_tool_all_scopes(srv_a, tool_name)
    with core._lock:
        for key in [k for k, v in core._servers.items() if v is srv_a]:
            core._servers.pop(key)
            core._server_scope_keys.pop(key, None)
            core._server_tool_scopes.pop(key, None)

    def fake_pass(new_servers):
        for name, config in new_servers.items():
            srv = _server(name, config)
            disc._adopt_server(name, srv)
            srv._registered_tool_names = reg._register_server_tools(name, srv, config)

    with patch.object(disc, "_run_discovery_pass", fake_pass), \
            patch.object(disc._loop, "_ensure_mcp_loop", lambda: None), \
            patch("tools.mcp_tool_config._load_mcp_config", lambda: {"x": cfg}):
        disc.register_mcp_servers({"x": cfg})

    # B never reloaded, yet has its tools back on the owner's new identical connection.
    two_profiles("b")
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
    assert disc.get_mcp_status({"x": cfg})[0]["status"] == "connected"
