"""After the hosted boot freezes the registries, no MCP path changes what the process serves.

Reconciling with config, restarting on new launch inputs, a scoped or named shutdown and the
removal primitives they share refuse before their first side effect. Keeping a connection alive
stays allowed: a server that parks or shuts down closes its connection but keeps its registrations,
a revival or a lazy server's first use publishes nothing new, and ``tools/list_changed`` is ignored.
The process-exit wildcard shutdown still runs. A local process never freezes.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import registry_freeze
from registry_freeze import RegistryFrozen

FROZEN = "MCP registry"


@pytest.fixture
def thawed(monkeypatch):
    # The freeze is one-way for a process; thawing first makes teardown restore an unfrozen flag.
    monkeypatch.setattr(registry_freeze, "_frozen", False)


@pytest.fixture
def core(monkeypatch, thawed):
    """Empty MCP ledgers for the test, so seeded state never leaks into another test."""
    from tools import mcp_tool
    for name in ("_servers", "_server_scope_keys", "_server_tool_scopes", "_lazy_server_configs",
                 "_lazy_server_fingerprints", "_lazy_server_tool_names", "_server_connect_errors",
                 "_server_trust_levels", "_server_cost_policies"):
        monkeypatch.setattr(mcp_tool, name, {})
    monkeypatch.setattr(mcp_tool, "_server_connecting", set())
    return mcp_tool


@pytest.fixture
def calls(monkeypatch):
    """Record (instead of run) teardown and discovery, to show a refused call reached neither."""
    from tools import mcp_tool_discovery as discovery
    from tools import mcp_tool_lifecycle as lifecycle
    seen = {"shutdown": [], "discover": []}
    monkeypatch.setattr(lifecycle, "shutdown_mcp_servers", lambda **kw: seen["shutdown"].append(kw))
    monkeypatch.setattr(discovery, "discover_mcp_tools", lambda *a, **k: seen["discover"].append(1) or [])
    return seen


def _tool(name):
    return SimpleNamespace(name=name, description="", inputSchema=None)


def _seed_lazy(core, key):
    core._lazy_server_configs[key] = {}
    core._lazy_server_fingerprints[key] = "fp"
    core._lazy_server_tool_names[key] = []
    return {k: dict(getattr(core, k)) for k in
            ("_lazy_server_configs", "_lazy_server_fingerprints", "_lazy_server_tool_names")}


def test_mcp_reconcile_refuses_before_removing_state(core, monkeypatch):
    from tools import mcp_tool_discovery as discovery
    seeded = _seed_lazy(core, "review-server")
    monkeypatch.setattr(discovery._config, "_load_mcp_config", dict)
    registry_freeze.freeze_registries()
    with pytest.raises(RegistryFrozen, match=FROZEN):
        discovery.reconcile_mcp_servers_with_config()
    assert {k: getattr(core, k) for k in seeded} == seeded


def test_reconcile_refuses_before_any_teardown(core, calls, monkeypatch):
    from tools import mcp_tool_discovery as discovery
    core._servers["gone"] = SimpleNamespace(session=object())
    core._server_scope_keys["gone"] = None
    monkeypatch.setattr(discovery._config, "_load_mcp_config", lambda: {"new": {"url": "https://n/mcp"}})
    registry_freeze.freeze_registries()
    with pytest.raises(RegistryFrozen, match=FROZEN):
        discovery.reconcile_mcp_servers_with_config()
    assert calls == {"shutdown": [], "discover": []}
    assert "gone" in core._servers


def test_reconcile_with_nothing_to_change_is_a_noop_after_freeze(core, calls, monkeypatch):
    from tools import mcp_tool_discovery as discovery
    core._servers["kept"] = SimpleNamespace(session=object())
    core._server_scope_keys["kept"] = None
    monkeypatch.setattr(discovery._config, "_load_mcp_config", lambda: {"kept": {"url": "https://k/mcp"}})
    registry_freeze.freeze_registries()
    assert discovery.reconcile_mcp_servers_with_config() == {"removed": [], "added": [], "pending": []}
    assert calls == {"shutdown": [], "discover": []}


def test_restart_refuses_before_disconnecting(core, calls, monkeypatch):
    from tools import mcp_tool_discovery as discovery
    core._servers["srv"] = SimpleNamespace(session=object(), _resolved_identity="connected-with")
    core._server_scope_keys["srv"] = None
    monkeypatch.setattr(discovery._config, "_load_mcp_config", lambda: {"srv": {"command": "srv"}})
    monkeypatch.setattr(discovery._registration, "_adopter_identity_digest", lambda name, config: "new-inputs")
    registry_freeze.freeze_registries()
    with pytest.raises(RegistryFrozen, match=FROZEN):
        discovery.restart_mcp_servers_with_new_inputs({"srv"})
    assert calls == {"shutdown": [], "discover": []}
    monkeypatch.setattr(discovery._registration, "_adopter_identity_digest", lambda name, config: "connected-with")
    assert discovery.restart_mcp_servers_with_new_inputs({"srv"}) == []  # nothing to restart: no refusal


def test_forget_lazy_server_refuses(core):
    from tools import mcp_tool_discovery as discovery
    seeded = _seed_lazy(core, "lazy")
    registry_freeze.freeze_registries()
    with pytest.raises(RegistryFrozen, match=FROZEN):
        discovery._forget_lazy_server("lazy")
    assert {k: getattr(core, k) for k in seeded} == seeded


def test_remove_server_scope_refuses(core):
    from tools import mcp_tool_registration as registration
    core._server_tool_scopes["shared"] = {"/owner", "/adopter"}
    registry_freeze.freeze_registries()
    with pytest.raises(RegistryFrozen, match=FROZEN):
        registration._remove_server_scope("shared", "/adopter")
    assert core._server_tool_scopes["shared"] == {"/owner", "/adopter"}


def test_deregister_one_tool_refuses_as_the_mcp_registry(core):
    from tools import mcp_tool_registration as registration
    registry_freeze.freeze_registries()
    with pytest.raises(RegistryFrozen, match=FROZEN):
        registration._deregister_mcp_tool_all_scopes("srv", "mcp__srv__t")


def test_scoped_or_named_shutdown_refuses_wildcard_runs(core):
    from tools import mcp_tool_lifecycle as lifecycle
    registry_freeze.freeze_registries()
    with pytest.raises(RegistryFrozen, match=FROZEN):
        lifecycle.shutdown_mcp_servers(scope="/profile")
    with pytest.raises(RegistryFrozen, match=FROZEN):
        lifecycle.shutdown_mcp_servers(names={"srv"})
    lifecycle.shutdown_mcp_servers()  # process exit: connections close, registrations stay


@pytest.fixture
def registered(core):
    """A connected server with one tool in a private registry, registered before the freeze."""
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_registration import _register_server_tools
    from tools.registry import ToolRegistry
    reg = ToolRegistry()
    server = MCPServerTask("frz")
    server._config = {}
    server._tools = [_tool("t")]
    server.session = MagicMock()
    with patch("tools.registry.registry", reg):
        core._servers["frz"] = server
        server._registered_tool_names = _register_server_tools("frz", server, {})
        before = sorted(reg.get_all_tool_names())
        assert "mcp__frz__t" in before
        registry_freeze.freeze_registries()
        yield server, reg, before


def test_closing_server_keeps_its_registrations_when_frozen(registered):
    server, reg, before = registered
    server._deregister_tools()
    assert sorted(server._registered_tool_names) == before
    assert sorted(reg.get_all_tool_names()) == before


@pytest.mark.asyncio
async def test_tools_list_changed_is_ignored_when_frozen(registered):
    server, reg, before = registered
    server._refresh_lock = asyncio.Lock()
    server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_tool("u")])))
    await server._refresh_tools()
    server.session.list_tools.assert_not_awaited()
    assert sorted(server._registered_tool_names) == before
    assert sorted(reg.get_all_tool_names()) == before


def test_revival_publishes_nothing_when_frozen(core, monkeypatch):
    from tools.mcp_tool import MCPServerTask
    from tools import mcp_tool_transport as transport
    published = []
    monkeypatch.setattr(transport._registration, "_register_server_tools",
                        lambda *a: published.append(a) or ["mcp__rev__t"])
    server = MCPServerTask("rev")
    server._config = {}
    core._servers["rev"] = server
    registry_freeze.freeze_registries()
    server._register_discovered_tools_if_needed()
    assert published == [] and server._registered_tool_names == []


def test_lazy_first_use_routes_cached_tools_to_live_server_when_frozen(core, monkeypatch):
    from tools import mcp_tool_discovery as discovery
    published = []
    live = SimpleNamespace(session=object(), _registered_tool_names=[])

    async def _connect(name, config):
        return live

    monkeypatch.setattr(discovery, "_connect_server", _connect)
    monkeypatch.setattr(discovery._registration, "_register_server_tools",
                        lambda *a: published.append(a) or ["mcp__lazy__other"])
    core._lazy_server_tool_names["lazy"] = ["mcp__lazy__cached"]
    registry_freeze.freeze_registries()
    assert asyncio.run(discovery._discover_and_register_server("lazy", {})) == ["mcp__lazy__cached"]
    assert published == []
    assert live._registered_tool_names == ["mcp__lazy__cached"] and core._servers["lazy"] is live


def _shared_owner_server(core, monkeypatch):
    from tools import mcp_tool_registration as registration
    server = SimpleNamespace(session=object(), _config={"command": "x"}, _resolved_identity="same",
                             _tools=[], tool_timeout=5)
    core._servers[("/owner", "srv")] = server
    monkeypatch.setattr(core, "_mcp_registry_scope", lambda: "/adopter")
    monkeypatch.setattr(registration, "_adopter_identity_digest", lambda *a: "same")
    monkeypatch.setattr(registration, "_select_utility_schemas", lambda *a: [])
    return registration, server


def test_shared_adoption_refuses_before_its_first_write(core, monkeypatch):
    """Adopting another profile's connection publishes visibility and trust for this scope; after
    the freeze the primitive refuses itself, whatever guarded caller reaches it."""
    from tools.registry import ToolRegistry
    registration, _server = _shared_owner_server(core, monkeypatch)
    with patch("tools.registry.registry", ToolRegistry()):
        registry_freeze.freeze_registries()
        with pytest.raises(RegistryFrozen, match=FROZEN):
            registration.register_connected_into_current_scope({"srv": {"command": "x", "trust": "untrusted"}})
    assert core._server_tool_scopes == {}
    assert core._server_trust_levels == {}


def test_shareable_connection_lookup_is_read_only_after_freeze(core, monkeypatch):
    """Finding an adoptable connection is a read, so it still answers after the freeze."""
    registration, server = _shared_owner_server(core, monkeypatch)
    registry_freeze.freeze_registries()
    found = registration.shareable_connection("srv", {"command": "x"}, "/adopter", resolved_identity="same")
    assert found == (("/owner", "srv"), server)
    assert registration.shareable_connection("missing", {"command": "x"}, "/adopter") is None
    assert core._server_tool_scopes == {}


def test_shared_adoption_still_works_unfrozen(core, monkeypatch):
    from tools.registry import ToolRegistry
    registration, _server = _shared_owner_server(core, monkeypatch)
    with patch("tools.registry.registry", ToolRegistry()):
        registration.register_connected_into_current_scope({"srv": {"command": "x"}})
    assert core._server_tool_scopes == {("/owner", "srv"): {"/adopter"}}
