"""A park/revive cycle must give ADOPTING profiles their tools back.

``_park`` deregisters the server's tools from every scope (owner and adopters), but the
revival path (``_register_discovered_tools_if_needed``) publishes only into the owner's
scope. The orphan-adopter heal covers an owner's scoped ``/reload-mcp`` teardown, not a
park — so an adopter sat tool-less behind a healthy connection until its own reload or a
gateway restart (observed live: exa parked 09:45, revived 09:53, the adopting profile's
deferred tool count stayed down by the server's tool count until a manual /reload-mcp).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override


def _tool():
    return SimpleNamespace(name="t", description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _server(name, cfg):
    from tools.mcp_tool_registration import _adopter_identity_digest
    return SimpleNamespace(name=name, session=object(), _config=cfg, _tools=[_tool()], tool_timeout=30,
                           initialize_result=None, _registered_tool_names=[], _sampling=None,
                           _was_parked=False,
                           _resolved_identity=_adopter_identity_digest(name, cfg))


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
               "_mcp_tool_server_names", "_orphaned_adopters", "_parallel_safe_servers",
               "_server_trust_levels", "_tool_read_only_hints")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    tokens = []

    def enter(which):
        tokens.append(set_hermes_home_override(homes[which]))
        return hermes_home_key(homes[which])

    yield enter
    for toolset in ("mcp-x",):
        for tool_name in list(registry.get_tool_names_for_toolset(toolset)):
            for home in homes.values():
                registry.deregister(tool_name, scope=hermes_home_key(home))
    for token in reversed(tokens):
        reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def _adopted_pair(two_profiles, cfg):
    """Owner A connects x; adopter B heals onto the same connection. Returns (srv, key)."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    scope_a = two_profiles("a")
    srv = _server("x", cfg)
    disc._adopt_server("x", srv)
    srv._registered_tool_names = reg._register_server_tools("x", srv, cfg)
    key = (scope_a, "x")
    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": dict(cfg)}) == 1
    return srv, key


def test_park_revive_heal_direct(two_profiles, tmp_path):
    """The heal itself: after a park wiped both overlays, re-registering the adopter's
    overlay from the live connection restores its tools without a discovery reconnect."""
    import tools.mcp_tool as core
    from tools import mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer same"}}
    # The heal re-loads each adopter's config under its own home: write B's before adoption.
    home_b = tmp_path / "profiles" / "b"
    (home_b / "config.yaml").write_text(
        "mcp_servers:\n  x:\n    url: https://mcp.example/x\n"
        "    headers:\n      Authorization: Bearer same\n", encoding="utf-8")
    srv, key = _adopted_pair(two_profiles, cfg)
    scope_b = hermes_home_key(home_b)

    # Park wipes both overlays.
    for tool_name in list(srv._registered_tool_names):
        reg._deregister_mcp_tool_all_scopes(srv, tool_name)
    srv._registered_tool_names = []
    srv._was_parked = True

    # Owner republishes its own tools (what _register_discovered_tools_if_needed does).
    two_profiles("a")
    srv._registered_tool_names = reg._register_server_tools("x", srv, cfg)
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
    two_profiles("b")
    assert registry.get_tool_names_for_toolset("mcp-x") == []

    # The new heal runs on the MCP loop (no ambient profile scope), like the real revival.
    lifecycle._reregister_parked_adopters(srv)
    two_profiles("b")
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]
    assert scope_b in core._server_tool_scopes[key]
    # The owner's overlay is untouched and no second connection was opened.
    assert core._servers[key] is srv
    two_profiles("a")
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]


def test_revive_without_park_does_not_touch_adopters(two_profiles, tmp_path):
    """A plain reconnect (never parked) keeps the adopter overlays intact; the heal is a no-op."""
    import tools.mcp_tool as core
    from tools import mcp_tool_lifecycle as lifecycle
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer same"}}
    srv, _key = _adopted_pair(two_profiles, cfg)
    assert srv._was_parked is False
    with patch.object(lifecycle, "_reregister_parked_adopters", wraps=lifecycle._reregister_parked_adopters) as spy:
        lifecycle._reregister_parked_adopters(srv)
        spy.assert_called_once()
    two_profiles("b")
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]


def test_transport_revival_hook_gated_on_park(two_profiles, tmp_path):
    """The transport hook: _register_discovered_tools_if_needed calls the heal exactly when
    the server was parked, and never on a plain first connect / reconnect."""
    import tools.mcp_tool as core
    from tools import mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg
    from tools.mcp_tool_transport import MCPServerTransportMixin

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer same"}}
    srv, _key = _adopted_pair(two_profiles, cfg)
    for tool_name in list(srv._registered_tool_names):
        reg._deregister_mcp_tool_all_scopes(srv, tool_name)
    srv._registered_tool_names = []
    two_profiles("a")  # the real revival runs in the owner's scope (run task context)

    with patch.object(lifecycle, "_reregister_parked_adopters") as heal:
        # Not parked: no heal (first connect / ordinary reconnect keeps adopter overlays).
        srv._was_parked = False
        MCPServerTransportMixin._register_discovered_tools_if_needed(srv)
        heal.assert_not_called()
        assert srv._registered_tool_names  # owner tools republished

        # Parked: the hook fires after the owner's republication.
        for tool_name in list(srv._registered_tool_names):
            reg._deregister_mcp_tool_all_scopes(srv, tool_name)
        srv._registered_tool_names = []
        srv._was_parked = True
        MCPServerTransportMixin._register_discovered_tools_if_needed(srv)
        heal.assert_called_once_with(srv)


def test_changed_adopter_credential_is_not_readopted(two_profiles, tmp_path):
    """The heal re-runs the identity gate: an adopter whose credential changed since adoption
    must NOT get the owner's connection back (it would call tools as the other user)."""
    import tools.mcp_tool as core
    from tools import mcp_tool_lifecycle as lifecycle
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer same"}}
    srv, key = _adopted_pair(two_profiles, cfg)
    scope_b = hermes_home_key(tmp_path / "profiles" / "b")
    for tool_name in list(srv._registered_tool_names):
        reg._deregister_mcp_tool_all_scopes(srv, tool_name)
    srv._registered_tool_names = []
    srv._was_parked = True
    two_profiles("a")
    srv._registered_tool_names = reg._register_server_tools("x", srv, cfg)

    # B's config now carries a different token: the heal must refuse the re-adoption.
    home_b = tmp_path / "profiles" / "b"
    (home_b / "config.yaml").write_text(
        "mcp_servers:\n  x:\n    url: https://mcp.example/x\n"
        "    headers:\n      Authorization: Bearer rotated\n", encoding="utf-8")
    lifecycle._reregister_parked_adopters(srv)
    two_profiles("b")
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert scope_b not in core._server_tool_scopes.get(key, set())