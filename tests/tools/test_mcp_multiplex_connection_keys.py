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


def test_oauth_server_is_not_adopted_across_profiles(two_profiles):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "auth": "oauth"}

    scope_a = two_profiles("a")
    srv_a = _server("x", cfg)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg)
    assert reg.register_connected_into_current_scope({"x": dict(cfg)}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]

    scope_b = two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": dict(cfg)}) == 0
    assert registry.get_tool_names_for_toolset("mcp-x") == []

    # Driven through the public entry point, B must open its own connection (its own OAuth
    # token) rather than adopt A's session.
    connected = []

    def fake_pass(new_servers):
        for name, config in new_servers.items():
            connected.append(_server(name, config))
            disc._adopt_server(name, connected[-1])

    with patch.object(disc, "_run_discovery_pass", fake_pass), \
            patch.object(disc._loop, "_ensure_mcp_loop", lambda: None):
        disc.register_mcp_servers({"x": dict(cfg)})
    assert connected and core._servers[(scope_b, "x")] is connected[0]
    assert core._servers[(scope_a, "x")] is srv_a


def test_same_named_server_with_other_mtls_identity_is_a_separate_connection(two_profiles):
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg_a = {
        "url": "https://mcp.example/x",
        "client_cert": "/certs/profile-a.pem",
        "client_key": "/certs/profile-a.key",
    }
    cfg_b = {
        "url": "https://mcp.example/x",
        "client_cert": "/certs/profile-b.pem",
        "client_key": "/certs/profile-b.key",
    }

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)

    two_profiles("b")
    reg.register_connected_into_current_scope({"x": cfg_b})
    assert "x" in disc._select_new_servers({"x": cfg_b})


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


def test_untrusted_adopter_of_a_full_profiles_connection_keeps_its_own_trust_gate(two_profiles, monkeypatch):
    """Trust is the consuming profile's policy: adopting A's ``trust: full`` connection must not let
    B's ``trust: untrusted`` write-capable call skip approval."""
    from tools import mcp_tool_discovery as disc, mcp_tool_handlers as handlers
    from tools import mcp_tool_registration as reg
    import tools.approval_prompt as approval_prompt

    route = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    cfg_a, cfg_b = dict(route, trust="full"), dict(route, trust="untrusted")
    asked = []
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent",
                        lambda *a, **k: asked.append(a) or "deny")

    two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)

    two_profiles("b")
    assert reg.register_connected_into_current_scope({"x": cfg_b}) == 1
    assert handlers._trust_gate_check("x", "t") is not None and asked

    two_profiles("a")
    assert handlers._trust_gate_check("x", "t") is None and len(asked) == 1


def test_parallel_safe_opt_in_is_per_profile(two_profiles):
    """B's ``supports_parallel_tool_calls`` on its own same-named server never makes A's serial
    server's tool parallel-safe (the batch planner would run two A calls concurrently)."""
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg

    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    cfg_b = dict(cfg_a, headers={"Authorization": "Bearer B"}, supports_parallel_tool_calls=True)

    two_profiles("a")
    disc._select_new_servers({"x": cfg_a})
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)

    two_profiles("b")
    disc._select_new_servers({"x": cfg_b})
    assert disc.is_mcp_tool_parallel_safe("mcp__x__t") is True

    two_profiles("a")
    assert disc.is_mcp_tool_parallel_safe("mcp__x__t") is False


def test_served_profile_without_multiplex_flag_gets_its_own_connection(two_profiles, monkeypatch):
    """A dashboard/desktop backend serves profiles through the HERMES_HOME override with
    ``gateway.multiplex_profiles`` off; a same-named server with other credentials must still be a
    separate connection there, or profile B calls the server as profile A (#111151). The launch
    profile itself (no override) keeps the bare, unscoped key."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.mcp_tool_scope import _resolve_server_key, _server_key
    from tools.registry import registry

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    cfg_b = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer B"}}

    scope_a = two_profiles("a")
    srv_a = _server("x", cfg_a)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)
    assert (scope_a, "x") in core._servers

    two_profiles("b")
    assert _resolve_server_key("x") != (scope_a, "x")
    assert registry.get_tool_names_for_toolset("mcp-x") == []
    assert "x" in disc._select_new_servers({"x": cfg_b})

    with patch("hermes_constants.get_hermes_home_override", return_value=None):
        assert core._mcp_registry_scope() is None
        assert _server_key("x") == "x"


def test_served_profile_check_fn_verdict_does_not_shadow_launch_profile(two_profiles, monkeypatch):
    """With multiplex off, a served profile's (correct) "not my connection" verdict must not sit in
    the process-wide check_fn cache under the launch profile's key: the cache scope has to follow
    the same served-profile predicate as the registry scope, or the owner loses its live tools."""
    import tools.registry as registry_mod
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    cfg_a = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer A"}}
    srv_a = _server("x", cfg_a)
    with patch("hermes_constants.get_hermes_home_override", return_value=None):
        disc._adopt_server("x", srv_a)
        srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg_a)
        entry = registry._tools["mcp__x__t"]
    registry_mod.invalidate_check_fn_cache()
    try:
        two_profiles("b")
        assert registry_mod.check_fn_cache_scope() is not None
        assert registry_mod._check_fn_cached(entry.check_fn) is False
        with patch("hermes_constants.get_hermes_home_override", return_value=None):
            assert registry_mod._check_fn_cached(entry.check_fn) is True
    finally:
        registry.deregister("mcp__x__t")
        registry_mod.invalidate_check_fn_cache()


def test_launch_profile_pruning_a_server_keeps_served_profiles_same_named_connection(two_profiles, monkeypatch):
    """The launch profile's registry scope is ``None``; when it drops server ``x`` from its config,
    ``reconcile_mcp_servers_with_config`` prunes with ``shutdown_mcp_servers(scope=None,
    names={"x"})``. ``scope=None`` must mean *the unscoped owner* there, not *every owner* —
    otherwise the dashboard's own profile silently tears down profile B's ``(B, "x")``."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle, mcp_tool_loop as loop

    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}

    scope_b = two_profiles("b")
    srv_b = _server("x", cfg)
    disc._adopt_server("x", srv_b)
    assert core._server_scope_keys[(scope_b, "x")] == scope_b

    with patch("hermes_constants.get_hermes_home_override", return_value=None):
        assert core._mcp_registry_scope() is None
        srv_launch = _server("x", cfg)
        disc._adopt_server("x", srv_launch)
        assert core._server_scope_keys["x"] is None
    # A third profile adopted the launch profile's connection: pruning it must remember the
    # adopter so the next discovery pass re-registers it.
    core._server_tool_scopes["x"] = {"c"}

    closed = []

    async def _shutdown(self):
        closed.append(self.name)

    for srv in (srv_b, srv_launch):
        srv.shutdown = _shutdown.__get__(srv)

    loop._ensure_mcp_loop()
    try:
        lifecycle.shutdown_mcp_servers(scope=None, names={"x"})
    finally:
        loop._stop_mcp_loop()

    assert "x" not in core._servers and "x" not in core._server_scope_keys
    assert core._servers[(scope_b, "x")] is srv_b
    assert core._server_scope_keys[(scope_b, "x")] == scope_b
    assert closed == ["x"]
    assert core._orphaned_adopters == {"c": {"x"}}


def test_adopter_scope_setup_failure_leaks_no_override_and_continues(two_profiles, monkeypatch):
    """A corrupt/removed adopter home raising inside ``build_profile_secret_scope`` must
    not leak that adopter's HERMES_HOME override into the caller's context, and the
    remaining adopters still get their re-registration pass."""
    import agent.secret_scope as ss
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_lifecycle as lifecycle

    core._orphaned_adopters.update({"/nonexistent/bad-home": {"x"}, "/nonexistent/good-home": {"x"}})

    def flaky_build(home):
        if "bad-home" in str(home):
            raise RuntimeError("corrupt profile home")
        return {}

    monkeypatch.setattr(ss, "build_profile_secret_scope", flaky_build)
    monkeypatch.setattr("tools.mcp_tool_config._load_mcp_config",
                        lambda: {"x": {"url": "https://mcp.example/x"}})
    registered = []
    monkeypatch.setattr(disc, "register_mcp_servers", lambda servers: registered.append(dict(servers)))

    lifecycle._reregister_orphaned_adopters()

    from hermes_constants import get_hermes_home_override
    assert get_hermes_home_override() is None
    assert ss.current_secret_scope() is None
    assert registered == [{"x": {"url": "https://mcp.example/x"}}]
