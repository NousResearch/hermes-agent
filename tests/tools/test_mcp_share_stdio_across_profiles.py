"""Identical stdio MCP entries in two multiplexed profiles share ONE child.

``_build_safe_env`` pins each profile's own ``HERMES_HOME`` into a stdio child's env, so the
resolved identity of two byte-identical entries always differed and every profile spawned its own
copy of every server (12 profiles x ~14 stdio servers = 124 children, ~12 GB). A server that opts in
(``shared: true`` or ``mcp.share_across_profiles``) leaves that injected home out of its identity,
so the second profile adopts the first one's live connection instead of spawning another.
"""

from __future__ import annotations

import sys
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
    for toolset in ("mcp-x", "mcp-s"):
        for tool_name in list(registry.get_tool_names_for_toolset(toolset)):
            for home in homes.values():
                registry.deregister(tool_name, scope=hermes_home_key(home))
    for token in reversed(tokens):
        reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])



def _cfg(**extra):
    return {"command": sys.executable, "args": ["-c", "pass"], **extra}


def _scopes(core, name="s"):
    (key,) = [k for k in core._servers if k[1] == name]
    return key, core._server_tool_scopes.get(key, set())


def _own_in_a(two_profiles, cfg, name="s"):
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    two_profiles("a")
    srv = _server(name, cfg)
    disc._adopt_server(name, srv)
    srv._registered_tool_names = reg._register_server_tools(name, srv, cfg)
    return core, reg


def test_identical_stdio_entry_without_opt_in_is_not_shared(two_profiles):
    core, reg = _own_in_a(two_profiles, _cfg())
    two_profiles("b")
    assert reg.register_connected_into_current_scope({"s": _cfg()}) == 0
    _, scopes = _scopes(core)
    assert core._mcp_registry_scope() not in scopes


def test_shared_true_adopts_the_live_child_of_another_profile(two_profiles):
    core, reg = _own_in_a(two_profiles, _cfg(shared=True))
    scope_a = core._mcp_registry_scope()
    key, _ = _scopes(core)
    two_profiles("b")
    scope_b = core._mcp_registry_scope()
    assert reg.register_connected_into_current_scope({"s": _cfg(shared=True)}) == 1
    # One connection, owned by A, visible from both; B did not get a connection of its own.
    assert list(core._servers) == [key] and key[0] == scope_a
    assert core._server_tool_scopes[key] >= {scope_a, scope_b}


def test_shared_server_with_other_env_is_not_shared(two_profiles):
    core, reg = _own_in_a(two_profiles, _cfg(shared=True, env={"TOKEN": "a"}))
    two_profiles("b")
    assert reg.register_connected_into_current_scope(
        {"s": _cfg(shared=True, env={"TOKEN": "b"})}) == 0
    assert core._mcp_registry_scope() not in _scopes(core)[1]


def test_explicit_hermes_home_in_env_keeps_the_server_per_profile(two_profiles):
    cfg = _cfg(shared=True, env={"HERMES_HOME": "/x"})
    _, reg = _own_in_a(two_profiles, cfg)
    # Equal explicit values are equal identities: still shareable.
    two_profiles("b")
    assert reg.register_connected_into_current_scope({"s": dict(cfg)}) == 1


def test_global_default_shares_and_per_server_false_opts_out(two_profiles):
    with patch("hermes_cli.config.load_config", return_value={"mcp": {"share_across_profiles": True}}):
        _, reg = _own_in_a(two_profiles, _cfg())
        two_profiles("b")
        assert reg.register_connected_into_current_scope({"s": _cfg()}) == 1
    with patch("hermes_cli.config.load_config", return_value={"mcp": {"share_across_profiles": True}}):
        _, reg2 = _own_in_a(two_profiles, _cfg(shared=False), name="o")
        two_profiles("b")
        assert reg2.register_connected_into_current_scope({"o": _cfg(shared=False)}) == 0


def test_adopter_leaving_keeps_the_owners_connection(two_profiles):
    core, reg = _own_in_a(two_profiles, _cfg(shared=True))
    key, _ = _scopes(core)
    two_profiles("b")
    scope_b = core._mcp_registry_scope()
    reg.register_connected_into_current_scope({"s": _cfg(shared=True)})
    reg._remove_server_scope(key, scope_b)
    assert key in core._servers and core._servers[key].session is not None
    assert scope_b not in core._server_tool_scopes.get(key, set())


def test_http_entries_ignore_the_flag(two_profiles):
    from tools import mcp_tool_registration as reg
    assert reg._share_across_profiles({"url": "https://x", "shared": True}) is False
