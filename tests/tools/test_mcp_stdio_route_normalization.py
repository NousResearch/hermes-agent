"""Regression for #127824: one stdio MCP server spelled two ways is ONE connection.

On Windows the gateway spawned every configured stdio server twice, one child per spelling of the
same launcher: ``config.yaml`` held the backslash spelling the user pasted, while the other copy
(plugin ``mcp.json`` / a profile config written by the desktop app) held the forward-slash one
(``F:/nodejs/node.exe``). Nothing in the duplicate checks treated those as the same server: the
plugin merge compared bare names, and the connection identity that lets a second profile reuse a
live connection hashed the raw config strings. Path spelling is not part of a server's identity.
"""

from __future__ import annotations

import json
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
    for toolset in ("mcp-ast-grep",):
        for tool_name in list(registry.get_tool_names_for_toolset(toolset)):
            for home in homes.values():
                registry.deregister(tool_name, scope=hermes_home_key(home))
    for token in reversed(tokens):
        reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


@pytest.mark.platforms("windows")
def test_forward_slash_spelling_shares_the_owners_connection(two_profiles):
    """Two profiles, one server, two spellings: the second profile adopts the live connection
    instead of opening a second child of the same gateway process."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from tools.registry import registry

    cfg_a = {"command": r"F:\nodejs\node.exe", "args": [r"C:\mcp\ast-grep.js", "--stdio"]}
    cfg_b = {"command": "F:/nodejs/node.exe", "args": ["C:/mcp/ast-grep.js", "--stdio"]}

    scope_a = two_profiles("a")
    with disc._owner_secret_scope():
        srv_a = _server("ast-grep", cfg_a)
    disc._adopt_server("ast-grep", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("ast-grep", srv_a, cfg_a)

    two_profiles("b")
    assert reg.register_connected_into_current_scope({"ast-grep": dict(cfg_b)}) == 1
    # One registration for the shared connection (the tool name is the sanitized server name).
    assert registry.get_tool_names_for_toolset("mcp-ast-grep") == ["mcp__ast_grep__t"]
    # A's live child serves both scopes, so B has nothing to spawn.
    assert "ast-grep" not in disc._select_new_servers({"ast-grep": dict(cfg_b)})
    assert (scope_a, "ast-grep") in core._servers


@pytest.mark.platforms("windows")
def test_portable_alias_of_a_native_route_registers_one_server(tmp_path):
    """A plugin server that names the same launcher as a native entry, spelled the other way, is
    the same server: ``_load_mcp_config`` returns one entry, so one child is spawned.

    The plugin side goes through the real ``_discover_mcp``, because that is the shape the merge
    actually sees: the loader hands every portable stdio entry an injected ``env``
    (``PLUGIN_ROOT``/``PLUGIN_DATA``) and a resolved ``cwd``. A hand-written portable dict without
    them is a shape the loader cannot produce, and it hid the fact that a whole-entry comparison
    never matched.
    """
    from hermes_cli.agent_plugins import _discover_mcp

    root = tmp_path / "plugin-root"
    root.mkdir()
    (root / "mcp.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
        "mcpServers": {"ast-grep": {"type": "stdio", "command": "npx",
                                    "args": ["-y", "F:/mcp/server.js"]}}}), encoding="utf-8")
    diagnostics: list = []
    portable = _discover_mcp(root, tmp_path / "plugin-data", diagnostics)
    assert diagnostics == []
    translated = portable["ast-grep"]
    assert {"env", "cwd"} <= set(translated)  # the injected fields a native entry does not carry

    native = {"ast-grep": {"command": "npx", "args": ["-y", r"F:\mcp\server.js"]}}
    manager = SimpleNamespace(get_portable_mcp_servers=lambda: {"ast_grep": translated})

    with (
        patch("hermes_cli.config.load_config", return_value={"mcp_servers": native}),
        patch("hermes_cli.plugins.discover_plugins"),
        patch("hermes_cli.plugins.get_plugin_manager", return_value=manager),
    ):
        from tools.mcp_tool_config import _load_mcp_config

        result = _load_mcp_config()

    assert list(result) == ["ast-grep"]


@pytest.mark.platforms("windows")
def test_case_and_package_specs_are_not_path_spelling():
    """Only a token spelled like a path has a spelling to fold: a flag value keeps its case and a
    package spec keeps its separators, so neither two profiles nor two entries collapse onto one
    route because of them."""
    from tools.mcp_schema_cache import config_fingerprint
    from tools.mcp_tool_config import _mcp_entry_key, _path_spelling_duplicate

    strict = {"command": "npx", "args": ["--mode", "Strict", "-y", "@mcp/github"]}
    lower = {"command": "npx", "args": ["--mode", "strict", "-y", "@mcp/github"]}
    assert config_fingerprint(strict) != config_fingerprint(lower)
    assert _path_spelling_duplicate(lower, {"strict": strict}) is None

    # ...while the same route spelled with the other slash is still one route.
    assert _mcp_entry_key({"command": "npx", "args": ["-y", "F:/mcp/server.js"]}) == \
        _mcp_entry_key({"command": "npx", "args": ["-y", r"F:\mcp\server.js"]})
