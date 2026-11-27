"""A lazy (schema-cache) MCP registration survives a served profile's repeat discovery.

Regression for #140265: ``_register_connected_into_current_scope`` judged every key in the
profile's overlay by its live connection, and a lazy registration has none by design, so the
second discovery pass removed its tools while the lazy markers stayed behind — from then on
``_select_new_servers`` counted the server as registered with nothing to call. A lazy key is
judged by its saved config instead: kept while the profile's entry is unchanged, forgotten
together with its overlay otherwise so discovery rebuilds it from the new entry.
"""

from __future__ import annotations

import sys

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _lazy_config(*extra_args: str) -> dict:
    return {"command": sys.executable, "args": ["-c", "pass", *extra_args], "lazy": True}


def _cache(config: dict, *tool_names: str) -> None:
    from tools.mcp_schema_cache import config_fingerprint, write_cache_entry

    write_cache_entry("example", config_fingerprint(config), tools=[
        {"name": t, "description": f"synthetic {t}", "inputSchema": {"type": "object"}} for t in tool_names])


@pytest.fixture
def served_profile(tmp_path, monkeypatch):
    """A routed profile home (the override names a home other than the process's own, so every
    discovery pass runs scoped), clean MCP ledgers; yields the scope and restores everything."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools.registry import registry

    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
               "_lazy_server_fingerprints", "_lazy_server_tool_names", "_mcp_tool_server_names",
               "_orphaned_adopters", "_parallel_safe_servers", "_server_trust_levels", "_tool_read_only_hints")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    home = tmp_path / "profiles" / "served"
    home.mkdir(parents=True)
    token = set_hermes_home_override(home)
    scope = core._mcp_registry_scope()
    assert scope is not None
    yield scope
    for tool_name in list(registry.get_tool_names_for_toolset("mcp-example")):
        registry.deregister(tool_name, scope=scope)
    reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def test_repeat_discovery_keeps_a_cached_lazy_server_registered(served_profile):
    import tools.mcp_tool as core
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.registry import registry

    config = _lazy_config()
    _cache(config, "read")
    assert register_mcp_servers({"example": config}) == ["mcp__example__read"]
    key = (served_profile, "example")
    assert key in core._lazy_server_configs

    # The second pass is where the overlay used to vanish; the third shows nothing had to recover.
    for _ in range(2):
        assert register_mcp_servers({"example": config}) == ["mcp__example__read"]
    assert registry.get_tool_names_for_toolset("mcp-example") == ["mcp__example__read"]
    assert key in core._lazy_server_configs
    assert served_profile in core._server_tool_scopes[key]
    assert not core._servers and not core._server_connecting  # still never connected


def test_disabled_lazy_entry_is_forgotten_with_its_overlay(served_profile):
    import tools.mcp_tool as core
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.registry import registry

    config = _lazy_config()
    _cache(config, "read")
    register_mcp_servers({"example": config})
    key = (served_profile, "example")

    assert register_mcp_servers({"example": {**config, "enabled": False}}) == []
    assert registry.get_tool_names_for_toolset("mcp-example") == []
    for ledger in (core._lazy_server_configs, core._lazy_server_fingerprints,
                   core._lazy_server_tool_names, core._server_tool_scopes):
        assert key not in ledger

    # Re-enabled, the entry registers from the cache again instead of being held by stale markers.
    assert register_mcp_servers({"example": config}) == ["mcp__example__read"]
    assert not core._servers


def test_edited_lazy_entry_is_rebuilt_from_the_new_config(served_profile):
    import tools.mcp_tool as core
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.registry import registry

    config_v1, config_v2 = _lazy_config(), _lazy_config("--v2")
    _cache(config_v1, "read")
    register_mcp_servers({"example": config_v1})
    key = (served_profile, "example")

    _cache(config_v2, "write")
    assert register_mcp_servers({"example": config_v2}) == ["mcp__example__write"]
    assert registry.get_tool_names_for_toolset("mcp-example") == ["mcp__example__write"]
    assert core._lazy_server_configs[key] == config_v2
    assert core._lazy_server_tool_names[key] == ["mcp__example__write"]
    assert not core._servers  # rebuilt from the cache, still without a connection


def test_saved_lazy_config_does_not_alias_the_callers_nested_values(served_profile):
    from tools.mcp_tool_discovery import register_mcp_servers

    config = _lazy_config()
    _cache(config, "read")
    register_mcp_servers({"example": config})

    # An in-place edit of the caller's own nested list: a shallow saved copy shared it and would
    # have compared equal to itself, keeping the stale registration.
    config["args"].append("--v2")
    _cache(config, "write")
    assert register_mcp_servers({"example": config}) == ["mcp__example__write"]
