"""A lazy (schema-cache) registration survives repeat scoped reconciliation (#140265).

Under a routed scope (multiplexed gateway, desktop backend, per-profile cron tick) the first
discovery registers a ``lazy: true`` server's tools from the schema cache **without** a live
connection and records the registering scope in ``_server_tool_scopes``. The next pass's heal
step (``register_connected_into_current_scope``) treated "no live server under this key" as a
stale overlay and removed the tools, while ``_lazy_server_configs`` / ``_lazy_server_fingerprints``
/ ``_lazy_server_tool_names`` survived — and ``_select_new_servers`` skips any key that carries a
lazy marker, so nothing rebuilt the registration. The profile stayed silently tool-less on every
later pass even though its config and the schema cache were intact.

The fix retains a lazy registration for its OWNING scope while the config is unchanged, and
revokes the scope TOGETHER with its lazy markers when the config changed or the server was
disabled, so normal discovery can rebuild it.
"""

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

pytestmark = pytest.mark.no_isolate

CONFIG = {"command": "/usr/bin/false", "lazy": True}
TOOL = {"name": "read", "description": "synthetic read", "inputSchema": {"type": "object"}}
TOOL_NAME = "mcp__example__read"


@pytest.fixture()
def routed_profile(tmp_path, monkeypatch):
    """A served profile home that is not the process home, so MCP resolves a profile scope."""
    from tools import mcp_tool
    from tools.registry import registry

    home = tmp_path / "profiles" / "secondary"
    home.mkdir(parents=True)
    # Pre-installed SDK mock: _ensure_mcp_sdk() returns without an import attempt.
    monkeypatch.setattr(mcp_tool, "ClientSession", object(), raising=False)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)

    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_lazy_server_configs", "_lazy_server_fingerprints",
               "_lazy_server_tool_names", "_mcp_tool_server_names", "_parallel_safe_servers",
               "_server_trust_levels", "_tool_read_only_hints")
    saved = {name: dict(getattr(mcp_tool, name)) for name in ledgers if isinstance(getattr(mcp_tool, name), dict)}
    saved_scoped_tools = {key: dict(value) for key, value in registry._scoped_tools.items()}

    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)
        for name, value in saved.items():
            live = getattr(mcp_tool, name)
            live.clear()
            live.update(value)
        mcp_tool._server_connecting.clear()
        registry._scoped_tools.clear()
        registry._scoped_tools.update(saved_scoped_tools)


def _write_cache(config=CONFIG):
    from tools.mcp_schema_cache import config_fingerprint, write_cache_entry

    write_cache_entry("example", config_fingerprint(config), tools=[dict(TOOL)])


def _register(config=None):
    from tools.mcp_tool_discovery import register_mcp_servers

    return register_mcp_servers({"example": dict(config or CONFIG)})


def test_repeat_reconciliation_keeps_the_lazy_registration(routed_profile):
    """Second and third discovery passes must not strip tools the first one cached (the defect)."""
    from tools.registry import registry

    _write_cache()

    assert TOOL_NAME in _register()
    assert TOOL_NAME in _register()
    assert TOOL_NAME in _register()
    assert registry.get_tool_names_for_toolset("mcp-example") == [TOOL_NAME]


def test_lazy_registration_never_spawns_a_connection(routed_profile):
    """Retention is for the cached registration only: no live server appears."""
    from tools import mcp_tool

    _write_cache()
    _register()
    _register()

    assert mcp_tool._servers == {}
    assert mcp_tool._lazy_server_tool_names


def test_changed_config_replaces_the_registration_instead_of_pinning_it(routed_profile):
    """A route change revokes the old overlay; the new route's manifest then registers.

    The revoked pass must also clear the lazy markers, or ``_select_new_servers`` keeps counting
    the key as known and the profile stays tool-less on the new route.
    """
    from tools import mcp_tool
    from tools.mcp_schema_cache import config_fingerprint
    from tools.registry import registry

    changed = {"command": "/usr/bin/false", "args": ["--other"], "lazy": True}
    _write_cache()
    assert TOOL_NAME in _register()

    # The cache describes the NEW route, so this pass revokes the stale registration (old
    # fingerprint) and rebuilds from the entry that matches the config it just judged.
    _write_cache(changed)
    assert TOOL_NAME in _register(changed)
    assert registry.get_tool_names_for_toolset("mcp-example") == [TOOL_NAME]
    assert mcp_tool._servers == {}
    assert set(mcp_tool._lazy_server_fingerprints.values()) == {config_fingerprint(changed)}


def test_disabled_server_revokes_the_registration_and_its_markers(routed_profile):
    """``enabled: false`` drops the overlay; re-enabling re-registers from the same cache entry."""
    from tools.registry import registry

    _write_cache()
    assert TOOL_NAME in _register()

    disabled = {**CONFIG, "enabled": False}
    assert TOOL_NAME not in _register(disabled)
    assert registry.get_tool_names_for_toolset("mcp-example") == []

    assert TOOL_NAME in _register()
