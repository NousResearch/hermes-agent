"""An adopter overlay refresh that races a shutdown must not leave phantom tools behind.

``_refresh_adopter_overlays`` runs off the MCP loop. It checks the connection's identity before
the (slow) per-profile registration; a shutdown or park can land in between and deregister the
adopter tools. Without a post-registration re-check, the just-published tools survive teardown
and the ledger re-adopts them. Deterministic: the shutdown is simulated inside the registration
call, no subprocess is started.
"""

from types import SimpleNamespace

import pytest

_NAME = "racesrv"
_KEY = ("racecfg", _NAME)


@pytest.fixture
def adopter_env(tmp_path, monkeypatch):
    from tools import mcp_tool
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    profile = tmp_path / "adopter"
    profile.mkdir()
    scope = str(profile)
    server = SimpleNamespace(_registered_tool_names=[], name=_NAME)
    with mcp_tool._lock:
        mcp_tool._servers[_KEY] = server
        # The adopter is recorded here by a real adoption (registration: _server_tool_scopes).
        mcp_tool._server_tool_scopes[_KEY] = {scope}
    monkeypatch.setattr(reg._config, "_load_mcp_config", lambda: {_NAME: {"command": "x"}})
    monkeypatch.setattr(reg, "mcp_server_enabled", lambda cfg: True)
    monkeypatch.setattr(reg, "_same_server_route", lambda s, c, **kw: True)
    try:
        yield SimpleNamespace(mcp_tool=mcp_tool, reg=reg, registry=registry,
                              server=server, scope=scope, monkeypatch=monkeypatch)
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(_KEY, None)
            mcp_tool._server_tool_scopes.pop(_KEY, None)
        registry.deregister(f"mcp_{_NAME}_phantom", scope=scope)


def _schema():
    return {"name": "phantom", "description": "adopter-only tool", "parameters": {"type": "object"}}


def test_adopter_refresh_racing_shutdown_retracts_published_tools(adopter_env):
    """Shutdown lands after the identity check but before the overlay is published: the overlay
    must be retracted, the ledger must not keep the name, and the adopter scope must be empty."""
    env = adopter_env
    tool_name = f"mcp_{_NAME}_phantom"
    toolset = f"mcp-{_NAME}"

    def racing_registration(name, key, server, config, scope):
        # Shutdown wins the race: the connection leaves the table and its tools are deregistered.
        with env.mcp_tool._lock:
            env.mcp_tool._servers.pop(key, None)
        # Then the in-flight registration publishes its tool into the adopter scope anyway.
        env.registry.register(name=tool_name, toolset=toolset, schema=_schema(),
                              handler=lambda *a, **k: "", scope=scope)
        return [tool_name]

    env.monkeypatch.setattr(env.reg, "_register_adopted_overlay", racing_registration)

    env.reg._refresh_adopter_overlays(_NAME, _KEY, env.server, {env.scope})

    assert env.registry.snapshot_registration(tool_name, scope=env.scope) is None, \
        "phantom adopter tool survived a shutdown that raced the refresh"
    assert tool_name not in env.server._registered_tool_names


def test_adopter_refresh_without_race_publishes_and_keeps_tools(adopter_env):
    """Control: with the connection still serving, the overlay is published and kept in the ledger."""
    env = adopter_env
    tool_name = f"mcp_{_NAME}_phantom"
    toolset = f"mcp-{_NAME}"

    def normal_registration(name, key, server, config, scope):
        env.registry.register(name=tool_name, toolset=toolset, schema=_schema(),
                              handler=lambda *a, **k: "", scope=scope)
        return [tool_name]

    env.monkeypatch.setattr(env.reg, "_register_adopted_overlay", normal_registration)

    env.reg._refresh_adopter_overlays(_NAME, _KEY, env.server, {env.scope})

    entry = env.registry.snapshot_registration(tool_name, scope=env.scope)
    assert entry is not None and entry.toolset == toolset
    assert tool_name in env.server._registered_tool_names
