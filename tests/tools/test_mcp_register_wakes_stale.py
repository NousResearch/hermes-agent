"""New sessions must wake parked/stale cached MCP servers immediately.

Regression for #50170: after a keepalive failure parks a server, its tools
are deregistered — so a NEW agent session starting up saw the tools silently
absent and had no way to trigger recovery until the next timed self-probe
(up to _PARKED_RETRY_INTERVAL later). register_mcp_servers now nudges any
cached entry whose session is None via _signal_reconnect.
"""

import pytest


@pytest.mark.no_isolate
def test_register_wakes_stale_cached_server(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools import mcp_tool_discovery as _mcp_discovery

    woken: list[str] = []

    class _Event:
        def __init__(self, name):
            self._name = name

        def set(self):
            woken.append(self._name)

    class _Stale:
        session = None

        def __init__(self, name):
            self.name = name
            self._reconnect_event = _Event(name)
            self._registered_tool_names: list[str] = []

    class _Alive:
        session = object()

        def __init__(self, name):
            self.name = name
            self._reconnect_event = _Event(name)
            self._registered_tool_names = [f"{name}__tool"]

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    stale = _Stale("parked-srv")
    alive = _Alive("healthy-srv")
    monkeypatch.setitem(mcp_tool._servers, "parked-srv", stale)
    monkeypatch.setitem(mcp_tool._servers, "healthy-srv", alive)

    try:
        result = _mcp_discovery.register_mcp_servers({
            "parked-srv": {"url": "http://127.0.0.1:9/mcp"},
            "healthy-srv": {"url": "http://127.0.0.1:9/mcp"},
        })
        # Both cached → no new connections attempted; existing names returned.
        assert "healthy-srv__tool" in result
        # The parked (session=None) entry got a reconnect nudge; the healthy
        # one was left alone.
        assert woken == ["parked-srv"]
    finally:
        mcp_tool._servers.pop("parked-srv", None)
        mcp_tool._servers.pop("healthy-srv", None)


def test_profile_owned_private_stale_server_is_woken(monkeypatch, tmp_path):
    from hermes_constants import hermes_home_key
    from tools import mcp_tool
    from tools import mcp_tool_discovery as discovery

    scope = hermes_home_key(tmp_path / "worker")
    private_key = f"shared::profile::{scope}"
    stale = type("Stale", (), {"session": None})()
    woken = []
    monkeypatch.setattr(mcp_tool, "_servers", {private_key: stale})
    monkeypatch.setattr(mcp_tool, "_server_public_names", {private_key: "shared"})
    monkeypatch.setattr(mcp_tool, "_server_scope_keys", {private_key: scope})
    monkeypatch.setattr(mcp_tool, "_server_tool_scopes", {private_key: {scope}})
    monkeypatch.setattr(mcp_tool, "_server_connecting", set())
    monkeypatch.setattr(mcp_tool, "_lazy_server_configs", {})
    monkeypatch.setattr(mcp_tool, "_server_connect_errors", {})
    monkeypatch.setattr(mcp_tool, "_server_connect_retry_after", {})
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: scope)
    monkeypatch.setattr(discovery._loop, "_signal_reconnect", woken.append)

    assert discovery._select_new_servers({"shared": {"auth": "oauth"}}) == {}
    assert woken == [stale]
