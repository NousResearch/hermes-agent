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
        result = mcp_tool.register_mcp_servers({
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


@pytest.mark.no_isolate
def test_register_rejects_source_bound_reuse_when_cached_config_missing(
    monkeypatch, tmp_path,
):
    """Missing legacy cache metadata must not weaken profile isolation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    woken: list[str] = []

    class _Event:
        def set(self):
            woken.append("gbrain")

    class _LegacyStale:
        session = None

        def __init__(self):
            self.name = "gbrain"
            self._reconnect_event = _Event()
            self._registered_tool_names: list[str] = []

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setitem(mcp_tool._servers, "gbrain", _LegacyStale())

    requested = {
        "gbrain": {
            "url": "http://127.0.0.1:7331/mcp",
            "auth": "oauth",
            "oauth": {
                "grant_type": "client_credentials",
                "token_url": "http://127.0.0.1:7331/token",
                "client_id": "profile-client",
            },
        }
    }
    try:
        with pytest.raises(RuntimeError, match="refusing to reuse"):
            mcp_tool.register_mcp_servers(requested)
        assert woken == []
    finally:
        mcp_tool._servers.pop("gbrain", None)
