"""Disabled MCP servers must not stay parked or get reconnect nudges.

Regression: setting ``enabled: false`` only skipped *new* connects, while
``register_mcp_servers`` still woke session=None cached entries via
``_signal_reconnect``. A disabled oauth_lab (etc.) kept retrying and
logging ``failed initial connection ... parking`` WARNINGs.
"""

from unittest.mock import MagicMock


def test_register_does_not_wake_disabled_parked_server(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    woken: list[str] = []

    class _Event:
        def __init__(self, name):
            self._name = name

        def set(self):
            woken.append(self._name)

    class _Parked:
        session = object()  # flipped to None after disable teardown

        def __init__(self, name):
            self.name = name
            self._reconnect_event = _Event(name)
            self._registered_tool_names: list[str] = []
            self._shutdown_event = MagicMock()

        def _deregister_tools(self):
            self._registered_tool_names = []

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    parked = _Parked("oauth_lab")
    # Parked = cached entry with no live session (the #50170 wake path).
    parked.session = None
    monkeypatch.setitem(mcp_tool._servers, "oauth_lab", parked)

    # Avoid scheduling work onto a real MCP loop for this unit test.
    monkeypatch.setattr(mcp_tool, "_mcp_loop", None)

    try:
        result = mcp_tool.register_mcp_servers({
            "oauth_lab": {
                "url": "http://127.0.0.1:9/mcp",
                "enabled": False,
            },
        })
        assert result == []
        assert woken == []
        assert "oauth_lab" not in mcp_tool._servers
    finally:
        mcp_tool._servers.pop("oauth_lab", None)
        mcp_tool._pending_disable.discard("oauth_lab")


def test_shutdown_mcp_server_marks_in_flight_connect_pending(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    mcp_tool._server_connecting.add("oauth_lab")
    try:
        assert mcp_tool.shutdown_mcp_server("oauth_lab") is True
        assert "oauth_lab" in mcp_tool._pending_disable
        assert "oauth_lab" not in mcp_tool._server_connecting
        assert mcp_tool._server_disabled_now("oauth_lab") is True
    finally:
        mcp_tool._server_connecting.discard("oauth_lab")
        mcp_tool._pending_disable.discard("oauth_lab")
