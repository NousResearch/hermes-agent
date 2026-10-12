"""``mcp_server_reconnecting`` (cron preflight's "is this outage healing on its own?" question)
must say True only for a transient park. A server parked on a PERMANENT error (revoked
credentials, endpoint gone) fails its self-probe identically every time, so treating it as
reconnecting would run the job tool-less forever with no blocked_config alert.
"""

import asyncio

import pytest

import tools.mcp_tool as core
from tools.mcp_tool_discovery import mcp_server_reconnecting


@pytest.fixture
def parked_server(monkeypatch):
    server = core.MCPServerTask("x")
    server._ever_connected = True
    server.session = None
    server._task = None
    monkeypatch.setitem(core._servers, "x", server)
    return server


def test_transient_park_is_reconnecting(parked_server):
    assert mcp_server_reconnecting("x") is True
    parked_server._park_reason = "from parked state"  # rapid-drop budget exhausted, still probing
    assert mcp_server_reconnecting("x") is True


def test_permanent_error_park_is_not_reconnecting(parked_server):
    parked_server._park_reason = "from parked state (permanent error)"
    assert mcp_server_reconnecting("x") is False


def test_park_records_reason_and_proven_session_clears_it(parked_server, monkeypatch):
    """The run loop hands ``_park`` its revival reason; a proven healthy session forgets it."""
    async def _shutdown_at_once(timeout=None):
        return "shutdown"
    monkeypatch.setattr(parked_server, "_wait_for_reconnect_or_shutdown", _shutdown_at_once)

    asyncio.run(parked_server._park("from parked state (permanent error)"))
    assert parked_server._park_reason == "from parked state (permanent error)"
    assert mcp_server_reconnecting("x") is False

    parked_server._mark_session_proven()
    assert parked_server._park_reason is None
