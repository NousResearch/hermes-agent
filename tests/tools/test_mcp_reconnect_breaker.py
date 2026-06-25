"""Regression tests: a keepalive-triggered reconnect must not trip the breaker.

When a server-side connection idles out, ``MCPServerTask``'s keepalive probe
fails and the run loop tears down and rebuilds the transport, setting
``session = None`` for a few hundred milliseconds. Tool handlers that observed
that transient ``None`` previously returned ``"not connected"`` *and* bumped
the circuit-breaker counter. ``_CIRCUIT_BREAKER_THRESHOLD`` such bumps opened
the breaker for the full cooldown, so a perfectly healthy server looked
unreachable for ~60s even though the only thing wrong was that a call landed
inside the reconnect gap.

These tests lock in the fix: while the background ``run()`` task is still
alive (reconnecting), handlers report a transient ``"reconnecting"`` status and
do NOT bump the breaker; only a genuinely dead server (task finished or never
started) is counted as a failure, preserving the original fast-fail behavior.
"""
import json
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_server(mcp_tool_module, name, *, session, task_done):
    """Install a fake server whose ``session`` and background-task state we
    control precisely.

    ``task_done`` drives ``_server_is_reconnecting``: ``False`` models a live
    ``run()`` task rebuilding the transport (reconnecting); ``True`` models a
    server whose task has exited (genuinely dead).
    """
    server = MagicMock()
    server.name = name
    server.session = session
    task = MagicMock()
    task.done.return_value = task_done
    server._task = task

    mcp_tool_module._servers[name] = server
    mcp_tool_module._server_error_counts.pop(name, None)
    mcp_tool_module._server_breaker_opened_at.pop(name, None)
    return server


def _cleanup(mcp_tool_module, name):
    mcp_tool_module._servers.pop(name, None)
    mcp_tool_module._server_error_counts.pop(name, None)
    mcp_tool_module._server_breaker_opened_at.pop(name, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reconnect_window_does_not_trip_breaker(monkeypatch, tmp_path):
    """session=None while the run() task is still alive == reconnecting.

    Every call in that window must report a transient "reconnecting" status
    and must NOT bump the breaker, so the breaker can never open on a healthy
    server that is merely rebuilding its transport.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    _install_server(mcp_tool, "srv", session=None, task_done=False)
    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)

        # More calls than the threshold: a pre-fix build would have opened
        # the breaker by now.
        for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD + 2):
            parsed = json.loads(handler({}))
            assert "reconnecting" in parsed.get("error", "").lower(), parsed

        assert mcp_tool._server_error_counts.get("srv", 0) == 0, (
            "reconnect window must not bump the breaker counter"
        )
        assert "srv" not in mcp_tool._server_breaker_opened_at, (
            "breaker must not open during a reconnect"
        )
    finally:
        _cleanup(mcp_tool, "srv")


def test_dead_server_still_trips_breaker(monkeypatch, tmp_path):
    """session=None and the run() task finished == genuinely dead.

    The original fast-fail + breaker behavior must be preserved: each call
    bumps the counter, and once the threshold is reached the breaker opens
    and short-circuits subsequent calls.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_tool_handler

    _install_server(mcp_tool, "srv", session=None, task_done=True)
    try:
        handler = _make_tool_handler("srv", "tool1", 10.0)

        for i in range(1, mcp_tool._CIRCUIT_BREAKER_THRESHOLD + 1):
            parsed = json.loads(handler({}))
            assert "not connected" in parsed.get("error", ""), parsed
            assert mcp_tool._server_error_counts["srv"] == i

        # Threshold reached → breaker open → next call short-circuits.
        assert "srv" in mcp_tool._server_breaker_opened_at
        parsed = json.loads(handler({}))
        assert "unreachable" in parsed.get("error", "").lower(), parsed
    finally:
        _cleanup(mcp_tool, "srv")


def test_resource_handler_reports_reconnecting(monkeypatch, tmp_path):
    """Non-tool handlers (which never bumped the breaker) should still report
    the transient reconnect state rather than a bare "not connected"."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool import _make_list_resources_handler

    _install_server(mcp_tool, "srv", session=None, task_done=False)
    try:
        handler = _make_list_resources_handler("srv", 10.0)
        parsed = json.loads(handler({}))
        assert "reconnecting" in parsed.get("error", "").lower(), parsed
    finally:
        _cleanup(mcp_tool, "srv")
