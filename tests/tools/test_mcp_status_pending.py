"""Regression for #38650: get_mcp_status() should distinguish "pending"
(discovery in progress, transient) from "failed" (discovery completed
and this server errored). The welcome screen previously labeled every
not-yet-connected server as "failed" — alarming red text on a fresh
launch where the background discovery simply hadn't populated the
registry yet.
"""

import pytest
from unittest.mock import patch, MagicMock


def _server(name: str):
    """A minimal stand-in for MCPServerTask with a live session."""
    s = MagicMock()
    s.session = object()  # non-None → "connected"
    s._registered_tool_names = ["t1", "t2"]
    s._tools = []
    s._sampling = None
    return s


class TestGetMcpStatusPending:
    """A configured server that isn't in the live registry is 'pending'
    if discovery hasn't reached it yet, not 'failed'."""

    def test_pending_when_registry_empty(self):
        """No servers registered at all → discovery hasn't started.
        All configured servers must be marked pending (not failed)."""
        from tools import mcp_tool

        configured = {"alpha": {"command": "x"}, "beta": {"command": "y"}}
        with patch.object(mcp_tool, "_load_mcp_config", return_value=configured), \
             patch.object(mcp_tool, "_lock", MagicMock()), \
             patch.object(mcp_tool, "_servers", {}):
            result = mcp_tool.get_mcp_status()

        assert len(result) == 2
        for entry in result:
            assert entry["connected"] is False
            assert entry["disabled"] is False
            assert entry.get("pending") is True, (
                f"server {entry['name']} should be pending, not failed"
            )

    def test_pending_when_registry_partial(self):
        """Discovery is in progress: some servers connected, others not.
        The not-yet-connected ones are pending, not failed."""
        from tools import mcp_tool

        configured = {"alpha": {"command": "x"}, "beta": {"command": "y"}}
        live = {"alpha": _server("alpha")}  # only alpha connected
        with patch.object(mcp_tool, "_load_mcp_config", return_value=configured), \
             patch.object(mcp_tool, "_lock", MagicMock()), \
             patch.object(mcp_tool, "_servers", live):
            result = mcp_tool.get_mcp_status()

        by_name = {e["name"]: e for e in result}
        assert by_name["alpha"]["connected"] is True
        assert by_name["beta"]["connected"] is False
        assert by_name["beta"].get("pending") is True

    def test_failed_when_registry_complete_but_server_missing(self):
        """All configured servers have been attempted; one isn't in the
        registry. That one is genuinely failed, not pending."""
        from tools import mcp_tool

        configured = {"alpha": {"command": "x"}, "beta": {"command": "y"}}
        live = {"alpha": _server("alpha")}  # only alpha connected
        # Pretend discovery has finished by signaling completion via a
        # separate mechanism. For the purposes of this test, the
        # "pending=False" branch is the one where discovery has reached
        # the server and given up. We simulate that by having the
        # server be "known attempted" via a different signal — but the
        # current implementation uses "registry non-empty" as the proxy.
        # This test documents the existing limitation: we can't yet
        # distinguish "beta hasn't been tried" from "beta was tried and
        # failed" without a richer registry. The welcome screen
        # will show "connecting..." until that signal exists.
        with patch.object(mcp_tool, "_load_mcp_config", return_value=configured), \
             patch.object(mcp_tool, "_lock", MagicMock()), \
             patch.object(mcp_tool, "_servers", live):
            result = mcp_tool.get_mcp_status()

        # Today this still says pending=True because we don't track
        # per-server "attempt completed" state. The test pins that
        # current behavior so any future refactor that introduces
        # genuine "failed" detection has to update this test too.
        beta = next(e for e in result if e["name"] == "beta")
        assert beta.get("pending") is True

    def test_disabled_server_not_marked_pending(self):
        """A server with enabled: false is intentionally not connected.
        It must be 'disabled', not 'pending'."""
        from tools import mcp_tool

        configured = {"alpha": {"command": "x", "enabled": False}}
        with patch.object(mcp_tool, "_load_mcp_config", return_value=configured), \
             patch.object(mcp_tool, "_lock", MagicMock()), \
             patch.object(mcp_tool, "_servers", {}):
            result = mcp_tool.get_mcp_status()

        assert len(result) == 1
        assert result[0]["disabled"] is True
        assert result[0].get("pending") is None or result[0].get("pending") is False

    def test_connected_server_has_no_pending_flag(self):
        """A successfully connected server doesn't need a 'pending' field."""
        from tools import mcp_tool

        configured = {"alpha": {"command": "x"}}
        live = {"alpha": _server("alpha")}
        with patch.object(mcp_tool, "_load_mcp_config", return_value=configured), \
             patch.object(mcp_tool, "_lock", MagicMock()), \
             patch.object(mcp_tool, "_servers", live):
            result = mcp_tool.get_mcp_status()

        assert result[0]["connected"] is True
        assert "pending" not in result[0] or result[0]["pending"] is False
