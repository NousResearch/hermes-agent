"""Wiring tests: tools.mcp_tool._run_stdio only enforces the MCP stdio
command allowlist (tools.mcp_command_guard) when it is enabled.

The allowlist is opt-in (default off — see tools/mcp_command_guard.py for
why): these tests confirm a disallowed command is let through unchanged
when the flag is off, and rejected before spawn when it's on.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_command_guard import DisallowedMcpCommandError


def _make_server(name="leak-guard"):
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask.__new__(MCPServerTask)
    server.name = name
    server._ready = MagicMock()
    server._shutdown_event = MagicMock()
    server._shutdown_event.is_set.return_value = True
    server._reconnect_event = MagicMock()
    server._sampling = None
    server._elicitation = None
    server._registered_tool_names = []
    return server


def _run_stdio_with_command(server, command, *, enabled):
    # A disallowed command ("bash") to make the allowlist's effect
    # observable; stdio_client raises immediately so no real subprocess is
    # spawned either way.
    config = {"command": command, "args": []}

    async def _run():
        with patch("tools.mcp_command_guard.is_enabled", return_value=enabled), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._build_safe_env", return_value={}), \
             patch("tools.mcp_tool._resolve_stdio_command",
                   return_value=(command, {})), \
             patch("tools.mcp_tool._write_stderr_log_header"), \
             patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None), \
             patch("tools.mcp_tool.check_package_for_malware",
                   return_value=None, create=True), \
             patch("tools.osv_check.check_package_for_malware",
                   return_value=None):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(side_effect=RuntimeError("stdio_client reached"))
            cm.__aexit__ = AsyncMock(return_value=False)
            with patch("tools.mcp_tool.stdio_client", return_value=cm):
                await server._run_stdio(config)

    return asyncio.run(_run())


class TestAllowlistWiringIsOptIn:
    def test_disallowed_command_passes_through_when_disabled(self):
        """Default-off: a disallowed command reaches the stdio_client spawn
        unchanged — it fails for the mocked spawn reason, not the guard."""
        server = _make_server()
        with pytest.raises(RuntimeError, match="stdio_client reached"):
            _run_stdio_with_command(server, "bash", enabled=False)

    def test_disallowed_command_rejected_when_enabled(self):
        """Once opted in, a disallowed command never reaches stdio_client."""
        server = _make_server()
        with pytest.raises(DisallowedMcpCommandError):
            _run_stdio_with_command(server, "bash", enabled=True)

    def test_allowed_command_passes_through_when_enabled(self):
        """Opting in doesn't affect a command that's already allowlisted."""
        server = _make_server()
        with pytest.raises(RuntimeError, match="stdio_client reached"):
            _run_stdio_with_command(server, "npx", enabled=True)
