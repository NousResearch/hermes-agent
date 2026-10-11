"""A stdio MCP server tracks exactly the child IT spawned, even when two servers connect at once.

The child is found as the delta of the process's children across the spawn. Two servers connecting
together on the shared MCP loop used to take both snapshots before either spawned, so each delta held
both children: stopping one server (a profile's ``/reload-mcp``, a server removed from config) then
reaped the OTHER server's live process as its orphan (#58414).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager
from unittest.mock import patch

from tools.mcp_tool import MCPServerTask


class _StopAfterSpawn:
    async def __aenter__(self):
        raise RuntimeError("stop after the spawn")

    async def __aexit__(self, *exc):
        return False


def _server(name: str) -> MCPServerTask:
    server = MCPServerTask.__new__(MCPServerTask)
    server.name = name
    server._sampling = None
    server._elicitation = None
    server._registered_tool_names = []
    return server


def test_two_servers_spawning_together_each_track_only_their_own_child(monkeypatch):
    tracked: dict[str, set] = {}
    children: dict[str, subprocess.Popen] = {}
    monkeypatch.setattr(MCPServerTask, "_track_spawned_children", lambda self, pids: tracked.setdefault(self.name, set(pids)))
    monkeypatch.setattr(MCPServerTask, "_release_spawned_children", lambda self, pids: None)

    @asynccontextmanager
    async def _spawning_stdio_client(params, errlog=None):
        await asyncio.sleep(0)  # the other server snapshots its children before this spawn
        children[params.args[0]] = await asyncio.to_thread(
            subprocess.Popen, [sys.executable, "-c", "import time; time.sleep(30)"])
        await asyncio.sleep(0.3)  # the other server spawns while this one is still connecting
        yield (None, None)

    async def _connect_both():
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool_config._build_safe_env", return_value={}), \
             patch("tools.mcp_tool_config._resolve_stdio_command", side_effect=lambda command, env: (command, env)), \
             patch("tools.mcp_tool_config._write_stderr_log_header"), \
             patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None), \
             patch("tools.osv_check.check_package_for_malware", return_value=None), \
             patch("tools.mcp_tool.stdio_client", _spawning_stdio_client), \
             patch("tools.mcp_tool.ClientSession", lambda *a, **k: _StopAfterSpawn()):
            await asyncio.gather(*(_server(name)._run_stdio({"command": "fixture", "args": [name]})
                                   for name in ("alpha", "beta")), return_exceptions=True)

    try:
        asyncio.run(_connect_both())
    finally:
        for child in children.values():
            child.kill()
            child.wait()

    assert tracked == {name: {child.pid} for name, child in children.items()}
