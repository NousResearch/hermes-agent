"""A stdio MCP server tracks exactly the child IT spawned, even when two servers connect at once.

The child is found as the delta of the process's children across the spawn. Two servers connecting
together on the shared MCP loop used to take both snapshots before either spawned, so each delta held
both children: stopping one server (a profile's ``/reload-mcp``, a server removed from config) then
reaped the OTHER server's live process as its orphan (#58414).
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

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

    spawned = {name: asyncio.Event() for name in ("alpha", "beta")}

    @asynccontextmanager
    async def _spawning_stdio_client(params, errlog=None):
        name = params.args[0]
        await asyncio.sleep(0)  # the other server snapshots its children before this spawn
        children[name] = await asyncio.to_thread(
            subprocess.Popen, [sys.executable, "-c", "import time; time.sleep(30)"])
        spawned[name].set()
        # Unserialized, the other server spawns now, inside this one's snapshot window; serialized, it
        # is still waiting for the window, so the wait runs out.
        other = "beta" if name == "alpha" else "alpha"
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(spawned[other].wait(), timeout=2)
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
            failures = await asyncio.gather(*(_server(name)._run_stdio({"command": "fixture", "args": [name]})
                                              for name in ("alpha", "beta")), return_exceptions=True)
            assert all(isinstance(exc, RuntimeError) and str(exc) == "stop after the spawn" for exc in failures)

    try:
        asyncio.run(_connect_both())
    finally:
        for child in children.values():
            child.kill()
            child.wait()

    assert set(children) == {"alpha", "beta"}
    assert tracked == {name: {child.pid} for name, child in children.items()}


def test_cancelling_a_queued_spawn_closes_stderr_without_releasing_another_spawn(monkeypatch):
    from tools import mcp_tool_config, mcp_tool_transport

    tees = []
    real_tee = mcp_tool_config._StderrTee

    def capture_tee(log):
        tee = real_tee(log)
        tees.append(tee)
        return tee

    async def cancel_waiter():
        waiting = asyncio.Event()

        class HeldSpawnLock(asyncio.Lock):
            async def acquire(self):
                waiting.set()
                return await super().acquire()

        lock = HeldSpawnLock()
        await lock.acquire()
        waiting.clear()
        monkeypatch.setattr(mcp_tool_transport, '_stdio_spawn_lock', lambda: lock)
        task = asyncio.create_task(_server('waiting')._run_stdio({'command': 'fixture'}))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(tees) == 1
            assert tees[0].sink.closed, 'a cancelled queued connection must close its stderr pipe'
            assert lock.locked(), 'the queued connection must leave the active spawn lock held'
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            lock.release()
            for tee in tees:
                tee.close()

    with patch('tools.mcp_tool._MCP_AVAILABLE', True), \
         patch('tools.mcp_tool_config._build_safe_env', return_value={}), \
         patch('tools.mcp_tool_config._resolve_stdio_command', side_effect=lambda command, env: (command, env)), \
         patch('tools.mcp_tool_config._write_stderr_log_header'), \
         patch('tools.mcp_tool._get_mcp_stderr_log', return_value=None), \
         patch('tools.mcp_tool_config._StderrTee', capture_tee), \
         patch('tools.osv_check.check_package_for_malware', return_value=None):
        asyncio.run(cancel_waiter())
