"""MCP spawn ownership must exclude unrelated concurrent gateway children."""
import asyncio
from contextlib import asynccontextmanager
import os
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("other_owner", [None, "another-connection"])
def test_stdio_connection_does_not_adopt_a_concurrent_cron_child(monkeypatch, other_owner):
    from tools import mcp_tool as mcp
    from tools import mcp_tool_transport as transport

    assert mcp._ensure_mcp_sdk()
    real_stdio = mcp.stdio_client
    observed = []
    bystanders = []

    @asynccontextmanager
    async def racing_stdio(params, errlog):
        async with real_stdio(params, errlog=errlog) as streams:
            # A scheduler thread can spawn a perfectly ordinary child while
            # the MCP task is opening its own process and pipes.
            env = dict(os.environ)
            if other_owner is not None:
                env[transport._lifecycle._STDIO_SPAWN_TOKEN_ENV] = other_owner
            bystanders.append(await asyncio.to_thread(
                subprocess.Popen,
                [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
                stdin=subprocess.PIPE, start_new_session=True, env=env,
            ))
            yield streams

    monkeypatch.setattr(mcp, "stdio_client", racing_stdio)
    monkeypatch.setattr(mcp, "_preflight_stdio_command", AsyncMock(
        return_value=(sys.executable, ["-c", "import sys; sys.stdin.buffer.read()"])))
    monkeypatch.setattr(transport, "_connect_inputs", lambda *_: ((sys.executable, {}, None), {}))
    server = mcp.MCPServerTask("owned-child-regression")
    monkeypatch.setattr(server, "_track_spawned_children", lambda pids: observed.append(set(pids)))
    monkeypatch.setattr(server, "_release_spawned_children", lambda _pids: None)
    monkeypatch.setattr(server, "_serve_session", AsyncMock(return_value="shutdown"))
    try:
        asyncio.run(server._run_stdio({"command": sys.executable}))
        assert observed and observed[0], "the actual MCP child must still be owned"
        assert bystanders[0].pid not in observed[0], "MCP adopted an unrelated scheduler child"
        assert bystanders[0].poll() is None
    finally:
        for child in bystanders:
            child.stdin.close()
            child.wait(timeout=10)


def test_spawn_identity_read_failure_never_grants_cleanup_ownership(monkeypatch):
    import psutil
    from tools import mcp_tool_lifecycle as lifecycle

    class UnreadableProcess:
        def environ(self):
            raise psutil.AccessDenied(123)

    monkeypatch.setattr(psutil, "Process", lambda _pid: UnreadableProcess())
    assert lifecycle._filter_mcp_children({123}, spawn_token="this-connection") == set()
