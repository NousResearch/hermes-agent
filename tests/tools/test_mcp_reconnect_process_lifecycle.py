"""Regression tests for bounded stdio MCP reconnect process lifecycles."""

import asyncio
import os
import signal
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_watchdog_child_stays_in_sdk_owned_process_group(monkeypatch):
    """The watchdog must not create a nested group the SDK cannot reap."""
    from tools import mcp_stdio_watchdog as watchdog

    proc = MagicMock()
    proc.wait.return_value = 0
    proc.poll.return_value = 0
    popen = MagicMock(return_value=proc)
    monkeypatch.setattr(watchdog.subprocess, "Popen", popen)
    monkeypatch.setattr(watchdog.threading.Thread, "start", lambda self: None)

    assert watchdog.main([
        "--ppid", str(os.getpid()),
        "--create-time", "1.0",
        "--", "fake-mcp-server",
    ]) == 0

    kwargs = popen.call_args.kwargs
    assert kwargs.get("start_new_session", False) is False, (
        "a nested process group lets the SDK kill only the watchdog while the "
        "real MCP server and Chromium descendants survive"
    )


def test_reaper_escalates_when_group_survives_dead_leader(monkeypatch):
    """A dead wrapper PID must not suppress SIGKILL of live descendants."""
    from tools import mcp_tool

    pid = 424242
    pgid = 424242
    with mcp_tool._lock:
        mcp_tool._stdio_pids.clear()
        mcp_tool._orphan_stdio_pids.clear()
        mcp_tool._orphan_stdio_pid_servers.clear()
        mcp_tool._stdio_pgids.clear()
        mcp_tool._orphan_stdio_pids.add(pid)
        mcp_tool._orphan_stdio_pid_servers[pid] = "browser"
        mcp_tool._stdio_pgids[pid] = pgid

    killpg = MagicMock()
    monkeypatch.setattr(mcp_tool.os, "killpg", killpg)
    monkeypatch.setattr(mcp_tool.time, "sleep", lambda _seconds: None)

    with patch("gateway.status._pid_exists", return_value=False):
        mcp_tool._kill_orphaned_mcp_children(server_name="browser")

    assert call(pgid, signal.SIGTERM) in killpg.call_args_list
    assert call(pgid, 0) in killpg.call_args_list
    assert call(pgid, signal.SIGKILL) in killpg.call_args_list


@pytest.mark.asyncio
async def test_run_stdio_reaps_same_server_before_spawn_and_after_exit(monkeypatch):
    """Each attempt must synchronously reap its predecessor before returning."""
    from tools import mcp_tool

    server = mcp_tool.MCPServerTask("browser")
    reaper = MagicMock()
    monkeypatch.setattr(mcp_tool, "_kill_orphaned_mcp_children", reaper)
    monkeypatch.setattr(mcp_tool, "_snapshot_child_pids", lambda: set())
    monkeypatch.setattr(mcp_tool, "_write_stderr_log_header", lambda _name: None)
    monkeypatch.setattr(mcp_tool, "_get_mcp_stderr_log", lambda: MagicMock())
    monkeypatch.setattr(mcp_tool, "_wrap_command_with_watchdog", lambda command, args: (command, args))
    monkeypatch.setattr(
        "tools.osv_check.check_package_for_malware",
        lambda _command, _args: None,
    )

    transport = MagicMock()
    transport.__aenter__ = AsyncMock(return_value=(object(), object()))
    transport.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mcp_tool, "stdio_client", lambda *a, **kw: transport)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.initialize = AsyncMock(side_effect=RuntimeError("handshake failed"))
    monkeypatch.setattr(mcp_tool, "ClientSession", lambda *a, **kw: session)

    with pytest.raises(RuntimeError, match="handshake failed"):
        await server._run_stdio({"command": "fake", "connect_timeout": 1})

    assert reaper.call_args_list == [
        call(include_active=True, server_name="browser"),
        call(include_active=True, server_name="browser"),
    ]


@pytest.mark.asyncio
async def test_reconnect_breaker_trips_at_configured_failure_count(monkeypatch):
    """The reconnect loop must park at N failures, not spawn attempt N+1."""
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_MAX_RECONNECT_RETRIES", 3)
    monkeypatch.setattr(mcp_tool.asyncio, "sleep", AsyncMock())
    calls = 0

    class FailingTask(mcp_tool.MCPServerTask):
        def _is_http(self):
            return False

        async def _run_stdio(self, config):
            nonlocal calls
            calls += 1
            if calls == 1:
                self._ready.set()  # establish once, then lose the connection
            raise RuntimeError(f"failure {calls}")

        def _deregister_tools(self):
            self._registered_tool_names = []

        async def _wait_for_reconnect_or_shutdown(self, timeout=None):
            return "shutdown"

    server = FailingTask("browser")
    await server.run({"command": "fake"})

    assert calls == 3
    assert server._reconnect_retries == 3


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
async def test_ten_stdio_reconnects_keep_one_tree_and_reap_previous_pids(
    monkeypatch, tmp_path
):
    """Ten real stdio reconnects must never accumulate MCP/browser trees."""
    import json
    import sys
    import time

    psutil = pytest.importorskip("psutil")
    from tools import mcp_tool

    records_path = tmp_path / "processes.jsonl"
    fixture_path = tmp_path / "mcp_fixture.py"
    fixture_path.write_text(
        "import json, os, signal, subprocess, sys, time\n"
        "from mcp.server.fastmcp import FastMCP\n"
        f"records_path = {str(records_path)!r}\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "with open(records_path, 'a', encoding='utf-8') as fh:\n"
        "    fh.write(json.dumps({'server': os.getpid(), 'child': child.pid, "
        "'pgid': os.getpgrp()}) + '\\n')\n"
        "    fh.flush()\n"
        "mcp = FastMCP('reconnect-fixture')\n"
        "@mcp.tool()\n"
        "def ping(): return 'pong'\n"
        "mcp.run(transport='stdio')\n"
    )

    monkeypatch.setattr(
        "tools.osv_check.check_package_for_malware",
        lambda _command, _args: None,
    )

    def records():
        if not records_path.exists():
            return []
        return [json.loads(line) for line in records_path.read_text().splitlines() if line]

    def alive(pid):
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    async def wait_for(predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.05)
        raise AssertionError("timed out waiting for subprocess lifecycle transition")

    server = mcp_tool.MCPServerTask("browser-regression")
    run_task = asyncio.create_task(server.run({
        "command": sys.executable,
        "args": [str(fixture_path)],
        "connect_timeout": 10,
    }))

    try:
        await asyncio.wait_for(server._ready.wait(), timeout=15)
        await wait_for(lambda: len(records()) == 1)

        for generation in range(1, 11):
            previous = records()[-1]
            server._reconnect_event.set()
            await wait_for(lambda: len(records()) == generation + 1)
            await wait_for(
                lambda: not alive(previous["server"]) and not alive(previous["child"])
            )

            seen = records()
            assert sum(alive(item["server"]) for item in seen) <= 1
            assert all(not alive(item["child"]) for item in seen[:-1])
    finally:
        await server.shutdown()
        await asyncio.wait_for(run_task, timeout=15)
        for item in records():
            try:
                os.killpg(item["pgid"], signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    assert all(
        not alive(pid)
        for item in records()
        for pid in (item["server"], item["child"])
    )


@pytest.mark.asyncio
async def test_shutdown_reaps_tree_before_awaiting_cancelled_transport(monkeypatch):
    """A timed-out aclose must be force-reaped before cancellation can hang."""
    from tools import mcp_tool

    events = []
    release = asyncio.Event()

    async def hung_transport():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("cancel-observed")
            await release.wait()

    server = mcp_tool.MCPServerTask("browser")
    server._task = asyncio.create_task(hung_transport())
    await asyncio.sleep(0)

    def reap(**kwargs):
        events.append("reap")
        release.set()

    monkeypatch.setattr(mcp_tool, "_MCP_SHUTDOWN_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(mcp_tool, "_MCP_CANCEL_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(mcp_tool, "_kill_orphaned_mcp_children", reap)

    await server.shutdown()

    assert events[:2] == ["reap", "cancel-observed"]


@pytest.mark.asyncio
async def test_same_named_stdio_tasks_serialize_their_process_lifecycle():
    """Concurrent task instances may not own two trees for one server name."""
    from tools import mcp_tool

    active = 0
    max_active = 0
    first_entered = asyncio.Event()
    release = asyncio.Event()

    class ProbeTask(mcp_tool.MCPServerTask):
        async def _run_stdio_locked(self, config):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            first_entered.set()
            await release.wait()
            active -= 1

    first = ProbeTask("browser")
    second = ProbeTask("browser")
    first_run = asyncio.create_task(first._run_stdio({"command": "fake"}))
    await first_entered.wait()
    second_run = asyncio.create_task(second._run_stdio({"command": "fake"}))
    await asyncio.sleep(0.05)

    assert max_active == 1
    release.set()
    await asyncio.gather(first_run, second_run)
    assert max_active == 1


@pytest.mark.asyncio
async def test_shutdown_force_reaps_server_owned_process_tree(monkeypatch):
    """Shutdown must reap even if transport teardown never recorded an orphan."""
    from tools import mcp_tool

    server = mcp_tool.MCPServerTask("browser")
    reaper = MagicMock()
    monkeypatch.setattr(mcp_tool, "_kill_orphaned_mcp_children", reaper)

    await server.shutdown()

    reaper.assert_called_once_with(include_active=True, server_name="browser")
