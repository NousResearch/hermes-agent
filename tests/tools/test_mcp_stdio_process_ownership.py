"""Concurrent MCP launches must not lend their child identities to siblings."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from tools import mcp_tool as core
from tools import mcp_tool_lifecycle as lifecycle
from tools import mcp_tool_transport as transport


def test_concurrent_stdio_transports_track_only_their_spawn(monkeypatch):
    """A sibling or unrelated child created during an await is not this server's."""
    assert core._ensure_mcp_sdk()
    from mcp.client import stdio

    async def run():
        both_spawned = asyncio.Event()
        served = {name: asyncio.Event() for name in ("owner-a", "owner-b")}
        release = asyncio.Event()
        entered = 0
        expected = {"owner-a": 11001, "owner-b": 11002}
        observed = {}

        async def spawn(*, command, **kwargs):
            return SimpleNamespace(pid=expected[command])

        @asynccontextmanager
        async def client(params, **kwargs):
            nonlocal entered
            await stdio._create_platform_compatible_process(command=params.command)
            entered += 1
            if entered == 2:
                both_spawned.set()
            await both_spawned.wait()
            yield object(), object()

        @asynccontextmanager
        async def session(*args, **kwargs):
            yield object()

        async def preflight(name, command, args):
            return command, args

        monkeypatch.setattr(stdio, "_create_platform_compatible_process", spawn)
        monkeypatch.setattr(core, "stdio_client", client)
        monkeypatch.setattr(core, "ClientSession", session)
        monkeypatch.setattr(core, "_preflight_stdio_command", preflight)
        monkeypatch.setattr(lifecycle, "_kill_orphaned_mcp_children", lambda: None)
        snapshots = iter([set(), set(), {11001, 11002, 11999}, {11001, 11002, 11999}])
        monkeypatch.setattr(lifecycle, "_snapshot_child_pids", lambda: next(snapshots))
        monkeypatch.setattr(transport, "_filter_mcp_children", lambda pids: pids, raising=False)
        monkeypatch.setattr("hermes_cli.process_identity.attach_self_to_kill_on_close_job", lambda: None)

        def make(name):
            task = core.MCPServerTask(name)
            task._session_kwargs = dict
            task._track_spawned_children = lambda pids: observed.setdefault(name, set()).update(pids)
            task._release_spawned_children = lambda pids: None

            async def serve(*args, **kwargs):
                served[name].set()
                await release.wait()
                return "shutdown"

            task._serve_session = serve
            return task

        a, b = make("owner-a"), make("owner-b")
        tasks = [asyncio.create_task(t._run_stdio({"command": t.name})) for t in (a, b)]
        try:
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in served.values())), 5)
            assert a._stdio_child_pids == {11001}
            assert b._stdio_child_pids == {11002}
            assert observed == {"owner-a": {11001}, "owner-b": {11002}}
        finally:
            release.set()
            await asyncio.gather(*tasks)

    asyncio.run(run())


def test_orphan_sweep_does_not_take_an_active_connector_child(monkeypatch):
    """A stale orphan record is not authority to reap a still-owned active PID."""
    monkeypatch.setattr(lifecycle, "_stdio_pids", {11001: "active"})
    monkeypatch.setattr(lifecycle, "_orphan_stdio_pids", {11001, 11002})
    monkeypatch.setattr(lifecycle, "_orphan_stdio_pid_servers", {11001: "old", 11002: "orphan"})
    monkeypatch.setattr(lifecycle, "_stdio_pgids", {})
    monkeypatch.setattr(lifecycle, "_stdio_starttimes", {})
    pids, _, _ = lifecycle._take_reapable_pids(False, None)
    assert pids == {11002: "orphan"}
    assert lifecycle._stdio_pids == {11001: "active"}


def test_release_preserves_another_transport_ownership(monkeypatch):
    owners = {11001: "other"}
    orphaned = set()
    monkeypatch.setattr(transport, "_stdio_pids", owners)
    monkeypatch.setattr(transport, "_orphan_stdio_pids", orphaned)
    monkeypatch.setattr(core, "_update_death_supervisor", lambda *args: None)
    core.MCPServerTask("closing")._release_spawned_children({11001})
    assert owners == {11001: "other"}
    assert orphaned == set()


def test_new_launch_clears_reused_orphan_record(monkeypatch):
    owners, orphaned, orphan_owners = {}, {11001}, {11001: "old"}
    monkeypatch.setattr(transport, "_stdio_pids", owners)
    monkeypatch.setattr(transport, "_stdio_pgids", {})
    monkeypatch.setattr(transport, "_stdio_starttimes", {})
    monkeypatch.setattr(transport, "_orphan_stdio_pids", orphaned)
    monkeypatch.setattr(transport, "_orphan_stdio_pid_servers", orphan_owners)
    monkeypatch.setattr(transport, "_leader_start_time", lambda pid: 123)
    monkeypatch.setattr(core, "_update_death_supervisor", lambda *args: None)
    monkeypatch.setattr("hermes_cli.process_identity.register_child", lambda *args: None)
    core.MCPServerTask("new")._track_spawned_children({11001})
    assert owners == {11001: "new"}
    assert orphaned == set()
    assert orphan_owners == {}


def test_final_sweep_consumes_overlapping_active_orphan_marker(monkeypatch):
    monkeypatch.setattr(lifecycle, "_stdio_pids", {11001: "active"})
    monkeypatch.setattr(lifecycle, "_orphan_stdio_pids", {11001})
    monkeypatch.setattr(lifecycle, "_orphan_stdio_pid_servers", {11001: "old"})
    monkeypatch.setattr(lifecycle, "_stdio_pgids", {})
    monkeypatch.setattr(lifecycle, "_stdio_starttimes", {11001: 123})
    pids, _, starts = lifecycle._take_reapable_pids(True, None)
    assert pids == {11001: "active"}
    assert starts == {11001: 123}
    assert lifecycle._take_reapable_pids(False, None) == ({}, {}, {})
