"""Regression coverage for concurrent stdio MCP subprocess attribution."""

import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import (
    MCPServerTask,
    _lock,
    _orphan_stdio_pid_servers,
    _orphan_stdio_pids,
    _stdio_pgids,
    _stdio_pids,
)


_TRACKING_STATE = (
    _stdio_pids,
    _orphan_stdio_pids,
    _orphan_stdio_pid_servers,
    _stdio_pgids,
)

_FAKE_SERVER_NAME = ContextVar("fake_mcp_server_name", default=None)


def _clear_tracking_state():
    with _lock:
        for state in _TRACKING_STATE:
            state.clear()


@pytest.fixture(autouse=True)
def clean_stdio_pid_tracking():
    """Keep the module-level subprocess bookkeeping hermetic."""
    _clear_tracking_state()
    yield
    _clear_tracking_state()


class _ConcurrentSpawnRig:
    """Fake stdio transports backed by real, independently owned children."""

    def __init__(self):
        self._srv_a_entered = asyncio.Event()
        self._srv_b_entered = asyncio.Event()
        self.processes = {}

    async def enter(self, server_name):
        # _run_stdio takes pids_before before invoking this context manager.
        # Force both coroutines to reach __aenter__ (and therefore take their
        # snapshots) before either child exists. Then always spawn srv_b first:
        # srv_b sees only B, while srv_a subsequently sees both A and B and
        # overwrites B's ownership on the buggy snapshot-delta implementation.
        if server_name == "srv_a":
            self._srv_a_entered.set()
            await asyncio.wait_for(self._srv_b_entered.wait(), timeout=2)
        elif server_name == "srv_b":
            await asyncio.wait_for(self._srv_a_entered.wait(), timeout=2)
            self._srv_b_entered.set()
        else:  # pragma: no cover - protects the test rig from accidental misuse
            raise AssertionError(f"unexpected fake server name: {server_name}")

        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        self.processes[server_name] = process
        return MagicMock(name=f"{server_name}_read"), MagicMock(
            name=f"{server_name}_write"
        )

    def stop(self, server_name):
        process = self.processes.get(server_name)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def cleanup(self):
        for server_name in list(self.processes):
            self.stop(server_name)


@contextmanager
def _patched_stdio_runtime(rig):
    @asynccontextmanager
    async def fake_stdio_client(server_params, **_kwargs):
        server_name = _FAKE_SERVER_NAME.get()
        streams = await rig.enter(server_name)
        try:
            yield streams
        finally:
            # Match the real transport's ownership boundary: exiting srv_a's
            # transport terminates A's child, but must not touch B's live child.
            rig.stop(server_name)

    @asynccontextmanager
    async def fake_client_session(*_args, **_kwargs):
        session = MagicMock()
        session.initialize = AsyncMock(
            return_value=MagicMock(capabilities=MagicMock(tools=None))
        )
        yield session

    def fake_server_parameters(**kwargs):
        return SimpleNamespace(**kwargs)

    with (
        patch("tools.mcp_tool.StdioServerParameters", side_effect=fake_server_parameters),
        patch("tools.mcp_tool.stdio_client", side_effect=fake_stdio_client),
        patch("tools.mcp_tool.ClientSession", side_effect=fake_client_session),
        patch(
            "tools.mcp_tool._wrap_command_with_watchdog",
            side_effect=lambda command, args: (command, args),
        ),
        patch("tools.osv_check.check_package_for_malware", return_value=None),
        patch("tools.mcp_tool._write_stderr_log_header"),
        patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None),
    ):
        yield


@asynccontextmanager
async def _running_server_pair():
    rig = _ConcurrentSpawnRig()
    srv_a = MCPServerTask("srv_a")
    srv_b = MCPServerTask("srv_b")
    config = {"command": "dummy", "args": []}

    async def start(server):
        # MCPServerTask.start() creates its long-lived run task while this
        # context is active, so the label is copied into that task without
        # making the two production configs artificially different.
        token = _FAKE_SERVER_NAME.set(server.name)
        try:
            await server.start(config)
        finally:
            _FAKE_SERVER_NAME.reset(token)

    with _patched_stdio_runtime(rig):
        try:
            await asyncio.gather(start(srv_a), start(srv_b))
            yield rig, srv_a, srv_b
        finally:
            await asyncio.gather(
                srv_a.shutdown(), srv_b.shutdown(), return_exceptions=True
            )
            rig.cleanup()


def test_concurrent_spawns_attribute_exactly_one_pid_per_server():
    async def run_test():
        async with _running_server_pair() as (rig, _srv_a, _srv_b):
            spawned = {
                server_name: process.pid
                for server_name, process in rig.processes.items()
            }
            with _lock:
                owners = {
                    server_name: {
                        pid
                        for pid, owner in _stdio_pids.items()
                        if owner == server_name
                    }
                    for server_name in ("srv_a", "srv_b")
                }

            owner_counts = {name: len(pids) for name, pids in owners.items()}
            assert owner_counts == {"srv_a": 1, "srv_b": 1}, (
                f"concurrent stdio PID attribution counts were {owner_counts}; "
                f"owners={owners}, spawned={spawned}"
            )
            assert owners == {
                "srv_a": {spawned["srv_a"]},
                "srv_b": {spawned["srv_b"]},
            }

    asyncio.run(run_test())


def test_teardown_of_one_server_does_not_orphan_siblings():
    async def run_test():
        async with _running_server_pair() as (rig, srv_a, _srv_b):
            sibling_pid = rig.processes["srv_b"].pid

            await srv_a.shutdown()

            with _lock:
                orphan_pids = set(_orphan_stdio_pids)
                orphan_servers = dict(_orphan_stdio_pid_servers)
                active_owners = dict(_stdio_pids)

            assert sibling_pid not in orphan_pids, (
                f"srv_a teardown orphaned live srv_b PID {sibling_pid}; "
                f"orphan_servers={orphan_servers}"
            )
            assert active_owners.get(sibling_pid) == "srv_b", (
                f"srv_b PID {sibling_pid} lost its active ownership after "
                f"srv_a teardown; active_owners={active_owners}"
            )

    asyncio.run(run_test())
