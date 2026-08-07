"""Regression coverage for concurrent stdio MCP subprocess attribution."""

import asyncio
import subprocess
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import (
    MCPServerTask,
    _ensure_mcp_loop,
    _lock,
    _orphan_stdio_pid_servers,
    _orphan_stdio_pids,
    _stdio_pgids,
    _stdio_pids,
    _stdio_spawn_attribution_guard,
    _stop_mcp_loop,
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

    def __init__(self, *, force_spawn_overlap=False):
        self._preflight_barrier = threading.Barrier(2, timeout=10)
        self._force_spawn_overlap = force_spawn_overlap
        self._enter_count = 0
        self._both_entered = None
        self.enter_attempts = {"srv_a": 0, "srv_b": 0}
        self.processes = {}

    def preflight(self, _command, _args):
        # The real OSV preflight runs in asyncio.to_thread before the spawn
        # lock. Rendezvous there so both server coroutines are ready to race
        # regardless of whether the production lock correctly serializes them.
        try:
            self._preflight_barrier.wait()
        except threading.BrokenBarrierError:
            pytest.fail("concurrent stdio starts did not rendezvous in OSV preflight")

    async def enter(self, server_name):
        if server_name not in self.enter_attempts:  # pragma: no cover
            raise AssertionError(f"unexpected fake server name: {server_name}")
        self.enter_attempts[server_name] += 1

        if self._force_spawn_overlap:
            # Used only with the production guard patched out. Do not let either
            # fake transport spawn until both unguarded _run_stdio tasks have
            # taken their pre-spawn PID snapshots and reached __aenter__.
            if self._both_entered is None:
                self._both_entered = asyncio.Event()
            self._enter_count += 1
            if self._enter_count == 2:
                self._both_entered.set()
            await asyncio.wait_for(self._both_entered.wait(), timeout=2)
        else:
            await asyncio.sleep(0.05)

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
        patch(
            "tools.osv_check.check_package_for_malware",
            side_effect=rig.preflight,
        ),
        patch("tools.mcp_tool._write_stderr_log_header"),
        patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None),
    ):
        yield


@asynccontextmanager
async def _running_server_pair(*, force_spawn_overlap=False):
    rig = _ConcurrentSpawnRig(force_spawn_overlap=force_spawn_overlap)
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
            assert rig.enter_attempts == {"srv_a": 1, "srv_b": 1}
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


def test_fixture_deterministically_reproduces_race_without_guard():
    @asynccontextmanager
    async def no_spawn_guard():
        yield

    async def run_test():
        with patch(
            "tools.mcp_tool._stdio_spawn_attribution_guard",
            side_effect=no_spawn_guard,
        ):
            async with _running_server_pair(force_spawn_overlap=True) as (
                _rig,
                _srv_a,
                _srv_b,
            ):
                with _lock:
                    owner_counts = {
                        name: sum(owner == name for owner in _stdio_pids.values())
                        for name in ("srv_a", "srv_b")
                    }
                assert sorted(owner_counts.values()) == [0, 2]

    asyncio.run(run_test())


def test_spawn_guard_serializes_overlapping_event_loops():
    """Old and replacement MCP loops must share one attribution guard."""
    rendezvous = threading.Barrier(2, timeout=10)
    state_lock = threading.Lock()
    errors = []
    active = 0
    max_active = 0

    async def worker():
        nonlocal active, max_active
        await asyncio.to_thread(rendezvous.wait)
        async with _stdio_spawn_attribution_guard():
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            with state_lock:
                active -= 1

    def run_worker():
        try:
            asyncio.run(worker())
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=run_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert max_active == 1


def test_loop_stop_blocks_replacement_loop_until_final_cleanup():
    """The old loop's final PID sweep must finish before publishing a new loop."""
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    ensure_done = threading.Event()
    stop_result = []

    def fake_finish_stopping(_loop, _thread):
        cleanup_started.set()
        assert release_cleanup.wait(timeout=10)
        return True

    def stop_loop():
        stop_result.append(_stop_mcp_loop())

    def ensure_replacement_loop():
        _ensure_mcp_loop()
        ensure_done.set()

    with patch(
        "tools.mcp_tool._finish_stopping_mcp_loop",
        side_effect=fake_finish_stopping,
    ):
        stop_thread = threading.Thread(target=stop_loop)
        stop_thread.start()
        assert cleanup_started.wait(timeout=10)
        ensure_thread = threading.Thread(target=ensure_replacement_loop)
        ensure_thread.start()
        assert not ensure_done.wait(timeout=0.05)
        release_cleanup.set()
        stop_thread.join(timeout=10)
        ensure_thread.join(timeout=10)

    assert not stop_thread.is_alive()
    assert not ensure_thread.is_alive()
    assert stop_result == [True]
    assert ensure_done.is_set()
    assert _stop_mcp_loop()


def test_cancelled_spawn_guard_waiter_does_not_strand_lock():
    async def run_test():
        holder_started = asyncio.Event()
        release_holder = asyncio.Event()

        async def hold_guard():
            async with _stdio_spawn_attribution_guard():
                holder_started.set()
                await release_holder.wait()

        async def wait_for_guard():
            async with _stdio_spawn_attribution_guard():
                pass

        holder = asyncio.create_task(hold_guard())
        await holder_started.wait()
        waiter = asyncio.create_task(wait_for_guard())
        await asyncio.sleep(0.03)
        assert not waiter.done()

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        release_holder.set()
        await holder

        # Cancellation while queued must not acquire the threading lock later
        # and strand it after the cancelled coroutine is gone.
        async with asyncio.timeout(1):
            async with _stdio_spawn_attribution_guard():
                pass

    asyncio.run(run_test())
