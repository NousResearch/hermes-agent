"""Ownership of the stdio MCP server's child process (#134611 and the PR review's F1/F2).

* A child another gateway thread spawns inside the connect window (a cron worker or its
  ``systemd-run`` wrapper) is never attributed to the MCP server.
* The before/after snapshot delta is only taken across two samples of the same scope: a failed
  after-read must not widen to the process-wide union and claim a foreign child that existed
  before the connection (F1).
* Each connection ledgers exactly the process its SDK ``stdio_client`` spawned, so overlapping
  connects on the shared MCP loop never claim, orphan or signal each other's child (F2).

Real processes are only ever our own disposable sleepers; foreign PIDs in the modeled-procfs
tests are synthetic and every cleanup signal is intercepted.
"""
import asyncio
import builtins
import contextlib
import io
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool, mcp_tool_transport
from tools import mcp_tool_lifecycle as lifecycle
from tools.mcp_tool_lifecycle import _ChildSnapshot

_SLEEPER = [sys.executable, "-c", "import sys; sys.stdin.read()"]
_LEDGERS = ("_stdio_pids", "_stdio_pgids", "_stdio_starttimes", "_orphan_stdio_pids", "_orphan_stdio_pid_servers")
_REAL_REAPER = lifecycle._kill_orphaned_mcp_children


class _ForeignSpawner:
    """A long-lived non-MCP thread that spawns an isolated child on demand, like the cron thread."""

    def __init__(self):
        self.spawned = threading.Event()
        self.release = threading.Event()
        self.proc = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.proc = subprocess.Popen(_SLEEPER, stdin=subprocess.PIPE, start_new_session=True)
        self.spawned.set()
        self.release.wait()

    def __enter__(self):
        self.thread.start()
        assert self.spawned.wait(10)
        return self

    def __exit__(self, *exc):
        self.proc.kill()
        self.proc.wait(10)
        self.release.set()
        self.thread.join(10)


@pytest.fixture
def clean_ledgers():
    for name in _LEDGERS:
        getattr(lifecycle, name).clear()
    yield
    for name in _LEDGERS:
        getattr(lifecycle, name).clear()


@contextlib.contextmanager
def _stdio_harness():
    """Drive the real ``_run_stdio`` with the handshake mocked out; records the supervisor and
    machine spawn-ledger side effects instead of performing them."""
    supervisor, registered = [], []
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    session_cm.__aexit__ = AsyncMock(return_value=False)

    async def passthrough_preflight(name, command, args):
        return command, args

    with (
        patch("tools.mcp_tool._preflight_stdio_command", passthrough_preflight),
        patch("tools.mcp_tool.ClientSession", return_value=session_cm),
        patch("tools.mcp_tool_lifecycle._kill_orphaned_mcp_children"),  # the pre-spawn sweep
        patch("tools.mcp_tool._update_death_supervisor",
              side_effect=lambda verb, pgids: supervisor.append((verb, set(pgids)))),
        patch("hermes_cli.process_identity.register_child", side_effect=lambda pid, kind: registered.append(pid)),
    ):
        yield SimpleNamespace(supervisor=supervisor, registered=registered)


def _server(name: str, serve):
    task = mcp_tool.MCPServerTask(name)
    task._serve_session = serve
    task._session_kwargs = dict
    return task


def _config():
    return {"command": _SLEEPER[0], "args": _SLEEPER[1:]}


# ---------------------------------------------------------------- thread scope (#134611)

@pytest.mark.platforms("linux")
def test_foreign_thread_child_spawned_in_window_is_not_attributed():
    async def scenario():
        before = lifecycle._snapshot_spawn_children()
        with _ForeignSpawner() as foreign:
            own = await asyncio.create_subprocess_exec(*_SLEEPER, stdin=subprocess.PIPE,
                                                       start_new_session=True)
            try:
                after = lifecycle._snapshot_spawn_children(before.scope)
                delta = lifecycle._filter_mcp_children(lifecycle._spawned_child_delta(before, after))
                # The process-wide reading sees both children: that is the window the fix closes.
                assert {own.pid, foreign.proc.pid} <= lifecycle._snapshot_child_pids()
            finally:
                own.kill()
                await own.wait()
        return before.scope, delta, own.pid, foreign.proc.pid

    scope, delta, own_pid, foreign_pid = asyncio.run(scenario())
    assert scope == "thread"
    assert own_pid in delta
    assert foreign_pid not in delta


@pytest.mark.platforms("linux")
def test_sdk_stdio_client_exposes_the_process_it_spawned():
    """The exact binding reads the SDK's own spawn; if an SDK bump renames it, fail here loudly
    rather than silently falling back to the snapshot delta."""
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario():
        params = StdioServerParameters(command=_SLEEPER[0], args=_SLEEPER[1:])
        with _ForeignSpawner() as foreign:
            transport_cm = stdio_client(params)
            async with transport_cm:
                pid = lifecycle._sdk_spawned_pid(transport_cm)
                thread_children = lifecycle._snapshot_spawn_children().pids
            return pid, thread_children, foreign.proc.pid

    pid, thread_children, foreign_pid = asyncio.run(scenario())
    assert pid is not None and pid in thread_children
    assert pid != foreign_pid


def test_sdk_spawned_pid_is_none_for_a_mocked_client():
    assert lifecycle._sdk_spawned_pid(MagicMock()) is None


# ------------------------------------------------- F1: never mix sample scopes in the delta

def test_delta_across_incompatible_or_failed_samples_is_rejected():
    foreign, ours = 900001, 900002
    thread_before = _ChildSnapshot(frozenset(), "thread")
    assert lifecycle._spawned_child_delta(thread_before, _ChildSnapshot(frozenset({foreign, ours}), "process")) == set()
    assert lifecycle._spawned_child_delta(thread_before, _ChildSnapshot(frozenset(), None)) == set()
    assert lifecycle._spawned_child_delta(_ChildSnapshot(frozenset(), None), _ChildSnapshot(frozenset({ours}), None)) == set()
    assert lifecycle._spawned_child_delta(thread_before, _ChildSnapshot(frozenset({ours}), "thread")) == {ours}


@contextlib.contextmanager
def _modeled_procfs(foreign_pid: int, *, thread_reads):
    """Model ``/proc``: the calling thread's ``children`` file answers ``thread_reads`` in order
    (a string, or an exception to raise; empty once they run out); another gateway thread's file lists ``foreign_pid``,
    a child that has been alive since before the connection."""
    task_dir = f"/proc/{os.getpid()}/task"
    own_tid, other_tid = threading.get_native_id(), 987654321
    reads = iter(thread_reads)
    real_open, real_listdir = builtins.open, os.listdir

    def fake_open(path, *args, **kwargs):
        if path == f"{task_dir}/{own_tid}/children":
            answer = next(reads, "")
            if isinstance(answer, BaseException):
                raise answer
            return io.StringIO(answer)
        if path == f"{task_dir}/{other_tid}/children":
            return io.StringIO(str(foreign_pid))
        return real_open(path, *args, **kwargs)

    def fake_listdir(path):
        return [str(own_tid), str(other_tid)] if path == task_dir else real_listdir(path)

    with patch("builtins.open", fake_open), patch.object(lifecycle.os, "listdir", fake_listdir):
        yield


@pytest.mark.platforms("linux")
def test_failed_after_read_is_not_widened_to_the_process_wide_union():
    foreign = 900001
    with _modeled_procfs(foreign, thread_reads=["", PermissionError("after-only fault")]):
        before = lifecycle._snapshot_spawn_children()
        after = lifecycle._snapshot_spawn_children(before.scope)
        assert foreign in lifecycle._snapshot_child_pids()  # what the old fallback would have read
    assert before.scope == "thread"
    assert after.scope is None
    assert lifecycle._spawned_child_delta(before, after) == set()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("thread_reads", [
    ["", PermissionError("after-only fault")],
    [PermissionError("before-only fault"), ""],
])
def test_preexisting_foreign_child_never_enters_the_ledger_when_a_sample_fails(clean_ledgers, thread_reads):
    """The review's replay, through the real ``_run_stdio``: the SDK binding is unavailable (mocked
    client), one thread sample fails, and the foreign child must stay out of every ledger and never
    be selected by the orphan reaper."""
    foreign = 900001
    signals = []
    stdio_cm = MagicMock()
    stdio_cm.__aenter__ = AsyncMock(return_value=(object(), object()))
    stdio_cm.__aexit__ = AsyncMock(return_value=False)

    async def serve(session, timeout, **kwargs):
        return "ok"

    with (
        _stdio_harness() as harness,
        patch("tools.mcp_tool.stdio_client", return_value=stdio_cm),
        _modeled_procfs(foreign, thread_reads=thread_reads),
        patch("psutil.Process", side_effect=lambda pid: SimpleNamespace(cmdline=lambda: ["python", "worker.py"])),
        patch.object(mcp_tool_transport.os, "getpgid", side_effect=lambda pid: pid),
        patch("gateway.status._pid_exists", return_value=True),
        patch.object(lifecycle, "_signal_mcp_process", side_effect=lambda pid, sig, *a: signals.append((pid, sig))),
        patch.object(lifecycle.time, "sleep"),
    ):
        server = _server("replay", serve)
        asyncio.run(server._run_stdio(_config()))
        _REAL_REAPER()

    assert foreign not in lifecycle._stdio_pids and foreign not in lifecycle._stdio_pgids
    assert foreign not in lifecycle._orphan_stdio_pids
    assert foreign not in server._stdio_child_pids
    assert foreign not in harness.registered
    assert not any(foreign in pgids for _verb, pgids in harness.supervisor)
    assert not any(pid == foreign for pid, _sig in signals)


@pytest.mark.platforms("linux")
def test_real_transport_child_keeps_cleanup_coverage_when_a_sample_fails(clean_ledgers):
    """With the after-sample unavailable the delta is rejected, but the SDK's own process is still
    ledgered (PID, group, supervisor) and reported to the dead-child fast-fail."""
    seen = {}

    async def serve(session, timeout, **kwargs):
        seen["ledger"] = dict(lifecycle._stdio_pids)
        seen["pgids"] = dict(lifecycle._stdio_pgids)
        seen["child_pids"] = set(server._stdio_child_pids)
        return "ok"

    with (
        _stdio_harness() as harness,
        patch.object(lifecycle, "_read_thread_children", side_effect=[set(), None, None]),
    ):
        server = _server("covered", serve)
        asyncio.run(server._run_stdio(_config()))

    assert len(seen["child_pids"]) == 1
    (pid,) = seen["child_pids"]
    assert seen["ledger"] == {pid: "covered"}
    assert harness.registered == [pid]
    assert ("register", {seen["pgids"][pid]}) in harness.supervisor
    # The SDK tore its child down on exit, so the release forgot it rather than orphaning it.
    assert pid not in lifecycle._stdio_pids and pid not in lifecycle._orphan_stdio_pids


# ----------------------------------------- F2: overlapping connects on the shared MCP loop

class _Connection:
    """One stdio server held in session until ``close`` is set."""

    def __init__(self, name):
        self.in_session = asyncio.Event()
        self.close = asyncio.Event()
        self.server = _server(name, self._serve)

    async def _serve(self, session, timeout, **kwargs):
        self.in_session.set()
        await asyncio.wait_for(self.close.wait(), 30)
        return "ok"


def _pid_of(name):
    return {pid for pid, owner in lifecycle._stdio_pids.items() if owner == name}


async def _assert_close_a_leaves_b_untouched(a, b, harness, signals):
    (pid_b,) = _pid_of("B")
    pgid_b = lifecycle._stdio_pgids[pid_b]
    a.close.set()
    await asyncio.wait_for(a.task, 30)
    await asyncio.to_thread(_REAL_REAPER)
    assert lifecycle._stdio_pids.get(pid_b) == "B"
    assert lifecycle._stdio_pgids.get(pid_b) == pgid_b
    assert pid_b not in lifecycle._orphan_stdio_pids
    assert b.server._stdio_child_pids == {pid_b}
    assert not any(verb == "unregister" and pgid_b in pgids for verb, pgids in harness.supervisor)
    assert not any(pid in (pid_b, pgid_b) for pid, _sig in signals)
    assert os.path.exists(f"/proc/{pid_b}")
    b.close.set()
    await asyncio.wait_for(b.task, 30)


@pytest.mark.platforms("linux")
def test_interleaved_sdk_spawns_each_ledger_only_their_own_child(clean_ledgers):
    """Exact binding on its own: the spawn window lock is disabled and both SDK spawns are forced
    to complete before either connection attributes its child."""
    import mcp.client.stdio as sdk_stdio
    real_spawn = sdk_stdio._create_platform_compatible_process
    signals = []

    async def scenario(harness):
        both_spawned = asyncio.Event()
        spawned = []

        async def spawn_then_wait_for_the_other(*args, **kwargs):
            proc = await real_spawn(*args, **kwargs)
            spawned.append(proc.pid)
            if len(spawned) == 2:
                both_spawned.set()
            await asyncio.wait_for(both_spawned.wait(), 30)
            return proc

        a, b = _Connection("A"), _Connection("B")
        with patch.object(sdk_stdio, "_create_platform_compatible_process", spawn_then_wait_for_the_other):
            a.task = asyncio.create_task(a.server._run_stdio(_config()))
            b.task = asyncio.create_task(b.server._run_stdio(_config()))
            await asyncio.wait_for(asyncio.gather(a.in_session.wait(), b.in_session.wait()), 30)
        assert sorted(spawned) == sorted(_pid_of("A") | _pid_of("B"))
        assert a.server._stdio_child_pids == _pid_of("A") and len(_pid_of("A")) == 1
        assert b.server._stdio_child_pids == _pid_of("B") and len(_pid_of("B")) == 1
        await _assert_close_a_leaves_b_untouched(a, b, harness, signals)

    with (
        _stdio_harness() as harness,
        patch.object(lifecycle, "_stdio_spawn_window_lock", side_effect=asyncio.Lock),
        patch.object(lifecycle, "_signal_mcp_process", side_effect=lambda pid, sig, *a: signals.append((pid, sig))),
    ):
        asyncio.run(scenario(harness))


async def _yield_loop(turns: int = 200):
    """Hand the loop to every other ready task for a bounded number of turns."""
    for _ in range(turns):
        await asyncio.sleep(0)


@contextlib.asynccontextmanager
async def _unbindable_stdio_client(server_params, errlog=None):
    """A stdio transport the exact binding cannot read (no SDK frame), so ``_run_stdio`` falls back
    to the snapshot delta. Spawns from the loop thread like the SDK, then yields the loop so an
    unserialized sibling connect would land its own spawn inside this window."""
    child = await asyncio.create_subprocess_exec(*_SLEEPER, stdin=subprocess.PIPE, start_new_session=True)
    try:
        await _yield_loop()
        yield object(), object()
    finally:
        child.kill()
        await child.wait()


@pytest.mark.platforms("linux")
def test_overlapping_fallback_connects_are_serialized_and_released_before_the_handshake(clean_ledgers):
    """Snapshot fallback: the spawn window lock keeps B's spawn out of A's delta, and is released
    before A's session (B connects while A is still in session)."""
    signals = []

    async def scenario(harness):
        a, b = _Connection("A"), _Connection("B")
        a.task = asyncio.create_task(a.server._run_stdio(_config()))
        b.task = asyncio.create_task(b.server._run_stdio(_config()))
        await asyncio.wait_for(asyncio.gather(a.in_session.wait(), b.in_session.wait()), 30)
        assert len(_pid_of("A")) == 1 and len(_pid_of("B")) == 1
        assert _pid_of("A").isdisjoint(_pid_of("B"))
        assert a.server._stdio_child_pids == _pid_of("A")
        assert b.server._stdio_child_pids == _pid_of("B")
        await _assert_close_a_leaves_b_untouched(a, b, harness, signals)

    with (
        _stdio_harness() as harness,
        patch("tools.mcp_tool.stdio_client", _unbindable_stdio_client),
        patch.object(lifecycle, "_signal_mcp_process", side_effect=lambda pid, sig, *a: signals.append((pid, sig))),
    ):
        asyncio.run(scenario(harness))


@pytest.mark.platforms("linux")
def test_cancelling_a_connect_inside_the_spawn_window_releases_it(clean_ledgers):
    entered = None
    calls = []

    @contextlib.asynccontextmanager
    async def first_hangs_then_unbindable(server_params, errlog=None):
        calls.append(server_params)
        if len(calls) == 1:
            entered.set()
            await asyncio.Event().wait()  # never set: A is cancelled inside the window
        async with _unbindable_stdio_client(server_params, errlog) as streams:
            yield streams

    async def scenario():
        nonlocal entered
        entered = asyncio.Event()
        a, b = _Connection("A"), _Connection("B")
        a.task = asyncio.create_task(a.server._run_stdio(_config()))
        await asyncio.wait_for(entered.wait(), 30)
        b.task = asyncio.create_task(b.server._run_stdio(_config()))
        await _yield_loop()
        assert len(calls) == 1 and not b.in_session.is_set()  # B is queued behind A's window
        a.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await a.task
        await asyncio.wait_for(b.in_session.wait(), 30)
        assert not lifecycle._stdio_spawn_window_lock().locked()
        assert _pid_of("A") == set() and len(_pid_of("B")) == 1
        b.close.set()
        await asyncio.wait_for(b.task, 30)

    with _stdio_harness(), patch("tools.mcp_tool.stdio_client", first_hangs_then_unbindable):
        asyncio.run(scenario())
