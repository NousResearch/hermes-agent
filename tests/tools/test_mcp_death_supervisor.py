"""Contract tests for the shared parent-death supervisor for stdio MCP servers.

The end-to-end tests here spawn real processes and really SIGKILL a real parent,
because the whole point of this module is behaviour that only exists when a
process dies without running any Python cleanup. A mocked parent death proves
nothing about the guarantee.
"""

import asyncio
import contextlib
import io
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_death_supervisor, mcp_tool

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the supervisor is POSIX-only (process groups)"
)

SUPERVISOR = os.path.join(os.path.dirname(mcp_tool.__file__), "mcp_death_supervisor.py")

# Long enough that nothing here can pass because the victim exited on its own.
_VICTIM = [sys.executable, "-c", "import time; time.sleep(300)"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


def _wait_gone(pid: int, timeout: float = 15.0) -> bool:
    """Wait for a process this test does NOT own to disappear."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _wait_exited(proc: subprocess.Popen, timeout: float = 15.0) -> bool:
    """Wait for a direct child of this test to exit.

    ``os.kill(pid, 0)`` cannot be used for our own children: a killed child
    stays a zombie until someone reaps it, and signalling a zombie succeeds.
    """
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


# ---------------------------------------------------------------------------
# Target safety: this process signals whole process GROUPS, so a bad target is
# unusually expensive. killpg(0, ...) would signal the supervisor's own group.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pgid", [0, 1, -1, -5])
def test_refuses_process_groups_that_are_never_a_valid_target(pgid):
    assert mcp_death_supervisor._is_safe_target(
        pgid, own_pgid=4242, parent_pgid=777
    ) is False


def test_refuses_its_own_group_and_the_parents_group():
    assert mcp_death_supervisor._is_safe_target(
        4242, own_pgid=4242, parent_pgid=777
    ) is False
    assert mcp_death_supervisor._is_safe_target(
        777, own_pgid=4242, parent_pgid=777
    ) is False


def test_accepts_an_unrelated_group():
    assert mcp_death_supervisor._is_safe_target(
        999, own_pgid=4242, parent_pgid=777
    ) is True


# ---------------------------------------------------------------------------
# Control protocol
# ---------------------------------------------------------------------------


def test_registrations_survive_to_eof_and_unregistrations_are_dropped():
    stream = io.StringIO("register 111\nregister 222\nunregister 111\n")

    still_registered = mcp_death_supervisor._serve(
        stream, own_pgid=4242, parent_pgid=777
    )

    assert still_registered == {222}


def test_garbage_lines_do_not_cost_us_the_other_registrations():
    # A corrupted byte on the control pipe must not take down reaping for every
    # other server -- that would turn a cosmetic bug into leaked processes.
    stream = io.StringIO(
        "register 111\n"
        "\n"
        "register\n"
        "register notanumber\n"
        "register 222 333\n"
        "explode 444\n"
        "register 555\n"
    )

    still_registered = mcp_death_supervisor._serve(
        stream, own_pgid=4242, parent_pgid=777
    )

    assert still_registered == {111, 555}


def test_a_writer_that_never_sends_a_newline_cannot_grow_us_without_bound():
    """Found for real: iterating the stream let /dev/zero reach 15 GB.

    The supervisor is the last line of defense against leaked MCP servers, so
    it must not be the process that dies under memory pressure -- and a reader
    that buffers until a newline arrives is exactly that risk.
    """
    huge = "register " + ("0" * 10_000_000) + "\nregister 222\n"

    still_registered = mcp_death_supervisor._serve(
        io.StringIO(huge), own_pgid=4242, parent_pgid=777
    )

    # The overlong line is skipped, and the stream resyncs on the next one.
    assert still_registered == {222}


def test_a_line_truncated_by_the_cap_is_never_acted_on():
    # Truncation must not turn one pgid into a different, valid-looking one:
    # "register 999999" clipped to "register 9" would reap the wrong group.
    stream = io.StringIO("register " + "9" * (mcp_death_supervisor._MAX_LINE_CHARS))

    assert mcp_death_supervisor._serve(
        stream, own_pgid=4242, parent_pgid=777
    ) == set()


def test_unsafe_targets_are_rejected_at_registration_time():
    stream = io.StringIO("register 0\nregister 777\nregister 999\n")

    still_registered = mcp_death_supervisor._serve(
        stream, own_pgid=4242, parent_pgid=777
    )

    assert still_registered == {999}


def test_refuses_to_run_inside_the_parents_own_process_group():
    # Started without start_new_session, a killpg of the parent's group would
    # take the supervisor out before it could reap. It must not pretend to work.
    proc = subprocess.run(
        [sys.executable, SUPERVISOR, "--parent-pgid", str(os.getpgid(0))],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2
    assert "process group" in proc.stderr


# ---------------------------------------------------------------------------
# End to end: real processes, real death
# ---------------------------------------------------------------------------


def test_reaps_a_registered_group_when_the_control_pipe_reaches_eof():
    victim = subprocess.Popen(_VICTIM, start_new_session=True)
    supervisor = subprocess.Popen(
        [sys.executable, SUPERVISOR, "--parent-pgid", str(os.getpgid(0))],
        stdin=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        supervisor.stdin.write(f"register {os.getpgid(victim.pid)}\n")
        supervisor.stdin.flush()
        assert victim.poll() is None, "victim should outlive registration"

        # EOF is the death signal, whatever closed the pipe.
        supervisor.stdin.close()

        assert _wait_exited(victim), "registered group survived parent death"
    finally:
        _kill(victim.pid)
        _kill(supervisor.pid)
        victim.wait(timeout=10)
        supervisor.wait(timeout=10)


def test_leaves_an_unregistered_group_alone_at_eof():
    # The other failure direction, and the more damaging one: a clean Hermes
    # shutdown unregisters as it tears each server down, so EOF must not become
    # a kill-everything event for servers that were handed back.
    survivor = subprocess.Popen(_VICTIM, start_new_session=True)
    supervisor = subprocess.Popen(
        [sys.executable, SUPERVISOR, "--parent-pgid", str(os.getpgid(0))],
        stdin=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        pgid = os.getpgid(survivor.pid)
        supervisor.stdin.write(f"register {pgid}\nunregister {pgid}\n")
        supervisor.stdin.flush()
        supervisor.stdin.close()

        supervisor.wait(timeout=15)
        assert survivor.poll() is None, "a cleanly unregistered server was killed"
    finally:
        _kill(survivor.pid)
        _kill(supervisor.pid)
        survivor.wait(timeout=10)
        supervisor.wait(timeout=10)


# A stand-in for Hermes: registers a real child, then blocks forever holding the
# only write end of the control pipe. SIGKILLing it is the scenario the whole
# module exists for -- no cleanup code of ours gets to run.
_FAKE_PARENT = """
import os, subprocess, sys, time

supervisor = sys.argv[1]
victim = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
)
sup = subprocess.Popen(
    [sys.executable, supervisor, "--parent-pgid", str(os.getpgid(0))],
    stdin=subprocess.PIPE, text=True, start_new_session=True,
)
sup.stdin.write("register %d\\n" % os.getpgid(victim.pid))
sup.stdin.flush()
print("%d %d" % (victim.pid, sup.pid), flush=True)
time.sleep(300)
"""


# Reparented-to-init processes are by definition outside this test's subtree,
# so cleaning them up trips conftest's live-system kill guard. Real signal
# delivery to a real orphan is the entire point of these two tests.
@pytest.mark.live_system_guard_bypass
def test_reaps_the_server_when_the_registering_parent_is_sigkilled(tmp_path):
    script = tmp_path / "fake_parent.py"
    script.write_text(_FAKE_PARENT)

    parent = subprocess.Popen(
        [sys.executable, str(script), SUPERVISOR],
        stdout=subprocess.PIPE,
        text=True,
    )
    victim_pid = supervisor_pid = None
    try:
        victim_pid, supervisor_pid = (
            int(x) for x in parent.stdout.readline().split()
        )
        assert _alive(victim_pid)

        # No graceful anything: the parent never runs another line of Python.
        parent.kill()
        parent.wait(timeout=10)

        assert _wait_gone(victim_pid), (
            "stdio MCP server survived kill -9 of its Hermes parent"
        )
    finally:
        for pid in (victim_pid, supervisor_pid):
            if pid is not None:
                _kill(pid)
        _kill(parent.pid)


@pytest.mark.live_system_guard_bypass
def test_reaps_a_grandchild_left_in_the_registered_group(tmp_path):
    # Real shape of the bug: mcp-remote exits but leaves the `node` it spawned
    # behind. The grandchild reparents to init but keeps the pgid, so killpg
    # still reaches it -- which is why we track groups and not pids.
    script = tmp_path / "leaky_server.py"
    script.write_text(
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c',"
        " 'import time; time.sleep(300)'])\n"
        "print(child.pid, flush=True)\n"
    )

    # start_new_session mirrors how the MCP SDK spawns stdio servers.
    server = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    grandchild_pid = int(server.stdout.readline())
    server.wait(timeout=10)  # the direct child exits; the grandchild does not

    supervisor = subprocess.Popen(
        [sys.executable, SUPERVISOR, "--parent-pgid", str(os.getpgid(0))],
        stdin=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert _alive(grandchild_pid), "grandchild should outlive its parent"
        # server.pid is its own pgid leader, captured at spawn time exactly as
        # mcp_tool records it -- still usable after the leader itself exited.
        supervisor.stdin.write(f"register {server.pid}\n")
        supervisor.stdin.flush()
        supervisor.stdin.close()

        assert _wait_gone(grandchild_pid), "orphaned grandchild was not reaped"
    finally:
        _kill(grandchild_pid)
        _kill(supervisor.pid)
        supervisor.wait(timeout=10)


# ---------------------------------------------------------------------------
# Client side: what mcp_tool tells the supervisor
# ---------------------------------------------------------------------------


class _FakeSupervisor:
    """Stands in for the supervisor process, recording the control stream."""

    def __init__(self, exited=False):
        self.stdin = io.StringIO()
        self.pid = 4242
        self._exited = exited
        self._sent = ""
        self.closed = False
        _real_close = self.stdin.close

        def _close():
            # Mirror a real pipe: capture what was written before the write
            # end goes away, so tests can still assert on the control stream.
            self._sent = self.stdin.getvalue()
            self.closed = True
            _real_close()

        self.stdin.close = _close

    def poll(self):
        return 1 if self._exited else None

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def lines(self):
        if self.closed:
            return self._sent.splitlines()
        return self.stdin.getvalue().splitlines()


@pytest.fixture(autouse=True)
def _reset_client_state():
    yield
    mcp_tool._death_supervisor = None
    mcp_tool._supervised_pgids.clear()


@pytest.fixture
def all_groups_alive(monkeypatch):
    """Answer every liveness probe with "this group exists".

    The protocol tests below register synthetic pgids that were never real
    process groups. Without this, the liveness prune correctly discards them
    before the control stream can be asserted on -- so state the precondition
    rather than letting these tests depend on pid-space luck.
    """
    monkeypatch.setattr(mcp_tool.os, "killpg", lambda pgid, sig: None)


def test_register_starts_the_supervisor_once_and_reuses_it(monkeypatch, all_groups_alive):
    spawned = []

    def _spawn():
        fake = _FakeSupervisor()
        spawned.append(fake)
        return fake

    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", _spawn)

    mcp_tool._update_death_supervisor("register", [111])
    mcp_tool._update_death_supervisor("register", [222])

    assert len(spawned) == 1, "each register spawned its own supervisor"
    assert spawned[0].lines() == ["register 111", "register 222"]


def test_unregister_is_forwarded(monkeypatch, all_groups_alive):
    fake = _FakeSupervisor()
    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", lambda: fake)

    mcp_tool._update_death_supervisor("register", [111])
    mcp_tool._update_death_supervisor("unregister", [111])

    assert fake.lines() == ["register 111", "unregister 111"]
    assert mcp_tool._supervised_pgids == set()


def test_supervisor_is_released_once_nothing_is_left_to_reap(monkeypatch, all_groups_alive):
    """An empty registration set must not keep a supervisor resident.

    A gateway that once connected a stdio server would otherwise carry a
    ~15 MB process and a live pipe for the rest of its life. Closing our
    write end is the same EOF the supervisor treats as parent death; with
    nothing registered it exits without reaping. The next register starts a
    fresh one, exactly like the dead-supervisor replay path.
    """
    spawned = []

    def _spawn():
        fake = _FakeSupervisor()
        spawned.append(fake)
        return fake

    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", _spawn)

    mcp_tool._update_death_supervisor("register", [111, 222])
    mcp_tool._update_death_supervisor("unregister", [111])
    assert not spawned[0].closed, "released the supervisor while a group was still registered"

    mcp_tool._update_death_supervisor("unregister", [222])
    assert spawned[0].closed, "supervisor kept resident with nothing left to reap"
    assert getattr(spawned[0], "waited", False), "released supervisor was never wait()ed -> zombie until the next Popen"
    assert spawned[0].lines()[-1] == "unregister 222", "release happened before the last unregister was sent"
    assert mcp_tool._death_supervisor is None

    mcp_tool._update_death_supervisor("register", [333])
    assert len(spawned) == 2 and spawned[1].lines() == ["register 333"]


def test_supervisor_survives_the_real_eof_release():
    """End to end: closing the control pipe with nothing registered exits cleanly."""
    if os.name != "posix":
        pytest.skip("POSIX-only supervisor")
    child = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        mcp_tool._update_death_supervisor("register", [os.getpgid(child.pid)])
        proc = mcp_tool._death_supervisor
        assert proc is not None and proc.poll() is None
        mcp_tool._update_death_supervisor("unregister", [os.getpgid(child.pid)])
        assert mcp_tool._death_supervisor is None
        assert proc.wait(timeout=10) == 0, "supervisor did not exit on the release EOF"
        assert child.poll() is None, "release reaped a group that had been unregistered"
    finally:
        _kill(child.pid)
        child.wait(timeout=10)


def test_unregister_alone_does_not_start_a_supervisor(monkeypatch):
    spawned = []
    monkeypatch.setattr(
        mcp_tool,
        "_spawn_death_supervisor",
        lambda: spawned.append(1) or _FakeSupervisor(),
    )

    mcp_tool._update_death_supervisor("unregister", [111])

    assert spawned == []


def test_a_dead_supervisor_is_replaced_and_live_coverage_replayed(monkeypatch, all_groups_alive):
    dead = _FakeSupervisor(exited=True)
    replacement = _FakeSupervisor()
    queue = [dead, replacement]
    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", lambda: queue.pop(0))

    mcp_tool._update_death_supervisor("register", [111])
    mcp_tool._update_death_supervisor("register", [222])

    # Losing the supervisor must not silently drop the server registered with
    # it -- the replacement has to be told about 111 as well as 222.
    assert set(replacement.lines()) == {"register 111", "register 222"}


def test_replay_does_not_resurrect_an_unregistered_group(monkeypatch, all_groups_alive):
    dead = _FakeSupervisor(exited=True)
    replacement = _FakeSupervisor()
    queue = [dead, replacement]
    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", lambda: queue.pop(0))

    mcp_tool._update_death_supervisor("register", [111])
    mcp_tool._update_death_supervisor("register", [222])
    mcp_tool._update_death_supervisor("unregister", [111])

    assert mcp_tool._supervised_pgids == {222}
    # 111 was legitimately replayed to the replacement (it was live when the
    # dead supervisor was swapped out), then unregistered. What must never
    # happen is a replay AFTER the unregister bringing it back.
    lines = replacement.lines()
    assert lines.index("unregister 111") > lines.index("register 111")
    assert "register 111" not in lines[lines.index("unregister 111") :]
    mcp_tool._update_death_supervisor("register", [333])  # any later replay/append
    assert "register 111" not in replacement.lines()[len(lines) :]


def test_a_broken_pipe_never_propagates_into_a_live_mcp_session(monkeypatch, all_groups_alive):
    class _BrokenPipe(_FakeSupervisor):
        def __init__(self):
            super().__init__()

            class _Stdin:
                def write(self, _payload):
                    raise BrokenPipeError("supervisor exited after poll()")

                def flush(self):
                    pass

            self.stdin = _Stdin()

    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", _BrokenPipe)

    mcp_tool._update_death_supervisor("register", [111])  # must not raise

    # Dropped, so the next registration respawns instead of writing into a
    # pipe that is known to be dead.
    assert mcp_tool._death_supervisor is None


def test_unregister_after_a_broken_pipe_rebuilds_coverage_for_survivors(monkeypatch, all_groups_alive):
    """A lost supervisor must be replaced by the NEXT lifecycle event, whatever its verb.

    Sequence from the #93517 review: two groups live, the control pipe dies
    (write fails, supervisor dropped, set retained), then a clean teardown
    unregisters one of them. Keying the no-spawn fast path on the verb left
    the survivor recorded but unsupervised; it must be keyed on the set.
    """
    spawned = []

    def _spawn():
        fake = _FakeSupervisor()
        spawned.append(fake)
        return fake

    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", _spawn)
    mcp_tool._update_death_supervisor("register", [111, 222])

    class _DeadStdin:
        def write(self, _payload):
            raise BrokenPipeError("supervisor died")

        def flush(self):
            pass

    spawned[0].stdin = _DeadStdin()
    mcp_tool._update_death_supervisor("register", [333])  # the write fails; supervisor dropped
    assert mcp_tool._death_supervisor is None
    assert mcp_tool._supervised_pgids == {111, 222, 333}

    mcp_tool._update_death_supervisor("unregister", [222])

    assert len(spawned) == 2, "unregister after a lost supervisor did not respawn one"
    assert sorted(spawned[1].lines()) == ["register 111", "register 333"], (
        "the replacement did not receive the surviving groups"
    )
    assert mcp_tool._death_supervisor is spawned[1]


def test_a_supervisor_that_cannot_start_is_not_fatal(monkeypatch, all_groups_alive):
    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", lambda: None)

    mcp_tool._update_death_supervisor("register", [111])  # must not raise

    assert mcp_tool._death_supervisor is None


@contextlib.contextmanager
def _stdio_connection(child_pid, fake_supervisor):
    """Drive the real MCPServerTask._run_stdio with a known spawned child.

    Only the MCP transport itself is mocked. Everything the supervisor wiring
    depends on -- child discovery, _filter_mcp_children, the real os.getpgid
    lookup -- runs for real against ``child_pid``, so the pgid asserted on is
    the pgid of an actual process rather than a fixture value.
    """
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    stdio_cm = MagicMock()
    stdio_cm.__aenter__ = AsyncMock(return_value=(object(), object()))
    stdio_cm.__aexit__ = AsyncMock(return_value=False)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("tools.mcp_tool.stdio_client", return_value=stdio_cm),
        patch("tools.mcp_tool.ClientSession", return_value=session_cm),
        # First call is the pids_before baseline; the second reports our child
        # as the newly spawned server.
        patch(
            "tools.mcp_tool._snapshot_child_pids",
            side_effect=[set(), {child_pid}],
        ),
        patch("tools.mcp_tool._write_stderr_log_header"),
        patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None),
        patch(
            "tools.mcp_tool._spawn_death_supervisor",
            return_value=fake_supervisor,
        ),
    ):
        yield mcp_tool.MCPServerTask("supervisor-wiring")


@pytest.mark.skipif(not mcp_tool._MCP_AVAILABLE, reason="MCP SDK not installed")
def test_connecting_a_stdio_server_registers_its_real_process_group():
    fake = _FakeSupervisor()
    child = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        with _stdio_connection(child.pid, fake) as server:
            asyncio.run(server.start({"command": "echo", "args": ["hi"]}))

        assert f"register {os.getpgid(child.pid)}" in fake.lines(), (
            "connecting a stdio server did not hand its process group to the "
            f"supervisor; control stream was {fake.lines()}"
        )
    finally:
        _kill(child.pid)
        child.wait(timeout=10)


@pytest.mark.skipif(not mcp_tool._MCP_AVAILABLE, reason="MCP SDK not installed")
def test_a_server_that_exited_is_released_on_teardown():
    fake = _FakeSupervisor()
    child = subprocess.Popen(_VICTIM, start_new_session=True)
    pgid = None
    try:
        with _stdio_connection(child.pid, fake) as server:

            async def _connect_then_lose_the_child():
                await server.start({"command": "echo", "args": ["hi"]})
                nonlocal pgid
                pgid = os.getpgid(child.pid)
                # The server exits while connected. Reap it here so the
                # teardown path sees a genuinely dead pid, not a zombie.
                child.kill()
                child.wait(timeout=10)
                await server.shutdown()

            asyncio.run(_connect_then_lose_the_child())

        assert f"register {pgid}" in fake.lines()
        assert f"unregister {pgid}" in fake.lines(), (
            "a stdio server with nothing left alive stayed registered, so the "
            f"supervisor would keep a stale group; stream was {fake.lines()}"
        )
    finally:
        _kill(child.pid)


@pytest.mark.skipif(not mcp_tool._MCP_AVAILABLE, reason="MCP SDK not installed")
def test_a_server_that_survived_teardown_stays_registered():
    # The case the whole module exists for: teardown did not manage to kill it.
    # Releasing it here would hand the orphan back to nobody.
    fake = _FakeSupervisor()
    child = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        with _stdio_connection(child.pid, fake) as server:

            async def _connect_then_shutdown():
                await server.start({"command": "echo", "args": ["hi"]})
                await server.shutdown()

            asyncio.run(_connect_then_shutdown())

        pgid = os.getpgid(child.pid)
        assert f"register {pgid}" in fake.lines()
        assert f"unregister {pgid}" not in fake.lines(), (
            "a server that outlived teardown was released from the supervisor, "
            "so an ungraceful exit would leave it running forever"
        )
    finally:
        _kill(child.pid)
        child.wait(timeout=10)


@pytest.mark.live_system_guard_bypass
def test_scoped_teardown_of_one_owner_keeps_the_other_owner_supervised(monkeypatch):
    """Two owners (profiles / agents) each hold a stdio group; tearing one down
    must release only that owner's group and leave the other covered, and the
    per-process supervisor must then still know about the survivor.

    Exercises the real registry + ``_kill_orphaned_mcp_children`` scoping
    rather than the control protocol alone (review request on #93517).
    """
    fake = _FakeSupervisor()
    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", lambda: fake)
    monkeypatch.setattr(mcp_tool.time, "sleep", lambda _s: None)  # skip the SIGTERM grace wait
    a = subprocess.Popen(_VICTIM, start_new_session=True)
    b = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        pg_a, pg_b = os.getpgid(a.pid), os.getpgid(b.pid)
        with mcp_tool._lock:
            mcp_tool._stdio_pids[a.pid] = "profile-a"
            mcp_tool._stdio_pids[b.pid] = "profile-b"
            mcp_tool._stdio_pgids[a.pid] = pg_a
            mcp_tool._stdio_pgids[b.pid] = pg_b
        mcp_tool._update_death_supervisor("register", [pg_a, pg_b])

        mcp_tool._kill_orphaned_mcp_children(include_active=True, server_name="profile-a")
        a.wait(timeout=10)

        assert b.poll() is None, "scoped teardown of profile-a killed profile-b's server"
        assert f"unregister {pg_a}" in fake.lines()
        assert f"unregister {pg_b}" not in fake.lines(), (
            "scoped teardown released the OTHER owner's group from the supervisor"
        )
        assert mcp_tool._supervised_pgids == {pg_b}
        assert b.pid in mcp_tool._stdio_pids and b.pid in mcp_tool._stdio_pgids
    finally:
        for p in (a, b):
            _kill(p.pid)
            try:
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        with mcp_tool._lock:
            for p in (a, b):
                mcp_tool._stdio_pids.pop(p.pid, None)
                mcp_tool._stdio_pgids.pop(p.pid, None)


@pytest.mark.live_system_guard_bypass
def test_a_group_with_nothing_left_alive_is_forgotten_and_unregistered(monkeypatch):
    """A dead group must not stay registered: its pgid can be recycled.

    Uses a real process so the liveness probe is answered by the kernel rather
    than a fixture -- the whole point is that we notice actual death.
    """
    fake = _FakeSupervisor()
    monkeypatch.setattr(mcp_tool, "_spawn_death_supervisor", lambda: fake)

    doomed = subprocess.Popen(_VICTIM, start_new_session=True)
    doomed_pgid = os.getpgid(doomed.pid)
    survivor = subprocess.Popen(_VICTIM, start_new_session=True)
    survivor_pgid = os.getpgid(survivor.pid)
    try:
        mcp_tool._update_death_supervisor("register", [doomed_pgid, survivor_pgid])
        assert mcp_tool._supervised_pgids == {doomed_pgid, survivor_pgid}

        # Reap it fully so the group is genuinely empty, not a zombie.
        doomed.kill()
        doomed.wait(timeout=10)

        # Any later registration change is when we notice.
        mcp_tool._update_death_supervisor("register", [survivor_pgid])

        assert doomed_pgid not in mcp_tool._supervised_pgids, (
            "a group with no members left stayed registered, so a recycled "
            "pgid could later be reaped as if it were an MCP server"
        )
        assert survivor_pgid in mcp_tool._supervised_pgids, (
            "pruning dropped a group that is still alive"
        )
        assert f"unregister {doomed_pgid}" in fake.lines(), (
            "the supervisor was never told to forget the dead group"
        )
    finally:
        _kill(survivor.pid)
        survivor.wait(timeout=10)
        _kill(doomed.pid)


def test_pruning_keeps_groups_it_cannot_prove_are_gone(monkeypatch):
    # An ambiguous probe (EPERM: exists but not ours) must not drop coverage --
    # losing a real registration is worse than keeping a doubtful one.
    monkeypatch.setattr(mcp_tool, "_supervised_pgids", {111, 222}, raising=False)

    def _probe(pgid, sig):
        if pgid == 111:
            raise PermissionError("exists, not ours")
        raise ProcessLookupError("gone")

    monkeypatch.setattr(mcp_tool.os, "killpg", _probe)

    stale = mcp_tool._prune_dead_supervised_pgids()

    assert stale == {222}
    assert mcp_tool._supervised_pgids == {111}


def test_no_pgids_is_a_no_op(monkeypatch):
    spawned = []
    monkeypatch.setattr(
        mcp_tool,
        "_spawn_death_supervisor",
        lambda: spawned.append(1) or _FakeSupervisor(),
    )

    mcp_tool._update_death_supervisor("register", [])

    assert spawned == []
