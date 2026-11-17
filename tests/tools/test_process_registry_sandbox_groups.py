"""A sandbox background job's kill stops its whole process group, or reports who survived."""
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import psutil
import pytest

from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_termination import (
    ENV_GONE_MARKER, ENV_STALE_MARKER, ENV_SURVIVORS_MARKER, env_group_kill_command, parse_env_group_kill)


class BashEnv:
    """The remote ``execute`` contract on a real non-interactive bash with no tty, the
    shape of a Sprites, SSH or Docker exec."""

    def __init__(self, directory):
        self.directory = directory

    def get_temp_dir(self):
        return str(self.directory)

    def execute(self, command, timeout=10, **_kwargs):
        # A file, not a pipe: the backgrounded job inherits the descriptor, and a pipe
        # read would wait for it past the launch shell's exit.
        with tempfile.TemporaryFile(mode="w+") as output:
            result = subprocess.run(
                ["bash", "-c", command], stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, timeout=timeout, check=False, start_new_session=True)
            output.seek(0)
            return {"output": output.read(), "returncode": result.returncode}


def _alive(proc):
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _writer_command(tmp_path, progress, *, ignore_term, shell_ignores_term=None):
    """A shell that runs a Python writer as its child, so the writer is a grandchild of
    the recorded PID (a kill of that PID alone leaves it writing)."""
    shell_ignores_term = ignore_term if shell_ignores_term is None else shell_ignores_term
    writer = tmp_path / "writer.py"
    writer.write_text(
        "import signal, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"with open({str(progress)!r}, 'a', buffering=1) as output:\n"
        "    while True:\n"
        "        output.write('tick\\n')\n"
        "        time.sleep(0.05)\n",
        encoding="utf-8")
    trap = "trap '' TERM; " if shell_ignores_term else ""
    return f"{trap}{shlex.quote(sys.executable)} {shlex.quote(str(writer))} & wait"


def _assert_quiet(progress):
    size = progress.stat().st_size
    # health: allow HX013 -- a quiet interval spanning several writes is the behaviour checked
    threading.Event().wait(0.3)
    assert progress.stat().st_size == size, "the workload kept writing after kill returned"


def _join_poller(session):
    if session._reader_thread is not None:
        session._reader_thread.join(timeout=12)


def _grace(seconds):
    return patch.object(ProcessRegistry, "_daemon_term_grace_seconds", staticmethod(lambda: seconds["value"]))


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("ignore_term", [False, True])
def test_kill_stops_the_workload_and_its_descendants(tmp_path, ignore_term):
    registry = ProcessRegistry()
    progress = tmp_path / "progress"
    tree = []
    with patch.object(registry, "_write_checkpoint"), _grace({"value": 0.3}):
        session = registry.spawn_via_env(
            BashEnv(tmp_path), _writer_command(tmp_path, progress, ignore_term=ignore_term))
        try:
            wait_until(lambda: progress.exists() and progress.stat().st_size > 0, what="writer started")
            root = psutil.Process(session.pid)
            tree = [root, *root.children(recursive=True)]
            assert len(tree) >= 3  # supervisor, command shell, writer

            result = registry.kill_process(session.id)

            assert result["status"] == "killed", result
            _assert_quiet(progress)
            assert not any(_alive(proc) for proc in tree)
            if not ignore_term:  # the supervisor outlived SIGTERM long enough to record the code
                exit_file = tmp_path / f"hermes_bg_{session.id}.exit"
                assert exit_file.read_text(encoding="utf-8").strip() == "143"
        finally:
            for proc in tree:
                if _alive(proc):
                    proc.kill()
            session.exited = True
            _join_poller(session)


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("shell_ignores_term", [True, False])
def test_survivors_are_reported_by_pid_and_the_session_stays_killable(tmp_path, shell_ignores_term):
    registry = ProcessRegistry()
    progress = tmp_path / "progress"
    tree = []
    grace = {"value": 0.0}  # 0 = no SIGKILL escalation, so the TERM-ignoring writer survives
    command = _writer_command(tmp_path, progress, ignore_term=True, shell_ignores_term=shell_ignores_term)
    with patch.object(registry, "_write_checkpoint"), _grace(grace):
        session = registry.spawn_via_env(BashEnv(tmp_path), command)
        try:
            wait_until(lambda: progress.exists() and progress.stat().st_size > 0, what="writer started")
            root = psutil.Process(session.pid)
            tree = [root, *root.children(recursive=True)]
            writer = next(proc for proc in tree if proc.name() != "bash")

            partial = registry.kill_process(session.id)

            assert partial["status"] == "error" and partial["process_running"] is True, partial
            if shell_ignores_term:
                # The supervisor traps TERM and waits on the TERM-ignoring shell, so all of it lives.
                assert set(partial["survivors"]) == {proc.pid for proc in tree}
            else:
                # The shell died and the supervisor recorded its exit; only the writer survives.
                assert partial["survivors"] == [writer.pid]
                exit_file = tmp_path / f"hermes_bg_{session.id}.exit"
                wait_until(lambda: exit_file.exists() and exit_file.stat().st_size > 0, what="supervisor exit")
                # health: allow HX013 -- spans two poller ticks (2 s each) after the supervisor exited
                threading.Event().wait(4.5)
            assert _alive(writer)
            assert registry.get(session.id) is session and not session.exited

            grace["value"] = 0.3
            assert registry.kill_process(session.id)["status"] == "killed"
            _assert_quiet(progress)
            assert not _alive(writer)
        finally:
            for proc in tree:
                if _alive(proc):
                    proc.kill()
            session.exited = True
            _join_poller(session)


def test_a_survivor_the_sandbox_cannot_list_is_reported_as_the_group():
    registry = ProcessRegistry()
    env = MagicMock()
    env.get_temp_dir.return_value = "/tmp"
    env.execute.return_value = {"returncode": 1, "output": "__HERMES_PROCESS_SURVIVORS__\n"}
    session = ProcessSession(id="proc_unlisted", command="writer", task_id="task",
                             pid=4321, pid_scope="sandbox", env_ref=env)
    registry._running[session.id] = session

    result = registry.kill_process(session.id)

    assert result["status"] == "error" and result["survivors"] == [4321], result
    assert registry.get(session.id) is session and not session.exited


@pytest.mark.parametrize("reply", [
    {"returncode": 255, "output": "connection lost"},
    {"returncode": 0, "output": ""},
    {"returncode": 1, "output": "__HERMES_PROCESS_TERMINATED__\n"},
    {"returncode": "invalid", "output": "__HERMES_PROCESS_TERMINATED__\n"},
    {"returncode": 0, "output": "__HERMES_PROCESS_TERMINATED__suffix\n"},
])
def test_unconfirmed_kill_keeps_the_session_running(reply):
    registry = ProcessRegistry()
    env = MagicMock()
    env.execute.return_value = reply
    session = ProcessSession(id="proc_unconfirmed", command="writer", task_id="task",
                             pid=4321, pid_scope="sandbox", env_ref=env)
    registry._running[session.id] = session

    result = registry.kill_process(session.id)

    assert result["status"] == "error" and result["process_running"] is True, result
    assert "survivors" not in result
    assert registry.get(session.id) is session and not session.exited


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("code", [7, 130])
def test_a_natural_exit_keeps_its_output_and_code(tmp_path, code):
    registry = ProcessRegistry()
    env = BashEnv(tmp_path)
    with patch.object(registry, "_write_checkpoint"):
        session = registry.spawn_via_env(env, f"printf 'output preserved\\n'; exit {code}")
        try:
            exit_file = tmp_path / f"hermes_bg_{session.id}.exit"
            wait_until(lambda: exit_file.exists() and exit_file.stat().st_size > 0, what="exit file")
            assert int(exit_file.read_text(encoding="utf-8")) == code
            assert (tmp_path / f"hermes_bg_{session.id}.log").read_text(encoding="utf-8") == "output preserved\n"
            registry._finish_exited(session, code)
            with patch.object(env, "execute", side_effect=AssertionError("signalled an exited group")):
                result = registry.kill_process(session.id)
            assert result["status"] == "already_exited"
            assert session.exit_code == code
        finally:
            session.exited = True
            _join_poller(session)


@pytest.mark.parametrize("pgid", [0, 1, -1, True])
def test_kill_command_refuses_group_ids_that_reach_other_processes(pgid):
    with pytest.raises(ValueError):
        env_group_kill_command(pgid, 1.0)


def test_a_supervisor_probe_that_raced_a_kill_is_repeated_on_the_group(monkeypatch):
    import tools.process_registry as module

    registry = ProcessRegistry()
    session = ProcessSession(id="proc_race", command="writer", task_id="task", pid=4321, pid_scope="sandbox")
    group_probes = []

    class Env:
        def execute(self, command, **_kwargs):
            group_probes.append(_probes_group(command))
            if len(group_probes) == 1:
                assert session.begin_group_kill()  # a kill begins while this supervisor probe runs
                return {"output": "0 0 exit:\n"}  # ...and has already stopped the supervisor
            return {"output": "0 0 run\n" if len(group_probes) == 2 else "0 0 exit:143\n"}

    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda *_a, **_k: None)
    session.env_ref = Env()
    registry._running[session.id] = session

    registry._env_poller_loop(session, session.env_ref, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")

    assert group_probes == [False, True, True]
    assert session.exit_code == 143


def _probes_group(command):
    """Whether a poll command probes the job's group rather than its supervisor alone."""
    from tools.process_registry_env_log import log_delta_command

    return command == log_delta_command("/tmp/bg.log", 0, "/tmp/bg.pid", "/tmp/bg.exit", group=True)


def test_a_supervisor_exit_verdict_cannot_finish_a_session_whose_kill_found_survivors(monkeypatch):
    """The poller has a supervisor-only exit verdict in hand when a kill switches it to the
    group and reports a survivor. The verdict must not finish the session the kill kept."""
    import tools.process_registry as module

    registry = ProcessRegistry()
    session = ProcessSession(id="proc_verdict_race", command="writer", task_id="task", pid=4321,
                             pid_scope="sandbox")
    verdict_in_hand, kill_returned = threading.Event(), threading.Event()
    real_commit = ProcessSession.mark_exited_if_probe_current

    def commit_after_the_kill(self, exit_code, probed_group):
        if not probed_group:
            verdict_in_hand.set()
            assert kill_returned.wait(5)
        return real_commit(self, exit_code, probed_group)

    class Env:
        def execute(self, command, **_kwargs):
            if ENV_SURVIVORS_MARKER in command:
                return {"output": f"{ENV_SURVIVORS_MARKER} 4322\n", "returncode": 1}
            if _probes_group(command):
                group_polled.set()
                raise RuntimeError("end the poll loop")  # it then settles as lost: not this test's concern
            return {"output": "0 0 exit:143\n"}

    group_polled = threading.Event()
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ProcessSession, "mark_exited_if_probe_current", commit_after_the_kill)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda *_a, **_k: None)
    session.env_ref = Env()
    registry._running[session.id] = session
    poller = threading.Thread(target=registry._env_poller_loop,
                              args=(session, session.env_ref, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit"))
    poller.start()
    assert verdict_in_hand.wait(5)

    result = registry.kill_process(session.id)
    kill_returned.set()
    assert group_polled.wait(5), "the stale supervisor verdict was recorded instead of re-probing the group"
    poller.join(5)

    assert result["status"] == "error" and result["survivors"] == [4322], result
    assert session.exit_code != 143, "the supervisor-only verdict finished the session"


def test_a_kill_after_the_poller_recorded_the_exit_signals_nothing():
    registry = ProcessRegistry()
    env = MagicMock()
    session = ProcessSession(id="proc_exited_first", command="w", task_id="t", pid=4321, pid_scope="sandbox",
                             env_ref=env)
    registry._running[session.id] = session
    real_begin = ProcessSession.begin_group_kill

    def exit_recorded_first(self):
        with self._lock:
            self.mark_exited(0)
        return real_begin(self)

    with patch.object(ProcessSession, "begin_group_kill", exit_recorded_first), \
            patch.object(registry, "_write_checkpoint"):
        result = registry.kill_process(session.id)

    assert result["status"] == "already_exited" and result["exit_code"] == 0, result
    env.execute.assert_not_called()


def test_a_kill_signals_nothing_once_the_pid_file_names_another_group(tmp_path):
    pid_file = tmp_path / "bg.pid"
    pid_file.write_text("999\n", encoding="utf-8")
    group = subprocess.Popen(["sleep", "30"], process_group=0)
    try:
        result = subprocess.run(["sh", "-c", env_group_kill_command(group.pid, 0.1, str(pid_file))],
                                capture_output=True, text=True, timeout=10, check=False)
        assert parse_env_group_kill({"output": result.stdout, "returncode": result.returncode}, group.pid) \
            == ENV_STALE_MARKER
        assert group.poll() is None
        pid_file.write_text(f"{group.pid}\n", encoding="utf-8")
        result = subprocess.run(["sh", "-c", env_group_kill_command(group.pid, 0.1, str(pid_file))],
                                capture_output=True, text=True, timeout=10, check=False)
        assert ENV_GONE_MARKER in result.stdout.splitlines(), result
    finally:
        group.kill()
        group.wait()


def test_a_kill_refused_by_a_recreated_sandbox_keeps_the_session_tracked():
    registry = ProcessRegistry()
    env = MagicMock()
    env.execute.return_value = {"returncode": 3, "output": f"{ENV_STALE_MARKER}\n"}
    session = ProcessSession(id="proc_stale", command="w", task_id="t", pid=4321, pid_scope="sandbox",
                             env_ref=env, env_pid_path="/tmp/hermes_bg_proc_stale.pid")
    registry._running[session.id] = session

    result = registry.kill_process(session.id)

    assert result["status"] == "error" and "nothing was signalled" in result["error"], result
    assert "/tmp/hermes_bg_proc_stale.pid" in env.execute.call_args.args[0]
    assert registry.get(session.id) is session and not session.exited


@pytest.mark.platforms("posix")
def test_a_sleep_without_fractions_still_waits_out_the_grace():
    script = env_group_kill_command(1234567, 2.0)
    integer_sleep = 'sleep() { case "$1" in *.*) return 1;; esac; command sleep "$1"; }; '
    group = subprocess.Popen(["bash", "-c", "trap '' TERM; sleep 30 & wait"], process_group=0)
    try:
        wait_until(lambda: len(psutil.Process(group.pid).children()) == 1, what="group started")
        started = time.monotonic()
        result = subprocess.run(["bash", "-c", integer_sleep + script.replace("1234567", str(group.pid))],
                                capture_output=True, text=True, timeout=30, check=False)
        elapsed = time.monotonic() - started
        assert ENV_GONE_MARKER in result.stdout.splitlines(), result
        assert 2.0 <= elapsed < 8, f"grace not honoured: {elapsed:.1f}s"
    finally:
        group.kill()
        group.wait()


def test_a_sweep_kills_sandbox_jobs_concurrently():
    registry = ProcessRegistry()
    in_flight, peak, lock = [0], [0], threading.Lock()
    both_started = threading.Barrier(2, timeout=5)

    class SlowEnv:
        def execute(self, command, **_kwargs):
            with lock:
                in_flight[0] += 1
                peak[0] = max(peak[0], in_flight[0])
            both_started.wait()  # a serial sweep never gets the second kill here
            with lock:
                in_flight[0] -= 1
            return {"output": f"{ENV_GONE_MARKER}\n", "returncode": 0}

    for n in (1, 2):
        session = ProcessSession(id=f"proc_sweep{n}", command="w", task_id="t", pid=4320 + n,
                                 pid_scope="sandbox", env_ref=SlowEnv())
        registry._running[session.id] = session
    with patch.object(registry, "_write_checkpoint"):
        assert registry.kill_all(source="gateway_shutdown") == 2
    assert peak[0] == 2


# Hides /proc and ps from the liveness shell, as in a sandbox that has neither.
_NO_MEMBER_LIST = ("ps() { return 1; }; ", "/proc/self/stat", "/nonexistent/stat")


def _poll_status(tmp_path, shell, pgid_text, prefix=""):
    from tools.process_registry_env_log import log_delta_command, parse_log_delta_reply

    for name, text in (("pid", pgid_text + "\n"), ("exit", "143\n"), ("log", "")):
        (tmp_path / f"bg.{name}").write_text(text, encoding="utf-8")
    quoted = {name: shlex.quote(str(tmp_path / f"bg.{name}")) for name in ("log", "pid", "exit")}
    command = log_delta_command(quoted["log"], 0, quoted["pid"], quoted["exit"], group=True)
    if prefix:
        command = prefix + command.replace(_NO_MEMBER_LIST[1], _NO_MEMBER_LIST[2])
    result = subprocess.run([shell, "-c", command], capture_output=True, text=True, timeout=10, check=False)
    reply = parse_log_delta_reply(result.stdout)
    assert reply is not None, result
    return "run" if reply.exit_code is None else "exit"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", ["bash", "sh"])
def test_the_group_poll_counts_live_members_only(tmp_path, shell):
    live = subprocess.Popen(["sleep", "30"], process_group=0)
    zombie = subprocess.Popen(["sleep", "0"], process_group=0)
    try:
        wait_until(lambda: psutil.Process(zombie.pid).status() == psutil.STATUS_ZOMBIE, what="zombie")
        assert _poll_status(tmp_path, shell, str(live.pid)) == "run"
        # A container whose PID 1 never reaps keeps a killed job's zombies: they are not the job.
        assert _poll_status(tmp_path, shell, str(zombie.pid)) == "exit"
        assert _poll_status(tmp_path, shell, "999999999") == "exit"
        assert _poll_status(tmp_path, shell, "not-a-pid") == "exit"
        # PID 1's group belongs to another user here (EPERM to signal): still a live member.
        assert _poll_status(tmp_path, shell, "1") == "run"
    finally:
        live.kill()
        live.wait()
        zombie.wait()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", ["bash", "sh"])
@pytest.mark.parametrize("diagnostic,status", [
    ("No such process", "exit"), ("Operation not permitted", "run"), (None, "run")])
def test_without_a_member_list_only_esrch_reads_as_gone(tmp_path, shell, diagnostic, status):
    # A shell function stands in for the kill builtin, so no real process is signalled.
    fake_kill = ("kill() { return 0; }; " if diagnostic is None else
                 "kill() { echo " + shlex.quote("kill: (-4321) - " + diagnostic) + " >&2; return 1; }; ")
    assert _poll_status(tmp_path, shell, "4321", prefix=_NO_MEMBER_LIST[0] + fake_kill) == status


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("shell", ["bash", "sh"])
@pytest.mark.parametrize("diagnostic,status", [("Operation not permitted", "run"), ("No such process", "exit")])
def test_a_member_the_proc_scan_cannot_see_keeps_the_group_running(tmp_path, shell, diagnostic, status):
    # /proc lists no member (another user's process behind hidepid), but kill -0 still reaches one.
    fake_kill = "kill() { echo " + shlex.quote("kill: (-4321) - " + diagnostic) + " >&2; return 1; }; "
    hidden = (fake_kill, "/proc/[0-9]*/stat", "/proc/nonexistent[0-9]*/stat")
    from tools.process_registry_env_log import log_delta_command, parse_log_delta_reply

    for name, text in (("pid", "4321\n"), ("exit", "143\n"), ("log", "")):
        (tmp_path / f"bg.{name}").write_text(text, encoding="utf-8")
    quoted = {name: shlex.quote(str(tmp_path / f"bg.{name}")) for name in ("log", "pid", "exit")}
    command = log_delta_command(quoted["log"], 0, quoted["pid"], quoted["exit"], group=True)
    assert hidden[1] in command
    command = hidden[0] + command.replace(hidden[1], hidden[2])
    result = subprocess.run([shell, "-c", command], capture_output=True, text=True, timeout=10, check=False)
    reply = parse_log_delta_reply(result.stdout)
    assert reply is not None, result
    assert ("run" if reply.exit_code is None else "exit") == status


@pytest.mark.platforms("posix")
def test_the_kill_treats_a_zombie_only_group_as_gone():
    zombie = subprocess.Popen(["sleep", "0"], process_group=0)
    try:
        wait_until(lambda: psutil.Process(zombie.pid).status() == psutil.STATUS_ZOMBIE, what="zombie")
        result = subprocess.run(["bash", "-c", env_group_kill_command(zombie.pid, 0.1)],
                                capture_output=True, text=True, timeout=10, check=False)
        assert result.returncode == 0 and "__HERMES_PROCESS_TERMINATED__" in result.stdout.splitlines(), result
    finally:
        zombie.wait()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("shell", ["bash", "sh"])
@pytest.mark.parametrize("name", [b"bad\nname", b"x) S 1 1 1"])
def test_a_process_name_cannot_hide_a_live_member(tmp_path, shell, name):
    from tools.process_registry_env_log import group_liveness_shell

    renamed = tmp_path / "renamed"
    # prctl(PR_SET_NAME) sets the comm the kernel prints in /proc/<pid>/stat.
    code = ("import ctypes, sys, time; ctypes.CDLL(None).prctl(15, bytes.fromhex(sys.argv[1])); "
            "open(sys.argv[2], 'w').close(); time.sleep(30)")
    member = subprocess.Popen([sys.executable, "-c", code, name.hex(), str(renamed)], process_group=0)
    try:
        wait_until(renamed.exists, what="renamed member")
        assert psutil.Process(member.pid).name().encode() == name
        result = subprocess.run([shell, "-c", group_liveness_shell(str(member.pid)) + "_hm"],
                                capture_output=True, text=True, timeout=10, check=False)
        assert result.stdout.split() == [str(member.pid)]
        assert _poll_status(tmp_path, shell, str(member.pid)) == "run"
    finally:
        member.kill()
        member.wait()


@pytest.mark.platforms("posix")
def test_a_slow_process_scan_does_not_stretch_the_grace():
    # Each member scan made to take 1 s, as a full process table does on a busy host. The
    # signal waits must not scan per tick, or the SIGKILL comes after the exec timeout.
    script = env_group_kill_command(1234567, 0.3)
    group = subprocess.Popen(["bash", "-c", "trap '' TERM; sleep 30 & wait"], process_group=0)
    try:
        wait_until(lambda: len(psutil.Process(group.pid).children()) == 1, what="group started")
        slow = script.replace("1234567", str(group.pid))
        assert "_hm() {" in slow, "the slowed scan must exist for this test's premise"
        slow = slow.replace("_hm() {", "_hm() { sleep 1; ", 1)
        started = time.monotonic()
        result = subprocess.run(["bash", "-c", slow], capture_output=True, text=True, timeout=30, check=False)
        elapsed = time.monotonic() - started
        assert "__HERMES_PROCESS_TERMINATED__" in result.stdout.splitlines(), result
        assert elapsed < 6, f"kill took {elapsed:.1f}s with 1 s scans"
    finally:
        group.kill()
        group.wait()


def test_the_kill_exec_timeout_is_whole_seconds():
    # Modal and Daytona reject a float exec timeout before running anything.
    registry = ProcessRegistry()
    env = MagicMock()
    env.execute.return_value = {"returncode": 0, "output": "__HERMES_PROCESS_TERMINATED__\n"}
    session = ProcessSession(id="proc_int_timeout", command="w", task_id="t", pid=4321, pid_scope="sandbox", env_ref=env)
    registry._running[session.id] = session
    with patch.object(registry, "_write_checkpoint"), _grace({"value": 2.5}):
        assert registry.kill_process(session.id)["status"] == "killed"
    timeout = env.execute.call_args.kwargs["timeout"]
    assert type(timeout) is int and timeout >= 2.5


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_a_slow_starting_supervisor_still_has_its_pid_recorded(tmp_path):
    # BASH_ENV runs before the supervisor's first command, as a heavy shell init would.
    slow_init = tmp_path / "slow_init.sh"
    slow_init.write_text("sleep 2\n", encoding="utf-8")

    class SlowInitEnv(BashEnv):
        def execute(self, command, timeout=10, **_kwargs):
            with tempfile.TemporaryFile(mode="w+") as output:
                result = subprocess.run(
                    ["bash", "-c", command], stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    timeout=timeout, check=False, start_new_session=True,
                    env={**os.environ, "BASH_ENV": str(slow_init)})
                output.seek(0)
                return {"output": output.read(), "returncode": result.returncode}

    registry = ProcessRegistry()
    with patch.object(registry, "_write_checkpoint"), _grace({"value": 0.3}):
        session = registry.spawn_via_env(SlowInitEnv(tmp_path), "sleep 30")
        try:
            assert session.pid and psutil.Process(session.pid).is_running()
            assert registry.kill_process(session.id)["status"] == "killed"
        finally:
            session.exited = True
            _join_poller(session)
