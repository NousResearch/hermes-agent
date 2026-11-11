"""A long-lived service inside a backend with an exec prefix (docker exec / ssh / a plugin prefix):
what its spawn hands the service (argv, exported environment, stdio, the recorded pid) and what its stop
signals inside the sandbox, through the real ``start_process`` / ``stop_process`` on an exec-prefix
environment faked on this host."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time

import pytest

from tests.waiting import wait_until
from tools.environments.base_processes import LongLivedProcessMixin


class _PrefixEnv(LongLivedProcessMixin):
    """A backend with an exec prefix: the "sandbox" is this host behind ``env``; ``execute`` is the
    sandbox-side command channel, run by POSIX ``sh`` as a backend's shell may be."""

    def __init__(self, tmp_path):
        self._tmp = str(tmp_path)
        self.executed: list[str] = []

    def exec_prefix(self, *, user=None, interactive=True):
        return ["env"]

    def get_temp_dir(self):
        return self._tmp

    def execute(self, command, cwd=None, timeout=30, scope=None):
        self.executed.append(command)
        return subprocess.run(["sh", "-c", command], capture_output=True, timeout=timeout, check=False).stdout.decode()

    def _kill_process(self, proc):
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()


def _pidfile_words(tmp_path, timeout: float = 10.0) -> tuple:
    deadline = time.monotonic() + timeout
    while True:
        files = list(tmp_path.glob("hermes_llp_*.pid"))
        if files and files[0].read_text().strip():
            return files[0], files[0].read_text().split()
        assert time.monotonic() < deadline, "the service never recorded its pid inside the sandbox"
        time.sleep(0.02)


_REPORT = ("import json,os,sys; print(json.dumps({'argv': sys.argv[1:], 'pid': os.getpid(), "
           "'probe': os.environ.get('HERMES_LLP_PROBE')}), flush=True); print('echo:' + sys.stdin.readline().strip(), "
           "flush=True)")


@pytest.mark.platforms("posix")  # POSIX argv, bash inside the "sandbox"
def test_the_service_gets_its_argv_environment_stdio_and_its_own_pid_recorded(tmp_path):
    env = _PrefixEnv(tmp_path)
    stream = env.start_process("svc", [sys.executable, "-c", _REPORT, "two words", "it's"],
                               child_env={"HERMES_LLP_PROBE": "a b'c"})
    try:
        seen = json.loads(stream.stdout.readline())
        assert seen["argv"] == ["two words", "it's"], "an argv word was re-split or re-quoted on the way in"
        assert seen["probe"] == "a b'c", "the service did not see its child_env value"
        _, words = _pidfile_words(tmp_path)
        assert int(words[0]) == seen["pid"], "the pidfile does not hold the pid of the process running the service"
        stream.stdin.write(b"ping\n")
        stream.stdin.flush()
        assert stream.stdout.readline() == b"echo:ping\n", "stdin/stdout are not the service's own"
    finally:
        env.stop_all_processes()


class _RecordOnlyEnv(_PrefixEnv):
    """The sandbox-side command channel only records: these tests start services that print and exit, and
    nothing in them runs the in-sandbox stop."""

    def execute(self, command, cwd=None, timeout=30, scope=None):
        self.executed.append(command)
        return ""


_ARGV = "import json,sys; print(json.dumps(sys.argv[1:]), flush=True)"


@pytest.mark.platforms("posix")  # POSIX argv, bash inside the "sandbox"
def test_a_child_env_path_without_bash_still_starts_a_service_given_by_absolute_path(tmp_path):
    env = _RecordOnlyEnv(tmp_path)
    no_bash = tmp_path / "bin"
    no_bash.mkdir()
    stream = env.start_process("svc", [sys.executable, "-c", _ARGV, "two words"], child_env={"PATH": str(no_bash)})
    try:
        line = stream.stdout.readline()
        assert line, "the service never started: the pid recording needed a PATH lookup"
        assert json.loads(line) == ["two words"]
    finally:
        env.stop_all_processes()


@pytest.mark.platforms("posix")  # POSIX argv, bash inside the "sandbox"
def test_a_bash_env_in_child_env_is_not_sourced_by_the_pid_recording(tmp_path):
    env = _RecordOnlyEnv(tmp_path)
    marker = tmp_path / "sourced"
    rcfile = tmp_path / "rc"
    rcfile.write_text(f"echo sourced > {shlex.quote(str(marker))}\n")
    stream = env.start_process("svc", [sys.executable, "-c", _ARGV, "x"], child_env={"BASH_ENV": str(rcfile)})
    try:
        assert json.loads(stream.stdout.readline()) == ["x"]
        assert not marker.exists(), "the wrapper that records the pid sourced the service's BASH_ENV"
    finally:
        env.stop_all_processes()


def _starttime(pid: int):
    """Field 22 of ``/proc/<pid>/stat`` (the 20th after the last ``)``), or None once the pid is gone or a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            fields = fh.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if fields[0] == "Z" else fields[19]


def _ticks() -> int:
    """The clock /proc start times count in: clock ticks since boot."""
    return int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))


def _wait_not(pid: int, starttime: str, timeout: float = 10.0) -> bool:
    """Wait until *pid* is no longer the process that started at *starttime*."""
    deadline = time.monotonic() + timeout
    while _starttime(pid) == starttime:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


class _DetachedClientEnv(_PrefixEnv):
    """A local client that is NOT the sandbox process (as with docker exec or ssh): it starts the service
    in a session of its own (no stdin, so no hangup either) and waits; killing the client alone leaves the
    service and its child running, so only the in-sandbox stop can end them."""

    def exec_prefix(self, *, user=None, interactive=True):
        return ["sh", "-c", 'setsid "$@" & wait', "client"]


def _start_service_with_child(env, tmp_path, started: list):
    """A running service with one child of its own. Appends (pid, start time) of the child and of the service to
    *started* as soon as each is known, so the caller's cleanup reaches them even if this raises part-way."""
    stream = env.start_process("svc", ["bash", "-c", "trap '' HUP; sleep 30 & echo $!; wait"])
    child = int(stream.stdout.readline())
    started.append((child, _starttime(child)))
    pidfile, words = _pidfile_words(tmp_path)
    service = int(words[0])
    started.append((service, _starttime(service)))
    assert service != env._long_lived()["svc"].pid, "the service must run apart from its local client"
    assert None not in (started[0][1], started[1][1])
    return pidfile, words


def _kill_if_same(started) -> None:
    """Cleanup: signal each pid only while it is still the process this test started."""
    for pid, starttime in started:
        if starttime is not None and _starttime(pid) == starttime:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.platforms("linux")  # reads /proc
@pytest.mark.live_system_guard_bypass  # the in-sandbox stop really signals (only processes this test started)
def test_a_stop_after_the_service_exited_never_signals_a_recycled_pid(tmp_path):
    env = _PrefixEnv(tmp_path)
    env.start_process("svc", ["true"])
    pidfile, words = _pidfile_words(tmp_path)
    env._long_lived()["svc"].wait(timeout=10)  # the service is gone; its pid is free for reuse
    # The unrelated process starts on a later clock tick (the unit of the start time) than the service did.
    gone_at = _ticks()
    wait_until(lambda: _ticks() > gone_at)
    sleeper = subprocess.Popen(["sleep", "30"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    try:
        assert _starttime(sleeper.pid) not in words[1:]
        # A recycled pid: the sleeper now has the pid, the recorded start time is the service's.
        pidfile.write_text(" ".join([str(sleeper.pid), *words[1:]]) + "\n")
        env.stop_process("svc")
        assert sleeper.poll() is None, "the stop signalled an unrelated process that reused the service's pid"
        assert not pidfile.exists(), "the stop left the pidfile behind"
    finally:
        sleeper.kill()
        sleeper.wait(timeout=10)
        env.stop_all_processes()


def _exited_service_pidfile(env, tmp_path):
    """Start a service that exits at once; return its pidfile once the service is gone."""
    env.start_process("svc", ["true"])
    pidfile, _ = _pidfile_words(tmp_path)
    env._long_lived()["svc"].wait(timeout=10)
    return pidfile


# Only a plain decimal pid greater than 1 is ever signalled. The pidfile values that would reach processes this
# test does not own (-1, 0, 1, a glob, shell metacharacters) are cleared by that same check and are never written
# here; a negative value is tested only as the negated id of a group that holds this test's own sleeper alone.
@pytest.mark.platforms("posix")  # process groups and sessions
@pytest.mark.live_system_guard_bypass  # the in-sandbox stop really runs (it could reach only this test's sleeper)
def test_a_negative_pid_in_the_pidfile_never_signals_a_process_group(tmp_path):
    env = _PrefixEnv(tmp_path)
    pidfile = _exited_service_pidfile(env, tmp_path)
    sleeper = subprocess.Popen(["sleep", "30"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               start_new_session=True)
    try:
        group = -sleeper.pid
        # Before anything can signal it: the value is a negated group id holding the sleeper alone.
        assert sleeper.pid > 1 and group < -1, "refusing to write a pidfile that could reach other processes"
        assert os.getpgid(sleeper.pid) == sleeper.pid, "the sleeper does not lead a group of its own"
        assert os.getpgid(sleeper.pid) != os.getpgid(0), "the sleeper shares this test's process group"
        pidfile.write_text(f"{group}\n")
        env.stop_process("svc")
        assert sleeper.poll() is None, "the stop signalled a whole process group named by a negative pid"
        assert not pidfile.exists(), "the stop left the pidfile behind"
    finally:
        sleeper.kill()
        sleeper.wait(timeout=10)
        env.stop_all_processes()


@pytest.mark.platforms("posix")  # POSIX sh inside the "sandbox"
@pytest.mark.live_system_guard_bypass  # the in-sandbox stop really runs (with no pid, it signals nothing)
@pytest.mark.parametrize("content", ["abc\n", ""], ids=["not-a-number", "empty"])
def test_a_pidfile_without_a_usable_pid_is_removed_and_signals_nothing(tmp_path, caplog, content):
    env = _PrefixEnv(tmp_path)
    pidfile = _exited_service_pidfile(env, tmp_path)
    try:
        pidfile.write_text(content)
        env.stop_process("svc")
        assert not pidfile.exists(), "the stop left the pidfile behind"
        assert "stop command failed" not in caplog.text
    finally:
        env.stop_all_processes()


@pytest.mark.platforms("posix")  # POSIX argv, bash inside the "sandbox"
def test_the_stop_command_turns_errexit_off_before_anything_can_fail(tmp_path):
    # A backend shell running with ``set -e`` must not abort the stop (an empty pidfile, a pid already gone, a
    # service with no children) before it removes the pidfile. Checked on the text only; nothing runs it here.
    env = _RecordOnlyEnv(tmp_path)
    try:
        _exited_service_pidfile(env, tmp_path)
        env.stop_process("svc")
        stops = [command for command in env.executed if "hermes_llp_" in command]
        assert stops, "the stop never reached the sandbox"
        assert all(command.startswith("(set +e;") for command in stops), stops
    finally:
        env.stop_all_processes()


@pytest.mark.platforms("linux")  # reads /proc
@pytest.mark.live_system_guard_bypass  # the in-sandbox stop really signals (only processes this test started)
def test_a_normal_stop_still_terminates_the_running_service_and_its_child(tmp_path):
    env = _DetachedClientEnv(tmp_path)
    started = []
    try:
        _start_service_with_child(env, tmp_path, started)
        assert env.stop_process("svc") is True
        assert all(_wait_not(pid, st) for pid, st in started), "the stop left the service or its child running"
    finally:
        env.stop_all_processes()
        _kill_if_same(started)


@pytest.mark.platforms("linux")  # reads /proc
@pytest.mark.live_system_guard_bypass  # the in-sandbox stop really signals (only processes this test started)
def test_a_pidfile_without_a_start_time_is_still_signalled(tmp_path):
    env = _DetachedClientEnv(tmp_path)
    started = []
    try:
        pidfile, words = _start_service_with_child(env, tmp_path, started)
        pidfile.write_text(words[0] + "\n")  # the old format: a sandbox with no /proc records only the pid
        assert env.stop_process("svc") is True
        assert all(_wait_not(pid, st) for pid, st in started), "a pid-only pidfile is no longer signalled"
        assert not pidfile.exists(), "the stop left the pidfile behind"
    finally:
        env.stop_all_processes()
        _kill_if_same(started)
