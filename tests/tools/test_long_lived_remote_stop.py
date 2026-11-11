"""A long-lived service inside a backend with an exec prefix (docker exec / ssh / a plugin prefix):
what its spawn hands the service (argv, exported environment, stdio, the recorded pid), through the
real ``start_process`` on an exec-prefix environment faked on this host."""

from __future__ import annotations

import contextlib
import json
import shlex
import subprocess
import sys
import time

import pytest

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
