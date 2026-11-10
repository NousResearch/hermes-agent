"""The long-lived process mixin: its table and lock are one owned state (a concurrent first start
never sees the table without the lock), a name reused after its leader exited reaps the old record
(its children and pipes do not outlive the record), and stopping a process that runs inside a
backend (docker exec / ssh / a plugin prefix) stops the service there, where its pid was recorded —
not only the local client."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time

import pytest

from tools.environments.base_processes import LongLivedProcessMixin


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


class _HostEnv(LongLivedProcessMixin):
    """A host backend: spawn like the local environment does, in an owned process group."""

    def __init__(self, tmp_path=None):
        self.cwd = os.getcwd()

    def _spawn_long_lived(self, argv, child_env):
        proc = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        proc._hermes_owned_pgid = proc.pid
        return proc

    def _kill_process(self, proc):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)


class _SandboxLikeEnv(LongLivedProcessMixin):
    """A backend with an exec prefix (docker exec / ssh shape): the service runs behind a local
    client. ``execute`` is the sandbox-side command channel, as a real backend's is."""

    def __init__(self, tmp_path):
        self._tmp = str(tmp_path)
        self.executed: list[str] = []

    def exec_prefix(self, *, user=None, interactive=True):
        return ["env"]

    def get_temp_dir(self):
        return self._tmp

    def execute(self, command, cwd=None, timeout=30, scope=None):
        self.executed.append(command)
        return subprocess.run(["bash", "-c", command], capture_output=True, timeout=timeout, check=False).stdout.decode()

    def _kill_process(self, proc):
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()


@pytest.mark.platforms("posix")  # POSIX argv and process-group signals
def test_a_first_start_never_sees_the_table_without_its_lock():
    env = _HostEnv()
    # The first-touch window the old two-key init could leave: another thread saw the table
    # before the starter installed its lock.
    env.__dict__["_long_lived_processes"] = {}
    env.start_process("svc", ["sleep", "5"])  # must not KeyError on the starter's lock
    try:
        assert env.attach_process("svc") is not None
    finally:
        env.stop_all_processes()


def test_attach_and_stop_take_the_same_lock_as_start():
    env = _HostEnv()
    lock = env._long_lived_lock()
    lock.acquire()
    try:
        probing = threading.Thread(target=lambda: env.attach_process("svc"), daemon=True)
        probing.start()
        probing.join(0.3)
        assert probing.is_alive(), "attach_process took a different lock than start_process"
    finally:
        lock.release()
    env.stop_all_processes()


@pytest.mark.platforms("posix")  # POSIX argv and process-group signals
def test_a_reused_name_reaps_the_old_records_children_and_pipes(tmp_path):
    pidfile = tmp_path / "child.pid"
    env = _HostEnv()
    env.start_process("svc", ["bash", "-c", f"sleep 60 & echo $! > {pidfile}"])
    deadline = time.monotonic() + 10
    while not (pidfile.exists() and pidfile.read_text().strip()):
        assert time.monotonic() < deadline, "the leader never started its child"
        time.sleep(0.05)
    child = int(pidfile.read_text().strip())
    old = env._long_lived()["svc"]
    deadline = time.monotonic() + 10
    while old.poll() is None:  # the leader exits; the child keeps its owned group alive
        assert time.monotonic() < deadline
        time.sleep(0.05)
    try:
        env.start_process("svc", ["sleep", "5"])  # name reuse after the leader exited
        assert _wait_gone(child), "reusing a name left the old record's children running"
        env.stop_all_processes()
    finally:
        if _alive(child):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child, signal.SIGKILL)


@pytest.mark.platforms("posix")  # POSIX argv and process-group signals
def test_a_failed_respawn_keeps_the_stale_record():
    """A respawn that fails (nonexistent executable) must keep the exited leader's record: its
    pipes and children are still the environment's to stop, and dropping the record would orphan
    both with nothing left to reap."""
    env = _HostEnv()
    env.start_process("svc", ["true"])
    old = env._long_lived()["svc"]
    assert old.stdin is not None and old.stdout is not None
    old.wait(timeout=10)  # the leader exits; its record is now a leftover, not a running process
    try:
        with pytest.raises(FileNotFoundError):
            env.start_process("svc", ["/nonexistent-hermes-review-executable"])
        assert env._long_lived().get("svc") is old, \
            "a failed spawn lost the stale record without reaping it"
        assert not old.stdin.closed, "a failed spawn must not close the stale record's pipes behind its back"
        env.stop_all_processes()
        assert old.stdin.closed and old.stdout.closed, "stop must still reap the stale record"
    finally:
        if not old.stdin.closed:
            old.stdin.close()
        if not old.stdout.closed:
            old.stdout.close()


@pytest.mark.platforms("posix")  # POSIX argv and process-group signals
def test_stopping_a_backend_process_stops_the_service_inside_the_sandbox(tmp_path):
    env = _SandboxLikeEnv(tmp_path)
    env.start_process("svc", ["sleep", "60"])
    deadline = time.monotonic() + 10
    while not list(tmp_path.glob("hermes_llp_*.pid")):
        assert time.monotonic() < deadline, "the service never recorded its pid inside the sandbox"
        time.sleep(0.05)
    pidfile = next(tmp_path.glob("hermes_llp_*.pid"))
    deadline = time.monotonic() + 10
    while not pidfile.read_text().strip():
        assert time.monotonic() < deadline, "the pid file stayed empty"
        time.sleep(0.05)
    service_pid = int(pidfile.read_text().strip())
    try:
        assert env.stop_process("svc") is True
        assert _wait_gone(service_pid), "stopping killed only the local client; the service still runs"
        assert any("kill" in command for command in env.executed), env.executed
    finally:
        env.stop_all_processes()
        if _alive(service_pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(service_pid, signal.SIGKILL)


@pytest.mark.platforms("posix")  # POSIX argv and process-group signals
@pytest.mark.parametrize("impl", [_HostEnv, lambda tmp: _SandboxLikeEnv(tmp)])
def test_streams_of_two_names_never_mix(tmp_path, impl):
    env = impl(tmp_path)
    one = env.start_process("one", ["bash", "-c", "read line; echo one:$line"])
    two = env.start_process("two", ["bash", "-c", "read line; echo two:$line"])
    try:
        one.stdin.write(b"x\n"), one.stdin.flush()
        two.stdin.write(b"y\n"), two.stdin.flush()
        assert one.stdout.readline() == b"one:x\n" and two.stdout.readline() == b"two:y\n"
    finally:
        env.stop_all_processes()
