"""A timed-out or killed docker foreground command must die inside the container.

Killing the host-side ``docker exec`` client does not signal the process it
started in the container (moby), so a command that outlived its timeout or a
/stop kept running in the shared, persistent container while the tool reported
it killed. The fake ``docker`` CLI below keeps exactly that contract: the
exec'd command runs in its own session (runc setsids it) and survives the
client being killed.
"""

import os
import stat
import sys
import time
import uuid

import psutil
import pytest

from tools.environments import docker as docker_env
from tools.environments.base import _EXECUTE_WAIT_BOUND_GRACE_S, BaseEnvironment

pytestmark = pytest.mark.platforms("posix")

_FAKE_DOCKER = """#!{python} -S
import os, sys, time
args = sys.argv[2:]  # drop "exec"
while args and args[0] in ("-i", "-e"):
    args = args[2:] if args[0] == "-e" else args[1:]
argv = args[1:]  # drop the container id
slow = os.path.join(os.path.dirname(sys.argv[0]), "slow-start")
delay = os.path.join(os.path.dirname(sys.argv[0]), "kill-delay")
if ".stop" in " ".join(argv) and os.path.exists(delay):  # a slow daemon: the kill's exec takes this long
    time.sleep(float(open(delay).read()))
pid = os.fork()
if pid == 0:
    os.setsid()
    # A test marks one command whose in-container shell starts late: the exec exists (its PID
    # is published) but the shell has not run yet when the kill lands.
    if os.path.exists(slow) and open(slow).read() in " ".join(argv):
        with open(slow + ".tmp", "w") as f:
            f.write(str(os.getpid()))
        os.rename(slow + ".tmp", slow + ".pid")
        time.sleep(2)
    os.execvp(argv[0], argv)
sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""


@pytest.fixture()
def env(tmp_path):
    fake = tmp_path / "docker"
    fake.write_text(_FAKE_DOCKER.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    # The container is the host here: skip ``docker run`` and keep every artifact in tmp_path.
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env.get_temp_dir = lambda: str(tmp_path)
    BaseEnvironment.__init__(env, cwd=str(tmp_path), timeout=60)
    env._snapshot_ready = True  # sourcing a missing snapshot is a no-op; avoids a host login shell
    env._forward_env = []
    env._prepare_command = lambda command: (command, None)
    env._container_id = "fake-container"
    env._docker_exe = str(fake)
    return env


def _sleeping(marker):
    """Live ``sleep <marker>`` processes: the command itself, not the client carrying it in argv."""
    found = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            if p.info["cmdline"] == ["sleep", marker] and p.status() != psutil.STATUS_ZOMBIE:
                found.append(p)
        except psutil.Error:
            pass
    return found


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _gone_within(marker, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _sleeping(marker):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture()
def marker():
    # A sleep duration nothing else on the host uses identifies our process. It outlasts the
    # 3s timeout, and a survivor (a regression) still ends on its own within half a minute.
    return f"{20 + uuid.uuid4().int % 10}.{os.getpid()}{uuid.uuid4().int % 10000:04d}"


def test_timed_out_command_does_not_survive_in_the_container(env, marker):
    result = env.execute(f"sleep {marker}", timeout=3)

    assert result["returncode"] == 124
    assert _gone_within(marker, 5), "the timed-out command is still running in the container"


def test_hard_exit_kill_reaches_the_command_in_the_container(env, marker):
    proc = env._run_bash(f"sleep {marker}")
    deadline = time.monotonic() + 10
    while not _sleeping(marker) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _sleeping(marker)

    env._force_kill_process(proc)

    assert _gone_within(marker, 5), "the command is still running in the container after a hard-exit kill"


@pytest.mark.parametrize("kill", ["_kill_process", "_force_kill_process"])
def test_kill_that_lands_before_the_shell_records_its_pid_still_stops_the_command(env, marker, tmp_path, kill):
    (tmp_path / "slow-start").write_text(marker)
    proc = env._run_bash(f"sleep {marker}")
    started = tmp_path / "slow-start.pid"
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    shell = int(started.read_text())

    getattr(env, kill)(proc)  # the exec exists, but its shell has not recorded a PID yet

    deadline = time.monotonic() + 10
    while _alive(shell):
        assert time.monotonic() < deadline, "the late-starting shell ran the command after the kill"
        time.sleep(0.1)
    assert not _sleeping(marker)


def test_slow_in_container_kill_does_not_hold_the_timeout_past_the_backstop(env, marker, tmp_path):
    """The in-container kill is a ``docker exec`` round trip; the timeout path runs it under a
    backstop that allows only ``_EXECUTE_WAIT_BOUND_GRACE_S`` past the timeout. A 3s kill must not
    make a 2s command return late or fire the backstop, and the command must still die."""
    (tmp_path / "kill-delay").write_text("3")
    fired = []
    real = env._kill_spawned_tree
    env._kill_spawned_tree = lambda spawned: (fired.append(spawned), real(spawned))

    started = time.monotonic()
    result = env.execute(f"sleep {marker}", timeout=2)
    elapsed = time.monotonic() - started

    assert result["returncode"] == 124
    assert not fired, "the backstop fired: the inner timeout path was blocked by the in-container kill"
    assert elapsed < 2 + _EXECUTE_WAIT_BOUND_GRACE_S, f"timed-out command returned after {elapsed:.2f}s"
    assert _gone_within(marker, 3 + 5), "the timed-out command is still running in the container"
