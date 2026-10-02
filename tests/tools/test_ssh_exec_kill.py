"""A timed-out or killed SSH foreground command must die on the remote host.

Killing the local ``ssh`` client only closes the channel: sshd does not signal a session without a
pty, so a command that outlived its timeout or a /stop kept running on the remote host while the
tool reported it killed. The fake ``ssh`` CLI below keeps exactly that contract: like sshd it runs
the remote command string through a shell in a new session, and that session survives the client
being killed. (Verified against a real OpenSSH 9.6 sshd on localhost as well.)
"""

import os
import stat
import sys
import time
import uuid

import psutil
import pytest

from tools.environments import ssh as ssh_env
from tools.environments.base import _EXECUTE_WAIT_BOUND_GRACE_S, BaseEnvironment

pytestmark = pytest.mark.platforms("posix")

_FAKE_SSH = """#!{python} -S
import os, sys
args = sys.argv[1:]
if "-O" in args:  # control-master commands (``-O exit``)
    sys.exit(0)
while args and args[0].startswith("-"):  # every option ssh gets here takes a value
    args = args[2:]
remote = " ".join(args[1:])  # drop user@host; ssh joins the rest into one remote command string
delay = os.path.join(os.path.dirname(sys.argv[0]), "kill-delay")
if ".stop" in remote and os.path.exists(delay):  # a slow link: the kill's round trip takes this long
    import time
    time.sleep(float(open(delay).read()))
pid = os.fork()
if pid == 0:
    os.setsid()  # sshd runs each session's command as a new session
    os.execvp("bash", ["bash", "-c", remote])
sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""


@pytest.fixture()
def env(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ssh"
    fake.write_text(_FAKE_SSH.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    # The remote host is this host: skip the connection probe and file sync, and keep every
    # artifact (control socket, snapshot, PID records) in tmp_path.
    env = ssh_env.SSHEnvironment.__new__(ssh_env.SSHEnvironment)
    env.get_temp_dir = lambda: str(tmp_path)
    BaseEnvironment.__init__(env, cwd=str(tmp_path), timeout=60)
    env.host, env.user, env.port, env.key_path = "remote.example", "alice", 22, ""
    env.control_socket = tmp_path / "control.sock"
    env._sync_manager = None
    env._snapshot_ready = True  # sourcing a missing snapshot is a no-op; avoids a login shell
    return env


def _sleeping(marker):
    """Live ``sleep <marker>`` processes: the command itself, not a shell carrying it in argv."""
    found = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            if p.info["cmdline"] == ["sleep", marker] and p.status() != psutil.STATUS_ZOMBIE:
                found.append(p)
        except psutil.Error:
            pass
    return found


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
    marker = f"{20 + uuid.uuid4().int % 10}.{os.getpid()}{uuid.uuid4().int % 10000:04d}"
    yield marker
    for p in _sleeping(marker):
        p.kill()


def test_timed_out_command_does_not_survive_on_the_remote_host(env, marker):
    result = env.execute(f"sleep {marker}", timeout=3)

    assert result["returncode"] == 124
    assert _gone_within(marker, 5), "the timed-out command is still running on the remote host"


@pytest.mark.parametrize("kill", ["_kill_process", "_force_kill_process"])
def test_interrupt_and_hard_exit_kills_reach_the_remote_command(env, marker, kill):
    proc = env._run_bash(f"sleep {marker}")
    deadline = time.monotonic() + 10
    while not _sleeping(marker) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _sleeping(marker)

    getattr(env, kill)(proc)

    assert _gone_within(marker, 5), f"the command is still running on the remote host after {kill}"


def _spy_backstop(env):
    """Record when the ``run_bounded_sync`` backstop (``_on_timeout``) had to kill."""
    fired = []
    real = env._kill_spawned_tree
    env._kill_spawned_tree = lambda spawned: (fired.append(spawned), real(spawned))
    return fired


def test_slow_remote_kill_does_not_hold_the_timeout_past_the_backstop(env, marker, tmp_path):
    """The remote kill is a network round trip; the timeout path runs it under a backstop that
    allows only ``_EXECUTE_WAIT_BOUND_GRACE_S`` past the timeout. A 3s kill must not make a 2s
    command return late or fire the backstop, and the command must still die."""
    (tmp_path / "bin" / "kill-delay").write_text("3")
    fired = _spy_backstop(env)

    started = time.monotonic()
    result = env.execute(f"sleep {marker}", timeout=2)
    elapsed = time.monotonic() - started

    assert result["returncode"] == 124
    assert not fired, "the backstop fired: the inner timeout path was blocked by the remote kill"
    assert elapsed < 2 + _EXECUTE_WAIT_BOUND_GRACE_S, f"timed-out command returned after {elapsed:.2f}s"
    assert _gone_within(marker, 3 + 5), "the timed-out command is still running on the remote host"


def test_shutdown_kills_of_several_commands_do_not_serialize_on_the_link(env, tmp_path):
    """``kill_live_foreground_processes`` calls ``_kill_process`` once per live command."""
    markers = [f"{20 + i}.{os.getpid()}{uuid.uuid4().int % 10000:04d}" for i in range(3)]
    try:
        procs = [env._run_bash(f"sleep {m}") for m in markers]
        deadline = time.monotonic() + 10
        while not all(_sleeping(m) for m in markers) and time.monotonic() < deadline:
            time.sleep(0.05)
        (tmp_path / "bin" / "kill-delay").write_text("2")

        started = time.monotonic()
        for proc in procs:
            env._kill_process(proc)
        assert time.monotonic() - started < 1.0

        for m in markers:
            assert _gone_within(m, 2 + 5), "a command survived the shutdown kill"
    finally:
        for m in markers:
            for p in _sleeping(m):
                p.kill()


def test_cleanup_lets_in_flight_kills_land_before_closing_the_master(env, marker, tmp_path):
    """``cleanup`` runs ``ssh -O exit`` right after the shutdown kills; a kill riding the
    ControlMaster must not be cut off by it."""
    proc = env._run_bash(f"sleep {marker}")
    deadline = time.monotonic() + 10
    while not _sleeping(marker) and time.monotonic() < deadline:
        time.sleep(0.05)
    (tmp_path / "bin" / "kill-delay").write_text("1")

    env._kill_process(proc)
    env.cleanup()

    assert not _sleeping(marker), "cleanup returned before the in-flight remote kill landed"
