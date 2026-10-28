"""Remote lifecycle contracts exercised through a real, local shell transport."""

import os
import shlex
import signal
import subprocess
import time
from contextlib import suppress

import psutil
import pytest

from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry

# The wrapper deliberately reparents our children; cleanup targets only PID
# artifacts created in this fixture's private sandbox.
pytestmark = pytest.mark.live_system_guard_bypass


class ShellEnv:
    """Execute remote shell strings locally, retaining real transport pipe semantics."""

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.signal_noop = False
        self.probe_command = None
        self.offline = False

    def get_temp_dir(self):
        return str(self.root)

    def execute(self, command, timeout=10, **kwargs):
        self.calls.append(command)
        if self.offline:
            raise OSError("transport disconnected")
        if self.signal_noop and command.startswith(("kill -TERM ", "kill -KILL ")):
            command = ":"  # A broken backend acknowledges the command without signalling.
        if command == "LC_ALL=C ps -eo pid=,pgid=,stat=" and self.probe_command:
            if self.probe_command == "raise":
                raise OSError("transport disconnected")
            command = self.probe_command
        proc = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-c", command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            stdin=subprocess.DEVNULL,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
            proc.stdout.close()
            proc.stderr.close()
            raise
        return {"output": out, "stderr": err, "returncode": proc.returncode}


@pytest.fixture
def remote(tmp_path):
    root = tmp_path / "remote space's sandbox"
    root.mkdir()
    env = ShellEnv(root)
    registry = ProcessRegistry()
    yield registry, env
    for path in root.glob("*.pid"):
        with suppress(ValueError, OSError):
            os.killpg(int(path.read_text().strip()), signal.SIGKILL)
    for session in list(registry._running.values()):
        session.exited = True
        if session._reader_thread:
            session._reader_thread.join(timeout=5)


@pytest.mark.platforms("linux")
def test_waiting_child_does_not_hold_launch_transport_pipes(remote):
    registry, env = remote
    session = registry.spawn_via_env(env, "sleep 60", timeout=2)
    assert not session.exited, session.output_buffer
    assert session.pid
    assert os.getpgid(session.pid) == session.pid


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("command", ["true", "exit 0"])
def test_successful_noop_writes_zero_exit_artifact(remote, command):
    registry, env = remote
    session = registry.spawn_via_env(env, command, timeout=2)
    artifact = env.root / f"hermes_bg_{session.id}.exit"
    wait_until(artifact.exists, timeout=3, what="remote exit artifact")
    assert artifact.read_text().strip() == "0"
    result = registry.wait(session.id, timeout=5)
    assert result["status"] == "exited", result
    assert result["exit_code"] == 0


def _live_group(pgid):
    members = []
    for proc in psutil.process_iter(["pid", "status"]):
        with suppress(psutil.NoSuchProcess, ProcessLookupError):
            if os.getpgid(proc.pid) == pgid and proc.status() != psutil.STATUS_ZOMBIE:
                members.append(proc.pid)
    return members


@pytest.mark.platforms("linux")
def test_term_resistant_child_is_reaped_with_bounded_escalation(remote):
    registry, env = remote
    ready = env.root / "ready"
    script = env.root / "worker.sh"
    script.write_text(
        "trap '' TERM\n"
        f"printf '%s' \"$$\" > {shlex.quote(str(ready))}\n"
        "while :; do sleep 60; done\n"
    )
    session = registry.spawn_via_env(env, f"bash {shlex.quote(str(script))}", timeout=2)
    wait_until(lambda: ready.exists() and ready.read_text(), timeout=3,
               what="TERM-resistant child ready")
    child_pid = int(ready.read_text())
    assert os.getpgid(child_pid) == session.pid
    start = time.monotonic()
    result = registry.kill_process(session.id)
    assert result["status"] == "killed", result
    assert not _live_group(session.pid), "killed receipt left live group members"
    assert time.monotonic() - start < 20


@pytest.mark.platforms("linux")
def test_post_kill_verification_detects_live_remote_group(remote):
    registry, env = remote
    session = registry.spawn_via_env(env, "sleep 60", timeout=2)
    assert registry._post_kill_survivors(session), "remote tree was never checked"


@pytest.mark.platforms("linux")
def test_exited_wrapper_does_not_hide_live_owned_group(remote):
    registry, env = remote
    session = registry.spawn_via_env(env, "sleep 60 &", timeout=2)
    artifact = env.root / f"hermes_bg_{session.id}.exit"
    wait_until(artifact.exists, timeout=3, what="wrapper exited")
    result = registry.wait(session.id, timeout=3)
    assert result["status"] == "timeout", result
    assert registry.kill_process(session.id)["status"] == "killed"
    assert not _live_group(session.pid)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("fault", ["noop", "false", "true", "printf unsupported", "raise", "offline"])
def test_unconfirmed_cleanup_stays_manageable_for_retry(remote, fault):
    registry, env = remote
    session = registry.spawn_via_env(env, "sleep 60", timeout=2)
    env.signal_noop = fault == "noop"
    env.probe_command = None if fault in ("noop", "offline") else fault
    env.offline = fault == "offline"
    result = registry.kill_process(session.id)
    assert result["status"] == "error", result
    assert "unconfirmed" in result["error"].lower()
    assert registry.get(session.id) is session
    assert not session.exited
    assert _live_group(session.pid)
    assert session.completion_reason != "killed"
    # Let the real poller run too: an unsupported probe must not discard the handle.
    assert registry.wait(session.id, timeout=3)["status"] == "timeout"
    env.signal_noop = False
    env.probe_command = None
    env.offline = False
    assert registry.kill_process(session.id)["status"] == "killed"
    assert not _live_group(session.pid)
