"""Sandbox background jobs retain ownership until their process tree is stopped."""
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from unittest.mock import MagicMock, patch

import psutil
import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.mark.parametrize("scope", ["sandbox", "sandbox_group"])
@pytest.mark.parametrize("response", [
    {"returncode": 0, "output": ""},
    {"returncode": 0, "output": "__HERMES_PROCESS_TERMINATED__suffix\n"},
    {"returncode": 1, "output": "__HERMES_PROCESS_TERMINATED__\n"},
    {"returncode": "invalid", "output": "__HERMES_PROCESS_TERMINATED__\n"},
    {"returncode": 0, "output": "__HERMES_PROCESS_TERMINATED__\n"},
])
def test_sandbox_kill_requires_confirmed_termination(tmp_path, scope, response):
    registry = ProcessRegistry()
    env = MagicMock()
    env.execute.return_value = response
    session = ProcessSession(
        id="owned-sandbox", command="sleep 60", task_id="test", started_at=time.time(),
        pid=4321, pid_scope=scope, env_ref=env,
    )
    registry._running[session.id] = session
    with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "processes.json"):
        result = registry.kill_process(session.id)
    confirmed = response["returncode"] == 0 and response["output"] == "__HERMES_PROCESS_TERMINATED__\n"
    assert result["status"] == ("killed" if confirmed else "error")
    assert session.exited is confirmed
    assert (session.id in registry._running) is not confirmed
    assert (session.id in registry._completion_consumed) is confirmed


# These intentionally detached children leave the test subtree; ownership is
# checked by UID and psutil creation-time identity before cleanup signals.
@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="requires POSIX Bash")
@pytest.mark.parametrize("operation", ["term", "kill", "exit7", "exit130"])
def test_sandbox_supervisor_stops_descendants_and_records_exit(tmp_path, operation):
    class BashEnv:
        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, **kwargs):
            # A file captures the transport output without waiting for inherited
            # pipe descriptors in a background descendant to close.
            with tempfile.TemporaryFile(mode="w+") as output:
                result = subprocess.run(
                    ["bash", "-c", command], stdout=output, stderr=output,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                    timeout=kwargs.get("timeout", 5),
                )
                output.seek(0)
                return {"returncode": result.returncode, "output": output.read()}

    registry = ProcessRegistry()
    child_pid_file = tmp_path / "child.pid"
    command = (
        ("trap '' TERM; " if operation == "kill" else "")
        + f"echo $$ > {shlex.quote(str(child_pid_file))}; sleep 60"
    )
    if operation.startswith("exit"):
        command = "printf 'output preserved\\n'; exit " + operation[4:]
    owned = []
    try:
        with patch("tools.process_registry.threading.Thread", return_value=MagicMock()), \
                patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "processes.json"):
            session = registry.spawn_via_env(BashEnv(), command)
            assert session.pid is not None
            if operation.startswith("exit"):
                exit_file = tmp_path / f"hermes_bg_{session.id}.exit"
                deadline = time.monotonic() + 3
                while not exit_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert int(exit_file.read_text()) == int(operation[4:])
                assert (tmp_path / f"hermes_bg_{session.id}.log").read_text() == "output preserved\n"
                return
            leader = psutil.Process(session.pid)
            assert leader.uids().real == os.getuid()
            owned.append(leader)
            deadline = time.monotonic() + 3
            while not child_pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            child = psutil.Process(int(child_pid_file.read_text()))
            assert child.uids().real == os.getuid()
            owned.append(child)
            assert child.is_running() and child.status() != psutil.STATUS_ZOMBIE
            result = registry.kill_process(session.id)
            deadline = time.monotonic() + 3
            while child.is_running() and child.status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
            assert result["status"] == "killed"
            assert session.pid_scope == "sandbox_group"
    finally:
        for proc in reversed(owned):
            try:
                # psutil checks creation time before signaling a possibly reused PID.
                if proc.uids().real == os.getuid():
                    proc.send_signal(signal.SIGKILL)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.skipif(not shutil.which("bash"), reason="requires Bash")
@pytest.mark.parametrize("diagnostic,confirmed", [
    ("No such process", True), ("Operation not permitted", False),
])
def test_sandbox_probe_distinguishes_missing_from_inaccessible(diagnostic, confirmed):
    # Model kernel errno output without signaling a process owned by another UID.
    probe = (
        "kill() { printf '%s\\n' "
        + shlex.quote("bash: kill: (-4321) - " + diagnostic)
        + " >&2; return 1; }; "
        + ProcessRegistry._env_termination_command(4321, process_group=True)
    )
    result = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, timeout=4)
    assert (result.returncode == 0) is confirmed
    assert ("__HERMES_PROCESS_TERMINATED__" in result.stdout) is confirmed
