"""Env-backed process kills must stop the command's writers, not just its wrapper."""

import shlex
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch

import psutil
import pytest

from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry


class BashEnvironment:
    """Exercise the remote execute contract using a real, noninteractive bash."""

    def __init__(self, directory):
        self.directory = directory

    def get_temp_dir(self):
        return str(self.directory)

    def execute(self, command, timeout=10, **_kwargs):
        # A remote exec can return while the background wrapper holds its descriptors.
        with tempfile.TemporaryFile(mode="w+") as output:
            result = subprocess.run(
                ["bash", "-c", command], stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
            output.seek(0)
            return {"output": output.read(), "returncode": result.returncode}


def _alive(process):
    return process.is_running() and process.status() != psutil.STATUS_ZOMBIE


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("ignore_term", [False, True])
def test_env_kill_stops_writer_and_children(tmp_path, ignore_term):
    writer = tmp_path / "writer.py"
    progress = tmp_path / "progress"
    writer.write_text(
        "import signal, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"with open({str(progress)!r}, 'a', buffering=1) as output:\n"
        "    while True:\n"
        "        output.write('tick\\n')\n"
        "        time.sleep(0.05)\n"
    )
    registry = ProcessRegistry()
    env = BashEnvironment(tmp_path)
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(writer))} & wait"
    with patch.object(registry, "_write_checkpoint"):
        session = registry.spawn_via_env(env, command)
        processes = []
        try:
            assert session.pid and not session.exited
            wait_until(lambda: progress.exists() and progress.stat().st_size > 0)
            root = psutil.Process(session.pid)
            processes = [root, *root.children(recursive=True)]
            assert len(processes) >= 3  # wrapper, command shell, actual writer
            result = registry.kill_process(session.id)
            assert result["status"] == "killed", result
            size_at_stop = progress.stat().st_size
            # Observe a quiet interval spanning several scheduled writes.
            # health: allow HX013 -- a quiet interval is the behavior being verified
            threading.Event().wait(0.3)
            assert progress.stat().st_size == size_at_stop, "writer continued after kill returned"
            wait_until(lambda: not any(_alive(proc) for proc in processes), what="command tree stopped")
        finally:
            for proc in reversed(processes):
                if _alive(proc):
                    proc.kill()
            session.exited = True
            session._reader_thread.join(timeout=12)


@pytest.mark.platforms("posix")
def test_env_kill_after_exit_preserves_result(tmp_path):
    registry = ProcessRegistry()
    env = BashEnvironment(tmp_path)
    with patch.object(registry, "_write_checkpoint"):
        session = registry.spawn_via_env(env, "exit 7")
        try:
            exit_file = tmp_path / f"hermes_bg_{session.id}.exit"
            wait_until(exit_file.exists)
            # The reader publishes the captured exit code; killing that completed
            # session must preserve it and must not send a signal to a reused PID.
            registry._finish_exited(session, int(exit_file.read_text()))
            with patch.object(env, "execute", side_effect=AssertionError("signal after exit")):
                result = registry.kill_process(session.id)
            assert result["status"] == "already_exited"
            assert session.exit_code == 7
        finally:
            session.exited = True
            session._reader_thread.join(timeout=12)


def test_env_kill_failure_keeps_session_running(tmp_path):
    from tools.process_registry import ProcessSession

    registry = ProcessRegistry()
    env = BashEnvironment(tmp_path)
    session = ProcessSession(id="proc_remote_failure", command="writer", task_id="task",
                             pid=123, pid_scope="sandbox", env_ref=env)
    registry._running[session.id] = session
    with patch.object(env, "execute", return_value={"returncode": 255, "output": "connection lost"}):
        result = registry.kill_process(session.id)
    assert result["status"] == "error"
    assert not session.exited
    assert registry.get(session.id) is session
