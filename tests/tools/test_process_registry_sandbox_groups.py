"""A sandbox background job's kill stops its whole process group, or reports who survived."""
import shlex
import subprocess
import sys
import tempfile
import threading
from unittest.mock import MagicMock, patch

import psutil
import pytest

from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_termination import env_group_kill_command


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


def _writer_command(tmp_path, progress, *, ignore_term):
    """A shell that runs a Python writer as its child, so the writer is a grandchild of
    the recorded PID (a kill of that PID alone leaves it writing)."""
    writer = tmp_path / "writer.py"
    writer.write_text(
        "import signal, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"with open({str(progress)!r}, 'a', buffering=1) as output:\n"
        "    while True:\n"
        "        output.write('tick\\n')\n"
        "        time.sleep(0.05)\n",
        encoding="utf-8")
    trap = "trap '' TERM; " if ignore_term else ""
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
def test_survivors_are_reported_by_pid_and_the_session_stays_killable(tmp_path):
    registry = ProcessRegistry()
    progress = tmp_path / "progress"
    tree = []
    grace = {"value": 0.0}  # 0 = no SIGKILL escalation, so the TERM-ignoring tree survives
    with patch.object(registry, "_write_checkpoint"), _grace(grace):
        session = registry.spawn_via_env(BashEnv(tmp_path), _writer_command(tmp_path, progress, ignore_term=True))
        try:
            wait_until(lambda: progress.exists() and progress.stat().st_size > 0, what="writer started")
            root = psutil.Process(session.pid)
            tree = [root, *root.children(recursive=True)]

            partial = registry.kill_process(session.id)

            assert partial["status"] == "error" and partial["process_running"] is True, partial
            # The supervisor traps TERM and waits on the TERM-ignoring command, so all of it lives.
            assert set(partial["survivors"]) == {proc.pid for proc in tree}
            assert all(_alive(proc) for proc in tree)
            assert registry.get(session.id) is session and not session.exited

            grace["value"] = 0.3
            assert registry.kill_process(session.id)["status"] == "killed"
            _assert_quiet(progress)
        finally:
            for proc in tree:
                if _alive(proc):
                    proc.kill()
            session.exited = True
            _join_poller(session)


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


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("diagnostic,gone", [("No such process", True), ("Operation not permitted", False)])
def test_only_esrch_proves_the_group_gone(diagnostic, gone):
    # A shell function stands in for the kill builtin, so no real process is signalled.
    script = (
        "kill() { printf '%s\\n' " + shlex.quote("bash: kill: (-4321) - " + diagnostic)
        + " >&2; return 1; }; " + env_group_kill_command(4321, 0.1))
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10, check=False)
    assert (result.returncode == 0) is gone
    assert ("__HERMES_PROCESS_TERMINATED__" in result.stdout.splitlines()) is gone


@pytest.mark.parametrize("pgid", [0, 1, -1, True])
def test_kill_command_refuses_group_ids_that_reach_other_processes(pgid):
    with pytest.raises(ValueError):
        env_group_kill_command(pgid, 1.0)
