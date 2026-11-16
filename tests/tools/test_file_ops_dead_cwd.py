"""Only the terminal tool may run in an ancestor of a deleted cwd; file tools must not follow it.

The fallback is safe for a terminal command because its result tells the model where it ran. A
file operation has no such notice: ``read_file("README.md")`` with the session cwd removed would
read the ancestor's ``README.md`` while read tracking records the original path. So a file op
fails like it always did, on the local backend and through the shared sandbox wrapper alike.
"""

import json
import shutil
import subprocess
import time

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.base_output import _pipe_stdin
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux", "macos")


class _BashSandbox(BaseEnvironment):
    """A sandbox backend reduced to a real bash: no host-side ``_usable_cwd``, the wrapper decides."""

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        proc = subprocess.Popen(["bash", "-c", cmd_string], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL if stdin_data is None else subprocess.PIPE, text=True, cwd="/")
        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)
        return proc

    def cleanup(self):
        pass


def _dead_cwd_with_ancestor_readme(tmp_path):
    (tmp_path / "README.md").write_text("ANCESTOR\n")
    doomed = tmp_path / "proj" / "sub"
    doomed.mkdir(parents=True)
    return doomed


def _make_local(doomed):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LocalEnvironment, "init_session", lambda self, *a, **k: None)
        return LocalEnvironment(cwd=str(doomed), timeout=30)


@pytest.mark.parametrize("native", ["1", "0"], ids=["native-read", "shell-read"])
def test_local_read_of_relative_path_in_deleted_cwd_errors_instead_of_reading_ancestor(tmp_path, monkeypatch, native):
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", native)
    doomed = _dead_cwd_with_ancestor_readme(tmp_path)
    env = _make_local(doomed)
    try:
        ops = ShellFileOperations(env, cwd=str(doomed))
        shutil.rmtree(tmp_path / "proj")
        env.execute("true", timeout=30)  # an internal command must not move the session either
        for _ in range(2):
            result = ops.read_file("README.md")
            assert "ANCESTOR" not in (result.content or ""), result
            assert result.error, result
        assert env.cwd == str(doomed)
    finally:
        env.cleanup()


def test_sandbox_read_of_relative_path_in_deleted_cwd_errors_instead_of_reading_ancestor(tmp_path):
    doomed = _dead_cwd_with_ancestor_readme(tmp_path)
    env = _BashSandbox(cwd=str(doomed), timeout=30)
    ops = ShellFileOperations(env, cwd=str(doomed))
    shutil.rmtree(tmp_path / "proj")

    result = ops.read_file("README.md")
    assert "ANCESTOR" not in (result.content or ""), result
    assert result.error and "does not exist" in result.error, result
    # The terminal tool's own command still recovers, and says so.
    assert env.execute("pwd", timeout=30, allow_cwd_fallback=True)["cwd_fallback"]["used"] == str(tmp_path.resolve())


def test_sandbox_background_start_in_deleted_cwd_carries_the_fallback_note(tmp_path):
    from tools.process_registry import process_registry
    from tools.terminal_tool_background import spawn_background_process

    doomed = tmp_path / "proj" / "sub"
    doomed.mkdir(parents=True)
    env = _BashSandbox(cwd=str(doomed), timeout=30)
    shutil.rmtree(tmp_path / "proj")
    started = json.loads(spawn_background_process(
        command="pwd -P", env=env, env_type="ssh", effective_task_id="bg-dead-cwd", task_id="bg-dead-cwd",
        session_key="", workdir=None, cwd=str(doomed), effective_pty=False, notify_on_complete=False,
        watch_patterns=None, approval_note=None, pty_disabled_reason=None))
    try:
        assert started["output"] == "Background process started", started
        assert str(doomed) in started["cwd_fallback"] and str(tmp_path.resolve()) in started["cwd_fallback"], started
        session = process_registry.get(started["session_id"])
        assert session is not None
        wait_until(lambda: session.exited, timeout=30, what="background job exit")
        assert session.exit_code == 0 and session.output_buffer.strip() == str(tmp_path.resolve())
    finally:
        process_registry.kill_all()


def test_sandbox_job_that_deletes_its_cwd_is_still_tracked_and_session_cwd_is_untouched(tmp_path):
    """The poller reads the job's log by absolute path from "/": a job removing its own cwd must not
    turn every poll into a cd failure (a false exit -1), and polling must not move ``env.cwd``."""
    from tools.process_registry import process_registry

    doomed = tmp_path / "proj"
    doomed.mkdir()
    env = _BashSandbox(cwd=str(doomed), timeout=30)
    session = process_registry.spawn_via_env(env, f"rm -rf {doomed}; sleep 3; echo done; exit 7", cwd=str(doomed))
    try:
        wait_until(lambda: session.exited, timeout=30, what="background job exit")
        assert session.exit_code == 7 and "done" in session.output_buffer, session.output_buffer
        assert env.cwd == str(doomed)
    finally:
        process_registry.kill_all()


@pytest.mark.parametrize("at_root", [False, True], ids=["project-cwd", "root-cwd"])
def test_sandbox_execute_code_per_call_leaves_the_session_cwd_alive(tmp_path, at_root):
    """The per-call ``execute_code`` launch enters a scratch dir and deletes it when done. If that dir
    became ``env.cwd``, the "/" cleanup's restore kept it, so every later file op, an absolute read
    included, failed at ``cd`` with "working directory ... does not exist"."""
    from tools.code_execution_tool import _run_remote_per_call

    project = tmp_path.resolve() / "proj"
    project.mkdir()
    (project / "notes.txt").write_text("NOTES\n")
    session_cwd = "/" if at_root else str(project)
    env = _BashSandbox(cwd=session_cwd, timeout=30)
    ran = json.loads(_run_remote_per_call(env, "ssh", "print('ran')", "per-call-cwd", frozenset(),
                                          timeout=30, max_tool_calls=1, exec_start=time.monotonic()))
    assert ran["status"] == "success" and ran["output"].strip() == "ran", ran

    read = ShellFileOperations(env).read_file(str(project / "notes.txt"))
    assert "NOTES" in (read.content or ""), read
    assert env.cwd == session_cwd
