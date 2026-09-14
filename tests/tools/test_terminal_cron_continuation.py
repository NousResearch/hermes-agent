"""Terminal transport contracts for cron continuations; remote I/O is simulated."""

import json
import subprocess
import sys
import threading

import pytest

from cron import continuations, executions, jobs, scheduler
from tools import process_registry as processes
from tools.terminal_tool_background import spawn_background_process


@pytest.fixture
def launch_context(tmp_path, monkeypatch):
    job = jobs.create_job(prompt="Build then inspect", schedule="every 1h", workdir=str(tmp_path))
    execution = executions.create_execution(job["id"], source="builtin")
    scope = scheduler._CronRunScope(job, job["id"], execution["id"])
    scope.enter()
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    try:
        identity = continuations.validate_request(True, scope.task_id)
        yield registry, scope, identity, tmp_path
    finally:
        registry.kill_all()
        scope.exit()


def launch(context, *, env_type="local", env=None, pty=False, notify=False, watch=None, opted=True):
    registry, scope, identity, cwd = context
    return json.loads(spawn_background_process(
        command="printf 'transport done'", env=env, env_type=env_type,
        effective_task_id=scope.task_id, task_id=scope.task_id, session_key="",
        workdir=str(cwd), cwd=str(cwd), effective_pty=pty,
        notify_on_complete=notify, watch_patterns=watch, approval_note=None,
        pty_disabled_reason=None, continuation=identity if opted else None))


@pytest.mark.parametrize("pty,notify,watch,opted", [
    (False, True, ["done"], True), (True, False, None, True),
    (False, True, None, False), (False, False, None, False),
])
def test_local_pipe_and_pty_keep_interactive_notifications_off(launch_context, pty, notify, watch, opted):
    registry, scope, identity, cwd = launch_context
    data = launch(launch_context, pty=pty, notify=notify, watch=watch, opted=opted)
    assert not data.get("error"), data
    session = registry.get(data["session_id"])
    assert session._completion_event.wait(20)
    assert not session.notify_on_complete and not session.watch_patterns
    assert registry.completion_queue.empty() and not registry.pending_watchers
    if pty:
        assert session._pty is not None  # Do not silently call a pipe fallback a PTY test.
    with executions._transaction() as conn:
        rows = conn.execute("SELECT * FROM process_continuations").fetchall()
        assert bool(rows) == opted
    assert bool(data.get("continue_on_complete")) == opted
    assert session._reader_thread.daemon == (not opted)


@pytest.mark.parametrize("backend", ["docker", "ssh"])
@pytest.mark.parametrize("outcome", ["success", "no_pid", "launch_error", "lost", "unknown_exit"])
def test_sandbox_transport_success_and_failure_contract(launch_context, backend, outcome):
    registry, scope, identity, cwd = launch_context

    class Backend:
        def __init__(self):
            self.started = False

        def get_temp_dir(self):
            return str(cwd)

        def execute(self, command, **kwargs):
            if not self.started:
                self.started = True
                if outcome == "launch_error":
                    raise OSError("backend unreachable")
                return {"output": "12345" if outcome != "no_pid" else "failed", "returncode": 0}
            if outcome == "lost":
                raise OSError("backend disconnected")
            if command.startswith("O="):
                return {"output": "14 0\ntransport done", "returncode": 0}
            if command.startswith("kill -0"):
                return {"output": "1", "returncode": 0}
            return {"output": "7" if outcome != "unknown_exit" else "", "returncode": 0}

    data = launch(launch_context, env_type=backend, env=Backend())
    if outcome in {"no_pid", "launch_error"}:
        assert data.get("error"), data
        assert not data.get("continue_on_complete")
    else:
        assert not data.get("error"), data
        session = registry.get(data["session_id"])
        assert session._completion_event.wait(20)
    executions.finish_execution(identity["execution_id"], success=True)
    pending = continuations.pending_jobs()
    assert bool(pending) == (outcome == "success")
    if pending:
        result = json.loads(pending[0]["_process_continuation"]["result"])
        assert result["exit_code"] == 7
        assert result["output"] == "transport done"
    assert registry.completion_queue.empty()


def test_local_launch_failure_does_not_create_a_completion(launch_context, monkeypatch):
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("spawn refused")))
    result = launch(launch_context)
    assert "spawn refused" in result["error"]
    with executions._transaction() as conn:
        assert conn.execute("SELECT count(*) FROM process_continuations").fetchone()[0] == 0


def test_reader_and_kill_race_persist_only_one_cancelled_result(launch_context):
    registry, scope, identity, cwd = launch_context
    # Real registry transition under contention, without timing a live shell.
    session = processes.ProcessSession(id="proc_race", command="build", exited=True,
        exit_code=-15, completion_reason="killed", cron_continuation=identity)
    registry._running[session.id] = session
    barrier = threading.Barrier(3)
    errors = []

    def finish():
        try:
            barrier.wait(15)
            registry._move_to_finished(session)
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=finish) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait(15)
    for worker in workers:
        worker.join(20)
        assert not worker.is_alive()
    assert not errors
    assert session._completion_event.is_set()
    with executions._transaction() as conn:
        assert conn.execute("SELECT count(*) FROM process_continuations").fetchone()[0] == 1
    assert continuations.pending_jobs() == []


def test_checkpoint_recovery_preserves_cron_identity_and_does_not_enable_notifications(launch_context):
    registry, scope, identity, cwd = launch_context
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    session = processes.ProcessSession(id="proc_checkpoint", command="test child", pid=child.pid,
        host_start_time=registry._safe_host_start_time(child.pid), cron_continuation=identity)
    registry._running[session.id] = session
    registry._write_checkpoint()
    recovered = processes.ProcessRegistry()
    try:
        assert recovered.recover_from_checkpoint() == 1
        restored = recovered.get(session.id)
        assert restored.cron_continuation == identity
        assert not restored.notify_on_complete
        assert not recovered.pending_watchers
        # PID recovery cannot recreate stdout or the exit code. It must not wake
        # the model with an invented success after that child disappears.
        child.terminate()
        child.wait(timeout=15)
        recovered.get(session.id)
        assert continuations.pending_jobs() == []
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=15)
        registry._running.clear()
        recovered._running.clear()
