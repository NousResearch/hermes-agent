"""Capture integrity survives kill/recovery and adoption retains bounded history."""
import json
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from tools import process_registry as module
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.registry import registry as tools
from tools.terminal_tool_background import spawn_background_process


@pytest.mark.parametrize("incomplete", [True, False])
def test_public_kill_keeps_prior_capture_integrity(monkeypatch, incomplete):
    registry = ProcessRegistry()
    monkeypatch.setattr(module, "process_registry", registry)
    env = Mock()
    session = ProcessSession(id="proc_capture_kill", command="job", env_ref=env,
                             pid=123, pid_scope="sandbox", output_buffer="retained-prefix")
    if incomplete:
        session.capture_failed("remote log read")
    registry._running[session.id] = session

    result = json.loads(tools.dispatch("process_manage", {"action": "kill", "session_id": session.id}))

    assert result["status"] == "killed"
    assert result["output"] == "retained-prefix"
    assert result["completion_reason"] == "killed"
    assert result["termination_source"] == "process.kill"
    assert session.id in registry._completion_consumed
    env.execute.assert_called_once()
    if incomplete:
        assert result.get("output_incomplete") is True
        assert result["capture_error"] == session.capture_error
    else:
        assert not result.get("output_incomplete")
        assert not result.get("capture_error")


@pytest.mark.parametrize("receipt", [OSError("private transport"), TimeoutError("private timeout"),
                                     {"output": "", "returncode": 0, "output_incomplete": True}])
def test_public_uncertain_launch_survives_receipt_failure(monkeypatch, receipt):
    registry = ProcessRegistry()
    monkeypatch.setattr(module, "process_registry", registry)
    env = Mock(cwd="/tmp", host_cwd=None)
    env.get_temp_dir.return_value = "/tmp"
    env.execute.side_effect = [{"output": "12", "returncode": 0, "output_incomplete": True}, receipt]

    result = json.loads(spawn_background_process(
        command="side_effect_once", env=env, env_type="ssh", effective_task_id="capture-recovery",
        task_id="capture-recovery", session_key="", workdir=None, cwd="/tmp", effective_pty=False,
        notify_on_complete=True, watch_patterns=None, approval_note=None, pty_disabled_reason=None))

    calls = [call.args[0] for call in env.execute.call_args_list]
    assert len(calls) == 2
    assert "side_effect_once" in calls[0]
    assert calls[1].startswith("cat ") and ".pid" in calls[1]
    assert "side_effect_once" not in calls[1]
    assert not registry._running
    assert result.get("output_incomplete") is True
    assert result["exit_code"] is None
    assert "may already have started" in result["error"]
    assert "private" not in result["error"]
    assert "Failed to start" not in result["error"]


@pytest.mark.parametrize("start_fails", [False, True])
def test_adoption_prunes_only_retirable_history_after_finalization(tmp_path, monkeypatch, start_fails):
    registry = ProcessRegistry()
    old = time.time() - module.FINISHED_TTL_SECONDS - 10
    expired = ProcessSession(id="proc_expired", command="done", started_at=old, exited=True,
                             output_buffer="x" * 1000)
    pending = ProcessSession(id="proc_pending", command="pending", started_at=old, exited=True)
    pending._finish_claimed = True
    scope = ProcessSession(id="proc_scope", command="scope", started_at=old, exited=True)
    scope._scope_stop_pending = True
    registry._finished.update({s.id: s for s in (expired, pending, scope)})
    registry._completion_consumed.add(expired.id)
    registry._poll_observed.add(expired.id)
    finalized = []
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read(); print('tail')"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def finalize():
        assert not registry._running
        finalized.append(True)

    try:
        if start_fails:
            with monkeypatch.context() as patcher:
                patcher.setattr("threading.Thread.start", Mock(side_effect=RuntimeError("cannot start reader")))
                with pytest.raises(RuntimeError, match="cannot start reader"):
                    registry.adopt_local(proc, command="job", cwd=str(tmp_path), finalize_capture=finalize)
            assert not registry._running and not finalized
            assert expired.id in registry._finished
            assert not proc.stdout.closed
        else:
            session = registry.adopt_local(proc, command="job", cwd=str(tmp_path), finalize_capture=finalize)
            assert finalized == [True]
            assert session.id in registry._running
            assert pending.id in registry._finished and scope.id in registry._finished
            assert expired.id not in registry._finished
            assert expired.id not in registry._completion_consumed
            assert expired.id not in registry._poll_observed
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
        if not start_fails:
            session._reader_thread.join(timeout=10)
            assert not session._reader_thread.is_alive()
        proc.stdout.close()
