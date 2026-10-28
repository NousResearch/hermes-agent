"""An interrupted spawn cannot report clean teardown while its scope is still live."""

import json
from types import SimpleNamespace

import pytest

import tools.process_registry as processes
import tools.terminal_tool as terminal
import tools.terminal_tool_background as background
from tools import interrupt


@pytest.mark.parametrize("scope_stops", [False, True], ids=["failed-stop", "stopped"])
def test_interrupted_spawn_reports_scope_cleanup_outcome(monkeypatch, tmp_path, scope_stops):
    registry = processes.ProcessRegistry()
    session = processes.ProcessSession(
        id="proc_interrupted_scope", command="daemonize", task_id="old-owner",
        session_key="agent:main:telegram:dm:scope",
        systemd_unit="hermes-worker-proc_interrupted_scope.scope",
    )
    session.mark_exited(0)
    registry._finished[session.id] = session
    monkeypatch.setattr(processes, "process_registry", registry)
    monkeypatch.setattr(processes, "_stop_systemd_unit", lambda _unit: scope_stops)
    checks = iter([False, True])
    monkeypatch.setattr(interrupt, "is_interrupted", lambda: next(checks))
    monkeypatch.setattr(background, "_spawn", lambda *_a, **_kw: session)
    monkeypatch.setattr(terminal, "_resolve_command_cwd", lambda **_kw: str(tmp_path))

    result = json.loads(background.spawn_background_process(
        command=session.command, env=SimpleNamespace(), env_type="local",
        effective_task_id=session.task_id, task_id=session.owner_task_id,
        session_key=session.session_key, workdir=None, cwd=str(tmp_path),
        effective_pty=False, notify_on_complete=False, watch_patterns=None,
        approval_note=None, pty_disabled_reason=None,
    ))

    if scope_stops:
        assert result["exit_code"] == 130
        assert result["error"] is None
        assert not session._scope_stop_pending
    else:
        assert result["exit_code"] == -1, result
        assert session.systemd_unit in result["error"]
        assert session._scope_stop_pending
