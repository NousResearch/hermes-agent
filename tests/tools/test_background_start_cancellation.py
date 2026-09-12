"""Interrupted detached starts must remain distinguishable and accounted for."""

import json
import subprocess
from types import SimpleNamespace

import pytest

import tools.process_registry as processes
from tools.interrupt import set_interrupt
from tools.terminal_tool_background import spawn_background_process


@pytest.mark.parametrize("pty", [False, True])
def test_interrupted_local_start_does_not_spawn_or_fall_back(monkeypatch, tmp_path, pty):
    registry = processes.ProcessRegistry()
    calls = []
    monkeypatch.setattr(processes, "process_registry", registry)
    monkeypatch.setattr(registry, "_scope_argv", lambda *_: ["inert"])
    monkeypatch.setattr(registry, "_spawn_env", lambda _: {})
    monkeypatch.setattr(registry, "_track_started", lambda *_: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: calls.append("pipe") or SimpleNamespace(pid=42))
    if pty:
        module = pytest.importorskip("winpty" if processes._IS_WINDOWS else "ptyprocess")
        monkeypatch.setattr(module.PtyProcess, "spawn", lambda *_a, **_kw: calls.append("pty") or SimpleNamespace(pid=42))
    set_interrupt(True)
    try:
        result = json.loads(spawn_background_process(
            command="true", env=SimpleNamespace(env={}), env_type="local", effective_task_id="test",
            task_id="test", session_key="", workdir=None, cwd=str(tmp_path), effective_pty=pty,
            notify_on_complete=False, watch_patterns=None, approval_note=None, pty_disabled_reason=None))
        assert result.get("status") == "cancelled", result
        assert result["exit_code"] == 130
        assert calls == []
        assert not registry._running
    finally:
        set_interrupt(False)


@pytest.mark.parametrize("stage", ["before", "pid", "no-pid", "natural", "natural-marker"])
def test_uncertain_sandbox_start_keeps_tracking_and_can_recover_pid(monkeypatch, tmp_path, stage):
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    monkeypatch.setattr(registry, "_env_temp_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    recorded = []

    def track(session, target, name, extra_args=()):
        registry._running[session.id] = session
        recorded.append((target, extra_args))

    monkeypatch.setattr(registry, "_track_started", track)
    result = {"output": "42", "returncode": 130}
    if stage == "before":
        result.update(output="", _process_start_cancelled=True)
    elif stage in ("pid", "no-pid"):
        result.update(output="42\n[Command interrupted]" if stage == "pid" else "[Command interrupted]",
                      _process_interrupted=True)
    elif stage == "natural-marker":
        result["output"] = "42\n[Command interrupted]"
    env = SimpleNamespace(execute=lambda *_a, **_kw: result)
    outcome = json.loads(spawn_background_process(
        command="true", env=env, env_type="docker", effective_task_id="test", task_id="test",
        session_key="", workdir=None, cwd=str(tmp_path), effective_pty=False,
        notify_on_complete=False, watch_patterns=None, approval_note=None, pty_disabled_reason=None))
    if stage == "before":
        assert outcome.get("status") == "cancelled", outcome
        assert not registry._running
        return
    session = registry.get(outcome["session_id"])
    assert session is not None and not session.exited
    if stage in ("pid", "no-pid"):
        assert outcome["exit_code"] == 130
        assert "may still be running" in outcome["error"]
        assert session.pid == (42 if stage == "pid" else None)
        if stage == "no-pid":
            replies = iter([{"output": ""}, {"output": "42"}, {"output": "0 0\n"},
                            {"output": "1"}, {"output": "7"}])
            env.execute = lambda *_a, **_kw: next(replies)
            monkeypatch.setattr(processes.time, "sleep", lambda _: None)
            target, args = recorded[0]
            target(session, *args)
            assert session.pid == 42
            assert session.exited and session.exit_code == 7
    else:
        assert outcome["exit_code"] == 0
        assert not getattr(session, "start_interrupted", False)
