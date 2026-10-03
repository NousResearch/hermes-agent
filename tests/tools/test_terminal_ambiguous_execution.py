"""Regression coverage for ambiguous terminal execution outcomes."""

import json

import pytest

import tools.environments.base as base_mod
import tools.terminal_tool as terminal_mod
from tools.environments.base import AmbiguousExecutionError, BaseEnvironment
from tools.terminal_tool import _ExecPlan, _run_foreground


class _PostSpawnFailureEnv(BaseEnvironment):
    def __init__(self, cwd: str):
        super().__init__(cwd=cwd, timeout=5)
        self.spawn_count = 0
        self.kill_count = 0

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        self.spawn_count += 1
        return object()

    def _wait_for_process(self, proc, timeout=120, **kwargs):
        raise RuntimeError("lost completion channel after spawn")

    def _kill_process(self, proc):
        self.kill_count += 1

    def cleanup(self):
        return None


def _plan(tmp_path):
    return _ExecPlan(
        config={},
        env_type="local",
        effective_task_id="ambiguous-execution-test",
        image="",
        cwd=str(tmp_path),
        host_cwd=None,
        effective_timeout=5,
    )


def test_environment_exception_after_spawn_is_ambiguous(tmp_path, monkeypatch):
    """A process handle is the point after which failure cannot prove no effect."""
    env = _PostSpawnFailureEnv(str(tmp_path))
    monkeypatch.setattr(base_mod, "_new_output_collector", lambda *args, **kwargs: object())

    with pytest.raises(AmbiguousExecutionError, match="outcome is unknown"):
        env.execute("printf mutation")

    assert env.spawn_count == 1
    assert env.kill_count == 1


def test_foreground_does_not_auto_retry_ambiguous_outcome(tmp_path):
    class EffectThenLostAck:
        host_cwd = None

        def __init__(self):
            self.calls = 0
            self.effects = 0

        def execute(self, command, **kwargs):
            self.calls += 1
            self.effects += 1
            raise AmbiguousExecutionError("acknowledgement lost")

    env = EffectThenLostAck()
    result = json.loads(
        _run_foreground(
            "mutate once",
            env,
            _plan(tmp_path),
            task_id=None,
            session_id=None,
            session_key="ambiguous-execution-test",
            workdir=None,
            approval_note=None,
            clear_interrupt=False,
            metered=False,
        )
    )

    assert env.calls == 1
    assert env.effects == 1
    assert result["status"] == "ambiguous"
    assert result["outcome_unknown"] is True
    assert "verify the target state before resending" in result["error"]


def test_pre_spawn_error_can_retry_but_ambiguity_stops_the_loop(tmp_path, monkeypatch):
    """Keep setup retries, but stop once an attempt may have landed."""
    class SetupThenAmbiguous:
        host_cwd = None

        def __init__(self):
            self.calls = 0

        def execute(self, command, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary pre-spawn setup failure")
            raise AmbiguousExecutionError("process spawned; completion channel lost")

    env = SetupThenAmbiguous()
    monkeypatch.setattr(terminal_mod.time, "sleep", lambda _: None)

    result = json.loads(
        _run_foreground(
            "mutate once",
            env,
            _plan(tmp_path),
            task_id=None,
            session_id=None,
            session_key="ambiguous-execution-test",
            workdir=None,
            approval_note=None,
            clear_interrupt=False,
            metered=False,
        )
    )

    assert env.calls == 2
    assert result["status"] == "ambiguous"
    assert result["outcome_unknown"] is True
