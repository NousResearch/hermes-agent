"""Regression coverage for ambiguous terminal execution outcomes."""

import json

import pytest

import tools.environments.base as base_mod
import tools.terminal_tool as terminal_mod
from tools.environments.base import BaseEnvironment
from tools.environments.local import LocalEnvironment


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


def test_environment_exception_after_spawn_is_ambiguous(tmp_path, monkeypatch):
    """A process handle is the point after which failure cannot prove no effect."""
    env = _PostSpawnFailureEnv(str(tmp_path))
    monkeypatch.setattr(base_mod, "_new_output_collector", lambda *args, **kwargs: object())

    with pytest.raises(RuntimeError, match="outcome is unknown") as exc:
        env.execute("printf mutation")

    assert isinstance(exc.value, base_mod.AmbiguousExecutionError)
    assert env.spawn_count == 1
    assert env.kill_count == 1


@pytest.mark.parametrize("pre_spawn_failures", [0, 1])
def test_foreground_does_not_replay_landed_effect(tmp_path, monkeypatch, pre_spawn_failures):
    """Real shell effects survive lost completion; only pre-spawn setup may retry."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    env = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    run_bash = env._run_bash
    wait_for_process = env._wait_for_process
    kill_process = env._kill_process
    attempts = []
    spawned = []
    killed = []

    def spawn(command, **kwargs):
        attempts.append(command)
        if len(attempts) <= pre_spawn_failures:
            raise RuntimeError("temporary pre-spawn setup failure")
        proc = run_bash(command, **kwargs)
        spawned.append(proc)
        return proc

    def lose_completion(proc, **kwargs):
        result = wait_for_process(proc, **kwargs)
        assert result["returncode"] == 0
        raise RuntimeError("lost completion channel after spawn")

    def kill(proc):
        killed.append(proc)
        kill_process(proc)

    monkeypatch.setattr(env, "_run_bash", spawn)
    monkeypatch.setattr(env, "_wait_for_process", lose_completion)
    monkeypatch.setattr(env, "_kill_process", kill)
    monkeypatch.setattr(terminal_mod, "_acquire_env", lambda plan, task_id: env)
    monkeypatch.setattr(terminal_mod, "_start_cleanup_thread", lambda: None)
    try:
        raw = terminal_mod.registry.dispatch("terminal", {
            "command": "printf 'effect\\n' >> effect.txt",
            "timeout": 5,
            "workdir": str(tmp_path),
        })
        assert isinstance(raw, str)
        result = json.loads(raw)
        assert (tmp_path / "effect.txt").read_text(encoding="utf-8-sig") == "effect\n"
        assert len(attempts) == pre_spawn_failures + 1
        assert len(spawned) == 1
        assert killed == spawned
        assert result["status"] == "ambiguous"
        assert result["outcome_unknown"] is True
        assert "verify the target state before resending" in result["error"]
    finally:
        env.cleanup()
