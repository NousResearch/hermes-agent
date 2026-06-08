"""execute_code shares the terminal env pool, so it has the same dead-container
reuse hazard: a cron `execute_code` task whose container was killed out-of-band
would otherwise keep RPC-ing into a corpse. ``_get_or_create_env`` must probe
liveness and rebuild, same as the terminal tool.
"""

import tools.terminal_tool as terminal_tool
import tools.code_execution_tool as cet


class _FakeEnv:
    def __init__(self, alive: bool, tag: str):
        self._alive = alive
        self.tag = tag

    def is_alive(self) -> bool:
        return self._alive


def _minimal_docker_config():
    return {
        "env_type": "docker",
        "docker_image": "python:3.11",
        "cwd": "/root",
        "timeout": 60,
    }


def setup_function():
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()


def teardown_function():
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()


def test_reuses_live_cached_env(monkeypatch):
    live = _FakeEnv(alive=True, tag="cached")
    terminal_tool._active_environments["default"] = live
    monkeypatch.setattr(terminal_tool, "_get_env_config", _minimal_docker_config)
    monkeypatch.setattr(
        terminal_tool, "_create_environment",
        lambda **k: (_ for _ in ()).throw(AssertionError("must not recreate a live env")),
    )

    env, env_type = cet._get_or_create_env("default")

    assert env is live
    assert env_type == "docker"


def test_evicts_dead_cached_env_and_recreates(monkeypatch):
    dead = _FakeEnv(alive=False, tag="dead")
    fresh = _FakeEnv(alive=True, tag="fresh")
    terminal_tool._active_environments["default"] = dead

    created = []

    def _fake_create(**kwargs):
        created.append(kwargs)
        return fresh

    monkeypatch.setattr(terminal_tool, "_get_env_config", _minimal_docker_config)
    monkeypatch.setattr(terminal_tool, "_create_environment", _fake_create)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    env, env_type = cet._get_or_create_env("default")

    assert env is fresh, "dead env must be evicted and rebuilt"
    assert len(created) == 1
    assert terminal_tool._active_environments["default"] is fresh
