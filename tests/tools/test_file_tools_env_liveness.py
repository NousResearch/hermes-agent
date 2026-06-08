"""file_tools caches ShellFileOperations keyed by task_id and reuses the shared
terminal env pool. Before this fix it only checked dict membership of
``_active_environments`` — not real container liveness — so an env whose
container was killed out-of-band would keep serving a stale, broken file_ops
(file reads/writes failing with "container is not running"). It must probe
liveness and rebuild, same as the terminal and execute_code paths.
"""

import tools.terminal_tool as terminal_tool
import tools.file_tools as file_tools


class _FakeEnv:
    def __init__(self, alive: bool, tag: str):
        self._alive = alive
        self.tag = tag
        self.cwd = "/root"

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
    file_tools._file_ops_cache.clear()


def teardown_function():
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()
    file_tools._file_ops_cache.clear()


def test_reuses_cached_file_ops_when_env_alive(monkeypatch):
    live = _FakeEnv(alive=True, tag="cached")
    terminal_tool._active_environments["default"] = live
    monkeypatch.setattr(
        terminal_tool, "_create_environment",
        lambda **k: (_ for _ in ()).throw(AssertionError("must not recreate a live env")),
    )

    first = file_tools._get_file_ops("default")
    second = file_tools._get_file_ops("default")

    assert first is second, "live env should reuse the cached file_ops"
    assert first.env is live


def test_dead_env_invalidates_file_ops_and_rebuilds(monkeypatch):
    dead = _FakeEnv(alive=False, tag="dead")
    fresh = _FakeEnv(alive=True, tag="fresh")
    terminal_tool._active_environments["default"] = dead

    # Seed a stale cached file_ops wrapping the dead env.
    file_tools._file_ops_cache["default"] = file_tools.ShellFileOperations(dead)

    created = []

    def _fake_create(**kwargs):
        created.append(kwargs)
        return fresh

    monkeypatch.setattr(terminal_tool, "_get_env_config", _minimal_docker_config)
    monkeypatch.setattr(terminal_tool, "_create_environment", _fake_create)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    file_ops = file_tools._get_file_ops("default")

    assert len(created) == 1, "dead env must trigger recreation"
    assert file_ops.env is fresh
    assert terminal_tool._active_environments["default"] is fresh
