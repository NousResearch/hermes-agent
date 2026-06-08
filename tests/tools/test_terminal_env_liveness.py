"""Reuse-path liveness checks for the terminal tool.

A sandbox container can be killed out-of-band (docker prune, OOM, manual rm).
The cached environment in ``_active_environments`` then points at a corpse, and
because ``docker exec`` into a dead container returns a normal non-zero result
(not an exception) the retry loop never rebuilds it — every subsequent command
reports "container is not running" until the gateway restarts.

``_acquire_cached_env`` closes that hole: it probes ``is_alive()`` before reuse
and evicts a dead env so the caller recreates it.
"""

import time

import tools.terminal_tool as terminal_tool


class _FakeEnv:
    def __init__(self, alive: bool):
        self._alive = alive
        self.execute_called = False

    def is_alive(self) -> bool:
        return self._alive

    def execute(self, *a, **k):  # pragma: no cover - must never run when dead
        self.execute_called = True
        return {"output": "", "returncode": 0}


def setup_function():
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()


def teardown_function():
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()


def test_returns_none_when_no_cached_env():
    assert terminal_tool._acquire_cached_env("default") is None


def test_reuses_live_env_and_refreshes_activity():
    env = _FakeEnv(alive=True)
    terminal_tool._active_environments["default"] = env
    terminal_tool._last_activity["default"] = 0.0

    result = terminal_tool._acquire_cached_env("default")

    assert result is env
    assert terminal_tool._active_environments["default"] is env
    assert terminal_tool._last_activity["default"] > 0.0


def test_env_without_probe_is_assumed_alive():
    """Env-likes that predate the is_alive() contract (or third-party doubles)
    keep the old reuse behavior instead of crashing the terminal call."""

    class _NoProbeEnv:
        pass

    env = _NoProbeEnv()
    terminal_tool._active_environments["default"] = env
    terminal_tool._last_activity["default"] = 0.0

    result = terminal_tool._acquire_cached_env("default")

    assert result is env
    assert terminal_tool._last_activity["default"] > 0.0


def test_evicts_dead_env_and_returns_none():
    dead = _FakeEnv(alive=False)
    terminal_tool._active_environments["default"] = dead
    terminal_tool._last_activity["default"] = time.time()

    result = terminal_tool._acquire_cached_env("default")

    assert result is None
    # Evicted so the caller falls through to recreation.
    assert "default" not in terminal_tool._active_environments
    assert "default" not in terminal_tool._last_activity
    assert dead.execute_called is False
