"""Plugin terminal backends with per-session sandboxes survive between turns.

A plugin backend that declares ``session_isolated_when_nonpersistent`` gets one sandbox per
session when ``terminal.container_persistent`` is false. Its ``cleanup()`` deletes that sandbox.
The end-of-turn cleanup skips environments ``is_persistent_env`` reports as persistent, so the
plugin's environment must count as session-scoped, as the Docker backend's does. Otherwise
every turn deletes the session's sandbox and the next turn starts a new, empty one.
"""

from types import SimpleNamespace

import pytest

from agent import terminal_env_registry as reg
from agent.chat_completion_helpers import cleanup_task_resources
from agent.terminal_env_provider import TerminalEnvironmentProvider
from tools import terminal_tool
from tools.terminal_tool_lifecycle import ensure_task_env, get_active_env

SESSION = "tui:sess-1"


class _SandboxEnv:
    """Stands in for a plugin sandbox: ``cleanup()`` is the call that deletes it."""

    def __init__(self, persistent: bool):
        self._persistent = persistent
        self.deleted = False

    def cleanup(self):
        self.deleted = True


class _IsolatingProvider(TerminalEnvironmentProvider):
    name = "isobox"
    session_isolated_when_nonpersistent = True

    def is_available(self):
        return True

    def create_environment(self, *, cwd, timeout, task_id="default", container_config=None, **_):
        # Same source the cua-spaces plugin reads its persistence from.
        return _SandboxEnv(bool((container_config or {}).get("container_persistent", True)))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reg._reset_for_tests()
    reg.register_provider(_IsolatingProvider())
    # The config->env bridge is one-shot; mark it done so tests control the env vars.
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    before = dict(terminal_tool._active_environments)
    yield
    with terminal_tool._env_lock:
        for key in set(terminal_tool._active_environments) - set(before):
            terminal_tool._active_environments.pop(key)
            terminal_tool._last_activity.pop(key, None)
    reg._reset_for_tests()


def _end_turn(monkeypatch, backend: str, task_id: str) -> _SandboxEnv:
    """Bring *backend*'s environment up through the real creation path, then end the turn."""
    monkeypatch.setenv("TERMINAL_ENV", backend)
    env = ensure_task_env(task_id)
    assert isinstance(env, _SandboxEnv)
    cleanup_task_resources(SimpleNamespace(verbose_logging=False), task_id)
    return env


def test_session_sandbox_survives_the_end_of_a_turn(monkeypatch):
    env = _end_turn(monkeypatch, "isobox", SESSION)
    assert env.deleted is False
    assert get_active_env(SESSION) is env  # the next turn reuses the same sandbox


def test_shared_default_sandbox_keeps_per_turn_teardown(monkeypatch):
    # No session key: every caller shares "default", which is not one session's sandbox.
    env = _end_turn(monkeypatch, "isobox", "default")
    assert env.deleted is True
