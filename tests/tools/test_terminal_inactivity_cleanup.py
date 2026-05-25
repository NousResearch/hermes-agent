"""Regression tests for monotonic terminal inactivity bookkeeping."""

import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_env_config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


class _DummyEnv:
    def __init__(self):
        self.cleaned = 0

    def cleanup(self):
        self.cleaned += 1


def test_cleanup_inactive_envs_ignores_wall_clock_jumps(monkeypatch):
    """Idle reaping should key off monotonic elapsed time, not wall clock."""
    import tools.terminal_tool as terminal_tool_module

    env = _DummyEnv()
    monkeypatch.setitem(
        sys.modules,
        "tools.process_registry",
        SimpleNamespace(
            process_registry=SimpleNamespace(
                has_active_processes=lambda _task_id: False
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.file_tools",
        SimpleNamespace(clear_file_ops_cache=lambda _task_id: None),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_active_environments",
        {"task-1": env},
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_last_activity",
        {"task-1": 1000.0},
    )
    monkeypatch.setattr(terminal_tool_module, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool_module, "_creation_locks_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool_module, "_env_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool_module, "_activity_now", lambda: 1100.0)
    monkeypatch.setattr(terminal_tool_module.time, "time", lambda: 10_000_000.0)

    terminal_tool_module._cleanup_inactive_envs(lifetime_seconds=300)

    assert "task-1" in terminal_tool_module._active_environments
    assert terminal_tool_module._last_activity["task-1"] == 1000.0
    assert env.cleaned == 0


def test_existing_env_refresh_uses_monotonic_activity_timestamp(monkeypatch):
    """Reusing an env should stamp _last_activity from the monotonic helper."""
    import tools.terminal_tool as terminal_tool_module

    env = MagicMock()
    env.execute.return_value = {"output": "done", "returncode": 0}
    last_activity = {"default": 0.0}

    monkeypatch.setattr(terminal_tool_module, "_activity_now", lambda: 123.0)
    monkeypatch.setattr(terminal_tool_module.time, "time", lambda: 99_999.0)

    with patch(
        "tools.terminal_tool._get_env_config",
        return_value=_make_env_config(),
    ), patch(
        "tools.terminal_tool._start_cleanup_thread",
        lambda: None,
    ), patch(
        "tools.terminal_tool._check_all_guards",
        return_value={"approved": True},
    ), patch(
        "tools.terminal_tool._active_environments",
        {"default": env},
    ), patch(
        "tools.terminal_tool._last_activity",
        last_activity,
    ):
        result = json.loads(terminal_tool_module.terminal_tool(command="echo hello"))
        assert result["error"] is None
        assert last_activity["default"] == 123.0
