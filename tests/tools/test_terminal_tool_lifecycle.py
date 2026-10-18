"""Lifecycle ownership contracts for strict terminal environment cleanup."""

import pytest


def test_strict_cleanup_retains_environment_after_teardown_failure(monkeypatch):
    from tools import terminal_tool
    from tools import terminal_tool_lifecycle as lifecycle

    task_id = "strict-cleanup-retry"

    class Environment:
        def __init__(self):
            self.attempts = 0

        def cleanup(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("environment teardown failed")

    env = Environment()
    terminal_tool._active_environments[task_id] = env
    terminal_tool._last_activity[task_id] = 1.0
    monkeypatch.setattr(lifecycle, "_clear_file_ops_cache", lambda _task_id: None)

    try:
        with pytest.raises(RuntimeError, match="environment teardown failed"):
            lifecycle.cleanup_vm(task_id, strict=True)

        assert terminal_tool._active_environments[task_id] is env
        lifecycle.cleanup_vm(task_id, strict=True)
        assert task_id not in terminal_tool._active_environments
        assert env.attempts == 2
    finally:
        terminal_tool._active_environments.pop(task_id, None)
        terminal_tool._last_activity.pop(task_id, None)
        terminal_tool._creation_locks.pop(task_id, None)
