from __future__ import annotations

from tools import terminal_tool


class _FailingEnvironment:
    def cleanup(self):
        return False


def test_cleanup_vm_retracks_environment_when_cleanup_fails():
    """Failed cleanup should not make Hermes forget a live container."""
    task_id = "cleanup-failed-task"
    env = _FailingEnvironment()

    with terminal_tool._env_lock:
        terminal_tool._active_environments[task_id] = env
        terminal_tool._last_activity[task_id] = 1.0

    try:
        assert terminal_tool.cleanup_vm(task_id) is False
        with terminal_tool._env_lock:
            assert terminal_tool._active_environments[task_id] is env
            assert task_id in terminal_tool._last_activity
    finally:
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop(task_id, None)
            terminal_tool._last_activity.pop(task_id, None)
