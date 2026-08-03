"""Apple Container routing coverage for execute_code."""

import threading
from unittest.mock import MagicMock, patch

import tools.code_execution_tool as code_execution


def _config() -> dict:
    return {
        "env_type": "apple_container",
        "apple_container_image": "python:3.12-slim",
        "apple_container_volumes": ["/host/data:/workspace/data:ro"],
        "cwd": "/workspace",
        "host_cwd": None,
        "timeout": 42,
        "container_cpu": 4,
        "container_memory": 6144,
        "container_disk": 51200,
        "container_persistent": True,
    }


def _capture(task_overrides=None):
    captured = {}
    environment = MagicMock()

    def create(**kwargs):
        captured.update(kwargs)
        return environment

    with patch("tools.terminal_tool._get_env_config", side_effect=_config), \
         patch("tools.terminal_tool._active_environments", {}), \
         patch("tools.terminal_tool._last_activity", {}), \
         patch("tools.terminal_tool._creation_locks", {}), \
         patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
         patch("tools.terminal_tool._task_env_overrides", task_overrides or {}), \
         patch("tools.terminal_tool._create_environment", side_effect=create), \
         patch("tools.terminal_tool._start_cleanup_thread"):
        code_execution._get_or_create_env("apple-execute")
    return captured


def test_execute_code_passes_apple_container_configuration():
    captured = _capture()
    assert captured["env_type"] == "apple_container"
    assert captured["image"] == "python:3.12-slim"
    assert captured["container_config"]["container_cpu"] == 4
    assert captured["container_config"]["container_memory"] == 6144
    assert captured["container_config"]["container_persistent"] is True
    assert captured["container_config"]["apple_container_volumes"] == [
        "/host/data:/workspace/data:ro"
    ]


def test_task_override_keeps_apple_volumes():
    captured = _capture(
        {"default": {"apple_container_image": "python:3.13-slim", "cwd": "/workspace/job"}}
    )
    assert captured["image"] == "python:3.13-slim"
    assert captured["cwd"] == "/workspace/job"
    assert captured["container_config"]["apple_container_volumes"] == [
        "/host/data:/workspace/data:ro"
    ]
