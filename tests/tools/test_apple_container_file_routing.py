"""Apple Container routing coverage for file operations."""

import threading
from unittest.mock import MagicMock, patch

import tools.file_tools as file_tools


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


def test_file_factory_passes_apple_container_configuration():
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
         patch("tools.terminal_tool._create_environment", side_effect=create), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.file_tools._file_ops_cache", {}), \
         patch("tools.file_tools._file_ops_lock", threading.Lock()):
        file_tools._get_file_ops("apple-files")

    assert captured["env_type"] == "apple_container"
    assert captured["image"] == "python:3.12-slim"
    assert captured["container_config"]["apple_container_image"] == "python:3.12-slim"
    assert captured["container_config"]["apple_container_volumes"] == [
        "/host/data:/workspace/data:ro"
    ]


def test_file_ops_reuses_apple_environment_for_same_task():
    environment = MagicMock()
    creation = MagicMock()
    with patch("tools.terminal_tool._get_env_config", side_effect=_config), \
         patch("tools.terminal_tool._active_environments", {"default": environment}), \
         patch("tools.terminal_tool._last_activity", {"default": 0}), \
         patch("tools.terminal_tool._create_environment", creation), \
         patch("tools.file_tools._file_ops_cache", {}), \
         patch("tools.file_tools._file_ops_lock", threading.Lock()):
        first = file_tools._get_file_ops("apple-files")
        second = file_tools._get_file_ops("apple-files")

    assert first is second
    assert first.env is environment
    creation.assert_not_called()
