"""Tests for docker container_config key propagation in code_execution_tool."""

from unittest.mock import MagicMock, patch
import threading

import tools.code_execution_tool as code_execution_tool


def _make_env_config(**overrides):
    base = {
        "env_type": "docker",
        "docker_image": "test-image:latest",
        "singularity_image": "docker://test",
        "modal_image": "test",
        "daytona_image": "test",
        "cwd": "/workspace",
        "host_cwd": None,
        "timeout": 180,
        "container_cpu": 2,
        "container_memory": 4096,
        "container_disk": 20480,
        "container_persistent": False,
        "docker_volumes": [],
        "docker_network_mode": "",
    }
    base.update(overrides)
    return base


class TestCodeExecutionContainerConfig:
    def _run(self, env_config, task_id):
        captured = {}
        mock_env = MagicMock()

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.code_execution_tool.time.time", return_value=123.0):
            code_execution_tool._get_or_create_env(task_id)

        return captured.get("container_config", {})

    def test_docker_network_mode_passed(self):
        """docker_network_mode is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_network_mode="none"), "t1")
        assert cc.get("docker_network_mode") == "none"
