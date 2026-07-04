"""Tests for docker container_config key propagation in code_execution_tool.

Mirrors tests/tools/test_file_tools_container_config.py — the execute_code
sandbox must forward the same docker_env / docker_forward_env /
docker_extra_args keys that terminal_tool already forwards, so containers it
spawns get host env vars and extra docker flags (e.g. AppArmor overrides).
"""

import threading
from unittest.mock import patch, MagicMock

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
        "vercel_runtime": "",
        "docker_volumes": [],
        "docker_run_as_host_user": False,
        "docker_env": {"FOO": "bar"},
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
        "docker_extra_args": ["--security-opt", "apparmor=unconfined"],
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
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch("tools.terminal_tool._resolve_container_task_id", lambda t: t), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            code_execution_tool._get_or_create_env(task_id)

        return captured.get("container_config", {})

    def test_docker_env_passed(self):
        """docker_env is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_env={"FOO": "bar"}), "t1")
        assert cc.get("docker_env") == {"FOO": "bar"}

    def test_docker_forward_env_passed(self):
        """docker_forward_env is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_forward_env=["MY_SECRET"]), "t2")
        assert cc.get("docker_forward_env") == ["MY_SECRET"]

    def test_docker_extra_args_passed(self):
        """docker_extra_args is forwarded to container_config."""
        cc = self._run(
            _make_env_config(docker_extra_args=["--security-opt", "apparmor=unconfined"]),
            "t3",
        )
        assert cc.get("docker_extra_args") == ["--security-opt", "apparmor=unconfined"]

    def test_docker_env_defaults_to_empty_dict(self):
        """docker_env defaults to {} when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_env"]
        cc = self._run(cfg, "t4")
        assert cc.get("docker_env") == {}

    def test_docker_forward_env_defaults_to_empty_list(self):
        """docker_forward_env defaults to [] when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_forward_env"]
        cc = self._run(cfg, "t5")
        assert cc.get("docker_forward_env") == []

    def test_docker_extra_args_defaults_to_empty_list(self):
        """docker_extra_args defaults to [] when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_extra_args"]
        cc = self._run(cfg, "t6")
        assert cc.get("docker_extra_args") == []
