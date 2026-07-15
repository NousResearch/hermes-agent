"""Tests for docker container_config key propagation in file_tools."""

import threading
from unittest.mock import patch, MagicMock
import tools.code_execution_tool as code_execution_tool
import tools.file_tools as file_tools


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
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
        "docker_env": {"A": "B"},
        "docker_extra_args": ["--shm-size=1g"],
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": False,
        "tenki_api_endpoint": "https://api.tenki.test",
        "tenki_workspace_id": "ws-123",
        "tenki_project_id": "prj-456",
        "tenki_name_prefix": "agent",
        "tenki_allow_inbound": True,
        "tenki_allow_outbound": False,
        "tenki_max_duration": 7200,
        "tenki_idle_timeout": 600,
        "tenki_pause_retention": 3600,
        "tenki_sync_hermes_home": True,
        "tenki_forward_env": ["GITHUB_TOKEN"],
    }
    base.update(overrides)
    return base


class TestFileToolsContainerConfig:
    def _run(self, env_config, task_id, task_env_overrides=None):
        captured = {}
        mock_env = MagicMock()

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", task_env_overrides or {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_disk_usage_warning"), \
             patch("tools.file_tools._file_ops_cache", {}), \
             patch("tools.file_tools._file_ops_lock", __import__("threading").Lock()):
            file_tools._get_file_ops(task_id)

        return captured

    def test_docker_mount_cwd_to_workspace_passed(self):
        """docker_mount_cwd_to_workspace is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_mount_cwd_to_workspace=True), "t1").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is True

    def test_docker_forward_env_passed(self):
        """docker_forward_env is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_forward_env=["MY_SECRET"]), "t2").get("container_config", {})
        assert cc.get("docker_forward_env") == ["MY_SECRET"]

    def test_docker_mount_cwd_defaults_to_false(self):
        """docker_mount_cwd_to_workspace defaults to False when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_mount_cwd_to_workspace"]
        cc = self._run(cfg, "t3").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is False

    def test_docker_forward_env_defaults_to_empty_list(self):
        """docker_forward_env defaults to [] when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_forward_env"]
        cc = self._run(cfg, "t4").get("container_config", {})
        assert cc.get("docker_forward_env") == []

    def test_shared_container_config_fields_are_forwarded(self):
        """File tools use the same container-config builder as terminal execution."""
        cc = self._run(_make_env_config(), "t5").get("container_config", {})

        assert cc.get("docker_env") == {"A": "B"}
        assert cc.get("docker_extra_args") == ["--shm-size=1g"]
        assert cc.get("docker_persist_across_processes") is False
        assert cc.get("docker_orphan_reaper") is False
        assert cc.get("tenki_name_prefix") == "agent"
        assert cc.get("tenki_api_endpoint") == "https://api.tenki.test"
        assert cc.get("tenki_workspace_id") == "ws-123"
        assert cc.get("tenki_project_id") == "prj-456"
        assert cc.get("tenki_allow_inbound") is True
        assert cc.get("tenki_allow_outbound") is False
        assert cc.get("tenki_max_duration") == 7200
        assert cc.get("tenki_idle_timeout") == 600
        assert cc.get("tenki_pause_retention") == 3600
        assert cc.get("tenki_sync_hermes_home") is True
        assert cc.get("tenki_forward_env") == ["GITHUB_TOKEN"]

    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"


class TestExecuteCodeContainerConfig:
    def test_execute_code_uses_shared_container_config_for_tenki(self):
        captured = {}
        mock_env = MagicMock()
        env_config = _make_env_config(
            env_type="tenki",
            tenki_image="tenki-image",
            cwd="/home/tenki",
        )

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._env_lock", threading.Lock()), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            env, env_type = code_execution_tool._get_or_create_env("exec-tenki")

        assert env is mock_env
        assert env_type == "tenki"
        assert captured["env_type"] == "tenki"
        assert captured["image"] == "tenki-image"
        assert captured["cwd"] == "/home/tenki"
        assert captured["task_id"] == "default"
        cc = captured["container_config"]
        assert cc["container_persistent"] is False
        assert cc["tenki_api_endpoint"] == "https://api.tenki.test"
        assert cc["tenki_workspace_id"] == "ws-123"
        assert cc["tenki_project_id"] == "prj-456"
        assert cc["tenki_name_prefix"] == "agent"
        assert cc["tenki_allow_inbound"] is True
        assert cc["tenki_allow_outbound"] is False
        assert cc["tenki_max_duration"] == 7200
        assert cc["tenki_idle_timeout"] == 600
        assert cc["tenki_pause_retention"] == 3600
        assert cc["tenki_sync_hermes_home"] is True
        assert cc["tenki_forward_env"] == ["GITHUB_TOKEN"]
