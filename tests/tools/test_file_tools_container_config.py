"""Tests for docker container_config key propagation in file_tools."""

from unittest.mock import patch, MagicMock
import tools.file_tools as file_tools


def _make_env_config(**overrides):
    base = {
        "env_type": "docker",
        "docker_image": "test-image:latest",
        "singularity_image": "docker://test",
        "modal_image": "test",
        "daytona_image": "test",
        "apple_container_image": "test",
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
        "apple_container_volumes": [],
        "apple_container_mount_cwd_to_workspace": False,
        "apple_container_forward_env": [],
        "apple_container_env": {},
        "apple_container_extra_args": [],
        "apple_container_run_as_host_user": False,
        "apple_container_persist_across_processes": False,
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

    def test_apple_container_config_passed(self):
        """Apple container backend-specific keys are forwarded to container_config."""
        cfg = _make_env_config(
            env_type="apple_container",
            apple_container_image="apple:test",
            apple_container_mount_cwd_to_workspace=True,
            apple_container_forward_env=["GITHUB_TOKEN"],
            apple_container_volumes=["/tmp/data:/data:ro"],
            apple_container_env={"DEBUG": "1"},
            apple_container_extra_args=["--rosetta"],
            apple_container_persist_across_processes=True,
        )

        captured = self._run(cfg, "apple-task")
        cc = captured.get("container_config", {})

        assert captured["image"] == "apple:test"
        assert cc.get("apple_container_mount_cwd_to_workspace") is True
        assert cc.get("apple_container_forward_env") == ["GITHUB_TOKEN"]
        assert cc.get("apple_container_volumes") == ["/tmp/data:/data:ro"]
        assert cc.get("apple_container_env") == {"DEBUG": "1"}
        assert cc.get("apple_container_extra_args") == ["--rosetta"]
        assert cc.get("apple_container_persist_across_processes") is True

    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"
