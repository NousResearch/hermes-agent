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
        "docker_env": {"APP_ENV": "test"},
        "docker_extra_args": ["--cap-drop=ALL"],
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": False,
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

    def test_docker_policy_fields_passed(self):
        """Docker policy values are forwarded to container_config unchanged."""
        cc = self._run(_make_env_config(), "t5").get("container_config", {})

        assert cc["docker_env"] == {"APP_ENV": "test"}
        assert cc["docker_extra_args"] == ["--cap-drop=ALL"]
        assert cc["docker_persist_across_processes"] is False
        assert cc["docker_orphan_reaper"] is False

    def test_docker_policy_fields_use_terminal_defaults(self):
        """Docker policy fields preserve terminal_tool defaults when absent."""
        cfg = _make_env_config()
        for key in (
            "docker_env",
            "docker_extra_args",
            "docker_persist_across_processes",
            "docker_orphan_reaper",
        ):
            del cfg[key]

        cc = self._run(cfg, "t6").get("container_config", {})

        assert cc["docker_env"] == {}
        assert cc["docker_extra_args"] == []
        assert cc["docker_persist_across_processes"] is True
        assert cc["docker_orphan_reaper"] is True

    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"
