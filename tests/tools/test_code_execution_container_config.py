"""Tests for docker container_config key propagation in code_execution_tool.

Mirrors test_file_tools_container_config.py: _get_or_create_env() builds its
own container_config dict independently of terminal_tool._create_environment's
caller, so it must be pinned separately to catch the same class of drift
(container comes up missing docker_env/docker_extra_args/etc. depending on
which tool wins the race to create the shared per-task container).
"""

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
        "docker_volumes": [],
        "docker_run_as_host_user": False,
        "docker_network": True,
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
        "docker_env": {},
        "docker_extra_args": [],
        "modal_mode": "auto",
        "docker_persist_across_processes": True,
        "docker_orphan_reaper": True,
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
             patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._env_lock", __import__("threading").Lock()), \
             patch("tools.terminal_tool._resolve_container_task_id", side_effect=lambda tid: tid):
            code_execution_tool._get_or_create_env(task_id)

        return captured

    def test_docker_env_passed(self):
        cc = self._run(_make_env_config(docker_env={"HTTP_PROXY": "http://proxy:8080"}), "t1").get("container_config", {})
        assert cc.get("docker_env") == {"HTTP_PROXY": "http://proxy:8080"}

    def test_docker_env_defaults_to_empty_dict(self):
        cfg = _make_env_config()
        del cfg["docker_env"]
        cc = self._run(cfg, "t2").get("container_config", {})
        assert cc.get("docker_env") == {}

    def test_docker_extra_args_passed(self):
        cc = self._run(_make_env_config(docker_extra_args=["--network=fetcher-net"]), "t3").get("container_config", {})
        assert cc.get("docker_extra_args") == ["--network=fetcher-net"]

    def test_docker_extra_args_defaults_to_empty_list(self):
        cfg = _make_env_config()
        del cfg["docker_extra_args"]
        cc = self._run(cfg, "t4").get("container_config", {})
        assert cc.get("docker_extra_args") == []

    def test_docker_mount_cwd_to_workspace_passed(self):
        cc = self._run(_make_env_config(docker_mount_cwd_to_workspace=True), "t5").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is True

    def test_docker_forward_env_passed(self):
        cc = self._run(_make_env_config(docker_forward_env=["MY_SECRET"]), "t6").get("container_config", {})
        assert cc.get("docker_forward_env") == ["MY_SECRET"]

    def test_docker_persist_across_processes_passed(self):
        cc = self._run(_make_env_config(docker_persist_across_processes=False), "t7").get("container_config", {})
        assert cc.get("docker_persist_across_processes") is False

    def test_docker_orphan_reaper_passed(self):
        cc = self._run(_make_env_config(docker_orphan_reaper=False), "t8").get("container_config", {})
        assert cc.get("docker_orphan_reaper") is False

    def test_modal_mode_passed(self):
        cc = self._run(_make_env_config(modal_mode="sandbox"), "t9").get("container_config", {})
        assert cc.get("modal_mode") == "sandbox"
