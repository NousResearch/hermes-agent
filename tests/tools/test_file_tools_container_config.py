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
        # Keys that previously did not reach file_tools' container_config — must
        # be wired through after the parity fix for issue #32848 item 6.
        "modal_mode": "auto",
        "docker_env": {"FOO": "bar", "BAZ": "qux"},
        "docker_extra_args": ["--gpus", "all"],
        "docker_persist_across_processes": True,
        "docker_orphan_reaper": True,
    }
    base.update(overrides)
    return base


class TestFileToolsContainerConfig:
    def _run(self, env_config, task_id):
        captured = {}
        mock_env = MagicMock()

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config),              patch("tools.terminal_tool._task_env_overrides", {}),              patch("tools.terminal_tool._active_environments", {}),              patch("tools.terminal_tool._creation_locks", {}),              patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()),              patch("tools.terminal_tool._create_environment", side_effect=fake_create_env),              patch("tools.terminal_tool._start_cleanup_thread"),              patch("tools.terminal_tool._check_disk_usage_warning"),              patch("tools.file_tools._file_ops_cache", {}),              patch("tools.file_tools._file_ops_lock", __import__("threading").Lock()):
            file_tools._get_file_ops(task_id)

        return captured.get("container_config", {})

    def test_docker_mount_cwd_to_workspace_passed(self):
        """docker_mount_cwd_to_workspace is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_mount_cwd_to_workspace=True), "t1")
        assert cc.get("docker_mount_cwd_to_workspace") is True

    def test_docker_forward_env_passed(self):
        """docker_forward_env is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_forward_env=["MY_SECRET"]), "t2")
        assert cc.get("docker_forward_env") == ["MY_SECRET"]

    def test_docker_mount_cwd_defaults_to_false(self):
        """docker_mount_cwd_to_workspace defaults to False when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_mount_cwd_to_workspace"]
        cc = self._run(cfg, "t3")
        assert cc.get("docker_mount_cwd_to_workspace") is False

    def test_docker_forward_env_defaults_to_empty_list(self):
        """docker_forward_env defaults to [] when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_forward_env"]
        cc = self._run(cfg, "t4")
        assert cc.get("docker_forward_env") == []

    # --- Issue #32848 item 6: parity with terminal_tool.py container_config ---
    # Before this fix, the following keys were silently dropped from file_tools'
    # container_config, so users who configured them (e.g. via TERMINAL_DOCKER_ENV)
    # saw them apply to terminal commands but be silently ignored for file ops.

    def test_docker_env_passed(self):
        """docker_env (custom env vars) is forwarded to container_config."""
        cc = self._run(
            _make_env_config(docker_env={"FOO": "bar", "BAZ": "qux"}), "t5"
        )
        assert cc.get("docker_env") == {"FOO": "bar", "BAZ": "qux"}

    def test_docker_env_defaults_to_empty_dict(self):
        cfg = _make_env_config()
        del cfg["docker_env"]
        cc = self._run(cfg, "t6")
        assert cc.get("docker_env") == {}

    def test_docker_extra_args_passed(self):
        """docker_extra_args (raw docker-run args) is forwarded."""
        cc = self._run(
            _make_env_config(docker_extra_args=["--gpus", "all", "--shm-size=2g"]),
            "t7",
        )
        assert cc.get("docker_extra_args") == ["--gpus", "all", "--shm-size=2g"]

    def test_docker_extra_args_defaults_to_empty_list(self):
        cfg = _make_env_config()
        del cfg["docker_extra_args"]
        cc = self._run(cfg, "t8")
        assert cc.get("docker_extra_args") == []

    def test_modal_mode_passed(self):
        """modal_mode is forwarded so file ops in modal env match terminal env."""
        cc = self._run(_make_env_config(modal_mode="sandbox"), "t9")
        assert cc.get("modal_mode") == "sandbox"

    def test_modal_mode_defaults_to_auto(self):
        cfg = _make_env_config()
        del cfg["modal_mode"]
        cc = self._run(cfg, "t10")
        assert cc.get("modal_mode") == "auto"

    def test_docker_persist_across_processes_passed(self):
        cc = self._run(
            _make_env_config(docker_persist_across_processes=False), "t11"
        )
        assert cc.get("docker_persist_across_processes") is False

    def test_docker_persist_across_processes_defaults_to_true(self):
        cfg = _make_env_config()
        del cfg["docker_persist_across_processes"]
        cc = self._run(cfg, "t12")
        assert cc.get("docker_persist_across_processes") is True

    def test_docker_orphan_reaper_passed(self):
        cc = self._run(_make_env_config(docker_orphan_reaper=False), "t13")
        assert cc.get("docker_orphan_reaper") is False

    def test_docker_orphan_reaper_defaults_to_true(self):
        cfg = _make_env_config()
        del cfg["docker_orphan_reaper"]
        cc = self._run(cfg, "t14")
        assert cc.get("docker_orphan_reaper") is True

    def test_parity_with_terminal_tool_keyset(self):
        """Regression guard: file_tools container_config must include every
        docker-related key that terminal_tool's container_config includes.

        If terminal_tool grows a new docker_ knob and this list isn't updated
        in lockstep, file ops will silently diverge from terminal ops — exactly
        the class of bug fixed by issue #32848 item 6.
        """
        cc = self._run(_make_env_config(), "t_parity")
        required_docker_keys = {
            "container_cpu",
            "container_memory",
            "container_disk",
            "container_persistent",
            "modal_mode",
            "docker_volumes",
            "docker_mount_cwd_to_workspace",
            "docker_forward_env",
            "docker_env",
            "docker_run_as_host_user",
            "docker_extra_args",
            "docker_persist_across_processes",
            "docker_orphan_reaper",
        }
        missing = required_docker_keys - set(cc.keys())
        assert not missing, (
            f"file_tools.container_config missing keys present in "
            f"terminal_tool: {sorted(missing)}"
        )
