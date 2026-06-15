"""Tests for docker container_config key propagation in code_execution_tool.

Regression coverage for issue #32848 item 6: TERMINAL_DOCKER_ENV /
TERMINAL_DOCKER_EXTRA_ARGS (and other docker_* knobs) configured at the
terminal layer were silently dropped on the path that builds the sandbox
for execute_code, so file-system writes from inside the sandbox could not
see env vars the user had configured globally.
"""

import threading
from unittest.mock import MagicMock, patch

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
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
        "docker_run_as_host_user": False,
        # Keys previously dropped from code_execution_tool.container_config
        # and now wired through after the parity fix for #32848 item 6.
        "modal_mode": "auto",
        "docker_env": {"FOO": "bar"},
        "docker_extra_args": ["--gpus", "all"],
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
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._env_lock", threading.Lock()), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
             patch("tools.terminal_tool._create_environment",
                   side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._resolve_container_task_id",
                   side_effect=lambda x: x):
            code_execution_tool._get_or_create_env(task_id)

        return captured.get("container_config", {})

    def test_docker_env_passed(self):
        cc = self._run(
            _make_env_config(docker_env={"FOO": "bar", "BAZ": "qux"}), "c1"
        )
        assert cc.get("docker_env") == {"FOO": "bar", "BAZ": "qux"}

    def test_docker_extra_args_passed(self):
        cc = self._run(
            _make_env_config(docker_extra_args=["--gpus", "all"]), "c2"
        )
        assert cc.get("docker_extra_args") == ["--gpus", "all"]

    def test_docker_forward_env_passed(self):
        cc = self._run(
            _make_env_config(docker_forward_env=["MY_SECRET"]), "c3"
        )
        assert cc.get("docker_forward_env") == ["MY_SECRET"]

    def test_docker_volumes_passed(self):
        cc = self._run(
            _make_env_config(docker_volumes=["/host:/container:ro"]), "c4"
        )
        assert cc.get("docker_volumes") == ["/host:/container:ro"]

    def test_docker_mount_cwd_to_workspace_passed(self):
        cc = self._run(
            _make_env_config(docker_mount_cwd_to_workspace=True), "c5"
        )
        assert cc.get("docker_mount_cwd_to_workspace") is True

    def test_modal_mode_passed(self):
        cc = self._run(_make_env_config(modal_mode="sandbox"), "c6")
        assert cc.get("modal_mode") == "sandbox"

    def test_docker_persist_across_processes_passed(self):
        cc = self._run(
            _make_env_config(docker_persist_across_processes=False), "c7"
        )
        assert cc.get("docker_persist_across_processes") is False

    def test_docker_orphan_reaper_passed(self):
        cc = self._run(_make_env_config(docker_orphan_reaper=False), "c8")
        assert cc.get("docker_orphan_reaper") is False

    def test_defaults_when_keys_absent(self):
        """When env_config omits the docker knobs, their documented defaults
        propagate to container_config (no KeyErrors, no surprise values)."""
        cfg = _make_env_config()
        for key in (
            "docker_env",
            "docker_extra_args",
            "modal_mode",
            "docker_persist_across_processes",
            "docker_orphan_reaper",
        ):
            del cfg[key]
        cc = self._run(cfg, "c_defaults")
        assert cc.get("docker_env") == {}
        assert cc.get("docker_extra_args") == []
        assert cc.get("modal_mode") == "auto"
        assert cc.get("docker_persist_across_processes") is True
        assert cc.get("docker_orphan_reaper") is True

    def test_parity_with_terminal_tool_keyset(self):
        """Regression guard: code_execution_tool container_config must include
        every docker-related key that terminal_tool's container_config includes.

        If terminal_tool grows a new docker_ knob and this list isn't updated
        in lockstep, execute_code will silently diverge from terminal — exactly
        the class of bug fixed by issue #32848 item 6.
        """
        cc = self._run(_make_env_config(), "c_parity")
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
            f"code_execution_tool.container_config missing keys present in "
            f"terminal_tool: {sorted(missing)}"
        )
