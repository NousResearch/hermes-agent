"""Tests for container configuration propagation in file tools."""

import threading
import unittest.mock as mock

import tools.file_tools as file_tools
import tools.terminal_tool_backends as terminal_tool_backends


def _make_env_config(**overrides):
    """Use non-default values so dropped settings cannot hide behind defaults."""
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
        "modal_mode": "managed",
        "vercel_runtime": "python3.13",
        "docker_volumes": ["/example-host:/example-container:ro"],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["EXAMPLE_FORWARD"],
        "docker_env": {"EXAMPLE_STATIC": "enabled"},
        "docker_run_as_host_user": True,
        "docker_extra_args": ["--label", "example=true"],
        "docker_shm_size": "2g",
        "docker_network": False,
        "docker_persist_across_processes": False,
        "docker_shared_container_key": "example-shared",
        "docker_orphan_reaper": False,
        "docker_snap_compat": True,
    }
    base.update(overrides)
    return base


class TestFileToolsContainerConfig:
    """Check settings on the real file-tool path with external creation isolated."""

    def test_file_first_creator_forwards_canonical_container_config(self):
        """File tools creating the shared sandbox must preserve every setting."""
        env_config = _make_env_config()

        captured = self._run(env_config, "file-first")

        # Compare the live creation path with the current canonical projection,
        # so future settings also have to survive the file-tool call boundary.
        assert captured["container_config"] == terminal_tool_backends._container_config_from_config(env_config)

    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"

    def _run(self, env_config, task_id, task_env_overrides=None):
        """Capture real file-tool creation without starting a sandbox or cleanup."""
        captured = {}
        mock_env = mock.MagicMock()

        def fake_create_env(**kwargs):
            """Capture the final factory arguments before external I/O."""
            captured.update(kwargs)
            return mock_env

        # Isolate shared task state so this call must create the environment.
        with (
            mock.patch("tools.terminal_tool._get_env_config", return_value=env_config),
            mock.patch("tools.terminal_tool._task_env_overrides", task_env_overrides or {}),
            mock.patch("tools.terminal_tool._active_environments", {}),
            mock.patch("tools.terminal_tool._last_activity", {}),
            mock.patch("tools.terminal_tool._creation_locks", {}),
            mock.patch("tools.terminal_tool._creation_locks_lock", threading.Lock()),
            # The lifecycle helper imports the backend factory at call time.
            mock.patch("tools.terminal_tool_backends._create_environment", side_effect=fake_create_env),
            mock.patch("tools.terminal_tool._start_cleanup_thread"),
            mock.patch("tools.terminal_tool._check_disk_usage_warning"),
            mock.patch("tools.file_tools._file_ops_cache", {}),
            mock.patch("tools.file_tools._file_ops_lock", threading.Lock()),
        ):
            file_tools._get_file_ops(task_id)

        return captured
