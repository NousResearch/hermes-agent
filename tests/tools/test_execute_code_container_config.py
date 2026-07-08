"""Tests for docker container_config key propagation in code_execution_tool.

Regression tests for the sibling-dict drift where execute_code's hand-copied
container_config lacked docker_forward_env/docker_env/docker_extra_args (and
the other terminal_tool keys). Because terminal, file, and execute_code tools
share one environment slot per task, an environment (re)created by
execute_code silently dropped every forwarded secret for all of them.
"""

import threading
from unittest.mock import MagicMock, patch

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
        # Non-default value for every container_config key so a dropped key
        # shows up as a mismatch instead of hiding behind the default.
        "container_cpu": 3,
        "container_memory": 1234,
        "container_disk": 4321,
        "container_persistent": False,
        "modal_mode": "sandbox",
        "docker_volumes": ["/host/secrets:/run/secrets:ro"],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
        "docker_env": {"KEY_PATH": "/run/secrets/key.pem"},
        "docker_run_as_host_user": True,
        "docker_extra_args": ["--cap-drop=ALL"],
        "docker_network": False,
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": False,
    }
    base.update(overrides)
    return base


# Every key terminal_tool puts in container_config; the sibling entry points
# must forward all of them or the shared environment loses settings depending
# on which tool happened to create it.
CONTAINER_CONFIG_KEYS = (
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
    "docker_network",
    "docker_persist_across_processes",
    "docker_orphan_reaper",
)


def _capture_container_config(run_entry_point):
    captured = {}
    mock_env = MagicMock()

    def fake_create_env(**kwargs):
        captured.update(kwargs)
        return mock_env

    with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
         patch("tools.terminal_tool._task_env_overrides", {}), \
         patch("tools.terminal_tool._active_environments", {}), \
         patch("tools.terminal_tool._last_activity", {}), \
         patch("tools.terminal_tool._creation_locks", {}), \
         patch("tools.terminal_tool._creation_locks_lock", threading.Lock()), \
         patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_disk_usage_warning"), \
         patch("tools.file_tools._file_ops_cache", {}), \
         patch("tools.file_tools._file_ops_lock", threading.Lock()):
        run_entry_point()

    return captured["container_config"]


class TestExecuteCodeContainerConfig:
    def test_every_key_taken_from_config(self):
        """No config key falls back to its default when explicitly set."""
        config = _make_env_config()
        cc = _capture_container_config(
            lambda: code_execution_tool._get_or_create_env("t-exec"))
        for key in CONTAINER_CONFIG_KEYS:
            assert key in cc, f"{key} missing from execute_code container_config"
            assert cc[key] == config[key], f"{key} not taken from config"

    def test_forwarded_secrets_survive_execute_code_creation(self):
        """The keys whose loss broke secret forwarding are passed through."""
        cc = _capture_container_config(
            lambda: code_execution_tool._get_or_create_env("t-exec2"))
        assert cc["docker_forward_env"] == ["MY_SECRET", "API_KEY"]
        assert cc["docker_env"] == {"KEY_PATH": "/run/secrets/key.pem"}
        assert cc["docker_extra_args"] == ["--cap-drop=ALL"]

    def test_execute_code_and_file_tools_build_identical_config(self):
        """The shared environment must not depend on which tool created it."""
        cc_exec = _capture_container_config(
            lambda: code_execution_tool._get_or_create_env("t-parity"))
        cc_file = _capture_container_config(
            lambda: file_tools._get_file_ops("t-parity"))
        assert cc_exec == cc_file
