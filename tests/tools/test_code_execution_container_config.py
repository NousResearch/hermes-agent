"""Tests for container configuration propagation in execute_code."""

import threading
import unittest.mock as mock

import tools.code_execution_tool as code_execution_tool
import tools.terminal_tool as terminal_tool


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
        "docker_orphan_reaper": False,
    }
    base.update(overrides)
    return base


def test_execute_code_replacement_forwards_canonical_container_config():
    """A post-cleanup execute_code creator must retain forwarded variables."""
    env_config = _make_env_config()
    captured = {}
    mock_env = mock.MagicMock()

    def fake_create_env(**kwargs):
        captured.update(kwargs)
        return mock_env

    with (
        mock.patch("tools.terminal_tool._get_env_config", return_value=env_config),
        mock.patch("tools.terminal_tool._task_env_overrides", {}),
        mock.patch("tools.terminal_tool._active_environments", {}),
        mock.patch("tools.terminal_tool._env_lock", threading.RLock()),
        mock.patch("tools.terminal_tool._creation_locks", {}),
        mock.patch("tools.terminal_tool._creation_locks_lock", threading.Lock()),
        mock.patch("tools.terminal_tool._create_environment", side_effect=fake_create_env),
        mock.patch("tools.terminal_tool._start_cleanup_thread"),
    ):
        env, env_type = code_execution_tool._get_or_create_env("execute-after-cleanup")

    assert env is mock_env
    assert env_type == "docker"
    assert captured["container_config"] == terminal_tool._container_config_from_config(env_config)
