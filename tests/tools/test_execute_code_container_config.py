"""Tests for container_config parity across sandbox entry points.

Regression tests for the sibling-dict drift where execute_code silently
dropped forwarded secrets from the environment shared by terminal and file
tools.  Each caller must pass the canonical builder output so a future config
addition cannot recreate the same divergence.
"""

import threading
from unittest.mock import MagicMock, patch

import tools.code_execution_tool as code_execution_tool
import tools.file_tools as file_tools
import tools.terminal_tool as terminal_tool


NON_DEFAULT_CONTAINER_SETTINGS = {
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
    }
    base.update(NON_DEFAULT_CONTAINER_SETTINGS)
    base.update(overrides)
    return base


def _capture_container_config(run_entry_point):
    captured = {}
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "", "returncode": 0}

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


class TestContainerConfigParity:
    def test_builder_preserves_every_container_setting(self):
        config = _make_env_config()
        built = terminal_tool._build_container_config(config)
        for key, value in NON_DEFAULT_CONTAINER_SETTINGS.items():
            assert built[key] == value

    def test_terminal_uses_shared_builder(self):
        config = _make_env_config()
        cc = _capture_container_config(
            lambda: terminal_tool.terminal_tool(
                command="true", task_id="t-terminal", force=True,
            )
        )
        assert cc == terminal_tool._build_container_config(config)

    def test_execute_code_uses_shared_builder(self):
        config = _make_env_config()
        cc = _capture_container_config(
            lambda: code_execution_tool._get_or_create_env("t-exec"))
        assert cc == terminal_tool._build_container_config(config)
        assert cc["docker_forward_env"] == ["MY_SECRET", "API_KEY"]
        assert cc["docker_env"] == {"KEY_PATH": "/run/secrets/key.pem"}
        assert cc["docker_extra_args"] == ["--cap-drop=ALL"]

    def test_file_tools_use_shared_builder(self):
        config = _make_env_config()
        cc = _capture_container_config(
            lambda: file_tools._get_file_ops("t-file"))
        assert cc == terminal_tool._build_container_config(config)


def test_execute_code_env_config_reaches_docker_factory(monkeypatch):
    """Exercise the real env parser, builder, and environment factory chain."""
    captured = {}

    class _FakeDockerEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CWD", "/workspace")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "python:3.11")
    monkeypatch.setenv(
        "TERMINAL_DOCKER_FORWARD_ENV", '["FORWARDED_SECRET", "API_TOKEN"]'
    )
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", '{"STATIC_KEY": "static-value"}')
    monkeypatch.setenv("TERMINAL_DOCKER_EXTRA_ARGS", '["--cap-drop=ALL"]')
    monkeypatch.setenv("TERMINAL_DOCKER_NETWORK", "false")
    monkeypatch.setattr(terminal_tool, "_DockerEnvironment", _FakeDockerEnvironment)
    monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda _config: None)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    _, env_type = code_execution_tool._get_or_create_env("t-exec-e2e")

    assert env_type == "docker"
    assert captured["forward_env"] == ["FORWARDED_SECRET", "API_TOKEN"]
    assert captured["env"] == {"STATIC_KEY": "static-value"}
    assert captured["extra_args"] == ["--cap-drop=ALL"]
    assert captured["network"] is False
