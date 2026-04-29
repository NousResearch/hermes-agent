"""Regression test for docker_forward_env / docker_env wiring through
the execute_code → _get_or_create_env → _create_environment path.

Mirrors the test_parse_env_var.py boundary check for the terminal_tool
path. Without these keys in container_config, env vars listed in
``terminal.docker_forward_env`` (or set in ``terminal.docker_env``) are
silently dropped when ``execute_code`` provisions a Docker sandbox —
even though they reach the same ``_DockerEnvironment`` constructor
correctly via the terminal_tool code path.
"""

from unittest.mock import patch

import tools.code_execution_tool as _ce_mod
import tools.terminal_tool as _tt_mod


def _make_env_config(forward_env, docker_env=None):
    """Build a minimal env-config dict matching _get_env_config()'s shape."""
    return {
        "env_type": "docker",
        "docker_image": "python:3.11",
        "cwd": "/root",
        "timeout": 60,
        "lifetime_seconds": 300,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "docker_volumes": [],
        "docker_forward_env": forward_env,
        "docker_env": docker_env or {},
    }


def test_get_or_create_env_passes_docker_forward_env_to_container_config():
    """execute_code's container_config must include docker_forward_env."""
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    with patch.object(_tt_mod, "_get_env_config",
                      return_value=_make_env_config(["GITHUB_TOKEN", "NPM_TOKEN"])), \
         patch.object(_tt_mod, "_create_environment", side_effect=fake_create), \
         patch.object(_tt_mod, "_active_environments", {}), \
         patch.object(_tt_mod, "_task_env_overrides", {}):
        _ce_mod._get_or_create_env("task-forward-env")

    assert captured["container_config"]["docker_forward_env"] == ["GITHUB_TOKEN", "NPM_TOKEN"]


def test_get_or_create_env_passes_docker_env_to_container_config():
    """execute_code's container_config must include docker_env."""
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    with patch.object(_tt_mod, "_get_env_config",
                      return_value=_make_env_config([], docker_env={"FOO": "bar"})), \
         patch.object(_tt_mod, "_create_environment", side_effect=fake_create), \
         patch.object(_tt_mod, "_active_environments", {}), \
         patch.object(_tt_mod, "_task_env_overrides", {}):
        _ce_mod._get_or_create_env("task-docker-env")

    assert captured["container_config"]["docker_env"] == {"FOO": "bar"}


def test_get_or_create_env_defaults_when_keys_absent():
    """When config omits the keys entirely, sensible empty defaults flow through."""
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    sparse_cfg = {
        "env_type": "docker",
        "docker_image": "python:3.11",
        "cwd": "/root",
        "timeout": 60,
        "lifetime_seconds": 300,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "docker_volumes": [],
        # docker_forward_env / docker_env intentionally omitted
    }

    with patch.object(_tt_mod, "_get_env_config", return_value=sparse_cfg), \
         patch.object(_tt_mod, "_create_environment", side_effect=fake_create), \
         patch.object(_tt_mod, "_active_environments", {}), \
         patch.object(_tt_mod, "_task_env_overrides", {}):
        _ce_mod._get_or_create_env("task-defaults")

    assert captured["container_config"]["docker_forward_env"] == []
    assert captured["container_config"]["docker_env"] == {}
