"""Regression tests for the Docker terminal network toggle.

Ported from NanoClaw PR #2713's opt-in egress lockdown idea. Hermes already
has DockerEnvironment(network=False), but the terminal config path did not
expose it, so operators could not request networkless Docker execution from
config.yaml.
"""

import tools.terminal_tool as terminal_tool
from tools.environments import docker as docker_env


def test_terminal_env_config_reads_docker_network_toggle(monkeypatch):
    monkeypatch.setenv("TERMINAL_DOCKER_NETWORK", "false")

    config = terminal_tool._get_env_config()

    assert config["docker_network"] is False


def test_create_environment_passes_docker_network_toggle(monkeypatch):
    captured = {}
    sentinel = object()

    def _fake_docker_environment(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(terminal_tool, "_DockerEnvironment", _fake_docker_environment)

    env = terminal_tool._create_environment(
        env_type="docker",
        image="python:3.11",
        cwd="/workspace",
        timeout=60,
        container_config={"docker_network": False},
    )

    assert env is sentinel
    assert captured["network"] is False


def test_docker_environment_adds_network_none_when_disabled(monkeypatch):
    commands = []

    def fake_run(cmd, *args, **kwargs):
        commands.append(cmd)

        class Result:
            returncode = 0
            stdout = "fake-container-id\n" if len(cmd) > 1 and cmd[1] == "run" else ""
            stderr = ""

        return Result()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_env.DockerEnvironment, "_storage_opt_supported", lambda self: False)

    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/workspace",
        timeout=60,
        task_id="network-none-test",
        network=False,
    )

    run_cmd = next(cmd for cmd in commands if len(cmd) > 2 and cmd[1:3] == ["run", "-d"])
    assert "--network=none" in run_cmd
    env.cleanup()


def test_docker_network_config_is_bridged_everywhere():
    from tests.tools.test_terminal_config_env_sync import (
        _cli_env_map_keys,
        _gateway_env_map_keys,
        _save_config_env_sync_keys,
        _terminal_tool_env_var_names,
    )

    assert "docker_network" in _cli_env_map_keys()
    assert "docker_network" in _gateway_env_map_keys()
    assert "docker_network" in _save_config_env_sync_keys()
    assert "TERMINAL_DOCKER_NETWORK" in _terminal_tool_env_var_names()
