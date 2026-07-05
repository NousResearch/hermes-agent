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


def test_sibling_container_config_sites_carry_docker_network():
    """Every container_config dict that carries docker_run_as_host_user must
    also carry docker_network — otherwise that code path silently falls back
    to networked containers while the terminal path honors the lockdown
    (the probe/exec asymmetry reported on issue #46358).
    """
    import ast
    import inspect

    import tools.code_execution_tool as code_execution_tool
    import tools.file_tools as file_tools

    for module in (terminal_tool, file_tools, code_execution_tool):
        tree = ast.parse(inspect.getsource(module))
        sites = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            if "docker_run_as_host_user" in keys:
                sites += 1
                assert "docker_network" in keys, (
                    f"{module.__name__} builds a container_config with "
                    f"docker_run_as_host_user but without docker_network "
                    f"(line {node.lineno})"
                )
        assert sites >= 1, f"expected at least one container_config site in {module.__name__}"


def _reuse_guard_harness(monkeypatch, *, existing_mode: str, network: bool):
    """Drive DockerEnvironment through the cross-process reuse path with a
    fake existing container whose NetworkMode is *existing_mode*.

    Returns the list of docker commands issued.
    """
    commands = []

    def fake_run(cmd, *args, **kwargs):
        commands.append(cmd)

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        if len(cmd) > 1 and cmd[1] == "ps":
            Result.stdout = "existing-container-id\trunning\n"
        elif len(cmd) > 1 and cmd[1] == "inspect":
            Result.stdout = f"{existing_mode}\n"
        elif len(cmd) > 1 and cmd[1] == "run":
            Result.stdout = "fresh-container-id\n"
        return Result()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_env.DockerEnvironment, "_storage_opt_supported", lambda self: False)

    docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/workspace",
        timeout=60,
        task_id="reuse-guard-test",
        network=network,
        persist_across_processes=True,
    )
    return commands


def test_reuse_rejects_networked_container_when_lockdown_requested(monkeypatch):
    commands = _reuse_guard_harness(monkeypatch, existing_mode="bridge", network=False)

    assert any(cmd[1:3] == ["rm", "-f"] for cmd in commands), (
        "bridge-networked container must be removed when docker_network=false"
    )
    run_cmd = next(cmd for cmd in commands if len(cmd) > 2 and cmd[1:3] == ["run", "-d"])
    assert "--network=none" in run_cmd


def test_reuse_keeps_airgapped_container_when_lockdown_requested(monkeypatch):
    commands = _reuse_guard_harness(monkeypatch, existing_mode="none", network=False)

    assert not any(cmd[1] == "rm" for cmd in commands)
    assert not any(cmd[1] == "run" for cmd in commands), "matching container must be reused"


def test_reuse_skips_inspect_when_network_enabled(monkeypatch):
    commands = _reuse_guard_harness(monkeypatch, existing_mode="none", network=True)

    # Default-network config never churns containers, even air-gapped ones
    # (operators may have created them via docker_extra_args).
    assert not any(cmd[1] == "inspect" for cmd in commands)
    assert not any(cmd[1] == "rm" for cmd in commands)
    assert not any(cmd[1] == "run" for cmd in commands)
