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


def test_parsed_container_config_reaches_docker_constructor(monkeypatch):
    """Docker creation must preserve parsed security and lifecycle settings."""
    env_values = {
        "TERMINAL_ENV": "docker",
        "TERMINAL_CWD": "/root",
        "TERMINAL_DOCKER_FORWARD_ENV": '["EXAMPLE_FORWARD"]',
        "TERMINAL_DOCKER_ENV": '{"EXAMPLE_STATIC": "enabled"}',
        "TERMINAL_DOCKER_EXTRA_ARGS": '["--label", "example=true"]',
        "TERMINAL_DOCKER_SHM_SIZE": "2g",
        "TERMINAL_DOCKER_NETWORK": "false",
        "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES": "false",
        "TERMINAL_DOCKER_ORPHAN_REAPER": "false",
    }
    for name, value in env_values.items():
        monkeypatch.setenv(name, value)

    # Keep this test focused on the real environment-variable parser instead
    # of allowing a developer's config.yaml to overwrite the isolated values.
    monkeypatch.setattr(terminal_tool, "_ensure_terminal_env_bridged", lambda: None)

    constructor_args = {}
    reaper_configs = []

    def fake_docker_environment(**kwargs):
        constructor_args.update(kwargs)
        return object()

    monkeypatch.setattr(terminal_tool, "_DockerEnvironment", fake_docker_environment)
    monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", reaper_configs.append)
    monkeypatch.setattr(terminal_tool, "_docker_session_isolation_enabled", lambda: False)

    config = terminal_tool._get_env_config()
    container_config = terminal_tool._container_config_from_config(config)
    terminal_tool._create_environment(
        env_type="docker",
        image=config["docker_image"],
        cwd=config["cwd"],
        timeout=config["timeout"],
        container_config=container_config,
        task_id="default",
    )

    assert constructor_args["forward_env"] == ["EXAMPLE_FORWARD"]
    assert constructor_args["env"] == {"EXAMPLE_STATIC": "enabled"}
    assert constructor_args["extra_args"] == ["--label", "example=true"]
    assert constructor_args["shm_size"] == "2g"
    assert constructor_args["network"] is False
    assert constructor_args["persist_across_processes"] is False
    assert reaper_configs == [container_config]
    assert reaper_configs[0]["docker_orphan_reaper"] is False


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
            # Matches the egress-aware reuse probe: with egress off the
            # format string is ID\tState\tEgressLabel and docker renders a
            # missing label as "<no value>".
            Result.stdout = "existing-container-id\trunning\t<no value>\n"
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
