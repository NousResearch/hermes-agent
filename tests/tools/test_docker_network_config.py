"""Regression tests for the Docker terminal network toggle.

Ported from NanoClaw PR #2713's opt-in egress lockdown idea. Hermes already
has DockerEnvironment(network=False), but the terminal config path did not
expose it, so operators could not request networkless Docker execution from
config.yaml.
"""

import tools.terminal_tool as terminal_tool
import tools.terminal_tool_backends as terminal_tool_backends
from tools.environments import docker as docker_env


def test_terminal_env_config_reads_docker_network_toggle(monkeypatch):
    monkeypatch.setenv("TERMINAL_DOCKER_NETWORK", "false")

    config = terminal_tool._get_env_config()

    assert config["docker_network"] is False


class TestParsedContainerConfig:
    """Exercise parsing and factory dispatch up to the external constructor."""

    def test_parsed_container_config_reaches_docker_constructor(self, monkeypatch):
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

        # Keep the parser real; bypass only the config.yaml-to-env bridge so
        # the isolated values above remain the source of this test's settings.
        monkeypatch.setattr(terminal_tool, "_ensure_terminal_env_bridged", lambda: None)
        constructor_args = {}
        reaper_configs = []

        def fake_docker_environment(**kwargs):
            """Capture constructor arguments without starting Docker."""
            constructor_args.update(kwargs)
            return object()

        # Patch only external creation and cleanup; the factory and projection
        # now live in terminal_tool_backends and must execute for real.
        monkeypatch.setattr(terminal_tool_backends, "_DockerEnvironment", fake_docker_environment)
        monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", reaper_configs.append)
        monkeypatch.setattr(terminal_tool, "_docker_session_isolation_enabled", lambda: False)
        config = terminal_tool._get_env_config()
        container_config = terminal_tool_backends._container_config_from_config(config)
        terminal_tool_backends._create_environment(
            env_type="docker",
            image=config["docker_image"],
            cwd=config["cwd"],
            timeout=config["timeout"],
            container_config=container_config,
            task_id="default",
        )

        # Independent expected values catch losses within the parser or factory,
        # which caller-to-projection equality alone cannot detect.
        assert constructor_args["forward_env"] == ["EXAMPLE_FORWARD"]
        assert constructor_args["env"] == {"EXAMPLE_STATIC": "enabled"}
        assert constructor_args["extra_args"] == ["--label", "example=true"]
        assert constructor_args["shm_size"] == "2g"
        assert constructor_args["network"] is False
        assert constructor_args["persist_across_processes"] is False
        assert reaper_configs == [container_config]
        assert reaper_configs[0]["docker_orphan_reaper"] is False


def test_every_sandbox_creator_passes_the_full_container_config(monkeypatch):
    """The terminal tool, execute_code and the prompt backend-probe must hand ``_create_environment``
    the SAME container_config keys. Each used to keep a private (key, default) table and drifted:
    the probe lost ``docker_network`` (bridge-networked probe under lockdown, #46358/#76906/#87995),
    execute_code lost ``docker_extra_args``/``docker_forward_env``/``docker_env`` (#84027/#100019)."""
    import agent.prompt_builder as prompt_builder
    import tools.code_execution_tool as code_execution_tool
    import tools.terminal_tool_backends as backends

    config = {"env_type": "docker", "cwd": "/root", "timeout": 60, "docker_network": False,
              "docker_extra_args": ["--user", "1009:1009"], "docker_forward_env": ["DATABASE_URL"],
              "docker_env": {"FOO": "bar"}, "docker_image": "debian:bookworm-slim"}
    expected = backends._container_config_from_config(config)
    seen: list = []

    class _Env:
        cwd = "/root"

        def execute(self, *a, **k):
            return {"output": "", "returncode": 0}

        def cleanup(self, **k):
            pass

    def _fake_create(**kwargs):
        seen.append(kwargs["container_config"])
        return _Env()

    # Both creators late-import their collaborators from terminal_tool / terminal_tool_backends.
    monkeypatch.setattr(backends, "_create_environment", _fake_create)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_select_image", lambda *a, **k: "img")
    monkeypatch.setattr(terminal_tool, "_resolve_task_host_cwd", lambda *a, **k: None)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})

    prompt_builder._run_backend_probe("docker", terminal_tool)
    code_execution_tool._get_or_create_env("cc-task")

    assert seen == [expected, expected]  # probe, then execute_code
    assert expected["docker_network"] is False and expected["docker_extra_args"] == ["--user", "1009:1009"]


def _reuse_guard_harness(
    monkeypatch, *, existing_mode: str, network: bool, extra_args=None
):
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
        extra_args=extra_args,
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


def test_extra_args_network_none_emits_flag_once(monkeypatch):
    """docker_network=false plus an operator ``--network none`` (either spelling) must emit the
    flag ONCE: Docker rejects a repeated --network with exit 125, so both together made every
    container start fail (#100248). Without an operator flag the implicit lockdown still applies."""
    for extra in (["--network=none", "--user", "1009:1009"], ["--network", "none"], ["--user", "1009:1009"]):
        commands = _reuse_guard_harness(monkeypatch, existing_mode="bridge", network=False, extra_args=list(extra))
        run_cmd = next(cmd for cmd in commands if len(cmd) > 2 and cmd[1:3] == ["run", "-d"])
        network_flags = [a for a in run_cmd if a in ("--network", "--net") or a.startswith(("--network=", "--net="))]
        assert len(network_flags) == 1, (extra, run_cmd)
        assert "--user" not in extra or "1009:1009" in run_cmd


def test_contradictory_network_request_fails_closed(monkeypatch):
    """docker_network=false with --network=host is contradictory: honouring the extra arg would
    defeat the lockdown and the reuse guard (NetworkMode == "none") would churn the container
    every startup. Fail loudly, naming both keys."""
    import pytest

    with pytest.raises(RuntimeError, match="docker_network"):
        _reuse_guard_harness(monkeypatch, existing_mode="bridge", network=False, extra_args=["--network=host"])
