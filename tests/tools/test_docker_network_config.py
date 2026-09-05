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

    # file_tools no longer builds its own container_config: it goes through
    # the shared _create_configured_env, which is what carries docker_network.
    assert "_create_configured_env(" in inspect.getsource(file_tools)

    for module in (terminal_tool, code_execution_tool):
        tree = ast.parse(inspect.getsource(module))
        sites = 0
        for node in ast.walk(tree):
            # Accept a dict literal or a (key, default) pair table that a dict
            # comprehension is built from.
            if isinstance(node, ast.Dict):
                keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            elif isinstance(node, ast.Tuple) and node.elts and all(
                isinstance(e, ast.Tuple) and len(e.elts) == 2 and isinstance(e.elts[0], ast.Constant)
                for e in node.elts
            ):
                keys = {e.elts[0].value for e in node.elts}
            else:
                continue
            if "docker_run_as_host_user" in keys:
                sites += 1
                assert "docker_network" in keys, (
                    f"{module.__name__} builds a container_config with "
                    f"docker_run_as_host_user but without docker_network "
                    f"(line {node.lineno})"
                )
        assert sites >= 1, f"expected at least one container_config site in {module.__name__}"


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
