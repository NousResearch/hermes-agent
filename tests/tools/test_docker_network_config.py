"""Regression tests for the Docker terminal network toggle.

Ported from NanoClaw PR #2713's opt-in egress lockdown idea. Hermes already
has DockerEnvironment(network=False), but the terminal config path did not
expose it, so operators could not request networkless Docker execution from
config.yaml.
"""

import json
import pytest

import tools.terminal_tool as terminal_tool
from tools.environments import docker as docker_env


@pytest.fixture(autouse=True)
def isolated_skill_snapshots(monkeypatch, tmp_path):
    """Use real staged snapshots while keeping tests outside the production data root."""
    monkeypatch.setenv("HERMES_DOCKER_DATA_ROOT", str(tmp_path / "docker-data"))
    original = docker_env._skill_mount_plan
    monkeypatch.setattr(docker_env, "_test_skill_mounts", [], raising=False)
    def plan(profile):
        args, digest = original(profile)
        mounts = []
        for spec in args:
            parsed = docker_env._split_volume_spec(spec)
            if parsed and any(p == "skills" or p.endswith("_skills") for p in parsed[1].split("/")):
                mounts.append({"Source": parsed[0], "Destination": parsed[1], "RW": False})
        docker_env._test_skill_mounts = mounts
        return args, digest
    monkeypatch.setattr(docker_env, "_skill_mount_plan", plan)


def test_terminal_env_config_reads_docker_network_toggle(monkeypatch):
    monkeypatch.setenv("TERMINAL_DOCKER_NETWORK", "false")

    config = terminal_tool._get_env_config()

    assert config["docker_network"] is False


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
    monkeypatch, *, existing_mode: str, network: bool, extra_args=None, existing_image: str = "python:3.11",
    image_pinned: bool = False, image_pullable: bool = True,
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
            # Reuse uses the portable ID/State format; labels are filters.
            Result.stdout = "existing-container-id\trunning\n"
        elif len(cmd) > 1 and cmd[1] == "inspect":
            # Two probes share `inspect`: image identity (must match for reuse) and network mode.
            if "{{json .Mounts}}" in cmd:
                Result.stdout = json.dumps(docker_env._test_skill_mounts)
            else:
                Result.stdout = f"{existing_image}\n" if ".Config.Image" in cmd[3] else f"{existing_mode}\n"
        elif len(cmd) > 2 and cmd[1:3] == ["image", "inspect"]:
            Result.returncode = 1  # never in the local store: the replacement must go through `pull`
        elif len(cmd) > 1 and cmd[1] == "pull":
            Result.returncode = 0 if image_pullable else 1
            Result.stderr = "" if image_pullable else "pull access denied"
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
        image_pinned=image_pinned,
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
    assert not any(cmd[1] == "inspect" and ".HostConfig.NetworkMode" in cmd[3] for cmd in commands)
    assert not any(cmd[1] == "rm" for cmd in commands)
    assert not any(cmd[1] == "run" for cmd in commands)


def test_reuse_recreates_container_built_from_another_image_when_pinned(monkeypatch):
    """The user changed docker_image (config.yaml / TERMINAL_DOCKER_IMAGE): the old container is not
    their sandbox any more and must be replaced, or the new image never takes effect."""
    commands = _reuse_guard_harness(monkeypatch, existing_mode="bridge", network=True,
                                    existing_image="old/image:1", image_pinned=True)

    assert any(cmd[1:3] == ["rm", "-f"] for cmd in commands), "container from another image must be removed"
    assert any(len(cmd) > 2 and cmd[1:3] == ["run", "-d"] for cmd in commands)


def test_reuse_pulls_the_replacement_before_removing_the_old_container(monkeypatch):
    """The old container's writable layer is the user's sandbox; it goes only once the image replacing it
    is in the local store, so a private/misspelled tag or a registry outage never leaves them with
    nothing."""
    commands = _reuse_guard_harness(monkeypatch, existing_mode="bridge", network=True,
                                    existing_image="old/image:1", image_pinned=True)
    kinds = [tuple(c[1:3]) for c in commands]
    assert kinds.index(("pull", "python:3.11")) < kinds.index(("rm", "-f"))


def test_reuse_keeps_the_old_container_when_the_replacement_cannot_be_pulled(monkeypatch, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger=docker_env.logger.name):
        commands = _reuse_guard_harness(monkeypatch, existing_mode="bridge", network=True,
                                        existing_image="old/image:1", image_pinned=True, image_pullable=False)

    assert any(cmd[1] == "pull" for cmd in commands)
    assert not any(cmd[1:3] == ["rm", "-f"] for cmd in commands), "nothing to replace it with: keep the sandbox"
    assert not any(len(cmd) > 2 and cmd[1:3] == ["run", "-d"] for cmd in commands)
    assert "could not be pulled" in caplog.text and "keeping the current sandbox" in caplog.text


def test_reuse_keeps_container_built_from_another_image_when_default_flipped(monkeypatch, caplog):
    """A DEFAULT flip (nikolaik base -> hermes-sandbox:desktop) must not replace a sandbox the user has
    state in: the existing container is kept and the log names the approval command. Same rule Modal
    (snapshot wins) and Daytona (labeled sandbox wins) already apply."""
    import logging
    with caplog.at_level(logging.WARNING, logger=docker_env.logger.name):
        commands = _reuse_guard_harness(monkeypatch, existing_mode="bridge", network=True,
                                        existing_image="old/image:1", image_pinned=False)

    assert not any(cmd[1:3] == ["rm", "-f"] for cmd in commands), "an unpinned default never removes a sandbox"
    assert not any(len(cmd) > 2 and cmd[1:3] == ["run", "-d"] for cmd in commands), "existing container reused"
    assert "hermes config set terminal.docker_image python:3.11" in caplog.text


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


def test_pin_verdict_reaches_docker_environment_through_the_terminal_tool(monkeypatch):
    """TERMINAL_DOCKER_IMAGE_PINNED must survive the terminal tool's config shaping
    (``_CONTAINER_KEYS`` is an allowlist): a pinned image that the builder never sees is a pin
    the runtime never honours, and the approval would silently do nothing."""
    from tools import terminal_tool as tt
    from tools.terminal_tool_backends import _build_docker_env, _container_config_from_config

    seen = {}
    monkeypatch.setattr("tools.terminal_tool_backends._DockerEnvironment",
                        lambda **kw: seen.update(kw) or object())
    monkeypatch.setattr(tt, "_maybe_reap_docker_orphans", lambda cc: None)
    monkeypatch.setattr(tt, "_docker_session_isolation_enabled", lambda: False)
    for pinned in ("1", "0"):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_IMAGE_PINNED", pinned)
        cc = _container_config_from_config(tt._get_env_config())
        _build_docker_env(image="x", cwd="/root", timeout=5, cc=cc, task_id="default", host_cwd=None)
        assert seen["image_pinned"] is (pinned == "1")
