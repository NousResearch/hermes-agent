import json
import logging
import subprocess

from unittest.mock import Mock

import pytest
from tools.environments import docker


def test_config_yaml_enables_zero_cap_and_rejects_overrides(tmp_path, monkeypatch):
    from hermes_cli.config import apply_terminal_config_to_env
    from tools import terminal_tool, terminal_tool_backends

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n  docker_zero_cap: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("TERMINAL_DOCKER_ZERO_CAP", raising=False)

    calls = []
    monkeypatch.setattr(docker, "find_docker", lambda: "docker")
    monkeypatch.setattr(docker, "_cgroup_limits_ok", True)

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="sandbox\n" if cmd[1] == "run" else "", stderr="")

    monkeypatch.setattr(docker.subprocess, "run", run)
    apply_terminal_config_to_env()
    config = terminal_tool._get_env_config()
    env = terminal_tool_backends._create_environment(
        env_type=config["env_type"],
        image=config["docker_image"],
        cwd=config["cwd"],
        timeout=config["timeout"],
        container_config=terminal_tool_backends._container_config_from_config(config),
    )

    assert env._zero_cap is True
    assert "--cap-add" not in env._all_run_args
    assert env._all_run_args[env._all_run_args.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in env._all_run_args
    for extra in (["--cap-add=SYS_ADMIN"], ["--privileged"], ["--security-opt", "no-new-privileges=false"]):
        with pytest.raises(ValueError, match="zero-cap"):
            docker.DockerEnvironment(image="fixture", zero_cap=True, extra_args=extra)


@pytest.fixture
def make_sandbox(monkeypatch):
    """Exercise real argv construction without contacting a Docker daemon."""
    monkeypatch.setattr(docker, "find_docker", lambda: "docker")
    monkeypatch.setattr(docker, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(docker, "_cgroup_limits_ok", True)
    monkeypatch.setattr(docker, "_image_uses_init_entrypoint", lambda *args: False)
    monkeypatch.setattr(docker, "_readonly_skill_mount_args", list)
    monkeypatch.setattr(docker, "reclaim_sandbox_files", Mock())
    monkeypatch.setattr(docker.DockerEnvironment, "_docker_run", Mock(return_value="sandbox"))
    monkeypatch.setattr(docker.DockerEnvironment, "init_session", Mock())
    def create(**kwargs):
        kwargs.setdefault("persist_across_processes", False)
        return docker.DockerEnvironment(image="fixture", **kwargs)

    return create


@pytest.mark.parametrize("enforce_egress", [False, True])
@pytest.mark.parametrize("extra, flag", [
    (["--privileged"], "--privileged"),
    (["--privileged=true"], "--privileged"),
    (["--privileged", "false"], "--privileged"),
    (["--cap-add", "SYS_ADMIN"], "--cap-add"),
    (["--cap-add=SYS_ADMIN"], "--cap-add"),
    (["--security-opt", "no-new-privileges=false"], "--security-opt"),
    (["--security-opt=seccomp=unconfined"], "--security-opt"),
    (["--pid=host"], "--pid"),
    (["--ipc", "host"], "--ipc"),
    (["--uts=host"], "--uts"),
    (["--userns=host"], "--userns"),
    (["--cgroupns=host"], "--cgroupns"),
    (["--network=host"], "--network"),
    (["--net", "container:other"], "--net"),
    (["-v", "/:/host"], "--volume"),
    (["-v/sensitive-source:/host"], "--volume"),
    (["-itv/sensitive-source:/host"], "--volume"),
    (["--volume=/sensitive-source:/host:ro"], "--volume"),
    (["--mount", "type=bind,src=/sensitive-source,dst=/host"], "--mount"),
    (["--volumes-from=other"], "--volumes-from"),
    (["--device=/dev/example"], "--device"),
])
def test_extra_args_warn_without_blocking_break_glass(
    make_sandbox, monkeypatch, caplog, extra, flag, enforce_egress,
):
    proxy_env = {"HTTPS_PROXY": "http://proxy.invalid:3128"} if enforce_egress else {}
    monkeypatch.setattr(docker, "_egress_proxy_args_for_docker", lambda: ([], proxy_env, []))
    monkeypatch.setattr(docker, "_egress_enforce_on_docker", lambda: enforce_egress)
    with caplog.at_level(logging.WARNING, logger=docker.__name__):
        if enforce_egress and flag in {"--network", "--net"}:
            # The existing egress refusal remains authoritative.
            with pytest.raises(RuntimeError, match="enforce_on_docker"):
                make_sandbox(extra_args=extra)
            env = None
        else:
            env = make_sandbox(extra_args=extra)
    warnings = [r.message for r in caplog.records if "sandbox hardening" in r.message]
    assert len(warnings) == 1
    assert flag in warnings[0]
    assert "docker_extra_args" in warnings[0]
    assert "Remove" in warnings[0]
    assert "break-glass" in warnings[0]
    assert "sensitive-source" not in warnings[0]
    if env is not None:
        assert env._all_run_args[-len(extra):] == extra
        env._docker_run.assert_called_once()


@pytest.mark.parametrize("extra", [
    ["--pid=host"],
    ["-v", "/:/host"],
    ["--mount=type=bind,src=/,dst=/host"],
])
def test_zero_cap_keeps_mount_and_namespace_break_glass_visible(make_sandbox, caplog, extra):
    with caplog.at_level(logging.WARNING, logger=docker.__name__):
        env = make_sandbox(zero_cap=True, extra_args=extra)
    assert "sandbox hardening" in caplog.text
    assert env._all_run_args[-len(extra):] == extra
    assert "--cap-add" not in env._all_run_args


@pytest.mark.parametrize("zero_cap", [False, True])
def test_ordinary_extra_args_do_not_warn(make_sandbox, caplog, zero_cap):
    extra = ["--gpus=all", "--shm-size", "2g", "--cap-drop=NET_RAW", "--pid=private",
             "--security-opt=no-new-privileges=true"]
    with caplog.at_level(logging.WARNING, logger=docker.__name__):
        env = make_sandbox(zero_cap=zero_cap, extra_args=extra)
    assert "sandbox hardening" not in caplog.text
    assert env._all_run_args[-len(extra):] == extra


def test_hardening_warning_is_deduplicated_after_type_filtering(make_sandbox, caplog):
    extra = [17, "--privileged", "--cap-add=SYS_ADMIN", "--cap-add", "SYS_ADMIN"]
    with caplog.at_level(logging.WARNING, logger=docker.__name__):
        env = make_sandbox(extra_args=extra)
    warnings = [r.message for r in caplog.records if "sandbox hardening" in r.message]
    assert len(warnings) == 1
    assert warnings[0].count("--cap-add") == 1
    assert "--privileged" in warnings[0]
    assert "SYS_ADMIN" not in warnings[0]
    assert env._all_run_args[-4:] == extra[1:]


def test_disabled_privileged_flag_does_not_warn(make_sandbox, caplog):
    with caplog.at_level(logging.WARNING, logger=docker.__name__):
        make_sandbox(extra_args=["--privileged=false"])
    assert "sandbox hardening" not in caplog.text


def test_hardening_warning_precedes_container_reuse(make_sandbox, monkeypatch, caplog):
    def attach(env, *args):
        assert "sandbox hardening" in caplog.text
        env._container_id = "existing"
        return True

    monkeypatch.setattr(docker.DockerEnvironment, "_attach_existing_container", attach)
    monkeypatch.setattr(docker, "strip_sandbox_privileges", Mock())
    with caplog.at_level(logging.WARNING, logger=docker.__name__):
        env = make_sandbox(extra_args=["--pid=host"], persist_across_processes=True)
    assert env._container_id == "existing"
    env._docker_run.assert_not_called()


def test_strict_reuse_requires_immutable_posture(monkeypatch):
    env = object.__new__(docker.DockerEnvironment)
    env._docker_exe = "docker"
    env._zero_cap = True
    env._labels = {}
    host = {"Privileged": False, "CapAdd": [], "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"]}
    def query(cmd, **kwargs):
        value = "sandbox\trunning\n" if cmd[1] == "ps" else json.dumps(host)
        return subprocess.CompletedProcess(cmd, 0, stdout=value, stderr="")
    monkeypatch.setattr(docker, "_docker_query", query)
    assert env._find_reusable_container("task", "profile", "off") == ("sandbox", "running")
    host["CapAdd"] = ["SYS_ADMIN"]
    assert env._find_reusable_container("task", "profile", "off") is None
