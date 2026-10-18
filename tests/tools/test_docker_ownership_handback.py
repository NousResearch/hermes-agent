"""Root-run Docker sandboxes hand their bind-mounted files back to the host user at teardown, and a
sandbox dir that still cannot be removed is reported instead of silently left behind."""

import importlib
import logging
import os
import subprocess

import pytest

from tools.environments import docker as docker_env


def _fake_docker(monkeypatch, security_options: str):
    docker_env._cgroup_limits_ok = True
    try:  # the daemon-mode probe is cached per process
        importlib.import_module("tools.environments.docker_ownership")._remapping_cache.clear()
    except ImportError:
        pass
    calls: list[list[str]] = []

    def _run(cmd, **kwargs):
        argv = [str(a) for a in cmd]
        calls.append(argv)
        sub = argv[1] if len(argv) > 1 else ""
        out = {"run": "fake-cid\n", "info": security_options}.get(sub, "")
        if kwargs.get("text") or kwargs.get("encoding"):
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=out.encode(), stderr=b"")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)  # Hermes itself is not root
    monkeypatch.setattr(os, "getegid", lambda: 1000)
    return calls


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(("run_as_host_user", "security_options", "expect_handback"), [
    (False, "[name=apparmor name=seccomp,profile=builtin name=cgroupns]", True),
    (True, "[name=apparmor name=seccomp,profile=builtin name=cgroupns]", False),
    (False, "[name=seccomp,profile=builtin name=rootless name=cgroupns]", False),
    (False, "[name=apparmor name=userns]", False),
])
def test_root_container_teardown_hands_every_sandbox_mount_back_before_removal(
        monkeypatch, tmp_path, run_as_host_user, security_options, expect_handback):
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    calls = _fake_docker(monkeypatch, security_options)
    nested_ro = tmp_path / "skills"
    nested_ro.mkdir()
    env = docker_env.DockerEnvironment(
        image="python:3.12-slim", cwd="/root", timeout=5, task_id="chown-back", persistent_filesystem=True,
        persist_across_processes=False, run_as_host_user=run_as_host_user,
        volumes=[f"{nested_ro}:/root/.hermes/skills:ro"])
    env.cleanup()
    assert env.wait_for_cleanup(5)

    subcommands = [argv[1] for argv in calls]
    handbacks = [argv for argv in calls if argv[1] == "exec" and "chown" in argv]
    if not expect_handback:
        assert handbacks == [], handbacks
        assert subcommands[-2:] == ["stop", "rm"]
        return
    assert len(handbacks) == 1, calls
    (handback,) = handbacks
    assert subcommands.index("exec", subcommands.index("run") + 1) < subcommands.index("stop")
    assert handback[2:5] == ["-u", "0:0", "fake-cid"]
    targets = handback[handback.index("find") + 1:handback.index("-xdev")]
    assert sorted(targets) == ["/root", "/workspace"]
    assert ["-path", "/root/.hermes/skills"] == handback[handback.index("-path"):handback.index("-path") + 2]
    assert "1000:1000" in handback


def test_remove_bind_dirs_warns_with_the_path_when_a_dir_survives(monkeypatch, tmp_path, caplog):
    survivor = tmp_path / "docker" / "task" / "home"
    survivor.mkdir(parents=True)
    monkeypatch.setattr(docker_env.shutil, "rmtree", lambda *a, **k: None)  # a root-owned tree rmtree cannot drop
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._workspace_dir, env._home_dir = None, str(survivor)

    with caplog.at_level(logging.WARNING, logger=docker_env.logger.name):
        env._remove_bind_dirs()

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(survivor) in w and "sudo rm -rf" in w for w in warnings), warnings
