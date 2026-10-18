"""Root-run Docker sandboxes hand their bind-mounted files back to the host user at teardown, and a
sandbox dir that still cannot be removed is reported instead of silently left behind."""

import importlib
import json
import logging
import os
import subprocess

import pytest

from tools.environments import docker as docker_env


_ROOT_OWNED = b"/root/.cache/x\0/workspace/out.txt\0"  # what a root walk of the sandbox finds
_HOST_OWNED = b"/workspace/repo/HEAD\0"  # a no-symlink walk for host-owned entries finds only this


def _fake_docker(monkeypatch, security_options: str, chown_times_out: bool = False):
    docker_env._cgroup_limits_ok = True
    try:  # the daemon-mode probe is cached per process
        importlib.import_module("tools.environments.docker_ownership")._remapping_cache.clear()
    except ImportError:
        pass
    calls: list[tuple[list[str], bytes]] = []

    def _mounts() -> str:  # like `docker inspect .Mounts`: every -v of the run, extra args included
        run = next(argv for argv, _ in calls if argv[1] == "run")
        return json.dumps([{"Destination": run[i + 1].split(":")[1]}
                           for i, a in enumerate(run[:-1]) if a == "-v"])

    def _run(cmd, **kwargs):
        argv = [str(a) for a in cmd]
        calls.append((argv, kwargs.get("input") or b""))
        sub = argv[1] if len(argv) > 1 else ""
        if chown_times_out and sub == "exec" and "chown" in argv:
            raise subprocess.TimeoutExpired(argv, 10)
        out = {"run": "fake-cid\n", "info": security_options}.get(sub, "")
        out = _mounts() if sub == "inspect" and "{{json .Mounts}}" in argv else out
        if sub == "exec" and "find" in argv and "-user" in argv:
            out = (_ROOT_OWNED if argv[argv.index("-user") + 1] == "0" else _HOST_OWNED).decode()
        if kwargs.get("text") or kwargs.get("encoding"):
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=out.encode(), stderr=b"")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)  # Hermes itself is not root
    monkeypatch.setattr(os, "getegid", lambda: 1000)
    return calls


_PLAIN = "[name=apparmor name=seccomp,profile=builtin name=cgroupns]"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(("run_as_host_user", "security_options", "scenario"), [
    (False, _PLAIN, "handback"),
    (True, _PLAIN, "none"),
    (False, "[name=seccomp,profile=builtin name=rootless name=cgroupns]", "none"),
    (False, "[name=apparmor name=userns]", "none"),
    (False, _PLAIN, "chown-timeout"),  # paths are recorded before they change owner
    (False, _PLAIN, "reclaim"),  # only recorded paths a no-symlink walk still finds go back to root
    (True, _PLAIN, "reclaim"),  # a host-user container keeps the handed-back owner
])
def test_root_container_teardown_hands_every_sandbox_mount_back_before_removal(
        monkeypatch, tmp_path, run_as_host_user, security_options, scenario):
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    calls = _fake_docker(monkeypatch, security_options, chown_times_out=scenario == "chown-timeout")
    nested_ro, shared = tmp_path / "skills", tmp_path / "shared"
    nested_ro.mkdir()
    shared.mkdir()
    record = tmp_path / "sandboxes" / "docker" / "chown-back" / ".hermes-owner-handback"
    if scenario == "reclaim":  # an earlier root teardown handed these back; "link" is now a symlink
        record.parent.mkdir(parents=True)
        record.write_bytes(_HOST_OWNED + b"/workspace/link/escaped\0")
    env = docker_env.DockerEnvironment(
        image="python:3.12-slim", cwd="/root", timeout=5, task_id="chown-back", persistent_filesystem=True,
        persist_across_processes=False, run_as_host_user=run_as_host_user,
        volumes=[f"{nested_ro}:/root/.hermes/skills:ro"], extra_args=["-v", f"{shared}:/workspace/shared"])
    if scenario == "reclaim":
        to_root = [stdin for argv, stdin in calls if argv[1] == "exec" and "chown" in argv and "0:0" in argv[argv.index("chown"):]]
        assert to_root == ([] if run_as_host_user else [_HOST_OWNED]), calls
        assert not record.exists()
        return
    env.cleanup()
    assert env.wait_for_cleanup(5)

    subcommands = [argv[1] for argv, _ in calls]
    chowns = [(argv, stdin) for argv, stdin in calls if argv[1] == "exec" and "chown" in argv]
    if scenario == "none":
        assert chowns == [], chowns
        assert subcommands[-2:] == ["stop", "rm"]
        return
    assert record.read_bytes() == _ROOT_OWNED
    if scenario == "chown-timeout":
        return
    (walk,) = [argv for argv, _ in calls if argv[1] == "exec" and "find" in argv]
    ((chown, stdin),) = chowns
    assert subcommands.index("exec", subcommands.index("run") + 1) < subcommands.index("stop")
    assert walk[2:5] == ["-u", "0:0", "fake-cid"] and chown[3:6] == ["-u", "0:0", "fake-cid"]
    assert walk[walk.index("find") + 1:walk.index("-xdev")] == ["-P", "/root", "/workspace"]
    pruned = [walk[i + 1] for i, a in enumerate(walk) if a == "-path"]
    assert {"/root/.hermes/skills", "/workspace/shared"} <= set(pruned)  # docker_volumes and extra-args binds
    assert "1000:1000" in chown and stdin == _ROOT_OWNED


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
