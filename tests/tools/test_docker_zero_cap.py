import inspect
import json
import subprocess

import pytest
from tools.environments import docker


def test_zero_cap_build_and_override_rejection(monkeypatch):
    assert "zero_cap" in inspect.signature(docker.DockerEnvironment).parameters
    calls = []
    monkeypatch.setattr(docker, "find_docker", lambda: "docker")
    monkeypatch.setattr(docker, "_cgroup_limits_ok", True)
    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="sandbox\n" if cmd[1] == "run" else "", stderr="")
    monkeypatch.setattr(docker.subprocess, "run", run)
    env = docker.DockerEnvironment(image="fixture", zero_cap=True)
    assert "--cap-add" not in env._all_run_args
    assert env._all_run_args[env._all_run_args.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in env._all_run_args
    for extra in (["--cap-add=SYS_ADMIN"], ["--privileged"], ["--security-opt", "no-new-privileges=false"]):
        with pytest.raises(ValueError, match="zero-cap"):
            docker.DockerEnvironment(image="fixture", zero_cap=True, extra_args=extra)


def test_strict_reuse_requires_immutable_posture(monkeypatch):
    env = object.__new__(docker.DockerEnvironment)
    env._docker_exe = "docker"
    env._zero_cap = True
    host = {"Privileged": False, "CapAdd": [], "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"]}
    def query(cmd, **kwargs):
        value = "sandbox\trunning\n" if cmd[1] == "ps" else json.dumps(host)
        return subprocess.CompletedProcess(cmd, 0, stdout=value, stderr="")
    monkeypatch.setattr(docker, "_docker_query", query)
    assert env._find_reusable_container("task", "profile", "off") == ("sandbox", "running")
    host["CapAdd"] = ["SYS_ADMIN"]
    assert env._find_reusable_container("task", "profile", "off") is None
