import json
import subprocess

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
