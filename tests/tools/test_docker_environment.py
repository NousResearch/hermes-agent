import logging
import subprocess
import sys
import types

import pytest

from tools.environments import docker as docker_env


def _make_dummy_env(**kwargs):
    """Helper to construct DockerEnvironment with minimal required args."""
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        cpu=kwargs.get("cpu", 0),
        memory=kwargs.get("memory", 0),
        disk=kwargs.get("disk", 0),
        persistent_filesystem=kwargs.get("persistent_filesystem", False),
        task_id=kwargs.get("task_id", "test-task"),
        volumes=kwargs.get("volumes", []),
        host_cwd=kwargs.get("host_cwd", ""),
        network=kwargs.get("network", True),
    )


def _install_fake_minisweagent(monkeypatch, captured):
    """Install a fake mini-swe-agent docker module to capture constructor args."""

    class _FakeDocker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.container_id = "fake-container"

        def cleanup(self):
            return None

    minisweagent_mod = types.ModuleType("minisweagent")
    environments_mod = types.ModuleType("minisweagent.environments")
    docker_mod = types.ModuleType("minisweagent.environments.docker")
    docker_mod.DockerEnvironment = _FakeDocker

    monkeypatch.setitem(sys.modules, "minisweagent", minisweagent_mod)
    monkeypatch.setitem(sys.modules, "minisweagent.environments", environments_mod)
    monkeypatch.setitem(sys.modules, "minisweagent.environments.docker", docker_mod)


def test_ensure_docker_available_logs_and_raises_when_not_found(monkeypatch, caplog):
    """When docker cannot be found, raise a clear error before mini-swe setup."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: None)
    monkeypatch.setattr(
        docker_env.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess.run should not be called when docker is missing"),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker executable not found in PATH or known install locations" in str(excinfo.value)
    assert any(
        "no docker executable was found in PATH or known install locations"
        in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_logs_and_raises_on_timeout(monkeypatch, caplog):
    """When docker version times out, surface a helpful error instead of hanging."""

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["/custom/docker", "version"], timeout=5)

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/custom/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _raise_timeout)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker daemon is not responding" in str(excinfo.value)
    assert any(
        "/custom/docker version' timed out" in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_uses_resolved_executable(monkeypatch):
    """When docker is found outside PATH, preflight should use that resolved path."""

    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/opt/homebrew/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    docker_env._ensure_docker_available()

    assert calls == [
        (["/opt/homebrew/bin/docker", "version"], {
            "capture_output": True,
            "text": True,
            "timeout": 5,
        })
    ]


def test_host_cwd_is_bind_mounted_to_workspace(monkeypatch, tmp_path):
    """A host cwd should be mounted at /workspace for Docker CLI sessions."""

    captured = {}
    _install_fake_minisweagent(monkeypatch, captured)
    monkeypatch.setattr(docker_env, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    env = _make_dummy_env(cwd="/workspace", host_cwd=str(tmp_path))

    assert env.cwd == "/workspace"
    assert captured["cwd"] == "/workspace"
    assert f"{tmp_path}:/workspace" in captured["run_args"]


def test_missing_host_cwd_falls_back_to_sandbox_workspace(monkeypatch, tmp_path):
    """Invalid host cwd should not produce a broken bind mount."""

    captured = {}
    _install_fake_minisweagent(monkeypatch, captured)
    monkeypatch.setattr(docker_env, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        "tools.environments.base.get_sandbox_dir",
        lambda: tmp_path / "sandboxes",
    )

    missing = tmp_path / "does-not-exist"
    _make_dummy_env(cwd="/workspace", host_cwd=str(missing), persistent_filesystem=True)

    assert f"{missing}:/workspace" not in captured["run_args"]
    assert any(str(tmp_path / "sandboxes" / "docker" / "test-task" / "workspace") in arg for arg in captured["run_args"])
