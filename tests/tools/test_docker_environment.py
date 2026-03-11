import logging
import subprocess

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
        network=kwargs.get("network", True),
    )


def test_ensure_docker_available_logs_and_raises_when_not_found(monkeypatch, caplog):
    """When docker is missing from PATH, we should raise a clear error and log it."""

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(docker_env, "subprocess", docker_env.subprocess)
    monkeypatch.setattr(docker_env.subprocess, "run", _raise_not_found)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    # Error message should mention that docker is not in PATH
    assert "Docker executable not found in PATH" in str(excinfo.value)
    assert any(
        "Docker backend selected but 'docker' executable was not found in PATH"
        in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_logs_and_raises_on_timeout(monkeypatch, caplog):
    """When docker version times out, surface a helpful error instead of hanging."""

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["docker", "version"], timeout=5)

    monkeypatch.setattr(docker_env.subprocess, "run", _raise_timeout)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker daemon is not responding" in str(excinfo.value)
    assert any(
        "docker version' timed out" in record.getMessage()
        for record in caplog.records
    )

