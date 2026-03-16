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


def test_auto_mount_host_cwd_adds_volume(monkeypatch, tmp_path):
    """When host_cwd is provided, it should be auto-mounted to /workspace."""
    import os

    # Create a temp directory to simulate user's project directory
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    # Mock Docker availability
    def _run_docker_version(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="Docker version", stderr="")

    def _run_docker_create(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="storage-opt not supported")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run_docker_version)

    # Mock the inner _Docker class to capture run_args
    captured_run_args = []

    class MockInnerDocker:
        container_id = "mock-container-123"
        config = type("Config", (), {"executable": "/usr/bin/docker", "forward_env": [], "env": {}})()

        def __init__(self, **kwargs):
            captured_run_args.extend(kwargs.get("run_args", []))

    monkeypatch.setattr(
        "minisweagent.environments.docker.DockerEnvironment",
        MockInnerDocker,
    )

    # Create environment with host_cwd
    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/workspace",
        timeout=60,
        persistent_filesystem=False,  # Non-persistent mode uses tmpfs, should be overridden
        task_id="test-auto-mount",
        volumes=[],
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
    )

    # Check that the host_cwd was added as a volume mount
    volume_mount = f"-v {project_dir}:/workspace"
    run_args_str = " ".join(captured_run_args)
    assert f"{project_dir}:/workspace" in run_args_str, f"Expected auto-mount in run_args: {run_args_str}"


def test_auto_mount_disabled_via_env(monkeypatch, tmp_path):
    """Auto-mount should be disabled when TERMINAL_DOCKER_NO_AUTO_MOUNT is set."""
    import os

    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setenv("TERMINAL_DOCKER_NO_AUTO_MOUNT", "true")

    def _run_docker_version(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run_docker_version)

    captured_run_args = []

    class MockInnerDocker:
        container_id = "mock-container-456"
        config = type("Config", (), {"executable": "/usr/bin/docker", "forward_env": [], "env": {}})()

        def __init__(self, **kwargs):
            captured_run_args.extend(kwargs.get("run_args", []))

    monkeypatch.setattr(
        "minisweagent.environments.docker.DockerEnvironment",
        MockInnerDocker,
    )

    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/workspace",
        timeout=60,
        persistent_filesystem=False,
        task_id="test-no-auto-mount",
        volumes=[],
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
    )

    # Check that the host_cwd was NOT added (because env var disabled it)
    run_args_str = " ".join(captured_run_args)
    assert f"{project_dir}:/workspace" not in run_args_str, f"Auto-mount should be disabled: {run_args_str}"


def test_auto_mount_skipped_when_workspace_already_mounted(monkeypatch, tmp_path):
    """Auto-mount should be skipped if /workspace is already mounted via user volumes."""
    import os

    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    def _run_docker_version(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run_docker_version)

    captured_run_args = []

    class MockInnerDocker:
        container_id = "mock-container-789"
        config = type("Config", (), {"executable": "/usr/bin/docker", "forward_env": [], "env": {}})()

        def __init__(self, **kwargs):
            captured_run_args.extend(kwargs.get("run_args", []))

    monkeypatch.setattr(
        "minisweagent.environments.docker.DockerEnvironment",
        MockInnerDocker,
    )

    # User already configured a volume mount for /workspace
    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/workspace",
        timeout=60,
        persistent_filesystem=False,
        task_id="test-workspace-exists",
        volumes=[f"{other_dir}:/workspace"],  # User explicitly mounted something to /workspace
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
    )

    # The user's explicit mount should be present
    run_args_str = " ".join(captured_run_args)
    assert f"{other_dir}:/workspace" in run_args_str

    # But the auto-mount should NOT add a duplicate
    assert run_args_str.count(":/workspace") == 1, f"Should only have one /workspace mount: {run_args_str}"

