"""Tests for docker_forward_env configuration forwarding to Docker containers.

Bug #12534: Docker sandbox never receives env vars from docker_forward_env config.
The environment variables listed in terminal.docker_forward_env config were read
but never actually passed to the container creation call.
"""
import subprocess
import pytest

from tools.environments import docker as docker_env


def _mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to intercept docker run -d and docker version calls.

    Returns a list of captured (cmd, kwargs) tuples for inspection.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def _make_dummy_env(forward_env=None, env=None, **kwargs):
    """Helper to construct DockerEnvironment with docker_forward_env support."""
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
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
        forward_env=forward_env,
        env=env,
    )


def _has_env_flag(run_cmd, key, value):
    """Check if -e key=value appears as consecutive args in run_cmd."""
    for i, arg in enumerate(run_cmd):
        if arg == "-e" and i + 1 < len(run_cmd):
            if run_cmd[i + 1] == f"{key}={value}":
                return True
    return False


def test_docker_forward_env_includes_configured_vars_in_container_run(
    monkeypatch,
):
    """docker_forward_env vars that exist in host env should be passed to docker run -e.

    This tests the fix for bug #12534 where docker_forward_env was read but
    never actually forwarded to the container creation call.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    # Set up environment variables that will be "forwarded"
    monkeypatch.setenv("HOME", "/home/testuser")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")
    monkeypatch.setenv("MY_VAR", "my_value")

    calls = _mock_subprocess_run(monkeypatch)

    # Create environment with docker_forward_env
    _make_dummy_env(
        forward_env=["HOME", "PATH", "MY_VAR"],
    )

    # Find the docker run call
    run_calls = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_calls, "docker run should have been called"
    run_cmd = run_calls[0][0]

    # Verify all forward_env vars appear as -e flags in the docker run command
    assert _has_env_flag(run_cmd, "HOME", "/home/testuser"), f"HOME should be forwarded: {run_cmd}"
    assert _has_env_flag(run_cmd, "PATH", "/usr/local/bin:/usr/bin:/bin"), f"PATH should be forwarded: {run_cmd}"
    assert _has_env_flag(run_cmd, "MY_VAR", "my_value"), f"MY_VAR should be forwarded: {run_cmd}"


def test_docker_forward_env_skips_missing_vars(monkeypatch):
    """docker_forward_env vars that don't exist in host env should be silently skipped.

    Missing environment variables should not cause errors or be passed as empty strings.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    # Set up some env vars but not all
    monkeypatch.setenv("HOME", "/home/testuser")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")
    # MY_VAR is intentionally NOT set
    monkeypatch.delenv("MY_VAR", raising=False)

    calls = _mock_subprocess_run(monkeypatch)

    # Create environment with docker_forward_env that includes a missing var
    _make_dummy_env(
        forward_env=["HOME", "PATH", "MY_VAR"],
    )

    # Find the docker run call
    run_calls = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_calls, "docker run should have been called"
    run_cmd = run_calls[0][0]

    # HOME and PATH should be forwarded
    assert _has_env_flag(run_cmd, "HOME", "/home/testuser")
    assert _has_env_flag(run_cmd, "PATH", "/usr/local/bin:/usr/bin:/bin")

    # MY_VAR should NOT appear (it doesn't exist in host env)
    assert not _has_env_flag(run_cmd, "MY_VAR", ""), \
        f"MY_VAR should not be forwarded when missing from host env: {run_cmd}"


def test_docker_forward_env_empty_list_forwards_nothing(monkeypatch):
    """When docker_forward_env is empty, no additional env vars should be forwarded."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setenv("HOME", "/home/testuser")

    calls = _mock_subprocess_run(monkeypatch)

    # Create environment with empty docker_forward_env
    _make_dummy_env(
        forward_env=[],
    )

    # Find the docker run call
    run_calls = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_calls, "docker run should have been called"
    run_cmd = run_calls[0][0]

    # With empty forward_env, HOME should NOT appear in -e flags
    # (it's not in docker_env, only in forward_env which is empty)
    assert not _has_env_flag(run_cmd, "HOME", "/home/testuser"), \
        f"HOME should not be forwarded with empty forward_env: {run_cmd}"


def test_docker_env_and_docker_forward_env_both_forwarded(monkeypatch):
    """Both docker_env and docker_forward_env should be passed to container.

    docker_env provides explicit key-value pairs, docker_forward_env provides
    host env vars to relay. Both should appear in the docker run command.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setenv("HOST_VAR", "from_host")

    calls = _mock_subprocess_run(monkeypatch)

    # Create environment with both docker_env (explicit) and docker_forward_env (host relay)
    _make_dummy_env(
        env={"EXPLICIT_VAR": "explicit_value"},
        forward_env=["HOST_VAR"],
    )

    # Find the docker run call
    run_calls = [
        c for c in calls
        if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"
    ]
    assert run_calls, "docker run should have been called"
    run_cmd = run_calls[0][0]

    # Both should appear
    assert _has_env_flag(run_cmd, "EXPLICIT_VAR", "explicit_value"), \
        f"docker_env var missing: {run_cmd}"
    assert _has_env_flag(run_cmd, "HOST_VAR", "from_host"), \
        f"docker_forward_env var missing: {run_cmd}"
