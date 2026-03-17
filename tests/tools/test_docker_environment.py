import logging
from io import StringIO
import subprocess
import sys
import types

import pytest

from tools.environments import docker as docker_env


def _install_fake_minisweagent(monkeypatch, captured_run_args):
    class MockInnerDocker:
        container_id = "fake-container"
        config = type("Config", (), {"executable": "/usr/bin/docker", "forward_env": [], "env": {}})()

        def __init__(self, **kwargs):
            captured_run_args.extend(kwargs.get("run_args", []))

        def cleanup(self):
            pass

    minisweagent_mod = types.ModuleType("minisweagent")
    environments_mod = types.ModuleType("minisweagent.environments")
    docker_mod = types.ModuleType("minisweagent.environments.docker")
    docker_mod.DockerEnvironment = MockInnerDocker

    monkeypatch.setitem(sys.modules, "minisweagent", minisweagent_mod)
    monkeypatch.setitem(sys.modules, "minisweagent.environments", environments_mod)
    monkeypatch.setitem(sys.modules, "minisweagent.environments.docker", docker_mod)


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
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
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
    """Opt-in docker cwd mounting should bind the host cwd to /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    def _run_docker_version(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run_docker_version)

    captured_run_args = []
    _install_fake_minisweagent(monkeypatch, captured_run_args)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
    )

    run_args_str = " ".join(captured_run_args)
    assert f"{project_dir}:/workspace" in run_args_str


def test_auto_mount_disabled_by_default(monkeypatch, tmp_path):
    """Host cwd should not be mounted unless the caller explicitly opts in."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    def _run_docker_version(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run_docker_version)

    captured_run_args = []
    _install_fake_minisweagent(monkeypatch, captured_run_args)

    _make_dummy_env(
        cwd="/root",
        host_cwd=str(project_dir),
        auto_mount_cwd=False,
    )

    run_args_str = " ".join(captured_run_args)
    assert f"{project_dir}:/workspace" not in run_args_str


def test_auto_mount_skipped_when_workspace_already_mounted(monkeypatch, tmp_path):
    """Explicit user volumes for /workspace should take precedence over cwd mount."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    def _run_docker_version(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run_docker_version)

    captured_run_args = []
    _install_fake_minisweagent(monkeypatch, captured_run_args)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        volumes=[f"{other_dir}:/workspace"],
    )

    run_args_str = " ".join(captured_run_args)
    assert f"{other_dir}:/workspace" in run_args_str
    assert run_args_str.count(":/workspace") == 1


def test_auto_mount_replaces_persistent_workspace_bind(monkeypatch, tmp_path):
    """Persistent mode should still prefer the configured host cwd at /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    def _run_docker_version(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run_docker_version)

    captured_run_args = []
    _install_fake_minisweagent(monkeypatch, captured_run_args)

    _make_dummy_env(
        cwd="/workspace",
        persistent_filesystem=True,
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        task_id="test-persistent-auto-mount",
    )

    run_args_str = " ".join(captured_run_args)
    assert f"{project_dir}:/workspace" in run_args_str
    assert "/sandboxes/docker/test-persistent-auto-mount/workspace:/workspace" not in run_args_str


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = StringIO("")
        self.stdin = None
        self.returncode = 0

    def poll(self):
        return self.returncode


def _make_execute_only_env(forward_env=None):
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env.cwd = "/root"
    env.timeout = 60
    env._forward_env = forward_env or []
    env._prepare_command = lambda command: (command, None)
    env._timeout_result = lambda timeout: {"output": f"timed out after {timeout}", "returncode": 124}
    env._inner = type("Inner", (), {
        "container_id": "test-container",
        "config": type("Cfg", (), {"executable": "/usr/bin/docker", "env": {}})(),
    })()
    return env


def test_execute_uses_hermes_dotenv_for_allowlisted_env(monkeypatch):
    env = _make_execute_only_env(["GITHUB_TOKEN"])
    popen_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"GITHUB_TOKEN": "value_from_dotenv"})
    monkeypatch.setattr(docker_env.subprocess, "Popen", _fake_popen)

    result = env.execute("echo hi")

    assert result["returncode"] == 0
    assert "GITHUB_TOKEN=value_from_dotenv" in popen_calls[0]


def test_execute_prefers_shell_env_over_hermes_dotenv(monkeypatch):
    env = _make_execute_only_env(["GITHUB_TOKEN"])
    popen_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setenv("GITHUB_TOKEN", "value_from_shell")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"GITHUB_TOKEN": "value_from_dotenv"})
    monkeypatch.setattr(docker_env.subprocess, "Popen", _fake_popen)

    env.execute("echo hi")

    assert "GITHUB_TOKEN=value_from_shell" in popen_calls[0]
    assert "GITHUB_TOKEN=value_from_dotenv" not in popen_calls[0]


def test_non_persistent_cleanup_removes_container(monkeypatch):
    """When container_persistent=false, cleanup() must run docker rm -f so the container is removed (Fixes #1679)."""
    run_calls = []

    def _run(cmd, **kwargs):
        run_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if cmd and getattr(cmd[0], '__str__', None) and 'docker' in str(cmd[0]):
            if len(cmd) >= 2 and cmd[1] == 'run':
                return subprocess.CompletedProcess(cmd, 0, stdout="abc123container\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')

    monkeypatch.setattr(docker_env, 'find_docker', lambda: '/usr/bin/docker')
    monkeypatch.setattr(docker_env.subprocess, 'run', _run)
    monkeypatch.setattr(docker_env.subprocess, 'Popen', lambda *a, **k: type('P', (), {'poll': lambda: None, 'wait': lambda **kw: None, 'returncode': 0, 'stdout': iter([]), 'stdin': None})())

    captured_run_args = []
    _install_fake_minisweagent(monkeypatch, captured_run_args)

    env = _make_dummy_env(persistent_filesystem=False, task_id='ephemeral-task')
    assert env._container_id
    container_id = env._container_id

    env.cleanup()

    rm_calls = [c for c in run_calls if isinstance(c[0], list) and len(c[0]) >= 4 and c[0][1:4] == ['rm', '-f', container_id]]
    assert len(rm_calls) >= 1, 'cleanup() should run docker rm -f <container_id> when container_persistent=false'
