"""Invariant: when a terminal backend's requirements check fails, the human reason is captured so the
CLI startup notice / doctor can tell the user WHY the terminal tool is missing (tools-runtime-01)."""

import subprocess

import pytest

from tools import terminal_tool
import tools.terminal_tool_backends as backends


@pytest.fixture(autouse=True)
def _reset_reason():
    backends._last_unavailable_reason = None
    yield
    backends._last_unavailable_reason = None


def test_docker_binary_missing_captures_reason(monkeypatch):
    monkeypatch.setitem(backends._BACKEND_SPECS, "docker",
                        {"binary": (lambda: None, "version", backends._BACKEND_SPECS["docker"]["binary"][2])})
    assert backends._check_requirements("docker", {}) is False
    reason = backends.terminal_backend_unavailable_reason()
    assert reason and "Docker" in reason


def test_docker_daemon_probe_failure_captures_daemon_reason(monkeypatch):
    monkeypatch.setitem(backends._BACKEND_SPECS, "docker",
                        {"binary": (lambda: "/usr/bin/docker", "version", "unused")})
    monkeypatch.setattr(backends.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, b"", b""))
    assert backends._check_requirements("docker", {}) is False
    assert backends.terminal_backend_unavailable_reason()


def test_ssh_unconfigured_captures_reason():
    assert backends._check_requirements("ssh", {"ssh_host": "", "ssh_user": ""}) is False
    assert backends.terminal_backend_unavailable_reason()


def test_successful_check_clears_reason():
    backends._last_unavailable_reason = "stale"
    assert backends._check_requirements("local", {}) is True
    assert backends.terminal_backend_unavailable_reason() is None


def test_check_terminal_requirements_exception_captures_reason(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert terminal_tool.check_terminal_requirements() is False
    assert "boom" in terminal_tool.terminal_backend_unavailable_reason()


def test_docker_probe_uses_shared_timeout(monkeypatch):
    """The docker probe waits DOCKER_PROBE_TIMEOUT, not the generic 5 s: a slow Docker Desktop CLI on
    Windows must not be reported as a dead daemon (and killed mid-request) after 5 s."""
    from tools.environments.docker import DOCKER_PROBE_TIMEOUT

    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["timeout"] = cmd, kwargs.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setitem(backends._BACKEND_SPECS, "docker",
                        {"binary": (lambda: "/usr/bin/docker", "version", "unused")})
    monkeypatch.setattr(backends.subprocess, "run", fake_run)
    assert backends._check_requirements("docker", {}) is True
    assert seen["cmd"] == ["/usr/bin/docker", "version"]
    assert seen["timeout"] == DOCKER_PROBE_TIMEOUT >= 20
