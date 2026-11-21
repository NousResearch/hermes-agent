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


def test_managed_modal_message_names_direct_credentials_that_exist():
    """Managed mode blocked with direct Modal credentials present: the text must not say there are none."""
    state = {"mode": None, "managed_mode_blocked": True, "has_direct": True}
    log, err = backends._modal_unavailable_reason(state)
    assert "no direct Modal credentials" not in log + err
    assert "Direct Modal credentials were found" in err
    state["has_direct"] = False
    assert "no direct Modal credentials/config were found" in backends._modal_unavailable_reason(state)[1]
