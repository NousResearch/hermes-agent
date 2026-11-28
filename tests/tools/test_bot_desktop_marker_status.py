"""Marker-owned Docker Bot Desktop status probes its recorded owner without making a new environment."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from tools.bot_desktop import placement, runtime, sandbox_host


def test_owner_identity_records_effective_docker_user(monkeypatch):
    env = SimpleNamespace(_container_id="recorded-container", _docker_exe="docker.exe")

    monkeypatch.setattr(sandbox_host, "_user_for", lambda value: "pn")
    assert sandbox_host._owner_identity(env) == {
        "backend": "SimpleNamespace", "container": "recorded-container",
        "docker": "docker.exe", "user": "pn",
    }

    monkeypatch.setattr(sandbox_host, "_user_for", lambda value: None)
    assert sandbox_host._owner_identity(env) == {
        "backend": "SimpleNamespace", "container": "recorded-container",
        "docker": "docker.exe", "user": None,
    }


@pytest.mark.parametrize(
    ("user_field", "expected_prefix"),
    [
        ("pn", ["docker.exe", "exec", "--user", "pn"]),
        (None, ["docker.exe", "exec"]),
        ("missing", ["docker.exe", "exec", "--user", "pn"]),
    ],
    ids=["explicit-pn", "docker-default", "legacy-marker"],
)
def test_marker_status_probes_use_recorded_user(monkeypatch, user_field, expected_prefix):
    marker = {
        "backend": "DockerEnvironment", "container": "recorded-container", "docker": "docker.exe",
        "dir": "/tmp/hermes-bot-desktop/lena-tabwright",
    }
    if user_field != "missing":
        marker["user"] = user_field
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "Published" in argv[-1] or "launcher.pid" in argv[-1]:
            output = b"DISPLAY=:20\nXAUTHORITY=/tmp/xauth\n"
        else:
            output = b"Xvnc\n"
        return subprocess.CompletedProcess(argv, 0, output, b"")

    monkeypatch.setattr(sandbox_host.subprocess, "run", run)

    published = sandbox_host.published_env_from_marker(marker)
    missing = sandbox_host.docker_marker_missing_binaries(marker)

    assert published == {"DISPLAY": ":20", "XAUTHORITY": "/tmp/xauth"}
    assert missing == ["Xvnc"]
    assert len(calls) == 2
    for argv, kwargs in calls:
        assert argv[:len(expected_prefix)] == expected_prefix
        assert argv[len(expected_prefix):len(expected_prefix) + 2] == ["recorded-container", "bash"]
        assert kwargs["check"] is False


def test_published_env_from_marker_execs_only_the_recorded_container(monkeypatch):
    marker = {
        "backend": "DockerEnvironment",
        "container": "recorded-container",
        "docker": "docker.exe",
        "dir": "/tmp/hermes-bot-desktop/lena-tabwright",
        "user": "pn",
    }
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b"DISPLAY=:20\nXAUTHORITY=/tmp/xauth\n", b"")

    monkeypatch.setattr(sandbox_host.subprocess, "run", run)

    assert sandbox_host.published_env_from_marker(marker) == {
        "DISPLAY": ":20", "XAUTHORITY": "/tmp/xauth"}
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:6] == ["docker.exe", "exec", "--user", "pn", "recorded-container", "bash"]
    assert "/tmp/hermes-bot-desktop/lena-tabwright" in argv[-1]
    assert kwargs["timeout"] == 15 and kwargs["check"] is False


def test_published_env_from_marker_fails_closed_on_missing_or_invalid_owner(monkeypatch):
    calls = []
    monkeypatch.setattr(sandbox_host.subprocess, "run", lambda *args, **kwargs: calls.append(args))

    assert sandbox_host.published_env_from_marker({}) == {}
    assert sandbox_host.published_env_from_marker(
        {"backend": "SSHEnvironment", "container": "id", "docker": "docker", "dir": "/tmp/screen"}) == {}
    assert calls == []


def test_published_env_from_marker_returns_empty_when_launcher_probe_fails(monkeypatch):
    marker = {
        "backend": "DockerEnvironment", "container": "recorded-container", "docker": "docker",
        "dir": "/tmp/hermes-bot-desktop/lena-tabwright",
    }
    monkeypatch.setattr(sandbox_host.subprocess, "run",
                        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, b"", b"not found"))

    assert sandbox_host.published_env_from_marker(marker) == {}


def test_published_env_from_marker_handles_docker_errors(monkeypatch):
    marker = {
        "backend": "DockerEnvironment", "container": "recorded-container", "docker": "docker",
        "dir": "/tmp/hermes-bot-desktop/lena-tabwright",
    }

    def fail_run(*args, **kwargs):
        raise OSError("docker unavailable")

    monkeypatch.setattr(sandbox_host.subprocess, "run", fail_run)

    assert sandbox_host.published_env_from_marker(marker) == {}


def test_standalone_status_uses_the_recorded_docker_container(monkeypatch):
    marker = {
        "backend": "DockerEnvironment", "container": "recorded-container", "docker": "docker",
        "dir": "/tmp/hermes-bot-desktop/lena-tabwright",
    }
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: marker)
    monkeypatch.setattr(sandbox_host, "marker_sandbox_alive", lambda value: True)
    probed = []
    monkeypatch.setattr(sandbox_host, "docker_marker_missing_binaries",
                        lambda value: probed.append(("missing", value)) or [])
    monkeypatch.setattr(sandbox_host, "published_env_from_marker",
                        lambda value: probed.append(("published", value)) or {"DISPLAY": ":20"})
    monkeypatch.setattr(runtime, "_sandbox_env",
                        lambda **kwargs: pytest.fail("status must not look up or create another terminal env"))

    status = runtime._sandbox_status("lena-tabwright", placement.Placement(placement.TERMINAL, "docker"))

    assert status.running and status.display == ":20" and status.profile == "lena-tabwright"
    assert probed == [("missing", marker), ("published", marker)]


def test_standalone_status_reflects_missing_marker_owned_docker_stack(monkeypatch):
    marker = {
        "backend": "DockerEnvironment", "container": "recorded-container", "docker": "docker",
        "dir": "/tmp/hermes-bot-desktop/lena-tabwright",
    }
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: marker)
    monkeypatch.setattr(sandbox_host, "marker_sandbox_alive", lambda value: True)
    monkeypatch.setattr(sandbox_host, "docker_marker_missing_binaries", lambda value: ["Xvnc"])
    monkeypatch.setattr(sandbox_host, "published_env_from_marker", lambda value: {})

    status = runtime._sandbox_status("lena-tabwright", placement.Placement(placement.TERMINAL, "docker"))

    assert not status.running and status.blocker and "Xvnc" in status.blocker
    assert "placement: gateway" in status.blocker  # non-running Docker marker retains image guidance.
