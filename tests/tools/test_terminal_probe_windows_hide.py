"""Regression tests for #81039 — Windows console window flash suppression.

When Hermes spawns a subprocess on Windows without ``creationflags`` set,
a black conhost window briefly appears on the user's desktop. Several
short-lived probe subprocess calls in ``tools.terminal_tool`` (Docker /
Singularity version checks) used to omit the flag, even though the
shared :func:`hermes_cli._subprocess_compat.windows_hide_flags` helper
already exists and is used by sibling call sites.

The fix routes the two probe calls through a tiny wrapper
(:func:`tools.terminal_tool._windows_hide_kwargs`) that returns
``{"creationflags": windows_hide_flags()}`` on Windows and an empty dict
on POSIX so the same call sites splat it unconditionally without leaking
POSIX-only kwargs.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest


def test_windows_hide_kwargs_returns_empty_on_posix(monkeypatch):
    """On POSIX platforms the wrapper must return an empty dict so it can
    be splatted unconditionally without affecting the subprocess call."""
    monkeypatch.setattr(sys, "platform", "linux")
    from tools.terminal_tool import _windows_hide_kwargs
    assert _windows_hide_kwargs() == {}


def test_windows_hide_kwargs_returns_creationflags_on_windows(monkeypatch):
    """On Windows the wrapper returns ``{"creationflags": <int>}`` where
    the int is whatever :func:`windows_hide_flags` returns."""
    monkeypatch.setattr(sys, "platform", "win32")
    sentinel_flag = 0x08000000  # CREATE_NO_WINDOW

    def fake_windows_hide_flags():
        return sentinel_flag

    # Patch the lazy import target (the helper imports inside the
    # function body to avoid a hard dependency from non-Windows
    # environments).
    fake_module = type(sys)("fake_subprocess_compat")
    fake_module.windows_hide_flags = fake_windows_hide_flags  # type: ignore[attr-defined]

    from tools.terminal_tool import _windows_hide_kwargs
    with patch.dict("sys.modules", {"hermes_cli._subprocess_compat": fake_module}):
        kwargs = _windows_hide_kwargs()
    assert kwargs == {"creationflags": sentinel_flag}


def test_docker_probe_passes_creationflags_on_windows(monkeypatch):
    """The Docker version probe must forward ``creationflags`` to
    ``subprocess.run`` on Windows. We don't actually need to invoke
    docker — we patch ``subprocess.run`` to capture the kwargs.
    """
    from tools.environments import docker as docker_mod
    monkeypatch.setattr(docker_mod, "find_docker", lambda: "docker")

    sentinel_flag = 0x08000000
    monkeypatch.setattr(sys, "platform", "win32")

    fake_module = type(sys)("fake_subprocess_compat")
    fake_module.windows_hide_flags = lambda: sentinel_flag  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli._subprocess_compat", fake_module)

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Force the function under test down the docker branch by configuring
    # env_type=docker on its config source.
    from tools.terminal_tool import check_terminal_requirements

    with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "docker"}):
        assert check_terminal_requirements() is True
    assert captured["kwargs"].get("creationflags") == sentinel_flag


def test_docker_probe_silent_on_posix(monkeypatch):
    """On POSIX the wrapper contributes no kwargs, so the subprocess
    call shape matches the previous (broken) call exactly."""
    from tools.environments import docker as docker_mod
    monkeypatch.setattr(docker_mod, "find_docker", lambda: "docker")
    monkeypatch.setattr(sys, "platform", "linux")

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    from tools.terminal_tool import check_terminal_requirements

    with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "docker"}):
        assert check_terminal_requirements() is True
    # No creationflags on POSIX — this is the contract that lets the
    # call site splat the wrapper unconditionally.
    assert "creationflags" not in captured["kwargs"]