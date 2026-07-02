"""Regression tests for Docker cold-start relative-path resolution.

The bug (#54354): on a container backend (docker/singularity/modal/daytona),
the very first relative file-tool path — before any terminal command has run,
so the live cwd registry is still empty and no TERMINAL_CWD/last-known anchor
exists — fell back to the HOST process cwd. That anchored container-bound
writes on the host filesystem.

Fix: when there is no authoritative workspace root AND the backend is a
container, anchor on the sanitized container cwd (e.g. ``/root``) from
``terminal_tool._get_env_config()`` instead of the host getcwd. Local/ssh keep
host getcwd. No Docker daemon required — only the path-resolution math runs.
"""

import os
from pathlib import Path

import pytest

import tools.file_tools as ft
import tools.terminal_tool as terminal_tool


@pytest.fixture
def _coldstart(tmp_path, monkeypatch):
    """Cold-start state: empty registries, no anchors, host cwd = decoy."""
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "file.py").write_text("HOST_DECOY\n")
    monkeypatch.chdir(decoy)
    # No live tracking, no overrides, no last-known, no configured TERMINAL_CWD.
    monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(ft, "_last_known_cwd", {})
    monkeypatch.setattr(ft, "_file_ops_cache", {})
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    return decoy


def test_docker_coldstart_resolves_in_container_not_host(_coldstart, monkeypatch):
    """First relative path on docker cold start anchors at /root, not host."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    resolved = ft._resolve_path_for_task("file.py", task_id="default")

    assert resolved == Path("/root/file.py")
    assert not str(resolved).startswith(str(_coldstart))


def test_local_coldstart_resolves_under_host_cwd(_coldstart, monkeypatch):
    """Local backend keeps host-cwd anchoring (unchanged behavior)."""
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    resolved = ft._resolve_path_for_task("file.py", task_id="default")

    assert resolved == (_coldstart / "file.py").resolve()


def test_local_explicit_coldstart_resolves_under_host_cwd(_coldstart, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")

    resolved = ft._resolve_path_for_task("file.py", task_id="default")

    assert resolved == (_coldstart / "file.py").resolve()
