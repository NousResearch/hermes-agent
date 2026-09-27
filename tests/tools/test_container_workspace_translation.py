"""Regression: container-mode file tools must translate the Windows host workspace
root to POSIX ``/workspace`` instead of POSIX-joining a drive-letter path onto the
process cwd (the doubled-path defect behind FORGE activation failure).

The tests exercise the resolution seam directly with ``container_paths=True`` —
no Docker daemon required — and prove normalize(normalize(path)) == normalize(path).
"""

import os

import pytest

from tools import file_tools_paths


@pytest.mark.windows_only
class TestContainerWorkspaceTranslation:
    def test_host_workspace_root_maps_to_container_workspace(self, monkeypatch):
        """A Windows host workspace root must become /workspace, not a POSIX-joined string."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setattr(
            file_tools_paths, "_authoritative_workspace_root",
            lambda task_id="default": r"C:\ForgeTask\work-001",
        )
        base = file_tools_paths._resolve_base_dir(container_paths=True)
        assert str(base) == "/workspace"

    def test_relative_path_resolves_inside_container_workspace(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setattr(
            file_tools_paths, "_authoritative_workspace_root",
            lambda task_id="default": r"C:\ForgeTask\work-001",
        )
        result = file_tools_paths._resolve_path_for_task("app.py")
        assert str(result) == "/workspace/app.py"

    def test_host_absolute_path_inside_root_translates(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setattr(
            file_tools_paths, "_authoritative_workspace_root",
            lambda task_id="default": r"C:\ForgeTask\work-001",
        )
        result = file_tools_paths._resolve_path_for_task(r"C:\ForgeTask\work-001\sub\file.py")
        assert str(result) == "/workspace/sub/file.py"

    def test_host_absolute_path_outside_root_rejected(self, monkeypatch):
        """A drive-letter path outside the mounted workspace must not silently resolve."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setattr(
            file_tools_paths, "_authoritative_workspace_root",
            lambda task_id="default": r"C:\ForgeTask\work-001",
        )
        with pytest.raises(ValueError, match="outside"):
            file_tools_paths._resolve_path_for_task(r"C:\Users\other\secret.txt")

    def test_already_container_path_passes_through(self, monkeypatch):
        """Idempotency: /workspace/... normalized twice is unchanged."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setattr(
            file_tools_paths, "_authoritative_workspace_root",
            lambda task_id="default": r"C:\ForgeTask\work-001",
        )
        once = file_tools_paths._resolve_path_for_task("/workspace/app.py")
        twice = file_tools_paths._resolve_path_for_task(str(once))
        assert str(once) == "/workspace/app.py"
        assert str(twice) == str(once)
