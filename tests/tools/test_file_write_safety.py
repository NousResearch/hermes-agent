"""Tests for file write safety and HERMES_WRITE_SAFE_ROOT sandboxing."""

import os
from pathlib import Path

import pytest

from tools.file_operations import _is_write_denied


class TestStaticDenyList:
    """Basic sanity checks for the static write deny list."""

    def test_temp_file_not_denied_by_default(self, tmp_path: Path):
        target = tmp_path / "regular.txt"
        # By default, arbitrary temp files should not be denied by the static list.
        assert _is_write_denied(str(target)) is False


class TestSafeWriteRoot:
    """HERMES_WRITE_SAFE_ROOT should sandbox writes to a specific subtree."""

    def test_writes_inside_safe_root_are_allowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        safe_root = tmp_path / "workspace"
        child = safe_root / "subdir" / "file.txt"

        # Simulate a workspace directory
        os.makedirs(child.parent, exist_ok=True)

        # Point safe root at the workspace
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

        # Writes inside the safe root (including nested dirs) are allowed
        assert _is_write_denied(str(child)) is False

    def test_writes_outside_safe_root_are_denied(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        safe_root = tmp_path / "workspace"
        outside = tmp_path / "other" / "file.txt"

        os.makedirs(safe_root, exist_ok=True)
        os.makedirs(outside.parent, exist_ok=True)

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

        # Any path outside the configured safe root must be denied
        assert _is_write_denied(str(outside)) is True

    def test_safe_root_env_ignores_empty_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        target = tmp_path / "regular.txt"
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "")

        # Empty env var should behave as if the feature is disabled
        assert _is_write_denied(str(target)) is False

