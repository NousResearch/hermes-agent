"""Tests for MSYS path normalization in file_tools._resolve_path_for_task.

On Windows + Git Bash/MSYS2, the LLM may pass MSYS-style paths like
``/d/Project/file.sh`` to file tools. Without normalization, Python's
pathlib treats ``/d/Project`` as "root \\ on the current drive" and
resolves it to e.g. ``D:\\d\\Project`` instead of ``D:\\Project``.

Regression test for #44726.
"""

from pathlib import Path
from unittest.mock import patch

from tools import file_tools as ft_mod
from tools.environments import local as local_mod


class TestResolvePathMsysNormalization:
    """_resolve_path_for_task normalizes MSYS paths before Path construction."""

    def test_msys_drive_path_normalized_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        with patch.object(ft_mod, "_resolve_base_dir", return_value=Path("C:/fallback")):
            result = ft_mod._resolve_path_for_task("/d/Project/file.sh")
        assert result == Path("D:\\Project\\file.sh").resolve()

    def test_msys_c_drive_path_normalized_on_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        with patch.object(ft_mod, "_resolve_base_dir", return_value=Path("C:/fallback")):
            result = ft_mod._resolve_path_for_task("/c/Users/dev/code.py")
        assert result == Path("C:\\Users\\dev\\code.py").resolve()

    def test_native_windows_path_unchanged(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        with patch.object(ft_mod, "_resolve_base_dir", return_value=Path("C:/fallback")):
            result = ft_mod._resolve_path_for_task("D:\\Project\\file.sh")
        assert result == Path("D:\\Project\\file.sh").resolve()

    def test_noop_on_non_windows(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        with patch.object(ft_mod, "_resolve_base_dir", return_value=Path("/home/user")):
            result = ft_mod._resolve_path_for_task("/d/Project/file.sh")
        assert result == Path("/d/Project/file.sh").resolve()
