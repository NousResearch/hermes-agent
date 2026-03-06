"""Tests for hermes_cli/clipboard.py — clipboard image extraction."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.clipboard import (
    save_clipboard_image,
    _linux_save,
    _macos_pngpaste,
    _macos_osascript,
)


class TestSaveClipboardImage:
    """Platform dispatch."""

    def test_dispatches_to_macos_on_darwin(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch("hermes_cli.clipboard._macos_save", return_value=False) as mock_mac:
                save_clipboard_image(dest)
                mock_mac.assert_called_once_with(dest)

    def test_dispatches_to_linux_on_linux(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("hermes_cli.clipboard._linux_save", return_value=False) as mock_linux:
                save_clipboard_image(dest)
                mock_linux.assert_called_once_with(dest)

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "deep" / "nested" / "out.png"
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("hermes_cli.clipboard._linux_save", return_value=False):
                save_clipboard_image(dest)
        assert dest.parent.exists()


class TestMacosPngpaste:
    def test_success(self, tmp_path):
        dest = tmp_path / "out.png"
        dest.write_bytes(b"fake png data")  # simulate pngpaste writing
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _macos_pngpaste(dest) is True

    def test_not_installed(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _macos_pngpaste(dest) is False

    def test_no_image_in_clipboard(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert _macos_pngpaste(dest) is False


class TestMacosOsascript:
    def test_no_image_type_in_clipboard(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="«class ut16», «class utf8»", returncode=0
            )
            assert _macos_osascript(dest) is False

    def test_clipboard_info_check_fails(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=Exception("fail")):
            assert _macos_osascript(dest) is False


class TestLinuxSave:
    def test_no_xclip_installed(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _linux_save(dest) is False

    def test_no_image_in_clipboard(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="text/plain\n", returncode=0)
            assert _linux_save(dest) is False

    def test_image_in_clipboard(self, tmp_path):
        dest = tmp_path / "out.png"
        
        def fake_run(cmd, **kwargs):
            if "TARGETS" in cmd:
                return MagicMock(stdout="image/png\ntext/plain\n", returncode=0)
            # Extract call — write fake data
            if "stdout" in kwargs and kwargs["stdout"]:
                kwargs["stdout"].write(b"fake png")
            return MagicMock(returncode=0)
        
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            # Create the file to simulate xclip writing
            dest.write_bytes(b"fake png")
            assert _linux_save(dest) is True
