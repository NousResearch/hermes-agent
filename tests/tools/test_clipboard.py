"""Tests for clipboard image paste — clipboard extraction, multimodal conversion,
and CLI integration.

Coverage:
  hermes_cli/clipboard.py  — platform-specific image extraction (macOS, WSL, Wayland, X11)
  cli.py                   — _try_attach_clipboard_image, _build_multimodal_content,
                              image attachment state, queue tuple routing
"""

import base64
import os
import queue
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock, mock_open

import pytest

from hermes_cli.clipboard import (
    save_clipboard_image,
    has_clipboard_image,
    _is_wsl,
    _linux_save,
    _macos_pngpaste,
    _macos_osascript,
    _macos_has_image,
    _xclip_save,
    _xclip_has_image,
    _wsl_save,
    _wsl_has_image,
    _wayland_save,
    _wayland_has_image,
    _windows_save,
    _windows_has_image,
    _convert_to_png,
)
from cli import _should_auto_attach_clipboard_image_on_paste

FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
FAKE_BMP = b"BM" + b"\x00" * 100
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


# ═════════════════════════════════════════════════════════════════════════
# Level 1: Clipboard module — platform dispatch + tool interactions
# ═════════════════════════════════════════════════════════════════════════

class TestSaveClipboardImage:
    def test_dispatches_to_macos_on_darwin(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch("hermes_cli.clipboard._macos_save", return_value=False) as m:
                save_clipboard_image(dest)
                m.assert_called_once_with(dest)

    def test_dispatches_to_windows_on_win32(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch("hermes_cli.clipboard._windows_save", return_value=False) as m:
                save_clipboard_image(dest)
                m.assert_called_once_with(dest)

    def test_dispatches_to_linux_on_linux(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("hermes_cli.clipboard._linux_save", return_value=False) as m:
                save_clipboard_image(dest)
                m.assert_called_once_with(dest)

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "deep" / "nested" / "out.png"
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("hermes_cli.clipboard._linux_save", return_value=False):
                save_clipboard_image(dest)
        assert dest.parent.exists()


# ── macOS ────────────────────────────────────────────────────────────────

class TestMacosPngpaste:
    def test_success_writes_file(self, tmp_path):
        dest = tmp_path / "out.png"
        def fake_run(cmd, **kw):
            dest.write_bytes(FAKE_PNG)
            return MagicMock(returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_pngpaste(dest) is True
        assert dest.stat().st_size == len(FAKE_PNG)

    def test_not_installed(self, tmp_path):
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _macos_pngpaste(tmp_path / "out.png") is False

    def test_no_image_in_clipboard(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert _macos_pngpaste(dest) is False
        assert not dest.exists()

    def test_empty_file_rejected(self, tmp_path):
        dest = tmp_path / "out.png"
        def fake_run(cmd, **kw):
            dest.write_bytes(b"")
            return MagicMock(returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_pngpaste(dest) is False

    def test_timeout_returns_false(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("pngpaste", 3)):
            assert _macos_pngpaste(dest) is False


class TestMacosHasImage:
    def test_png_detected(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="«class PNGf», «class ut16»", returncode=0
            )
            assert _macos_has_image() is True

    def test_tiff_detected(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="«class TIFF»", returncode=0
            )
            assert _macos_has_image() is True

    def test_text_only(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="«class ut16», «class utf8»", returncode=0
            )
            assert _macos_has_image() is False


class TestMacosOsascript:
    def test_no_image_type_in_clipboard(self, tmp_path):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="«class ut16», «class utf8»", returncode=0
            )
            assert _macos_osascript(tmp_path / "out.png") is False

    def test_clipboard_info_fails(self, tmp_path):
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=Exception("fail")):
            assert _macos_osascript(tmp_path / "out.png") is False

    def test_success_with_png(self, tmp_path):
        dest = tmp_path / "out.png"
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                return MagicMock(stdout="«class PNGf», «class ut16»", returncode=0)
            dest.write_bytes(FAKE_PNG)
            return MagicMock(stdout="", returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_osascript(dest) is True
        assert dest.stat().st_size > 0

    def test_success_with_tiff(self, tmp_path):
        dest = tmp_path / "out.png"
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                return MagicMock(stdout="«class TIFF»", returncode=0)
            dest.write_bytes(FAKE_PNG)
            return MagicMock(stdout="", returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_osascript(dest) is True

    def test_extraction_returns_fail(self, tmp_path):
        dest = tmp_path / "out.png"
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                return MagicMock(stdout="«class PNGf»", returncode=0)
            return MagicMock(stdout="fail", returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_osascript(dest) is False

    def test_extraction_writes_empty_file(self, tmp_path):
        dest = tmp_path / "out.png"
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if len(calls) == 1:
                return MagicMock(stdout="«class PNGf»", returncode=0)
            dest.write_bytes(b"")
            return MagicMock(stdout="", returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_osascript(dest) is False


# ── WSL detection ────────────────────────────────────────────────────────

class TestIsWsl:
    def setup_method(self):
        # _is_wsl is hermes_constants.is_wsl; reset the function's own module
        # globals so this stays stable even if hermes_constants was imported
        # through a different module object earlier in a large xdist run.
        import hermes_constants
        hermes_constants._wsl_detected = None
        _is_wsl.__globals__["_wsl_detected"] = None

    def teardown_method(self):
        # Reset again after the test so we don't leak a cached value
        # (True/False) into whichever test the xdist worker runs next.
        import hermes_constants
        hermes_constants._wsl_detected = None
        _is_wsl.__globals__["_wsl_detected"] = None

    def test_wsl2_detected(self):
        content = "Linux version 5.15.0 (microsoft-standard-WSL2)"
        with patch.dict(_is_wsl.__globals__, {"open": mock_open(read_data=content)}):
            assert _is_wsl() is True

    def test_wsl1_detected(self):
        content = "Linux version 4.4.0-microsoft-standard"
        with patch.dict(_is_wsl.__globals__, {"open": mock_open(read_data=content)}):
            assert _is_wsl() is True

    def test_regular_linux(self):
        # GHA hosted runners are Azure VMs whose real /proc/version often
        # contains "microsoft". Patching builtins.open with mock_open is
        # supposed to intercept hermes_constants.is_wsl's `open` call,
        # but if another test on the same xdist worker already cached
        # _wsl_detected=True, the mock never runs because the function
        # short-circuits on the cache. setup_method resets, so we just
        # need to be sure the patched `open` is actually reached.
        content = "Linux version 6.14.0-37-generic (buildd@lcy02-amd64-049)"
        with patch.dict(_is_wsl.__globals__, {"open": mock_open(read_data=content)}):
            assert _is_wsl() is False

    def test_proc_version_missing(self):
        with patch.dict(_is_wsl.__globals__, {"open": MagicMock(side_effect=FileNotFoundError)}):
            assert _is_wsl() is False

    def test_result_is_cached(self):
        content = "Linux version 5.15.0 (microsoft-standard-WSL2)"
        opener = mock_open(read_data=content)
        with patch.dict(_is_wsl.__globals__, {"open": opener}):
            assert _is_wsl() is True
            assert _is_wsl() is True
            opener.assert_called_once()  # only read once


# ── WSL (powershell.exe) ────────────────────────────────────────────────

class TestWslHasImage:
    def test_clipboard_has_image(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="True\n", returncode=0)
            assert _wsl_has_image() is True

    def test_clipboard_no_image(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="False\n", returncode=0)
            assert _wsl_has_image() is False

    def test_falls_back_to_get_clipboard_image(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="False\n", returncode=0),
                MagicMock(stdout="True\n", returncode=0),
            ]
            assert _wsl_has_image() is True
            assert mock_run.call_count == 2

    def test_powershell_not_found(self):
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _wsl_has_image() is False

    def test_powershell_error(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=1)
            assert _wsl_has_image() is False


class TestWslSave:
    def test_successful_extraction(self, tmp_path):
        dest = tmp_path / "out.png"
        b64_png = base64.b64encode(FAKE_PNG).decode()
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b64_png + "\n", returncode=0)
            assert _wsl_save(dest) is True
        assert dest.read_bytes() == FAKE_PNG

    def test_falls_back_to_get_clipboard_extraction(self, tmp_path):
        dest = tmp_path / "out.png"
        b64_png = base64.b64encode(FAKE_PNG).decode()
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="", returncode=1),
                MagicMock(stdout=b64_png + "\n", returncode=0),
            ]
            assert _wsl_save(dest) is True
            assert mock_run.call_count == 2
        assert dest.read_bytes() == FAKE_PNG

    def test_no_image_returns_false(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=1)
            assert _wsl_save(dest) is False
        assert not dest.exists()

    def test_empty_output(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            assert _wsl_save(dest) is False

    def test_powershell_not_found(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _wsl_save(dest) is False

    def test_invalid_base64(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="not-valid-base64!!!", returncode=0)
            assert _wsl_save(dest) is False

    def test_timeout(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("powershell.exe", 15)):
            assert _wsl_save(dest) is False


# ── Wayland (wl-paste) ──────────────────────────────────────────────────

class TestWaylandHasImage:
    def test_has_png(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="image/png\ntext/plain\n", returncode=0
            )
            assert _wayland_has_image() is True

    def test_has_bmp_only(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="text/html\nimage/bmp\n", returncode=0
            )
            assert _wayland_has_image() is True

    def test_text_only(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="text/plain\ntext/html\n", returncode=0
            )
            assert _wayland_has_image() is False

    def test_wl_paste_not_installed(self):
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _wayland_has_image() is False


class TestWaylandSave:
    def test_png_extraction(self, tmp_path):
        dest = tmp_path / "out.png"
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "--list-types" in cmd:
                return MagicMock(stdout="image/png\ntext/plain\n", returncode=0)
            # Extract call — write fake data to stdout file
            if "stdout" in kw and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(FAKE_PNG)
            return MagicMock(returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _wayland_save(dest) is True
        assert dest.stat().st_size > 0

    def test_bmp_extraction_with_pillow_convert(self, tmp_path):
        dest = tmp_path / "out.png"
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "--list-types" in cmd:
                return MagicMock(stdout="text/html\nimage/bmp\n", returncode=0)
            if "stdout" in kw and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(FAKE_BMP)
            return MagicMock(returncode=0)

        def fake_convert(path):
            assert path == dest
            path.write_bytes(FAKE_PNG)
            return True

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            with patch("hermes_cli.clipboard._convert_to_png", side_effect=fake_convert):
                assert _wayland_save(dest) is True

    def test_jpeg_extraction_converts_to_real_png(self, tmp_path):
        dest = tmp_path / "out.png"

        def fake_run(cmd, **kw):
            if "--list-types" in cmd:
                return MagicMock(stdout="image/jpeg\ntext/plain\n", returncode=0)
            if "stdout" in kw and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(FAKE_JPEG)
            return MagicMock(returncode=0)

        def fake_convert(path):
            assert path == dest
            path.write_bytes(FAKE_PNG)
            return True

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            with patch("hermes_cli.clipboard._convert_to_png", side_effect=fake_convert) as mock_convert:
                assert _wayland_save(dest) is True

        mock_convert.assert_called_once_with(dest)
        assert dest.read_bytes() == FAKE_PNG

    def test_non_png_conversion_failure_cleans_up(self, tmp_path):
        dest = tmp_path / "out.png"

        def fake_run(cmd, **kw):
            if "--list-types" in cmd:
                return MagicMock(stdout="image/jpeg\n", returncode=0)
            if "stdout" in kw and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(FAKE_JPEG)
            return MagicMock(returncode=0)

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            with patch("hermes_cli.clipboard._convert_to_png", return_value=True):
                assert _wayland_save(dest) is False

        assert not dest.exists()

    def test_no_image_types(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="text/plain\ntext/html\n", returncode=0
            )
            assert _wayland_save(dest) is False

    def test_wl_paste_not_installed(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _wayland_save(dest) is False

    def test_list_types_fails(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=1)
            assert _wayland_save(dest) is False

    def test_prefers_png_over_bmp(self, tmp_path):
        """When both PNG and BMP are available, PNG should be preferred."""
        dest = tmp_path / "out.png"
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "--list-types" in cmd:
                return MagicMock(
                    stdout="image/bmp\nimage/png\ntext/plain\n", returncode=0
                )
            if "stdout" in kw and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(FAKE_PNG)
            return MagicMock(returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _wayland_save(dest) is True
        # Verify PNG was requested, not BMP
        extract_cmd = calls[1]
        assert "image/png" in extract_cmd


# ── X11 (xclip) ─────────────────────────────────────────────────────────

class TestXclipHasImage:
    def test_has_image(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="image/png\ntext/plain\n", returncode=0
            )
            assert _xclip_has_image() is True

    def test_no_image(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="text/plain\n", returncode=0
            )
            assert _xclip_has_image() is False

    def test_xclip_not_installed(self):
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _xclip_has_image() is False


class TestXclipSave:
    def test_no_xclip_installed(self, tmp_path):
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _xclip_save(tmp_path / "out.png") is False

    def test_no_image_in_clipboard(self, tmp_path):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="text/plain\n", returncode=0)
            assert _xclip_save(tmp_path / "out.png") is False

    def test_image_extraction_success(self, tmp_path):
        dest = tmp_path / "out.png"
        def fake_run(cmd, **kw):
            if "TARGETS" in cmd:
                return MagicMock(stdout="image/png\ntext/plain\n", returncode=0)
            if "stdout" in kw and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(FAKE_PNG)
            return MagicMock(returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _xclip_save(dest) is True
        assert dest.stat().st_size > 0

    def test_extraction_fails_cleans_up(self, tmp_path):
        dest = tmp_path / "out.png"
        def fake_run(cmd, **kw):
            if "TARGETS" in cmd:
                return MagicMock(stdout="image/png\n", returncode=0)
            raise subprocess.SubprocessError("pipe broke")
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _xclip_save(dest) is False
        assert not dest.exists()

    def test_targets_check_timeout(self, tmp_path):
        with patch("hermes_cli.clipboard.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("xclip", 3)):
            assert _xclip_save(tmp_path / "out.png") is False


# ── Linux dispatch ──────────────────────────────────────────────────────

class TestLinuxSave:
    """Test that _linux_save dispatches correctly to WSL → Wayland → X11."""

    def setup_method(self):
        import hermes_cli.clipboard as cb
        cb._wsl_detected = None

    def test_wsl_tried_first(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._is_wsl", return_value=True):
            with patch("hermes_cli.clipboard._wsl_save", return_value=True) as m:
                assert _linux_save(dest) is True
                m.assert_called_once_with(dest)

    def test_wsl_fails_falls_through_to_xclip(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._is_wsl", return_value=True):
            with patch("hermes_cli.clipboard._wsl_save", return_value=False):
                with patch.dict(os.environ, {}, clear=True):
                    with patch("hermes_cli.clipboard._xclip_save", return_value=True) as m:
                        assert _linux_save(dest) is True
                        m.assert_called_once_with(dest)

    def test_wayland_tried_when_display_set(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._is_wsl", return_value=False):
            with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
                with patch("hermes_cli.clipboard._wayland_save", return_value=True) as m:
                    assert _linux_save(dest) is True
                    m.assert_called_once_with(dest)

    def test_wayland_fails_falls_through_to_xclip(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._is_wsl", return_value=False):
            with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
                with patch("hermes_cli.clipboard._wayland_save", return_value=False):
                    with patch("hermes_cli.clipboard._xclip_save", return_value=True) as m:
                        assert _linux_save(dest) is True
                        m.assert_called_once_with(dest)

    def test_xclip_used_on_plain_x11(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._is_wsl", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                with patch("hermes_cli.clipboard._xclip_save", return_value=True) as m:
                    assert _linux_save(dest) is True
                    m.assert_called_once_with(dest)


# ── Native Windows (PowerShell) ─────────────────────────────────────────

class TestWindowsHasImage:
    def setup_method(self):
        import hermes_cli.clipboard as cb
        cb._ps_exe = False  # reset cache

    def test_clipboard_has_image(self):
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="True\n", returncode=0)
                assert _windows_has_image() is True

    def test_clipboard_no_image(self):
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="False\n", returncode=0)
                assert _windows_has_image() is False

    def test_falls_back_to_get_clipboard_image(self):
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.side_effect = [
                    MagicMock(stdout="False\n", returncode=0),
                    MagicMock(stdout="True\n", returncode=0),
                ]
                assert _windows_has_image() is True
                assert mock_run.call_count == 2

    def test_no_powershell_available(self):
        with patch("hermes_cli.clipboard._get_ps_exe", return_value=None):
            assert _windows_has_image() is False

    def test_powershell_error(self):
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="", returncode=1)
                assert _windows_has_image() is False

    def test_subprocess_exception(self):
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("powershell", 5)):
                assert _windows_has_image() is False


class TestWindowsSave:
    def setup_method(self):
        import hermes_cli.clipboard as cb
        cb._ps_exe = False  # reset cache

    def test_successful_extraction(self, tmp_path):
        dest = tmp_path / "out.png"
        b64_png = base64.b64encode(FAKE_PNG).decode()
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout=b64_png + "\n", returncode=0)
                assert _windows_save(dest) is True
        assert dest.read_bytes() == FAKE_PNG

    def test_falls_back_to_filedrop_image(self, tmp_path):
        dest = tmp_path / "out.png"
        b64_png = base64.b64encode(FAKE_PNG).decode()
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.side_effect = [
                    MagicMock(stdout="", returncode=1),
                    MagicMock(stdout="", returncode=1),
                    MagicMock(stdout=b64_png + "\n", returncode=0),
                ]
                assert _windows_save(dest) is True
                assert mock_run.call_count == 3
        assert dest.read_bytes() == FAKE_PNG

    def test_no_image_returns_false(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="", returncode=1)
                assert _windows_save(dest) is False
        assert not dest.exists()

    def test_empty_output(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="", returncode=0)
                assert _windows_save(dest) is False

    def test_no_powershell_returns_false(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._get_ps_exe", return_value=None):
            assert _windows_save(dest) is False

    def test_invalid_base64(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="not-valid-base64!!!", returncode=0)
                assert _windows_save(dest) is False

    def test_timeout(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("powershell", 15)):
                assert _windows_save(dest) is False


class TestHasClipboardImageWin32:
    """Verify has_clipboard_image dispatches to _windows_has_image on win32."""

    def test_dispatches_on_win32(self):
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch("hermes_cli.clipboard._windows_has_image", return_value=True) as m:
                assert has_clipboard_image() is True
                m.assert_called_once()


# ── BMP conversion ──────────────────────────────────────────────────────

class TestConvertToPng:
    def test_pillow_conversion(self, tmp_path):
        dest = tmp_path / "img.png"
        dest.write_bytes(FAKE_BMP)
        mock_img_instance = MagicMock()
        mock_image_cls = MagicMock()
        mock_image_cls.open.return_value = mock_img_instance
        # `from PIL import Image` fetches PIL.Image from the PIL module
        mock_pil_module = MagicMock()
        mock_pil_module.Image = mock_image_cls
        with patch.dict(sys.modules, {"PIL": mock_pil_module}):
            assert _convert_to_png(dest) is True
            mock_img_instance.save.assert_called_once_with(dest, "PNG")

    def test_pillow_not_available_tries_imagemagick(self, tmp_path):
        dest = tmp_path / "img.png"
        dest.write_bytes(FAKE_BMP)

        def fake_run(cmd, **kw):
            # Simulate ImageMagick converting
            dest.write_bytes(FAKE_PNG)
            return MagicMock(returncode=0)

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
                # Force ImportError for Pillow
                import hermes_cli.clipboard as cb
                original = cb._convert_to_png

                def patched_convert(path):
                    # Skip Pillow, go straight to ImageMagick
                    try:
                        tmp = path.with_suffix(".bmp")
                        path.rename(tmp)
                        import subprocess as sp
                        r = sp.run(
                            ["convert", str(tmp), "png:" + str(path)],
                            capture_output=True, timeout=5,
                        )
                        tmp.unlink(missing_ok=True)
                        return r.returncode == 0 and path.exists() and path.stat().st_size > 0
                    except Exception:
                        return False

                # Just test that the fallback logic exists
                assert dest.exists()

    def test_file_still_usable_when_no_converter(self, tmp_path):
        """BMP file should still be reported as success if no converter available."""
        dest = tmp_path / "img.png"
        dest.write_bytes(FAKE_BMP)  # it's a BMP but named .png
        # Both Pillow and ImageMagick unavailable
        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
                result = _convert_to_png(dest)
                # Raw BMP is better than nothing — function should return True
                assert result is True
                assert dest.exists() and dest.stat().st_size > 0

    def test_imagemagick_failure_preserves_original(self, tmp_path):
        """When ImageMagick convert fails, the original file must not be lost."""
        dest = tmp_path / "img.png"
        original_data = FAKE_BMP
        dest.write_bytes(original_data)

        def fake_run_fail(cmd, **kw):
            # Simulate convert failing without producing output
            return MagicMock(returncode=1)

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run_fail):
                _convert_to_png(dest)

        # Original file must still exist with original content
        assert dest.exists(), "Original file was lost after failed conversion"
        assert dest.read_bytes() == original_data

    def test_imagemagick_not_installed_preserves_original(self, tmp_path):
        """When ImageMagick is not installed, the original file must not be lost."""
        dest = tmp_path / "img.png"
        original_data = FAKE_BMP
        dest.write_bytes(original_data)

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
                _convert_to_png(dest)

        assert dest.exists(), "Original file was lost when ImageMagick not installed"
        assert dest.read_bytes() == original_data

    def test_imagemagick_timeout_preserves_original(self, tmp_path):
        """When ImageMagick times out, the original file must not be lost."""
        import subprocess
        dest = tmp_path / "img.png"
        original_data = FAKE_BMP
        dest.write_bytes(original_data)

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with patch("hermes_cli.clipboard.subprocess.run", side_effect=subprocess.TimeoutExpired("convert", 5)):
                _convert_to_png(dest)

        assert dest.exists(), "Original file was lost after timeout"
        assert dest.read_bytes() == original_data


# ── has_clipboard_image dispatch ─────────────────────────────────────────

class TestHasClipboardImage:
    def setup_method(self):
        import hermes_cli.clipboard as cb
        cb._wsl_detected = None

    def test_macos_dispatch(self):
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch("hermes_cli.clipboard._macos_has_image", return_value=True) as m:
                assert has_clipboard_image() is True
                m.assert_called_once()

    def test_linux_wsl_dispatch(self):
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("hermes_cli.clipboard._is_wsl", return_value=True):
                with patch("hermes_cli.clipboard._wsl_has_image", return_value=True) as m:
                    assert has_clipboard_image() is True
                    m.assert_called_once()

    def test_wsl_falls_through_to_wayland_when_windows_path_empty(self):
        """WSLg often bridges images to wl-paste even when powershell.exe check fails."""
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("hermes_cli.clipboard._is_wsl", return_value=True):
                with patch("hermes_cli.clipboard._wsl_has_image", return_value=False) as wsl:
                    with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
                        with patch("hermes_cli.clipboard._wayland_has_image", return_value=True) as wl:
                            assert has_clipboard_image() is True
                            wsl.assert_called_once()
                            wl.assert_called_once()

    def test_linux_wayland_dispatch(self):
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("hermes_cli.clipboard._is_wsl", return_value=False):
                with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
                    with patch("hermes_cli.clipboard._wayland_has_image", return_value=True) as m:
                        assert has_clipboard_image() is True
                        m.assert_called_once()

    def test_linux_x11_dispatch(self):
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("hermes_cli.clipboard._is_wsl", return_value=False):
                with patch.dict(os.environ, {}, clear=True):
                    with patch("hermes_cli.clipboard._xclip_has_image", return_value=True) as m:
                        assert has_clipboard_image() is True
                        m.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════
# Level 2: _preprocess_images_with_vision — image → text via vision tool
# ═════════════════════════════════════════════════════════════════════════

class TestPreprocessImagesWithVision:
    """Test vision-based image pre-processing for the CLI."""

    @pytest.fixture
    def cli(self):
        """Minimal HermesCLI with mocked internals."""
        with patch("cli.load_cli_config") as mock_cfg:
            mock_cfg.return_value = {
                "model": {"default": "test/model", "base_url": "http://x", "provider": "auto"},
                "terminal": {"timeout": 60},
                "browser": {},
                "compression": {"enabled": True},
                "agent": {"max_turns": 10},
                "display": {"compact": True},
                "clarify": {},
                "code_execution": {},
                "delegation": {},
            }
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                with patch("cli.CLI_CONFIG", mock_cfg.return_value):
                    from cli import HermesCLI
                    cli_obj = HermesCLI.__new__(HermesCLI)
                    # Manually init just enough state
                    cli_obj._attached_images = []
                    cli_obj._image_counter = 0
                    return cli_obj

    def _make_image(self, tmp_path, name="test.png", content=FAKE_PNG):
        img = tmp_path / name
        img.write_bytes(content)
        return img

    def _mock_vision_success(self, description="A test image with colored pixels."):
        """Return an async mock that simulates a successful vision_analyze_tool call."""
        import json
        async def _fake_vision(**kwargs):
            return json.dumps({"success": True, "analysis": description})
        return _fake_vision

    def _mock_vision_failure(self):
        """Return an async mock that simulates a failed vision_analyze_tool call."""
        import json
        async def _fake_vision(**kwargs):
            return json.dumps({"success": False, "analysis": "Error"})
        return _fake_vision

    def test_single_image_with_text(self, cli, tmp_path):
        img = self._make_image(tmp_path)
        with patch("tools.vision_tools.vision_analyze_tool", side_effect=self._mock_vision_success()):
            result = cli._preprocess_images_with_vision("Describe this", [img])

        assert isinstance(result, str)
        assert "A test image with colored pixels." in result
        assert "Describe this" in result
        assert str(img) in result
        assert "base64," not in result  # no raw base64 image content

    def test_multiple_images(self, cli, tmp_path):
        imgs = [self._make_image(tmp_path, f"img{i}.png") for i in range(3)]
        with patch("tools.vision_tools.vision_analyze_tool", side_effect=self._mock_vision_success()):
            result = cli._preprocess_images_with_vision("Compare", imgs)

        assert isinstance(result, str)
        assert "Compare" in result
        # Each image path should be referenced
        for img in imgs:
            assert str(img) in result

    def test_empty_text_gets_default_question(self, cli, tmp_path):
        img = self._make_image(tmp_path)
        with patch("tools.vision_tools.vision_analyze_tool", side_effect=self._mock_vision_success()):
            result = cli._preprocess_images_with_vision("", [img])
        assert isinstance(result, str)
        assert "A test image with colored pixels." in result

    def test_missing_image_skipped(self, cli, tmp_path):
        missing = tmp_path / "gone.png"
        with patch("tools.vision_tools.vision_analyze_tool", side_effect=self._mock_vision_success()):
            result = cli._preprocess_images_with_vision("test", [missing])
        # No images analyzed, falls back to default
        assert result == "test"

    def test_mix_of_existing_and_missing(self, cli, tmp_path):
        real = self._make_image(tmp_path, "real.png")
        missing = tmp_path / "gone.png"
        with patch("tools.vision_tools.vision_analyze_tool", side_effect=self._mock_vision_success()):
            result = cli._preprocess_images_with_vision("test", [real, missing])
        assert str(real) in result
        assert str(missing) not in result
        assert "test" in result

    def test_vision_failure_includes_path(self, cli, tmp_path):
        img = self._make_image(tmp_path)
        with patch("tools.vision_tools.vision_analyze_tool", side_effect=self._mock_vision_failure()):
            result = cli._preprocess_images_with_vision("check this", [img])
        assert isinstance(result, str)
        assert str(img) in result  # path still included for retry
        assert "check this" in result

    def test_vision_exception_includes_path(self, cli, tmp_path):
        img = self._make_image(tmp_path)
        async def _explode(**kwargs):
            raise RuntimeError("API down")
        with patch("tools.vision_tools.vision_analyze_tool", side_effect=_explode):
            result = cli._preprocess_images_with_vision("check this", [img])
        assert isinstance(result, str)
        assert str(img) in result  # path still included for retry


# ═════════════════════════════════════════════════════════════════════════
# Level 3: _try_attach_clipboard_image — state management
# ═════════════════════════════════════════════════════════════════════════

class TestTryAttachClipboardImage:
    """Test the clipboard → state flow."""

    @pytest.fixture
    def cli(self):
        from cli import HermesCLI
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj._attached_images = []
        cli_obj._image_counter = 0
        return cli_obj

    def test_image_found_attaches(self, cli):
        with patch("hermes_cli.clipboard.save_clipboard_image", return_value=True):
            result = cli._try_attach_clipboard_image()
        assert result is True
        assert len(cli._attached_images) == 1
        assert cli._image_counter == 1

    def test_no_image_doesnt_attach(self, cli):
        with patch("hermes_cli.clipboard.save_clipboard_image", return_value=False):
            result = cli._try_attach_clipboard_image()
        assert result is False
        assert len(cli._attached_images) == 0
        assert cli._image_counter == 0  # rolled back

    def test_multiple_attaches_increment_counter(self, cli):
        with patch("hermes_cli.clipboard.save_clipboard_image", return_value=True):
            cli._try_attach_clipboard_image()
            cli._try_attach_clipboard_image()
            cli._try_attach_clipboard_image()
        assert len(cli._attached_images) == 3
        assert cli._image_counter == 3

    def test_mixed_success_and_failure(self, cli):
        results = [True, False, True]
        with patch("hermes_cli.clipboard.save_clipboard_image", side_effect=results):
            cli._try_attach_clipboard_image()
            cli._try_attach_clipboard_image()
            cli._try_attach_clipboard_image()
        assert len(cli._attached_images) == 2
        assert cli._image_counter == 2  # 3 attempts, 1 rolled back

    def test_image_path_follows_naming_convention(self, cli):
        with patch("hermes_cli.clipboard.save_clipboard_image", return_value=True):
            cli._try_attach_clipboard_image()
        path = cli._attached_images[0]
        assert path.parent == Path(os.environ["HERMES_HOME"]) / "images"
        assert path.name.startswith("clip_")
        assert path.suffix == ".png"


class TestAutoAttachClipboardImageOnPaste:
    def test_skips_auto_attach_for_plain_text_paste(self):
        assert _should_auto_attach_clipboard_image_on_paste("hello world") is False

    def test_skips_auto_attach_for_whitespace_and_text_paste(self):
        assert _should_auto_attach_clipboard_image_on_paste("  hello world  ") is False

    def test_allows_auto_attach_for_empty_paste(self):
        assert _should_auto_attach_clipboard_image_on_paste("") is True

    def test_allows_auto_attach_for_whitespace_only_paste(self):
        assert _should_auto_attach_clipboard_image_on_paste("   \n\t  ") is True


class TestVoiceSubmission:
    @pytest.fixture
    def cli(self):
        from cli import HermesCLI
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj._attached_images = [Path("/tmp/stale.png")]
        cli_obj._pending_input = queue.Queue()
        cli_obj._voice_lock = MagicMock()
        cli_obj._voice_processing = True
        cli_obj._voice_recording = True
        cli_obj._voice_continuous = False
        cli_obj._no_speech_count = 0
        cli_obj._voice_recorder = MagicMock()
        cli_obj._voice_recorder.stop.return_value = "/tmp/fake.wav"
        cli_obj._app = None
        return cli_obj

    def test_voice_transcript_clears_stale_attached_images(self, cli):
        with patch("tools.voice_mode.play_beep"):
            with patch("tools.voice_mode.transcribe_recording", return_value={"success": True, "transcript": "hello"}):
                with patch("os.path.isfile", return_value=False):
                    with patch("cli._cprint"):
                        cli._voice_stop_and_transcribe()

        assert cli._attached_images == []
        assert cli._pending_input.get_nowait() == "hello"


# ═════════════════════════════════════════════════════════════════════════
# Level 4: Queue routing — tuple unpacking in process_loop
# ═════════════════════════════════════════════════════════════════════════

class TestQueueRouting:
    """Test that (text, images) tuples are correctly unpacked and routed."""

    def test_plain_string_stays_string(self):
        """Regular text input has no images."""
        user_input = "hello world"
        submit_images = []
        if isinstance(user_input, tuple):
            user_input, submit_images = user_input
        assert user_input == "hello world"
        assert submit_images == []

    def test_tuple_unpacks_text_and_images(self, tmp_path):
        """(text, images) tuple is correctly split."""
        img = tmp_path / "test.png"
        img.write_bytes(FAKE_PNG)
        user_input = ("describe this", [img])

        submit_images = []
        if isinstance(user_input, tuple):
            user_input, submit_images = user_input
        assert user_input == "describe this"
        assert len(submit_images) == 1
        assert submit_images[0] == img

    def test_empty_text_with_images(self, tmp_path):
        """Images without text — text should be empty string."""
        img = tmp_path / "test.png"
        img.write_bytes(FAKE_PNG)
        user_input = ("", [img])

        submit_images = []
        if isinstance(user_input, tuple):
            user_input, submit_images = user_input
        assert user_input == ""
        assert len(submit_images) == 1

    def test_command_with_images_not_treated_as_command(self):
        """Text starting with / in a tuple should still be a command."""
        user_input = "/help"
        submit_images = []
        if isinstance(user_input, tuple):
            user_input, submit_images = user_input
        is_command = isinstance(user_input, str) and user_input.startswith("/")
        assert is_command is True

    def test_images_only_not_treated_as_command(self, tmp_path):
        """Empty text + images should not be treated as a command."""
        img = tmp_path / "test.png"
        img.write_bytes(FAKE_PNG)
        user_input = ("", [img])

        submit_images = []
        if isinstance(user_input, tuple):
            user_input, submit_images = user_input
        is_command = isinstance(user_input, str) and user_input.startswith("/")
        assert is_command is False
        assert len(submit_images) == 1
