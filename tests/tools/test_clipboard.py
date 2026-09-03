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
from unittest.mock import patch, MagicMock, mock_open

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

    def test_empty_file_rejected(self, tmp_path):
        dest = tmp_path / "out.png"
        def fake_run(cmd, **kw):
            dest.write_bytes(b"")
            return MagicMock(returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_pngpaste(dest) is False


class TestMacosHasImage:
    @pytest.mark.parametrize("stdout, expected", [
        ("«class PNGf», «class ut16»", True),
        ("«class ut16», «class utf8»", False),
    ])
    def test_image_class_detection(self, stdout, expected):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=stdout, returncode=0)
            assert _macos_has_image() is expected


class TestMacosOsascript:
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

    @pytest.mark.parametrize("content, expected", [
        ("Linux version 5.15.0 (microsoft-standard-WSL2)", True),
        # GHA hosted runners are Azure VMs whose real /proc/version often
        # contains "microsoft", so the patched `open` must actually be reached
        # (setup_method clears the cache that would short-circuit it).
        ("Linux version 6.14.0-37-generic (buildd@lcy02-amd64-049)", False),
    ])
    def test_detection_from_proc_version(self, content, expected):
        with patch.dict(_is_wsl.__globals__, {"open": mock_open(read_data=content)}):
            assert _is_wsl() is expected


    def test_result_is_cached(self):
        content = "Linux version 5.15.0 (microsoft-standard-WSL2)"
        opener = mock_open(read_data=content)
        with patch.dict(_is_wsl.__globals__, {"open": opener}):
            assert _is_wsl() is True
            assert _is_wsl() is True
            opener.assert_called_once()  # only read once


# ── WSL (powershell.exe) ────────────────────────────────────────────────

class TestWslHasImage:
    @pytest.mark.parametrize("stdout, expected", [
        ("True\n", True),
        ("False\n", False),
    ])
    def test_clipboard_image_probe(self, stdout, expected):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=stdout, returncode=0)
            assert _wsl_has_image() is expected

    def test_falls_back_to_get_clipboard_image(self):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="False\n", returncode=0),
                MagicMock(stdout="True\n", returncode=0),
            ]
            assert _wsl_has_image() is True
            assert mock_run.call_count == 2


class TestWslSave:
    def test_successful_extraction(self, tmp_path):
        dest = tmp_path / "out.png"
        b64_png = base64.b64encode(FAKE_PNG).decode()
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b64_png + "\n", returncode=0)
            assert _wsl_save(dest) is True
        assert dest.read_bytes() == FAKE_PNG


    def test_invalid_base64(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="not-valid-base64!!!", returncode=0)
            assert _wsl_save(dest) is False


# ── Wayland (wl-paste) ──────────────────────────────────────────────────

class TestWaylandHasImage:
    @pytest.mark.parametrize("types, expected", [
        ("image/png\ntext/plain\n", True),
        ("text/html\nimage/bmp\n", True),   # non-PNG image types count too
        ("text/plain\ntext/html\n", False),
    ])
    def test_type_list_detection(self, types, expected):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=types, returncode=0)
            assert _wayland_has_image() is expected


class TestWaylandSave:
    def test_png_extraction(self, tmp_path):
        dest = tmp_path / "out.png"
        def fake_run(cmd, **kw):
            if "--list-types" in cmd:
                return MagicMock(stdout="image/png\ntext/plain\n", returncode=0)
            # Extract call — write fake data to stdout file
            if "stdout" in kw and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(FAKE_PNG)
            return MagicMock(returncode=0)
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _wayland_save(dest) is True
        assert dest.stat().st_size > 0


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
    @pytest.mark.parametrize("targets, expected", [
        ("image/png\ntext/plain\n", True),
        ("text/plain\n", False),
    ])
    def test_targets_detection(self, targets, expected):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=targets, returncode=0)
            assert _xclip_has_image() is expected


class TestXclipSave:
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

    def test_wayland_fails_falls_through_to_xclip(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard._is_wsl", return_value=False):
            with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
                with patch("hermes_cli.clipboard._wayland_save", return_value=False):
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

    def test_falls_back_to_get_clipboard_image(self):
        with patch("hermes_cli.clipboard._get_ps_exe", return_value="powershell"):
            with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
                mock_run.side_effect = [
                    MagicMock(stdout="False\n", returncode=0),
                    MagicMock(stdout="True\n", returncode=0),
                ]
                assert _windows_has_image() is True
                assert mock_run.call_count == 2


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


    @pytest.mark.parametrize("failure", ["nonzero-exit", "timeout"])
    def test_imagemagick_failure_preserves_original(self, tmp_path, failure):
        """When ImageMagick can't convert, the original file must not be lost."""
        dest = tmp_path / "img.png"
        dest.write_bytes(FAKE_BMP)

        side_effect = (
            (lambda cmd, **kw: MagicMock(returncode=1))
            if failure == "nonzero-exit"
            else subprocess.TimeoutExpired("convert", 5)
        )

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with patch("hermes_cli.clipboard.subprocess.run", side_effect=side_effect):
                _convert_to_png(dest)

        # Original file must still exist with original content
        assert dest.exists(), "Original file was lost after failed conversion"
        assert dest.read_bytes() == FAKE_BMP


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

    def test_single_image_with_text(self, cli, tmp_path):
        img = self._make_image(tmp_path)
        with patch("tools.vision_tools.vision_analyze_tool", side_effect=self._mock_vision_success()):
            result = cli._preprocess_images_with_vision("Describe this", [img])

        assert isinstance(result, str)
        assert "A test image with colored pixels." in result
        assert "Describe this" in result
        assert str(img) in result
        assert "base64," not in result  # no raw base64 image content


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


    def test_image_path_follows_naming_convention(self, cli):
        with patch("hermes_cli.clipboard.save_clipboard_image", return_value=True):
            cli._try_attach_clipboard_image()
        path = cli._attached_images[0]
        assert path.parent == Path(os.environ["HERMES_HOME"]) / "images"
        assert path.name.startswith("clip_")
        assert path.suffix == ".png"


class TestAutoAttachClipboardImageOnPaste:
    @pytest.mark.parametrize("pasted, expected", [
        ("  hello world  ", False),   # real text paste — don't hijack it
        ("   \n\t  ", True),          # whitespace-only paste may be an image
    ])
    def test_auto_attach_decision(self, pasted, expected):
        assert _should_auto_attach_clipboard_image_on_paste(pasted) is expected


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
        queued = cli._pending_input.get_nowait()
        # Voice transcripts are wrapped in the _VoiceInputMessage sentinel
        # (#65827) so process_loop can distinguish STT output from typed text.
        from cli import _VoiceInputMessage
        assert isinstance(queued, _VoiceInputMessage)
        assert queued.text == "hello"
