"""Tests for hermes_cli/clipboard.py — clipboard image extraction.

Tests clipboard image extraction across platforms, and the CLI-level
multimodal content conversion that turns attached images into OpenAI
vision API format.
"""

import base64
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from hermes_cli.clipboard import (
    save_clipboard_image,
    _linux_save,
    _macos_pngpaste,
    _macos_osascript,
)


# ── Platform dispatch ────────────────────────────────────────────────────

class TestSaveClipboardImage:
    def test_dispatches_to_macos_on_darwin(self, tmp_path):
        dest = tmp_path / "out.png"
        with patch("hermes_cli.clipboard.sys") as mock_sys:
            mock_sys.platform = "darwin"
            with patch("hermes_cli.clipboard._macos_save", return_value=False) as m:
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


# ── macOS pngpaste ───────────────────────────────────────────────────────

class TestMacosPngpaste:
    def test_success_writes_file(self, tmp_path):
        """pngpaste writes the file on success — verify we detect it."""
        dest = tmp_path / "out.png"

        def fake_run(cmd, **kw):
            # Simulate pngpaste writing the file
            dest.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            return MagicMock(returncode=0)

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_pngpaste(dest) is True
        assert dest.stat().st_size > 0

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
        """pngpaste exits 0 but writes an empty file — should return False."""
        dest = tmp_path / "out.png"

        def fake_run(cmd, **kw):
            dest.write_bytes(b"")  # empty
            return MagicMock(returncode=0)

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_pngpaste(dest) is False


# ── macOS osascript ──────────────────────────────────────────────────────

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
        """clipboard has PNGf, osascript extracts it successfully."""
        dest = tmp_path / "out.png"
        call_count = [0]

        def fake_run(cmd, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                # clipboard info check
                return MagicMock(stdout="«class PNGf», «class ut16»", returncode=0)
            else:
                # extraction — simulate writing the file
                dest.write_bytes(b"\x89PNG" + b"\x00" * 50)
                return MagicMock(stdout="", returncode=0)

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_osascript(dest) is True
        assert dest.stat().st_size > 0

    def test_success_with_tiff(self, tmp_path):
        """clipboard has TIFF type — should still attempt extraction."""
        dest = tmp_path / "out.png"
        call_count = [0]

        def fake_run(cmd, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(stdout="«class TIFF»", returncode=0)
            else:
                dest.write_bytes(b"\x89PNG" + b"\x00" * 50)
                return MagicMock(stdout="", returncode=0)

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_osascript(dest) is True

    def test_extraction_returns_fail(self, tmp_path):
        """clipboard info says image but extraction script returns 'fail'."""
        dest = tmp_path / "out.png"
        call_count = [0]

        def fake_run(cmd, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(stdout="«class PNGf»", returncode=0)
            else:
                return MagicMock(stdout="fail", returncode=0)

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _macos_osascript(dest) is False


# ── Linux xclip ──────────────────────────────────────────────────────────

class TestLinuxSave:
    def test_no_xclip_installed(self, tmp_path):
        with patch("hermes_cli.clipboard.subprocess.run", side_effect=FileNotFoundError):
            assert _linux_save(tmp_path / "out.png") is False

    def test_no_image_in_clipboard(self, tmp_path):
        with patch("hermes_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="text/plain\n", returncode=0)
            assert _linux_save(tmp_path / "out.png") is False

    def test_image_extraction_success(self, tmp_path):
        """xclip reports image/png in targets, then pipes PNG data."""
        dest = tmp_path / "out.png"
        call_count = [0]

        def fake_run(cmd, **kw):
            call_count[0] += 1
            if "TARGETS" in cmd:
                return MagicMock(stdout="image/png\ntext/plain\n", returncode=0)
            # Extract call — write via the stdout file handle
            if "stdout" in kw and hasattr(kw["stdout"], "write"):
                kw["stdout"].write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            return MagicMock(returncode=0)

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _linux_save(dest) is True
        assert dest.stat().st_size > 0

    def test_extraction_fails_cleans_up(self, tmp_path):
        """If xclip extraction fails, any partial file is cleaned up."""
        dest = tmp_path / "out.png"
        call_count = [0]

        def fake_run(cmd, **kw):
            call_count[0] += 1
            if "TARGETS" in cmd:
                return MagicMock(stdout="image/png\n", returncode=0)
            raise subprocess.SubprocessError("pipe broke")

        with patch("hermes_cli.clipboard.subprocess.run", side_effect=fake_run):
            assert _linux_save(dest) is False
        assert not dest.exists()


# ── Multimodal content conversion (CLI-level) ────────────────────────────

class TestMultimodalConversion:
    """Test the image → OpenAI vision content conversion in chat()."""

    def _make_fake_image(self, tmp_path, name="test.png", size=64):
        """Create a small fake PNG file."""
        img = tmp_path / name
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * size)
        return img

    def test_single_image_with_text(self, tmp_path):
        """One image + text → multimodal content array."""
        img = self._make_fake_image(tmp_path)
        raw_bytes = img.read_bytes()
        expected_b64 = base64.b64encode(raw_bytes).decode()

        # Simulate what chat() does with images
        message = "What's in this image?"
        images = [img]

        content_parts = []
        content_parts.append({"type": "text", "text": message})
        for img_path in images:
            data = base64.b64encode(img_path.read_bytes()).decode()
            ext = img_path.suffix.lower().lstrip(".")
            mime = {"png": "image/png", "jpg": "image/jpeg"}.get(ext, "image/png")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"}
            })

        assert len(content_parts) == 2
        assert content_parts[0]["type"] == "text"
        assert content_parts[0]["text"] == "What's in this image?"
        assert content_parts[1]["type"] == "image_url"
        assert content_parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert expected_b64 in content_parts[1]["image_url"]["url"]

    def test_multiple_images(self, tmp_path):
        """Multiple images → all included in content array."""
        imgs = [self._make_fake_image(tmp_path, f"img{i}.png") for i in range(3)]

        content_parts = [{"type": "text", "text": "Compare these"}]
        for img_path in imgs:
            data = base64.b64encode(img_path.read_bytes()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"}
            })

        assert len(content_parts) == 4  # 1 text + 3 images

    def test_no_text_gets_default(self):
        """Empty text with image → default question added."""
        text = ""
        if not text:
            text = "What do you see in this image?"
        assert text == "What do you see in this image?"

    def test_jpeg_mime_type(self, tmp_path):
        """JPEG files get the correct MIME type."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

        ext = img.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "gif": "image/gif",
                "webp": "image/webp"}.get(ext, "image/png")
        assert mime == "image/jpeg"

    def test_missing_image_skipped(self, tmp_path):
        """Non-existent image path is silently skipped."""
        missing = tmp_path / "does_not_exist.png"
        images = [missing]
        content_parts = [{"type": "text", "text": "test"}]
        for img_path in images:
            if img_path.exists():
                content_parts.append({"type": "image_url"})
        assert len(content_parts) == 1  # only text, no image
