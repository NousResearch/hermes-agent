"""Tests for AMR conversion in TTS tool (_convert_to_amr, platform detection, extension config).

Covers:
  - Successful AMR-NB conversion
  - Input file missing / empty / corrupt
  - No ffmpeg available
  - AMR-WB path
  - Subprocess timeout
  - _VOICE_EXTS includes .amr
  - want_amr platform detection for wecom/weixin
"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tools import tts_tool
from tools.send_message_tool import _VOICE_EXTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_mp3(tmp_path: Path) -> Path:
    """Write a minimal valid MP3 header so os.path.getsize > 0."""
    p = tmp_path / "speech.mp3"
    p.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 256)
    return p


def _mock_ffmpeg_success(tmp_path: Path):
    """Return a subprocess.run mock that writes a small .amr file and returns rc=0."""

    def _run(*args, **kwargs):
        # Extract the output path from the ffmpeg args
        cmd = args[0] if args else kwargs.get("args", [])
        amr_path = cmd[-2] if len(cmd) >= 2 else None
        if amr_path:
            Path(amr_path).write_bytes(b"#!AMR\n" + b"\x00" * 100)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    return Mock(side_effect=_run)


# ---------------------------------------------------------------------------
# Test 1: Successful AMR-NB conversion
# ---------------------------------------------------------------------------

def test_convert_to_amr_nb_success(tmp_path, monkeypatch):
    """A valid MP3 file converts to AMR-NB and returns the .amr path."""
    mp3 = _make_valid_mp3(tmp_path)
    monkeypatch.setattr(tts_tool, "_has_ffmpeg", lambda: True)
    monkeypatch.setattr(subprocess, "run", _mock_ffmpeg_success(tmp_path))

    result = tts_tool._convert_to_amr(str(mp3), amr_nb=True)

    expected_amr = str(tmp_path / "speech.amr")
    assert result == expected_amr
    assert os.path.isfile(expected_amr)
    assert os.path.getsize(expected_amr) > 0


# ---------------------------------------------------------------------------
# Test 2: Input file missing
# ---------------------------------------------------------------------------

def test_convert_to_amr_missing_file(tmp_path, monkeypatch):
    """Returns None when the MP3 file does not exist."""
    monkeypatch.setattr(tts_tool, "_has_ffmpeg", lambda: True)
    nonexistent = str(tmp_path / "does_not_exist.mp3")
    result = tts_tool._convert_to_amr(nonexistent)
    assert result is None


# ---------------------------------------------------------------------------
# Test 3: Empty (zero-byte) input file
# ---------------------------------------------------------------------------

def test_convert_to_amr_empty_file(tmp_path, monkeypatch):
    """Returns None when the MP3 file is zero bytes."""
    monkeypatch.setattr(tts_tool, "_has_ffmpeg", lambda: True)
    p = tmp_path / "empty.mp3"
    p.write_bytes(b"")
    result = tts_tool._convert_to_amr(str(p))
    assert result is None


# ---------------------------------------------------------------------------
# Test 4: Corrupted file / ffmpeg failure
# ---------------------------------------------------------------------------

def test_convert_to_amr_ffmpeg_failure(tmp_path, monkeypatch):
    """Returns None when ffmpeg exits with non-zero return code."""
    mp3 = _make_valid_mp3(tmp_path)
    monkeypatch.setattr(tts_tool, "_has_ffmpeg", lambda: True)

    def _fail(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout=b"",
            stderr=b"Invalid data found when processing input",
        )

    monkeypatch.setattr(subprocess, "run", _fail)
    result = tts_tool._convert_to_amr(str(mp3))
    assert result is None


# ---------------------------------------------------------------------------
# Test 5: No ffmpeg on the system
# ---------------------------------------------------------------------------

def test_convert_to_amr_no_ffmpeg(tmp_path, monkeypatch):
    """Returns None immediately when ffmpeg is not in PATH."""
    mp3 = _make_valid_mp3(tmp_path)
    monkeypatch.setattr(tts_tool, "_has_ffmpeg", lambda: False)
    result = tts_tool._convert_to_amr(str(mp3))
    assert result is None


# ---------------------------------------------------------------------------
# Test 6: AMR-WB path
# ---------------------------------------------------------------------------

def test_convert_to_amr_wb_path(tmp_path, monkeypatch):
    """amr_nb=False produces AMR-WB with the correct codec arguments."""
    mp3 = _make_valid_mp3(tmp_path)
    monkeypatch.setattr(tts_tool, "_has_ffmpeg", lambda: True)

    captured_cmd = []

    def _capture(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        captured_cmd.append(cmd)
        amr_path = cmd[-2] if len(cmd) >= 2 else None
        if amr_path:
            Path(amr_path).write_bytes(b"#!AMR-WB\n" + b"\x00" * 100)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _capture)
    result = tts_tool._convert_to_amr(str(mp3), amr_nb=False)

    assert result is not None
    assert result.endswith(".amr")
    # Verify WB codec args were used
    assert any("libvo_amrwbenc" in str(c) for c in captured_cmd)
    assert any("16000" in str(c) for c in captured_cmd)


# ---------------------------------------------------------------------------
# Test 7: Subprocess timeout
# ---------------------------------------------------------------------------

def test_convert_to_amr_timeout(tmp_path, monkeypatch):
    """Returns None when ffmpeg exceeds the 30s timeout."""
    mp3 = _make_valid_mp3(tmp_path)
    monkeypatch.setattr(tts_tool, "_has_ffmpeg", lambda: True)
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired(
        cmd=["ffmpeg"], timeout=30, output=b"", stderr=b"",
    )))
    result = tts_tool._convert_to_amr(str(mp3))
    assert result is None


# ---------------------------------------------------------------------------
# Test 8: _VOICE_EXTS configuration contains .amr
# ---------------------------------------------------------------------------

def test_voice_extensions_include_amr():
    """The global _VOICE_EXTS set includes '.amr' for WeCom/WeChat voice delivery."""
    assert ".amr" in _VOICE_EXTS
    assert ".ogg" in _VOICE_EXTS
    assert ".opus" in _VOICE_EXTS


# ---------------------------------------------------------------------------
# Test 9: Platform detection sets want_amr for wecom/weixin
# ---------------------------------------------------------------------------

def test_tts_platform_detection_want_amr(monkeypatch, tmp_path):
    """When platform is wecom or weixin, want_amr triggers AMR conversion."""
    from gateway.session_context import _UNSET, _VAR_MAP

    # Clean state
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)

    # Use Edge TTS (outputs MP3, needs ffmpeg for AMR)
    monkeypatch.setattr(tts_tool, "_has_ffmpeg", lambda: True)

    out = tmp_path / "speech.mp3"

    # Mock Edge TTS to produce an MP3
    async def _edge_tts(_text, path, _cfg):
        Path(path).write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 256)
        return path

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _edge_tts)

    # Mock ffmpeg AMR conversion to succeed
    monkeypatch.setattr(subprocess, "run", _mock_ffmpeg_success(tmp_path))

    # Test wecom
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "wecom")
    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert result["voice_compatible"] is True
    assert result["file_path"].endswith(".amr")
    assert result["media_tag"].startswith("[[audio_as_voice]]")
    assert "MEDIA:" in result["media_tag"]

    # Test weixin
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "weixin")
    result2 = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(tmp_path / "speech2.mp3")))

    assert result2["success"] is True
    assert result2["voice_compatible"] is True
    assert result2["file_path"].endswith(".amr")
