"""Tests for transcription_tools.py — local (faster-whisper) and OpenAI providers.

Tests cover provider selection, config loading, validation, and transcription
dispatch.  All external dependencies (faster_whisper, openai) are mocked.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


class TestGetProvider:
    """_get_provider() picks the right backend based on config + availability."""

    def test_local_when_available(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "local"

    def test_local_fallback_to_openai(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "openai"

    def test_local_nothing_available(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "none"

    def test_openai_when_key_set(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openai"}) == "openai"

    def test_openai_fallback_to_local(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openai"}) == "local"

    def test_default_provider_is_local(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------


class TestValidateAudioFile:

    def test_missing_file(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        result = _validate_audio_file(str(tmp_path / "nope.ogg"))
        assert result is not None
        assert "not found" in result["error"]

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_bytes(b"data")
        from tools.transcription_tools import _validate_audio_file
        result = _validate_audio_file(str(f))
        assert result is not None
        assert "Unsupported" in result["error"]

    def test_valid_file_returns_none(self, tmp_path):
        f = tmp_path / "test.ogg"
        f.write_bytes(b"fake audio data")
        from tools.transcription_tools import _validate_audio_file
        assert _validate_audio_file(str(f)) is None

    def test_too_large(self, tmp_path):
        import stat as stat_mod
        f = tmp_path / "big.ogg"
        f.write_bytes(b"x")
        from tools.transcription_tools import _validate_audio_file, MAX_FILE_SIZE
        real_stat = f.stat()
        with patch.object(type(f), "stat", return_value=os.stat_result((
            real_stat.st_mode, real_stat.st_ino, real_stat.st_dev,
            real_stat.st_nlink, real_stat.st_uid, real_stat.st_gid,
            MAX_FILE_SIZE + 1,  # st_size
            real_stat.st_atime, real_stat.st_mtime, real_stat.st_ctime,
        ))):
            result = _validate_audio_file(str(f))
        assert result is not None
        assert "too large" in result["error"]


# ---------------------------------------------------------------------------
# Local transcription
# ---------------------------------------------------------------------------


class TestTranscribeLocal:

    def test_successful_transcription(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 2.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools.WhisperModel", return_value=mock_model), \
             patch("tools.transcription_tools._local_model", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio_file), "base")

        assert result["success"] is True
        assert result["transcript"] == "Hello world"

    def test_not_installed(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local("/tmp/test.ogg", "base")
        assert result["success"] is False
        assert "not installed" in result["error"]


# ---------------------------------------------------------------------------
# OpenAI transcription
# ---------------------------------------------------------------------------


class TestTranscribeOpenAI:

    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        from tools.transcription_tools import _transcribe_openai
        result = _transcribe_openai("/tmp/test.ogg", "whisper-1")
        assert result["success"] is False
        assert "VOICE_TOOLS_OPENAI_KEY" in result["error"]

    def test_successful_transcription(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "Hello from OpenAI"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(str(audio_file), "whisper-1")

        assert result["success"] is True
        assert result["transcript"] == "Hello from OpenAI"


# ---------------------------------------------------------------------------
# Main transcribe_audio() dispatch
# ---------------------------------------------------------------------------


class TestTranscribeAudio:

    def test_dispatches_to_local(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "local"}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local", return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(audio_file))

        assert result["success"] is True
        mock_local.assert_called_once()

    def test_dispatches_to_openai(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openai"}), \
             patch("tools.transcription_tools._get_provider", return_value="openai"), \
             patch("tools.transcription_tools._transcribe_openai", return_value={"success": True, "transcript": "hi"}) as mock_openai:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(audio_file))

        assert result["success"] is True
        mock_openai.assert_called_once()

    def test_no_provider_returns_error(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="none"):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(audio_file))

        assert result["success"] is False
        assert "No STT provider" in result["error"]

    def test_invalid_file_returns_error(self):
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/nonexistent/file.ogg")
        assert result["success"] is False
        assert "not found" in result["error"]
