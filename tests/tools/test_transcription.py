"""Tests for transcription_tools.py — local (faster-whisper) and OpenAI providers.

Tests cover provider selection, config loading, validation, and transcription
dispatch.  All external dependencies (faster_whisper, openai) are mocked.
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, mock_open

import pytest


def _fake_faster_whisper_module(mock_model):
    return SimpleNamespace(WhisperModel=MagicMock(return_value=mock_model))


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class TestGetProvider:
    """_get_provider() picks the right backend based on config + availability."""

    def test_local_when_available(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "local"

    def test_explicit_local_no_cloud_fallback(self, monkeypatch):
        """Explicit local provider must not silently fall back to cloud."""
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._has_local_command", return_value=False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "none"

    def test_local_nothing_available(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", False), \
             patch("tools.transcription_tools._has_local_command", return_value=False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "none"

    def test_openai_when_key_set(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openai"}) == "openai"

    def test_explicit_openai_no_key_returns_none(self, monkeypatch):
        """Explicit openai without key returns none — no cross-provider fallback."""
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openai"}) == "none"

    def test_default_provider_is_local(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"

    def test_disabled_config_returns_none(self):
        from tools.transcription_tools import _get_provider
        assert _get_provider({"enabled": False, "provider": "openai"}) == "none"


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

        fake_fw = _fake_faster_whisper_module(mock_model)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch.dict("sys.modules", {"faster_whisper": fake_fw}), \
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
             patch("openai.OpenAI", return_value=mock_client):
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

    def test_disabled_config_returns_disabled_error(self, tmp_path):
        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch("tools.transcription_tools._load_stt_config", return_value={"enabled": False}), \
             patch("tools.transcription_tools._get_provider", return_value="none"):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(audio_file))

        assert result["success"] is False
        assert "disabled" in result["error"].lower()

    def test_invalid_file_returns_error(self):
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/nonexistent/file.ogg")
        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# Model name normalisation for local providers
# ---------------------------------------------------------------------------


class TestNormalizeLocalModel:
    """_normalize_local_model() maps cloud-only names to the local default."""

    def test_openai_model_name_maps_to_default(self):
        from tools.transcription_tools import _normalize_local_model, DEFAULT_LOCAL_MODEL
        assert _normalize_local_model("whisper-1") == DEFAULT_LOCAL_MODEL

    def test_groq_model_name_maps_to_default(self):
        from tools.transcription_tools import _normalize_local_model, DEFAULT_LOCAL_MODEL
        assert _normalize_local_model("whisper-large-v3-turbo") == DEFAULT_LOCAL_MODEL

    def test_valid_local_model_preserved(self):
        from tools.transcription_tools import _normalize_local_model
        for size in ("tiny", "base", "small", "medium", "large-v3"):
            assert _normalize_local_model(size) == size

    def test_none_maps_to_default(self):
        from tools.transcription_tools import _normalize_local_model, DEFAULT_LOCAL_MODEL
        assert _normalize_local_model(None) == DEFAULT_LOCAL_MODEL

    def test_warning_emitted_for_cloud_model(self, caplog):
        import logging
        from tools.transcription_tools import _normalize_local_model
        with caplog.at_level(logging.WARNING, logger="tools.transcription_tools"):
            _normalize_local_model("whisper-1")
        assert any("whisper-1" in r.message for r in caplog.records)

    def test_local_transcribe_normalises_model(self):
        """transcribe_audio with local provider must not pass 'whisper-1' to WhisperModel."""
        import tempfile, os
        from unittest.mock import MagicMock, patch

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(b"x")
            audio_file = f.name
        try:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = (iter([]), MagicMock(language="en", duration=1.0))
            with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
                 patch("tools.transcription_tools._load_stt_config", return_value={
                     "enabled": True,
                     "provider": "local",
                     "local": {"model": "whisper-1"},
                 }), \
                 patch("tools.transcription_tools._local_model", None), \
                 patch("tools.transcription_tools._local_model_name", None), \
                 patch.dict("sys.modules", {"faster_whisper": _fake_faster_whisper_module(mock_model)}):
                mock_cls = __import__("faster_whisper").WhisperModel
                from tools.transcription_tools import transcribe_audio
                transcribe_audio(audio_file)
                # WhisperModel must NOT have been called with "whisper-1"
                call_args = mock_cls.call_args
                assert call_args is not None
                assert call_args[0][0] != "whisper-1", (
                    "WhisperModel was called with the cloud-only name 'whisper-1'"
                )
        finally:
            os.unlink(audio_file)
