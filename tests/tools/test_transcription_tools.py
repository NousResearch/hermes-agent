"""Tests for tools.transcription_tools -- provider resolution and model correction."""

import os
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal valid WAV file (1 second of silence at 16kHz)."""
    wav_path = tmp_path / "test.wav"
    n_frames = 16000
    silence = struct.pack(f"<{n_frames}h", *([0] * n_frames))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(silence)

    return str(wav_path)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no real API keys leak into tests."""
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


# ============================================================================
# _resolve_stt_provider
# ============================================================================

class TestResolveSTTProvider:
    def test_openai_preferred_over_groq(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        from tools.transcription_tools import _resolve_stt_provider
        key, url, provider = _resolve_stt_provider()

        assert provider == "openai"
        assert key == "sk-test"
        assert "openai.com" in url

    def test_groq_fallback(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        from tools.transcription_tools import _resolve_stt_provider
        key, url, provider = _resolve_stt_provider()

        assert provider == "groq"
        assert key == "gsk-test"
        assert "groq.com" in url

    def test_no_keys_returns_none(self):
        from tools.transcription_tools import _resolve_stt_provider
        key, url, provider = _resolve_stt_provider()

        assert provider == "none"
        assert key is None
        assert url is None


# ============================================================================
# transcribe_audio -- no API key
# ============================================================================

class TestTranscribeAudioNoKey:
    def test_returns_error_when_no_key(self):
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/tmp/test.wav")

        assert result["success"] is False
        assert "No STT API key" in result["error"]

    def test_returns_error_for_missing_file(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/nonexistent/audio.wav")

        assert result["success"] is False
        assert "not found" in result["error"]


# ============================================================================
# Model auto-correction
# ============================================================================

class TestModelAutoCorrection:
    def test_groq_corrects_openai_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import transcribe_audio, DEFAULT_GROQ_STT_MODEL
            result = transcribe_audio(sample_wav, model="whisper-1")

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        # Verify the model was corrected to Groq default
        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_openai_corrects_groq_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import transcribe_audio, DEFAULT_STT_MODEL
            result = transcribe_audio(sample_wav, model="whisper-large-v3-turbo")

        assert result["success"] is True
        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_STT_MODEL

    def test_none_model_uses_provider_default(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import transcribe_audio, DEFAULT_GROQ_STT_MODEL
            transcribe_audio(sample_wav, model=None)

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_compatible_model_not_overridden(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_wav, model="whisper-large-v3")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-large-v3"


# ============================================================================
# transcribe_audio -- success path
# ============================================================================

class TestTranscribeAudioSuccess:
    def test_successful_transcription(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_wav)

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        assert result["provider"] == "groq"

    def test_api_error_returns_failure(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        with patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_wav)

        assert result["success"] is False
        assert "API error" in result["error"]

    def test_whitespace_transcript_stripped(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  hello world  \n"

        with patch("openai.OpenAI", return_value=mock_client):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_wav)

        assert result["transcript"] == "hello world"
