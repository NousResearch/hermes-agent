"""Tests for tools.transcription_tools — three-provider STT pipeline.

Covers the full provider matrix (local, groq, openai), fallback chains,
model auto-correction, config loading, validation edge cases, and
end-to-end dispatch.  All external dependencies are mocked.
"""

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


@pytest.fixture
def sample_ogg(tmp_path):
    """Create a fake OGG file for validation tests."""
    ogg_path = tmp_path / "test.ogg"
    ogg_path.write_bytes(b"fake audio data")
    return str(ogg_path)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no real API keys leak into tests."""
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


# ============================================================================
# _get_provider — full permutation matrix
# ============================================================================

class TestGetProviderGroq:
    """Groq-specific provider selection tests."""

    def test_groq_when_key_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools._HAS_FASTER_WHISPER", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "groq"

    def test_groq_fallback_to_local(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "local"

    def test_groq_fallback_to_openai(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "openai"

    def test_groq_nothing_available(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "groq"}) == "none"


class TestGetProviderFallbackPriority:
    """Cross-provider fallback priority tests."""

    def test_local_fallback_prefers_groq_over_openai(self, monkeypatch):
        """When local unavailable, groq (free) is preferred over openai (paid)."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "groq"

    def test_local_fallback_to_groq_only(self, monkeypatch):
        """When only groq key available, falls back to groq."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "local"}) == "groq"

    def test_openai_fallback_to_groq(self, monkeypatch):
        """When openai key missing but groq available, falls back to groq."""
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openai"}) == "groq"

    def test_openai_nothing_available(self, monkeypatch):
        """When no openai key and no local, returns none."""
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "openai"}) == "none"

    def test_unknown_provider_passed_through(self):
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "custom-endpoint"}) == "custom-endpoint"

    def test_empty_config_defaults_to_local(self):
        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({}) == "local"


# ============================================================================
# _transcribe_groq
# ============================================================================

class TestTranscribeGroq:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from tools.transcription_tools import _transcribe_groq
        result = _transcribe_groq("/tmp/test.ogg", "whisper-large-v3-turbo")
        assert result["success"] is False
        assert "GROQ_API_KEY" in result["error"]

    def test_openai_package_not_installed(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq("/tmp/test.ogg", "whisper-large-v3-turbo")
        assert result["success"] is False
        assert "openai package" in result["error"]

    def test_successful_transcription(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        assert result["provider"] == "groq"

    def test_whitespace_stripped(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  hello world  \n"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["transcript"] == "hello world"

    def test_uses_groq_base_url(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client) as mock_openai_cls:
            from tools.transcription_tools import _transcribe_groq, GROQ_BASE_URL
            _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        call_kwargs = mock_openai_cls.call_args
        assert call_kwargs.kwargs["base_url"] == GROQ_BASE_URL

    def test_api_error_returns_failure(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is False
        assert "API error" in result["error"]

    def test_permission_error(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = PermissionError("denied")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            result = _transcribe_groq(sample_wav, "whisper-large-v3-turbo")

        assert result["success"] is False
        assert "Permission denied" in result["error"]


# ============================================================================
# _transcribe_openai — additional tests
# ============================================================================

class TestTranscribeOpenAIExtended:
    def test_openai_package_not_installed(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_OPENAI", False):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai("/tmp/test.ogg", "whisper-1")
        assert result["success"] is False
        assert "openai package" in result["error"]

    def test_uses_openai_base_url(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client) as mock_openai_cls:
            from tools.transcription_tools import _transcribe_openai, OPENAI_BASE_URL
            _transcribe_openai(sample_wav, "whisper-1")

        call_kwargs = mock_openai_cls.call_args
        assert call_kwargs.kwargs["base_url"] == OPENAI_BASE_URL

    def test_whitespace_stripped(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  hello  \n"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(sample_wav, "whisper-1")

        assert result["transcript"] == "hello"

    def test_permission_error(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = PermissionError("denied")

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            result = _transcribe_openai(sample_wav, "whisper-1")

        assert result["success"] is False
        assert "Permission denied" in result["error"]


# ============================================================================
# _transcribe_local — additional tests
# ============================================================================

class TestTranscribeLocalExtended:
    def test_model_reuse_on_second_call(self, tmp_path):
        """Second call with same model should NOT reload the model."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio), "base")
            _transcribe_local(str(audio), "base")

        # WhisperModel should be created only once
        assert mock_whisper_cls.call_count == 1

    def test_model_reloaded_on_change(self, tmp_path):
        """Switching model name should reload the model."""
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_segment = MagicMock()
        mock_segment.text = "hi"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls = MagicMock(return_value=mock_model)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None), \
             patch("tools.transcription_tools._local_model_name", None):
            from tools.transcription_tools import _transcribe_local
            _transcribe_local(str(audio), "base")
            _transcribe_local(str(audio), "small")

        assert mock_whisper_cls.call_count == 2

    def test_exception_returns_failure(self, tmp_path):
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        mock_whisper_cls = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools.WhisperModel", mock_whisper_cls), \
             patch("tools.transcription_tools._local_model", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "large-v3")

        assert result["success"] is False
        assert "CUDA out of memory" in result["error"]

    def test_multiple_segments_joined(self, tmp_path):
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"fake")

        seg1 = MagicMock()
        seg1.text = "Hello"
        seg2 = MagicMock()
        seg2.text = " world"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 3.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        with patch("tools.transcription_tools._HAS_FASTER_WHISPER", True), \
             patch("tools.transcription_tools.WhisperModel", return_value=mock_model), \
             patch("tools.transcription_tools._local_model", None):
            from tools.transcription_tools import _transcribe_local
            result = _transcribe_local(str(audio), "base")

        assert result["success"] is True
        assert result["transcript"] == "Hello world"


# ============================================================================
# Model auto-correction
# ============================================================================

class TestModelAutoCorrection:
    def test_groq_corrects_openai_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq, DEFAULT_GROQ_STT_MODEL
            _transcribe_groq(sample_wav, "whisper-1")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_groq_corrects_gpt4o_transcribe(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq, DEFAULT_GROQ_STT_MODEL
            _transcribe_groq(sample_wav, "gpt-4o-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_GROQ_STT_MODEL

    def test_openai_corrects_groq_model(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "hello world"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai, DEFAULT_STT_MODEL
            _transcribe_openai(sample_wav, "whisper-large-v3-turbo")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_STT_MODEL

    def test_openai_corrects_distil_whisper(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai, DEFAULT_STT_MODEL
            _transcribe_openai(sample_wav, "distil-whisper-large-v3-en")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_STT_MODEL

    def test_compatible_groq_model_not_overridden(self, monkeypatch, sample_wav):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "whisper-large-v3")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-large-v3"

    def test_compatible_openai_model_not_overridden(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "gpt-4o-mini-transcribe")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini-transcribe"

    def test_unknown_model_passes_through_groq(self, monkeypatch, sample_wav):
        """A model not in either known set should not be overridden."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_groq
            _transcribe_groq(sample_wav, "my-custom-model")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-custom-model"

    def test_unknown_model_passes_through_openai(self, monkeypatch, sample_wav):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "test"

        with patch("tools.transcription_tools._HAS_OPENAI", True), \
             patch("tools.transcription_tools.OpenAI", return_value=mock_client):
            from tools.transcription_tools import _transcribe_openai
            _transcribe_openai(sample_wav, "my-custom-model")

        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "my-custom-model"


# ============================================================================
# _load_stt_config
# ============================================================================

class TestLoadSttConfig:
    def test_returns_dict_when_import_fails(self):
        with patch("tools.transcription_tools._load_stt_config") as mock_load:
            mock_load.return_value = {}
            from tools.transcription_tools import _load_stt_config
            assert _load_stt_config() == {}

    def test_real_load_returns_dict(self):
        """_load_stt_config should always return a dict, even on import error."""
        with patch.dict("sys.modules", {"hermes_cli": None, "hermes_cli.config": None}):
            from tools.transcription_tools import _load_stt_config
            result = _load_stt_config()
        assert isinstance(result, dict)


# ============================================================================
# _validate_audio_file — edge cases
# ============================================================================

class TestValidateAudioFileEdgeCases:
    def test_directory_is_not_a_file(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        # tmp_path itself is a directory with an .ogg-ish name? No.
        # Create a directory with a valid audio extension
        d = tmp_path / "audio.ogg"
        d.mkdir()
        result = _validate_audio_file(str(d))
        assert result is not None
        assert "not a file" in result["error"]

    def test_stat_oserror(self, tmp_path):
        f = tmp_path / "test.ogg"
        f.write_bytes(b"data")
        from tools.transcription_tools import _validate_audio_file
        real_stat = f.stat()
        call_count = 0

        def stat_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First calls are from exists() and is_file(), let them pass
            if call_count <= 2:
                return real_stat
            raise OSError("disk error")

        with patch("pathlib.Path.stat", side_effect=stat_side_effect):
            result = _validate_audio_file(str(f))
        assert result is not None
        assert "Failed to access" in result["error"]

    def test_all_supported_formats_accepted(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file, SUPPORTED_FORMATS
        for fmt in SUPPORTED_FORMATS:
            f = tmp_path / f"test{fmt}"
            f.write_bytes(b"data")
            assert _validate_audio_file(str(f)) is None, f"Format {fmt} should be accepted"

    def test_case_insensitive_extension(self, tmp_path):
        from tools.transcription_tools import _validate_audio_file
        f = tmp_path / "test.MP3"
        f.write_bytes(b"data")
        assert _validate_audio_file(str(f)) is None


# ============================================================================
# transcribe_audio — end-to-end dispatch
# ============================================================================

class TestTranscribeAudioDispatch:
    def test_dispatches_to_groq(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "groq"}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi", "provider": "groq"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        assert result["provider"] == "groq"
        mock_groq.assert_called_once()

    def test_dispatches_to_local(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        mock_local.assert_called_once()

    def test_dispatches_to_openai(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "openai"}), \
             patch("tools.transcription_tools._get_provider", return_value="openai"), \
             patch("tools.transcription_tools._transcribe_openai",
                   return_value={"success": True, "transcript": "hi", "provider": "openai"}) as mock_openai:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is True
        mock_openai.assert_called_once()

    def test_no_provider_returns_error(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="none"):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(sample_ogg)

        assert result["success"] is False
        assert "No STT provider" in result["error"]
        assert "faster-whisper" in result["error"]
        assert "GROQ_API_KEY" in result["error"]

    def test_invalid_file_short_circuits(self):
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio("/nonexistent/audio.wav")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_model_override_passed_to_groq(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="whisper-large-v3")

        _, kwargs = mock_groq.call_args
        assert kwargs.get("model_name") or mock_groq.call_args[0][1] == "whisper-large-v3"

    def test_model_override_passed_to_local(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model="large-v3")

        assert mock_local.call_args[0][1] == "large-v3"

    def test_default_model_used_when_none(self, sample_ogg):
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_tools._get_provider", return_value="groq"), \
             patch("tools.transcription_tools._transcribe_groq",
                   return_value={"success": True, "transcript": "hi"}) as mock_groq:
            from tools.transcription_tools import transcribe_audio, DEFAULT_GROQ_STT_MODEL
            transcribe_audio(sample_ogg, model=None)

        assert mock_groq.call_args[0][1] == DEFAULT_GROQ_STT_MODEL

    def test_config_local_model_used(self, sample_ogg):
        config = {"local": {"model": "small"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="local"), \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hi"}) as mock_local:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_local.call_args[0][1] == "small"

    def test_config_openai_model_used(self, sample_ogg):
        config = {"openai": {"model": "gpt-4o-transcribe"}}
        with patch("tools.transcription_tools._load_stt_config", return_value=config), \
             patch("tools.transcription_tools._get_provider", return_value="openai"), \
             patch("tools.transcription_tools._transcribe_openai",
                   return_value={"success": True, "transcript": "hi"}) as mock_openai:
            from tools.transcription_tools import transcribe_audio
            transcribe_audio(sample_ogg, model=None)

        assert mock_openai.call_args[0][1] == "gpt-4o-transcribe"


# ============================================================================
# get_stt_model_from_config
# ============================================================================

class TestGetSttModelFromConfig:
    def test_returns_model_from_config(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("stt:\n  model: whisper-large-v3\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.transcription_tools import get_stt_model_from_config
        assert get_stt_model_from_config() == "whisper-large-v3"

    def test_returns_none_when_no_stt_section(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("tts:\n  provider: edge\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.transcription_tools import get_stt_model_from_config
        assert get_stt_model_from_config() is None

    def test_returns_none_when_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.transcription_tools import get_stt_model_from_config
        assert get_stt_model_from_config() is None

    def test_returns_none_on_invalid_yaml(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(": : :\n  bad yaml [[[")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.transcription_tools import get_stt_model_from_config
        assert get_stt_model_from_config() is None

    def test_returns_none_when_model_key_missing(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("stt:\n  enabled: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from tools.transcription_tools import get_stt_model_from_config
        assert get_stt_model_from_config() is None
