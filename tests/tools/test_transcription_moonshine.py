"""Tests for the Moonshine STT provider in transcription_tools.

Covers provider selection, language normalization, and the
``_transcribe_moonshine`` dispatch path. The moonshine_voice library
itself is not loaded — we patch ``_transcribe_moonshine`` at the module
boundary to keep these tests dependency-free.

Mirrors the StepFun provider shape: a deterministic, opt-in handler
that requires the ``moonshine_voice`` package to be installed. English
models are MIT-licensed; non-English models ship under the Moonshine
Community License (non-commercial).
"""

from unittest.mock import patch

import pytest


pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")


# ---------------------------------------------------------------------------
# Provider selection — moonshine
# ---------------------------------------------------------------------------


class TestGetProviderMoonshine:
    """``_get_provider`` honours an explicit ``provider: moonshine`` config."""

    def test_moonshine_when_package_installed(self):
        with patch("tools.transcription_tools._HAS_MOONSHINE", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "moonshine"}) == "moonshine"

    def test_moonshine_when_package_missing_returns_none(self):
        with patch("tools.transcription_tools._HAS_MOONSHINE", False):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "moonshine"}) == "none"

    def test_moonshine_does_not_silently_fall_back_to_cloud(self, monkeypatch):
        """Explicit moonshine without the package must not silently route elsewhere."""
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        with patch("tools.transcription_tools._HAS_MOONSHINE", False), \
             patch("tools.transcription_tools._HAS_OPENAI", True):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "moonshine"}) == "none"

    def test_moonshine_not_in_builtin_set(self):
        """Moonshine is not a BUILTIN_STT_PROVIDER (mirrors stepfun) — keeps the
        registry/dispatcher sync invariant free of moonshine-specific entries."""
        from tools.transcription_tools import BUILTIN_STT_PROVIDERS
        assert "moonshine" not in BUILTIN_STT_PROVIDERS


# ---------------------------------------------------------------------------
# Language normalization
# ---------------------------------------------------------------------------


class TestNormalizeMoonshineLanguage:
    """``_normalize_moonshine_language`` accepts Moonshine's 2-letter codes
    plus the ``"auto"`` sentinel."""

    def test_known_codes_pass_through(self):
        from tools.transcription_tools import _normalize_moonshine_language
        for code in ("ar", "es", "en", "ja", "ko", "vi", "uk", "zh"):
            assert _normalize_moonshine_language(code) == code

    def test_auto_sentinel(self):
        from tools.transcription_tools import _normalize_moonshine_language
        assert _normalize_moonshine_language("auto") == "auto"

    def test_case_insensitive(self):
        from tools.transcription_tools import _normalize_moonshine_language
        assert _normalize_moonshine_language("KO") == "ko"
        assert _normalize_moonshine_language("En") == "en"

    def test_empty_and_none_fall_back_to_auto(self):
        from tools.transcription_tools import _normalize_moonshine_language
        assert _normalize_moonshine_language(None) == "auto"
        assert _normalize_moonshine_language("") == "auto"
        assert _normalize_moonshine_language("   ") == "auto"

    def test_unsupported_code_warns_and_falls_back(self, caplog):
        from tools.transcription_tools import _normalize_moonshine_language
        with caplog.at_level("WARNING", logger="tools.transcription_tools"):
            result = _normalize_moonshine_language("fr")
        assert result == "auto"
        assert "fr" in caplog.text and "not supported" in caplog.text


# ---------------------------------------------------------------------------
# Dispatch — transcribe_audio() routes provider=moonshine to _transcribe_moonshine
# ---------------------------------------------------------------------------


class TestTranscribeAudioMoonshineDispatch:
    """When ``provider: moonshine`` is configured, ``transcribe_audio`` must
    call ``_transcribe_moonshine`` (and not any other handler)."""

    def test_dispatch_routes_to_moonshine_handler(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_STT_LANGUAGE_ENV", "")  # no env override
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)  # any bytes; handler is mocked

        # Pytest's conftest redirects HERMES_HOME to a tempdir, so we have
        # to inject the moonshine config explicitly. The handler itself is
        # mocked so we don't need a real Moonshine model.
        moonshine_cfg = {
            "enabled": True,
            "provider": "moonshine",
            "moonshine": {"model": "tiny", "language": "ko", "timeout": 60},
        }
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=moonshine_cfg), \
             patch(
                 "tools.transcription_tools._transcribe_moonshine",
                 return_value={"success": True, "transcript": "안녕하세요", "provider": "moonshine"},
             ) as mock_handler:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(wav))

        assert result["success"] is True
        assert result["provider"] == "moonshine"
        assert result["transcript"] == "안녕하세요"
        # The handler was called exactly once with the audio path
        assert mock_handler.call_count == 1
        assert str(wav) in mock_handler.call_args.args[0]

    def test_dispatch_uses_configured_model_and_language(self, tmp_path, monkeypatch):
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        cfg = {
            "enabled": True,
            "provider": "moonshine",
            "moonshine": {"model": "medium", "language": "ja", "timeout": 60},
        }
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch(
                 "tools.transcription_tools._transcribe_moonshine",
                 return_value={"success": True, "transcript": "こんにちは", "provider": "moonshine"},
             ) as mock_handler:
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(wav))

        # model + moonshine_cfg should be passed through
        args, kwargs = mock_handler.call_args
        assert args[0] == str(wav)
        # model_name and moonshine_cfg are positional/keyword
        assert args[1] == "medium"  # model name
        assert args[2] == {"model": "medium", "language": "ja", "timeout": 60}
        assert result["transcript"] == "こんにちは"

    def test_handler_not_called_when_provider_is_other(self, tmp_path, monkeypatch):
        """Regression guard: moonshine handler must not be invoked for non-moonshine providers."""
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        cfg = {"enabled": True, "provider": "local"}
        with patch("tools.transcription_tools._validate_audio_file", return_value=None), \
             patch("tools.transcription_tools._load_stt_config", return_value=cfg), \
             patch("tools.transcription_tools._transcribe_moonshine") as mock_moonshine, \
             patch("tools.transcription_tools._transcribe_local",
                   return_value={"success": True, "transcript": "hello", "provider": "local"}):
            from tools.transcription_tools import transcribe_audio
            result = transcribe_audio(str(wav))

        assert result["provider"] == "local"
        mock_moonshine.assert_not_called()


# ---------------------------------------------------------------------------
# Error envelope shape — mirrors the StepFun / ElevenLabs / xai handlers
# ---------------------------------------------------------------------------


class TestTranscribeMoonshineErrorEnvelope:
    """The handler must return the standard error envelope shape so the
    gateway/CLI can rely on ``{success, transcript, error}``."""

    def test_missing_package_returns_helpful_error(self, tmp_path):
        wav = tmp_path / "voice.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("tools.transcription_tools._HAS_MOONSHINE", False):
            from tools.transcription_tools import _transcribe_moonshine
            result = _transcribe_moonshine(str(wav), "tiny", {})

        assert result["success"] is False
        assert result["transcript"] == ""
        assert "moonshine_voice" in result["error"]
        assert "pip install" in result["error"]

    def test_missing_file_returns_error(self):
        from tools.transcription_tools import _transcribe_moonshine
        result = _transcribe_moonshine("/nonexistent/path/voice.wav", "tiny", {})
        assert result["success"] is False
        assert "not found" in result["error"].lower()
