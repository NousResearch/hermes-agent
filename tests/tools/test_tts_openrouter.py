"""Tests for the OpenRouter TTS provider in tools/tts_tool.py."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "TTS_OPENROUTER_BASE_URL",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_openai_client():
    """Patch _import_openai_client to return a fake OpenAI class.

    The fake stream_to_file just touches the output_path so the dispatcher's
    "file produced" check succeeds.
    """

    def _stream_to_file(path):
        with open(path, "wb") as fh:
            fh.write(b"fake-audio-mp3-bytes")

    response = MagicMock()
    response.stream_to_file.side_effect = _stream_to_file

    client = MagicMock()
    client.audio.speech.create.return_value = response

    fake_class = MagicMock(return_value=client)
    with patch("tools.tts_tool._import_openai_client", return_value=fake_class):
        yield client, fake_class


class TestGenerateOpenRouterTts:
    def test_missing_api_key_raises(self, tmp_path, mock_openai_client):
        from tools.tts_tool import _generate_openrouter_tts

        out = str(tmp_path / "out.mp3")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            _generate_openrouter_tts("hi", out, {})

    def test_successful_generation(self, tmp_path, mock_openai_client, monkeypatch):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        client, _ = mock_openai_client

        out = str(tmp_path / "out.mp3")
        result = _generate_openrouter_tts("Hello world", out, {})

        assert result == out
        assert (tmp_path / "out.mp3").exists()
        client.audio.speech.create.assert_called_once()
        kwargs = client.audio.speech.create.call_args.kwargs
        assert kwargs["input"] == "Hello world"
        # OR only accepts mp3 / pcm — provider always requests mp3 and lets
        # downstream ffmpeg convert to .ogg when the platform needs Opus.
        assert kwargs["response_format"] == "mp3"

    def test_default_model_and_voice(self, tmp_path, mock_openai_client, monkeypatch):
        from tools.tts_tool import (
            DEFAULT_OPENROUTER_TTS_MODEL,
            DEFAULT_OPENROUTER_TTS_VOICE,
            _generate_openrouter_tts,
        )

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        client, _ = mock_openai_client

        _generate_openrouter_tts("hi", str(tmp_path / "out.mp3"), {})

        kwargs = client.audio.speech.create.call_args.kwargs
        assert kwargs["model"] == DEFAULT_OPENROUTER_TTS_MODEL
        assert kwargs["voice"] == DEFAULT_OPENROUTER_TTS_VOICE

    def test_custom_model_and_voice(self, tmp_path, mock_openai_client, monkeypatch):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        client, _ = mock_openai_client

        cfg = {"openrouter": {"model": "google/gemini-flash-tts", "voice": "Kore"}}
        _generate_openrouter_tts("hi", str(tmp_path / "out.mp3"), cfg)

        kwargs = client.audio.speech.create.call_args.kwargs
        assert kwargs["model"] == "google/gemini-flash-tts"
        assert kwargs["voice"] == "Kore"

    def test_default_base_url(self, tmp_path, mock_openai_client, monkeypatch):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        _, fake_class = mock_openai_client

        _generate_openrouter_tts("hi", str(tmp_path / "out.mp3"), {})

        # The OpenAI client was constructed with OR's base_url
        ctor_kwargs = fake_class.call_args.kwargs
        assert ctor_kwargs["api_key"] == "or-test-key"
        assert "openrouter.ai/api/v1" in ctor_kwargs["base_url"]

    def test_base_url_override_via_config(self, tmp_path, mock_openai_client, monkeypatch):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        _, fake_class = mock_openai_client

        cfg = {"openrouter": {"base_url": "https://or.example.com/v1"}}
        _generate_openrouter_tts("hi", str(tmp_path / "out.mp3"), cfg)

        assert fake_class.call_args.kwargs["base_url"] == "https://or.example.com/v1"

    def test_base_url_override_via_env(self, tmp_path, mock_openai_client, monkeypatch):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        monkeypatch.setenv("TTS_OPENROUTER_BASE_URL", "https://env.example.com/v1")
        _, fake_class = mock_openai_client

        _generate_openrouter_tts("hi", str(tmp_path / "out.mp3"), {})

        assert fake_class.call_args.kwargs["base_url"] == "https://env.example.com/v1"

    def test_response_format_always_mp3_even_for_ogg_path(
        self, tmp_path, mock_openai_client, monkeypatch
    ):
        """OR's API does not accept opus — provider must request mp3 regardless of
        the output path extension. The downstream ffmpeg pass converts to Opus."""
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        client, _ = mock_openai_client

        _generate_openrouter_tts("hi", str(tmp_path / "out.ogg"), {})

        kwargs = client.audio.speech.create.call_args.kwargs
        assert kwargs["response_format"] == "mp3"

    def test_speed_clamped(self, tmp_path, mock_openai_client, monkeypatch):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        client, _ = mock_openai_client

        _generate_openrouter_tts(
            "hi", str(tmp_path / "out.mp3"), {"openrouter": {"speed": 99.0}}
        )

        kwargs = client.audio.speech.create.call_args.kwargs
        assert kwargs["speed"] == 4.0  # upper-clamp

    def test_speed_default_omitted(self, tmp_path, mock_openai_client, monkeypatch):
        """speed=1.0 should not be sent (matches OpenAI provider behavior)."""
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        client, _ = mock_openai_client

        _generate_openrouter_tts("hi", str(tmp_path / "out.mp3"), {})

        kwargs = client.audio.speech.create.call_args.kwargs
        assert "speed" not in kwargs


class TestTtsDispatcherOpenRouter:
    def test_dispatcher_routes_to_openrouter(
        self, tmp_path, mock_openai_client, monkeypatch
    ):
        """text_to_speech_tool with provider=openrouter routes through the new
        _generate_openrouter_tts function."""
        import json

        import tools.tts_tool as tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
        cfg = {"provider": "openrouter", "openrouter": {"voice": "alloy"}}

        with patch.object(tts, "_load_tts_config", return_value=cfg):
            result_json = tts.text_to_speech_tool("hello dispatcher")

        result = json.loads(result_json)
        assert result["success"] is True
        assert result["provider"] == "openrouter"

    def test_openrouter_in_builtin_provider_set(self):
        """openrouter must be in BUILTIN_TTS_PROVIDERS so user-declared
        ``tts.providers.openrouter`` command blocks can never shadow it."""
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        assert "openrouter" in BUILTIN_TTS_PROVIDERS

    def test_openrouter_has_max_text_length(self):
        from tools.tts_tool import PROVIDER_MAX_TEXT_LENGTH

        assert PROVIDER_MAX_TEXT_LENGTH["openrouter"] == 4096
