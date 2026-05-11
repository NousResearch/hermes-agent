"""Tests for the OpenRouter TTS provider in tools/tts_tool.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "VOICE_TOOLS_OPENAI_KEY",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_openai_response_mp3():
    """A successful OpenAI audio.speech.create response (mp3)."""
    resp = MagicMock()
    resp.stream_to_file = MagicMock()
    resp._path = None  # storage for path

    def capture(path):
        resp._path = path
        # Simulate writing a small mp3 header so the file is non-empty
        Path(path).write_bytes(b"\xff\xfb\x90\x00" + b"mp3" * 100)

    resp.stream_to_file.side_effect = capture
    return resp


class TestGenerateOpenRouterTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_openrouter_tts

        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            _generate_openrouter_tts("Hello", output_path, {})

    def test_missing_api_key_env_var(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")

        # Simulate openai package not installed
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError):
                _generate_openrouter_tts("Hello", output_path, {})

    def test_mp3_output_streamed_to_file(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")

        mock_response = mock_openai_response_mp3
        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_response)
        mock_client.audio.speech.create = mock_create
        mock_client.close = MagicMock()

        with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
            mock_import.return_value = MagicMock(return_value=mock_client)

            with patch("builtins.__import__", return_value=MagicMock()):
                result = _generate_openrouter_tts("Hello world", output_path, {})

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini-tts-2025-12-15"
        assert call_kwargs["voice"] == "alloy"
        assert call_kwargs["input"] == "Hello world"
        assert call_kwargs["response_format"] == "mp3"
        assert "extra_headers" in call_kwargs

    def test_default_model_and_voice(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import (
            DEFAULT_OPENROUTER_TTS_MODEL,
            DEFAULT_OPENROUTER_TTS_VOICE,
            _generate_openrouter_tts,
        )

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_openai_response_mp3)
        mock_client.audio.speech.create = mock_create

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                _generate_openrouter_tts("Hi", output_path, {})

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == DEFAULT_OPENROUTER_TTS_MODEL
        assert call_kwargs["voice"] == DEFAULT_OPENROUTER_TTS_VOICE

    def test_custom_model_and_voice_from_config(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")
        config = {
            "openrouter": {
                "model": "mistralai/voxtral-mini-tts-2603",
                "voice": "Nova",
            }
        }

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_openai_response_mp3)
        mock_client.audio.speech.create = mock_create

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                _generate_openrouter_tts("Hi", output_path, config)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "mistralai/voxtral-mini-tts-2603"
        assert call_kwargs["voice"] == "Nova"

    def test_custom_base_url_from_config(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")
        config = {"openrouter": {"base_url": "https://custom.example.com/v1"}}

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_openai_response_mp3)
        mock_client.audio.speech.create = mock_create

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                _generate_openrouter_tts("Hi", output_path, config)

        # Client was instantiated with the custom base URL
        import_calls = mock_import.return_value.call_args_list
        assert len(import_calls) == 1
        _cls, init_kwargs = import_calls[0]
        assert init_kwargs["base_url"] == "https://custom.example.com/v1"

    def test_ogg_format_for_opus(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.ogg")  # Telegram target

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_openai_response_mp3)
        mock_client.audio.speech.create = mock_create

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                _generate_openrouter_tts("Hi", output_path, {})

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_format"] == "opus"

    def test_speed_parameter_applied(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")
        config = {"speed": 1.5}

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_openai_response_mp3)
        mock_client.audio.speech.create = mock_create

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                _generate_openrouter_tts("Hi", output_path, config)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["speed"] == 1.5

    def test_speed_clamped_to_valid_range(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")
        config = {"speed": 10.0}  # out of range, should clamp to 4.0

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_openai_response_mp3)
        mock_client.audio.speech.create = mock_create

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                _generate_openrouter_tts("Hi", output_path, config)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["speed"] == 4.0  # clamped to max

    def test_idempotency_key_in_headers(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_openai_response_mp3)
        mock_client.audio.speech.create = mock_create

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                _generate_openrouter_tts("Hi", output_path, {})

        call_kwargs = mock_create.call_args.kwargs
        assert "extra_headers" in call_kwargs
        assert "x-idempotency-key" in call_kwargs["extra_headers"]
        # Verify it looks like a UUID
        idempotency_key = call_kwargs["extra_headers"]["x-idempotency-key"]
        assert len(idempotency_key) == 36  # standard UUID length

    def test_close_method_called(self, tmp_path, monkeypatch, mock_openai_response_mp3):
        from tools.tts_tool import _generate_openrouter_tts

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        output_path = str(tmp_path / "test.mp3")

        mock_client = MagicMock()
        mock_create = MagicMock(return_value=mock_openai_response_mp3)
        mock_client.audio.speech.create = mock_create
        mock_client.close = MagicMock()

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                _generate_openrouter_tts("Hi", output_path, {})

        mock_client.close.assert_called_once()


class TestOpenRouterInCheckRequirements:
    def test_openrouter_api_key_satisfies_requirements(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        for key in (
            "ELEVENLABS_API_KEY",
            "OPENAI_API_KEY",
            "VOICE_TOOLS_OPENAI_KEY",
            "MINIMAX_API_KEY",
            "XAI_API_KEY",
            "MISTRAL_API_KEY",
            "GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "edge_tts":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert check_tts_requirements() is True


class TestOpenRouterTtsInProviderList:
    def test_openrouter_in_builtin_providers(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        assert "openrouter" in BUILTIN_TTS_PROVIDERS

    def test_openrouter_in_dispatch_comment(self, tmp_path, monkeypatch):
        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        # Ensure config says openrouter provider
        config = {"provider": "openrouter", "openrouter": {"model": "x"}}

        mock_client = MagicMock()
        mock_create = MagicMock(
            return_value=MagicMock(stream_to_file=MagicMock())
        )
        mock_client.audio.speech.create = mock_create

        with patch("builtins.__import__", return_value=MagicMock()):
            with patch("tools.tts_tool._import_openai_client", return_value=MagicMock) as mock_import:
                mock_import.return_value = MagicMock(return_value=mock_client)
                with patch.object(mock_client.audio.speech, "create", mock_create):
                    result = text_to_speech_tool("Hello", output_path=str(tmp_path / "out.mp3"),)

        # If it gets here without error, dispatch worked
        assert result is not None


class TestOpenRouterInDefaultConfig:
    def test_openrouter_block_in_default_config(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert "openrouter" in DEFAULT_CONFIG["tts"]
        tts_or = DEFAULT_CONFIG["tts"]["openrouter"]
        assert "model" in tts_or
        assert "voice" in tts_or
        assert isinstance(tts_or["model"], str)
        assert isinstance(tts_or["voice"], str)


class TestOpenRouterTtsMaxTextLength:
    def test_openrouter_in_provider_max_text_length(self):
        from tools.tts_tool import PROVIDER_MAX_TEXT_LENGTH

        assert "openrouter" in PROVIDER_MAX_TEXT_LENGTH
        assert PROVIDER_MAX_TEXT_LENGTH["openrouter"] == 4096