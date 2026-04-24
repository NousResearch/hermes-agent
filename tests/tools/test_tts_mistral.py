"""Tests for the Mistral (Voxtral) TTS provider in tools/tts_tool.py."""

import base64
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("MISTRAL_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_mistral_module():
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_mistral_cls = MagicMock(return_value=mock_client)
    fake_module = MagicMock()
    fake_module.Mistral = mock_mistral_cls
    with patch.dict("sys.modules", {"mistralai": fake_module, "mistralai.client": fake_module}):
        yield mock_client


class TestGenerateMistralTts:
    def test_missing_api_key_raises_value_error(self, tmp_path, mock_mistral_module):
        from tools.tts_tool import _generate_mistral_tts

        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            _generate_mistral_tts("Hello", output_path, {})

    def test_successful_generation(self, tmp_path, mock_mistral_module, monkeypatch):
        from tools.tts_tool import _generate_mistral_tts

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        audio_content = b"fake-audio-bytes"
        mock_mistral_module.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(audio_content).decode()
        )

        output_path = str(tmp_path / "test.mp3")
        result = _generate_mistral_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.mp3").read_bytes() == audio_content
        mock_mistral_module.audio.speech.complete.assert_called_once()
        mock_mistral_module.__exit__.assert_called_once()
        call_kwargs = mock_mistral_module.audio.speech.complete.call_args[1]
        assert call_kwargs["input"] == "Hello world"
        assert call_kwargs["response_format"] == "mp3"

    @pytest.mark.parametrize(
        "extension, expected_format",
        [(".ogg", "opus"), (".wav", "wav"), (".flac", "flac"), (".mp3", "mp3")],
    )
    def test_response_format_from_extension(
        self, tmp_path, mock_mistral_module, monkeypatch, extension, expected_format
    ):
        from tools.tts_tool import _generate_mistral_tts

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(b"data").decode()
        )

        output_path = str(tmp_path / f"test{extension}")
        _generate_mistral_tts("Hi", output_path, {})

        call_kwargs = mock_mistral_module.audio.speech.complete.call_args[1]
        assert call_kwargs["response_format"] == expected_format

    def test_voice_id_passed_when_configured(
        self, tmp_path, mock_mistral_module, monkeypatch
    ):
        from tools.tts_tool import _generate_mistral_tts

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(b"data").decode()
        )

        config = {"mistral": {"voice_id": "my-voice-uuid"}}
        _generate_mistral_tts("Hi", str(tmp_path / "test.mp3"), config)

        call_kwargs = mock_mistral_module.audio.speech.complete.call_args[1]
        assert call_kwargs["voice_id"] == "my-voice-uuid"

    def test_default_voice_id_when_absent(
        self, tmp_path, mock_mistral_module, monkeypatch
    ):
        from tools.tts_tool import DEFAULT_MISTRAL_TTS_VOICE_ID, _generate_mistral_tts

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(b"data").decode()
        )

        _generate_mistral_tts("Hi", str(tmp_path / "test.mp3"), {})

        call_kwargs = mock_mistral_module.audio.speech.complete.call_args[1]
        assert call_kwargs["voice_id"] == DEFAULT_MISTRAL_TTS_VOICE_ID

    def test_default_voice_id_when_empty_string(
        self, tmp_path, mock_mistral_module, monkeypatch
    ):
        from tools.tts_tool import DEFAULT_MISTRAL_TTS_VOICE_ID, _generate_mistral_tts

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(b"data").decode()
        )

        config = {"mistral": {"voice_id": ""}}
        _generate_mistral_tts("Hi", str(tmp_path / "test.mp3"), config)

        call_kwargs = mock_mistral_module.audio.speech.complete.call_args[1]
        assert call_kwargs["voice_id"] == DEFAULT_MISTRAL_TTS_VOICE_ID

    def test_api_error_sanitized(self, tmp_path, mock_mistral_module, monkeypatch):
        from tools.tts_tool import _generate_mistral_tts

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.speech.complete.side_effect = RuntimeError(
            "secret-key-in-error"
        )

        with pytest.raises(RuntimeError, match="RuntimeError") as exc_info:
            _generate_mistral_tts("Hello", str(tmp_path / "test.mp3"), {})
        assert "secret-key-in-error" not in str(exc_info.value)

    def test_default_model_used(self, tmp_path, mock_mistral_module, monkeypatch):
        from tools.tts_tool import DEFAULT_MISTRAL_TTS_MODEL, _generate_mistral_tts

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(b"data").decode()
        )

        _generate_mistral_tts("Hi", str(tmp_path / "test.mp3"), {})

        call_kwargs = mock_mistral_module.audio.speech.complete.call_args[1]
        assert call_kwargs["model"] == DEFAULT_MISTRAL_TTS_MODEL

    def test_model_from_config_overrides_default(
        self, tmp_path, mock_mistral_module, monkeypatch
    ):
        from tools.tts_tool import _generate_mistral_tts

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(b"data").decode()
        )

        config = {"mistral": {"model": "voxtral-large-tts-9999"}}
        _generate_mistral_tts("Hi", str(tmp_path / "test.mp3"), config)

        call_kwargs = mock_mistral_module.audio.speech.complete.call_args[1]
        assert call_kwargs["model"] == "voxtral-large-tts-9999"


class TestTtsDispatcherMistral:
    def test_dispatcher_routes_to_mistral(
        self, tmp_path, mock_mistral_module, monkeypatch
    ):
        import json

        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        mock_mistral_module.audio.speech.complete.return_value = MagicMock(
            audio_data=base64.b64encode(b"audio").decode()
        )

        output_path = str(tmp_path / "out.mp3")
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "mistral"}):
            result = json.loads(text_to_speech_tool("Hello", output_path=output_path))

        assert result["success"] is True
        assert result["provider"] == "mistral"
        mock_mistral_module.audio.speech.complete.assert_called_once()

    def test_dispatcher_returns_error_when_sdk_not_installed(self, tmp_path, monkeypatch):
        import json

        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch(
            "tools.tts_tool._import_mistral_client", side_effect=ImportError("no module")
        ), patch("tools.tts_tool._load_tts_config", return_value={"provider": "mistral"}):
            result = json.loads(
                text_to_speech_tool("Hello", output_path=str(tmp_path / "out.mp3"))
            )

        assert result["success"] is False
        assert "mistralai" in result["error"]


class TestCheckTtsRequirementsMistral:
    def test_mistral_sdk_and_key_returns_true(self, mock_mistral_module, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False):
            assert check_tts_requirements() is True

    def test_mistral_key_missing_returns_false(self, mock_mistral_module):
        from tools.tts_tool import check_tts_requirements

        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False):
            assert check_tts_requirements() is False
