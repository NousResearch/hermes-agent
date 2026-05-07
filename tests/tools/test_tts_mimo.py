"""Tests for the Xiaomi MiMo TTS provider in tools/tts_tool.py."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("XIAOMI_API_KEY", "XIAOMI_BASE_URL", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


def _make_mimo_response(audio_bytes: bytes = b"fake-wav-audio") -> dict:
    """Build a fake MiMo TTS chat completions response."""
    return {
        "id": "test-id",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "",
                    "role": "assistant",
                    "audio": {
                        "data": base64.b64encode(audio_bytes).decode(),
                    },
                },
            }
        ],
        "model": "mimo-v2.5-tts",
        "usage": {"completion_tokens": 5, "prompt_tokens": 100, "total_tokens": 105},
    }


class TestGenerateMimoTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_mimo_tts

        output_path = str(tmp_path / "test.wav")
        with pytest.raises(ValueError, match="XIAOMI_API_KEY"):
            _generate_mimo_tts("Hello", output_path, {})

    def test_successful_generation(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key-123")
        audio_content = b"RIFF fake-wav-data"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_mimo_response(audio_content)
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            output_path = str(tmp_path / "test.wav")
            result = _generate_mimo_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.wav").read_bytes() == audio_content
        mock_post.assert_called_once()

    def test_default_voice_and_model(self, tmp_path, monkeypatch):
        from tools.tts_tool import DEFAULT_MIMO_MODEL, DEFAULT_MIMO_VOICE, _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = _make_mimo_response()
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_mimo_tts("Hi", str(tmp_path / "test.wav"), {})

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == DEFAULT_MIMO_MODEL
        assert payload["audio"]["voice"] == DEFAULT_MIMO_VOICE

    def test_custom_voice_from_config(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = _make_mimo_response()
        mock_response.raise_for_status = MagicMock()

        config = {"mimo": {"voice": "苏打"}}
        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_mimo_tts("Hi", str(tmp_path / "test.wav"), config)

        payload = mock_post.call_args[1]["json"]
        assert payload["audio"]["voice"] == "苏打"

    def test_custom_model_from_config(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = _make_mimo_response()
        mock_response.raise_for_status = MagicMock()

        config = {"mimo": {"model": "mimo-v2.5-tts-voicedesign"}}
        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_mimo_tts("Hi", str(tmp_path / "test.wav"), config)

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "mimo-v2.5-tts-voicedesign"

    def test_custom_base_url_from_env(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        monkeypatch.setenv("XIAOMI_BASE_URL", "https://custom.mimo.api/v1")
        mock_response = MagicMock()
        mock_response.json.return_value = _make_mimo_response()
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_mimo_tts("Hi", str(tmp_path / "test.wav"), {})

        call_url = mock_post.call_args[0][0]
        assert call_url == "https://custom.mimo.api/v1/chat/completions"

    def test_custom_base_url_from_config(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = _make_mimo_response()
        mock_response.raise_for_status = MagicMock()

        config = {"mimo": {"base_url": "https://config-url/v1"}}
        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_mimo_tts("Hi", str(tmp_path / "test.wav"), config)

        call_url = mock_post.call_args[0][0]
        assert call_url == "https://config-url/v1/chat/completions"

    def test_api_key_header_format(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "my-secret-key")
        mock_response = MagicMock()
        mock_response.json.return_value = _make_mimo_response()
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_mimo_tts("Hi", str(tmp_path / "test.wav"), {})

        headers = mock_post.call_args[1]["headers"]
        assert headers["api-key"] == "my-secret-key"

    def test_request_payload_shape(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = _make_mimo_response()
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_mimo_tts("Hello world", str(tmp_path / "test.wav"), {})

        payload = mock_post.call_args[1]["json"]
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][1]["content"] == "Hello world"
        assert payload["audio"]["format"] == "wav"

    def test_http_error_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(Exception, match="401"):
                _generate_mimo_tts("Hi", str(tmp_path / "test.wav"), {})

    def test_malformed_response_raises_runtime_error(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "bad request"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="unexpected response format"):
                _generate_mimo_tts("Hi", str(tmp_path / "test.wav"), {})

    def test_empty_audio_data_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_mimo_tts

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = _make_mimo_response(b"")
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            output_path = str(tmp_path / "test.wav")
            _generate_mimo_tts("Hi", output_path, {})
            assert (tmp_path / "test.wav").exists()


class TestMimoInCheckRequirements:
    def test_returns_true_when_api_key_present(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
        assert check_tts_requirements() is True

    def test_returns_false_when_no_providers(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        for key in (
            "XIAOMI_API_KEY", "XAI_API_KEY", "MINIMAX_API_KEY",
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY",
            "ELEVENLABS_API_KEY", "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        result = check_tts_requirements()
        assert isinstance(result, bool)


class TestMimoInBuiltinProviders:
    def test_mimo_in_builtin_set(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        assert "mimo" in BUILTIN_TTS_PROVIDERS

    def test_mimo_max_text_length(self):
        from tools.tts_tool import PROVIDER_MAX_TEXT_LENGTH

        assert "mimo" in PROVIDER_MAX_TEXT_LENGTH
        assert PROVIDER_MAX_TEXT_LENGTH["mimo"] > 0
