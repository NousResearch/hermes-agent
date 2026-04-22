"""Tests for the Deepgram (Aura) TTS provider in tools/tts_tool.py."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("DEEPGRAM_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


class TestGenerateDeepgramTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_deepgram_tts

        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
            _generate_deepgram_tts("Hello", output_path, {})

    def test_successful_generation(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_deepgram_tts

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        audio_content = b"fake-mp3-bytes"
        mock_response = MagicMock()
        mock_response.content = audio_content
        mock_response.status_code = 200

        with patch("tools.tts_tool.requests.post", return_value=mock_response) as mock_post:
            output_path = str(tmp_path / "test.mp3")
            result = _generate_deepgram_tts("Hello world", output_path, {})

            assert result == output_path
            assert (tmp_path / "test.mp3").read_bytes() == audio_content
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["json"] == {"text": "Hello world"}
            assert call_kwargs[1]["params"]["model"] == "aura-2-thalia-en"
            assert call_kwargs[1]["params"]["encoding"] == "mp3"

    @pytest.mark.parametrize(
        "extension, expected_encoding",
        [(".ogg", "opus"), (".wav", "linear16"), (".flac", "flac"), (".mp3", "mp3")],
    )
    def test_encoding_from_extension(self, tmp_path, monkeypatch, extension, expected_encoding):
        from tools.tts_tool import _generate_deepgram_tts

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.status_code = 200

        with patch("tools.tts_tool.requests.post", return_value=mock_response) as mock_post:
            output_path = str(tmp_path / f"test{extension}")
            _generate_deepgram_tts("Test", output_path, {})
            assert mock_post.call_args[1]["params"]["encoding"] == expected_encoding

    def test_model_from_config_overrides_default(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_deepgram_tts

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.status_code = 200

        with patch("tools.tts_tool.requests.post", return_value=mock_response) as mock_post:
            output_path = str(tmp_path / "test.mp3")
            _generate_deepgram_tts("Test", output_path, {"deepgram": {"model": "aura-2-orion-en"}})
            assert mock_post.call_args[1]["params"]["model"] == "aura-2-orion-en"

    def test_http_error_raises_runtime_error(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_deepgram_tts

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = Exception("400")

        with patch("tools.tts_tool.requests.post", return_value=mock_response):
            output_path = str(tmp_path / "test.mp3")
            with pytest.raises(RuntimeError):
                _generate_deepgram_tts("Test", output_path, {})

    def test_auth_header_uses_token_not_bearer(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_deepgram_tts

        monkeypatch.setenv("DEEPGRAM_API_KEY", "my-secret-key")
        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.status_code = 200

        with patch("tools.tts_tool.requests.post", return_value=mock_response) as mock_post:
            output_path = str(tmp_path / "test.mp3")
            _generate_deepgram_tts("Test", output_path, {})
            headers = mock_post.call_args[1]["headers"]
            assert headers["Authorization"] == "Token my-secret-key"
            assert "Bearer" not in headers["Authorization"]


class TestTtsDispatcherDeepgram:
    def test_dispatcher_routes_to_deepgram(self, tmp_path, monkeypatch):
        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.content = b"fake-audio"
        mock_response.status_code = 200

        with patch("tools.tts_tool.requests.post", return_value=mock_response):
            monkeypatch.setenv("HERMES_SESSION_PLATFORM", "")
            result = text_to_speech_tool("Hello", config_override={"tts": {"provider": "deepgram"}})
            import json
            data = json.loads(result)
            assert data["success"] is True
            assert data["provider"] == "deepgram"
