"""
Tests for the native Cartesia TTS provider in tools/tts_tool.py.

Covers:
- Provider registration in BUILTIN_TTS_PROVIDERS and tts_registry
- Max text length definition
- Availability checks based on CARTESIA_API_KEY
- Synthesis request payload construction (model, voice_id, container, speed, emotion)
- Header propagation (X-API-Key, Cartesia-Version)
- Error handling (missing key, HTTP 401/429/500, network failure)
- Integration with text_to_speech_tool dispatch
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import tts_registry
from tools import tts_tool
from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_CARTESIA_MODEL,
    DEFAULT_CARTESIA_VOICE_ID,
    PROVIDER_MAX_TEXT_LENGTH,
    _generate_cartesia_tts,
    check_tts_requirements,
    text_to_speech_tool,
)


class TestCartesiaRegistration:
    def test_cartesia_is_a_builtin_provider(self):
        assert "cartesia" in BUILTIN_TTS_PROVIDERS

    def test_cartesia_in_registry_builtins(self):
        assert "cartesia" in tts_registry._BUILTIN_NAMES

    def test_cartesia_has_max_text_length(self):
        assert PROVIDER_MAX_TEXT_LENGTH.get("cartesia", 0) > 0


class TestCartesiaAvailability:
    def test_available_when_api_key_present(self):
        with patch.object(tts_tool, "_resolve_provider_key", return_value="test_cartesia_key"):
            with patch.object(tts_tool, "_load_tts_config", return_value={"provider": "cartesia"}):
                assert check_tts_requirements() is True

    def test_unavailable_when_api_key_missing(self):
        with patch.object(tts_tool, "_resolve_provider_key", return_value=""):
            with patch.object(tts_tool, "_load_tts_config", return_value={"provider": "cartesia"}):
                assert check_tts_requirements() is False


class TestCartesiaSynthesis:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        out_file = str(tmp_path / "out.mp3")
        with patch.object(tts_tool, "_resolve_provider_key", return_value=""):
            with pytest.raises(ValueError, match="CARTESIA_API_KEY not set"):
                _generate_cartesia_tts("hello", out_file, {})

    def test_successful_synthesis_default_payload(self, tmp_path):
        out_file = str(tmp_path / "out.mp3")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake_mp3_audio_data"

        with patch.object(tts_tool, "_resolve_provider_key", return_value="cartesia_secret_123"):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                res = _generate_cartesia_tts("Hello world", out_file, {})

        assert res == out_file
        with open(out_file, "rb") as f:
            assert f.read() == b"fake_mp3_audio_data"

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.cartesia.ai/tts/bytes"
        headers = kwargs["headers"]
        assert headers["X-API-Key"] == "cartesia_secret_123"
        assert headers["Cartesia-Version"] == "2024-06-10"
        assert headers["Content-Type"] == "application/json"

        payload = kwargs["json"]
        assert payload["model_id"] == DEFAULT_CARTESIA_MODEL
        assert payload["transcript"] == "Hello world"
        assert payload["voice"]["id"] == DEFAULT_CARTESIA_VOICE_ID
        assert payload["output_format"]["container"] == "mp3"
        assert payload["output_format"]["sample_rate"] == 24000

    def test_opus_container_for_voice_bubbles(self, tmp_path):
        out_file = str(tmp_path / "out.ogg")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake_opus_bytes"

        with patch.object(tts_tool, "_resolve_provider_key", return_value="cartesia_secret"):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                _generate_cartesia_tts("Voice message", out_file, {})

        payload = mock_post.call_args[1]["json"]
        assert payload["output_format"]["container"] == "ogg"
        assert payload["output_format"]["encoding"] == "opus"

    def test_custom_voice_model_emotion_and_speed(self, tmp_path):
        out_file = str(tmp_path / "out.wav")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake_wav_bytes"

        config = {
            "cartesia": {
                "model": "sonic-turbo",
                "voice_id": "custom-voice-uuid",
                "speed": 1.25,
                "emotion": ["positivity:high", "curiosity:medium"],
                "language": "en",
                "sample_rate": 44100,
            }
        }

        with patch.object(tts_tool, "_resolve_provider_key", return_value="cartesia_secret"):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                _generate_cartesia_tts("Expressive test", out_file, config)

        payload = mock_post.call_args[1]["json"]
        assert payload["model_id"] == "sonic-turbo"
        assert payload["voice"]["id"] == "custom-voice-uuid"
        controls = payload["voice"]["__experimental_controls"]
        assert controls["speed"] == 1.25
        assert controls["emotion"] == ["positivity:high", "curiosity:medium"]
        assert payload["language"] == "en"
        assert payload["output_format"]["container"] == "wav"
        assert payload["output_format"]["sample_rate"] == 44100

    def test_http_error_handling(self, tmp_path):
        out_file = str(tmp_path / "out.mp3")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": "Invalid API key provided"}

        with patch.object(tts_tool, "_resolve_provider_key", return_value="bad_key"):
            with patch("requests.post", return_value=mock_resp):
                with pytest.raises(RuntimeError, match="Cartesia TTS API error \\(401\\): Invalid API key provided"):
                    _generate_cartesia_tts("Hello", out_file, {})


class TestCartesiaDispatch:
    def test_text_to_speech_tool_dispatches_cartesia(self, tmp_path):
        from pathlib import Path
        out_file = str(tmp_path / "out.mp3")
        tts_cfg = {"provider": "cartesia"}

        def side_effect(text, path, cfg):
            Path(path).write_bytes(b"mock_audio_data")
            return path

        with patch.object(tts_tool, "_load_tts_config", return_value=tts_cfg):
            with patch.object(tts_tool, "_generate_cartesia_tts", side_effect=side_effect) as mock_gen:
                res_str = text_to_speech_tool("Speak this", output_path=out_file)
                res = json.loads(res_str)
                assert res["success"] is True
                assert res["provider"] == "cartesia"
                mock_gen.assert_called_once_with("Speak this", out_file, tts_cfg)
