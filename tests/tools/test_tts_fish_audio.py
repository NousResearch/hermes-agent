"""Tests for the Fish Audio TTS provider in tools/tts_tool.py."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "FISH_AUDIO_API_KEY",
        "FISH_API_KEY",
        "FISH_AUDIO_BASE_URL",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


def _mock_audio_response(content=b"fish-audio"):
    response = MagicMock()
    response.status_code = 200
    response.content = content
    response.text = ""
    return response


class TestGenerateFishAudioTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_fish_audio_tts

        with pytest.raises(ValueError, match="FISH_AUDIO_API_KEY"):
            _generate_fish_audio_tts("Hello", str(tmp_path / "out.mp3"), {})

    def test_successful_generation_posts_documented_payload(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_fish_audio_tts

        monkeypatch.setenv("FISH_AUDIO_API_KEY", "test-key")
        response = _mock_audio_response(b"audio-bytes")
        output_path = str(tmp_path / "out.mp3")

        with patch("requests.post", return_value=response) as mock_post:
            result = _generate_fish_audio_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "out.mp3").read_bytes() == b"audio-bytes"
        mock_post.assert_called_once()

        call_args = mock_post.call_args
        assert call_args.args[0] == "https://api.fish.audio/v1/tts"
        assert call_args.kwargs["headers"] == {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
            "model": "s2-pro",
        }
        assert call_args.kwargs["json"] == {
            "text": "Hello world",
            "format": "mp3",
            "latency": "normal",
        }

    def test_reference_id_model_speed_and_alias_config(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_fish_audio_tts

        monkeypatch.setenv("FISH_API_KEY", "fallback-key")
        config = {
            "speed": 0.8,
            "fish_audio": {
                "model": "s1",
                "reference_id": "voice-id",
                "speed": 1.5,
                "temperature": 0.4,
                "format": "wav",
            },
        }

        with patch("requests.post", return_value=_mock_audio_response()) as mock_post:
            _generate_fish_audio_tts("Hi", str(tmp_path / "out.mp3"), config)

        headers = mock_post.call_args.kwargs["headers"]
        payload = mock_post.call_args.kwargs["json"]
        assert headers["Authorization"] == "Bearer fallback-key"
        assert headers["model"] == "s1"
        assert payload["reference_id"] == "voice-id"
        assert payload["format"] == "wav"
        assert payload["prosody"] == {"speed": 1.5}
        assert payload["temperature"] == 0.4

    @pytest.mark.parametrize(
        "filename, expected_format",
        [("out.mp3", "mp3"), ("out.ogg", "opus"), ("out.wav", "wav"), ("out.pcm", "pcm")],
    )
    def test_output_format_from_extension(
        self, tmp_path, monkeypatch, filename, expected_format
    ):
        from tools.tts_tool import _generate_fish_audio_tts

        monkeypatch.setenv("FISH_AUDIO_API_KEY", "test-key")
        with patch("requests.post", return_value=_mock_audio_response()) as mock_post:
            _generate_fish_audio_tts("Hi", str(tmp_path / filename), {})

        assert mock_post.call_args.kwargs["json"]["format"] == expected_format

    def test_api_error_message_is_surfaced(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_fish_audio_tts

        monkeypatch.setenv("FISH_AUDIO_API_KEY", "test-key")
        response = MagicMock()
        response.status_code = 422
        response.text = "bad request"
        response.json.return_value = {"message": "missing voice"}

        with patch("requests.post", return_value=response):
            with pytest.raises(RuntimeError, match="missing voice"):
                _generate_fish_audio_tts("Hi", str(tmp_path / "out.mp3"), {})


class TestTtsDispatcherFishAudio:
    def test_dispatcher_routes_to_fish(self, tmp_path, monkeypatch):
        from tools import tts_tool

        output_path = str(tmp_path / "out.mp3")

        def fake_generate(_text, path, _config):
            with open(path, "wb") as f:
                f.write(b"audio")
            return path

        monkeypatch.setenv("FISH_AUDIO_API_KEY", "test-key")
        with patch(
            "tools.tts_tool._load_tts_config", return_value={"provider": "fish"}
        ), patch.object(tts_tool, "_generate_fish_audio_tts", side_effect=fake_generate):
            result = json.loads(tts_tool.text_to_speech_tool("Hello", output_path=output_path))

        assert result["success"] is True
        assert result["provider"] == "fish"
        assert result["file_path"] == output_path


class TestCheckTtsRequirementsFishAudio:
    def test_fish_audio_key_returns_true_when_other_providers_unavailable(
        self, monkeypatch
    ):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("FISH_AUDIO_API_KEY", "test-key")
        with patch("tools.tts_tool._has_any_command_tts_provider", return_value=False), \
             patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._import_mistral_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False), \
             patch("tools.tts_tool._check_kittentts_available", return_value=False), \
             patch("tools.tts_tool._check_piper_available", return_value=False):
            assert check_tts_requirements() is True
