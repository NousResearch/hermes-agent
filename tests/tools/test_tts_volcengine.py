"""Tests for the Volcengine / Doubao TTS provider in tools/tts_tool.py."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("VOLCENGINE_TTS_API_KEY", "VOLC_ACCESS_KEY", "VOLC_APP_ID", "VOLCENGINE_TTS_APP_ID", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


class _MockStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line


class _MockJsonResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestGenerateVolcengineTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_volcengine_tts

        with pytest.raises(ValueError, match="VOLCENGINE_TTS_API_KEY / VOLC_ACCESS_KEY"):
            _generate_volcengine_tts("Hello", str(tmp_path / "test.mp3"), {})

    def test_missing_app_id_raises_value_error(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_volcengine_tts

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        with pytest.raises(ValueError, match="VOLCENGINE_TTS_APP_ID"):
            _generate_volcengine_tts("Hello", str(tmp_path / "test.mp3"), {})

    def test_successful_generation_writes_audio(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_volcengine_tts

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        monkeypatch.setenv("VOLC_APP_ID", "app-123")
        response = _MockJsonResponse({
            "code": 3000,
            "message": "Success",
            "data": base64.b64encode(b"hello-world").decode(),
        })

        with patch("requests.post", return_value=response) as mock_post:
            output_path = str(tmp_path / "test.mp3")
            result = _generate_volcengine_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.mp3").read_bytes() == b"hello-world"

        kwargs = mock_post.call_args[1]
        assert kwargs["headers"]["Authorization"] == "Bearer;test-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["json"]["app"]["appid"] == "app-123"
        assert kwargs["json"]["app"]["cluster"] == "volcano_tts"
        assert kwargs["json"]["audio"]["voice_type"] == "zh_female_vv_uranus_bigtts"
        assert kwargs["json"]["audio"]["encoding"] == "mp3"
        assert kwargs["json"]["request"]["text"] == "Hello world"
        assert kwargs["json"]["request"]["operation"] == "query"
        assert kwargs["timeout"] == 60

    def test_ogg_extension_uses_ogg_opus_format(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_volcengine_tts

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        monkeypatch.setenv("VOLC_APP_ID", "app-123")
        response = _MockJsonResponse({
            "code": 3000,
            "message": "Success",
            "data": base64.b64encode(b"audio").decode(),
        })

        with patch("requests.post", return_value=response) as mock_post:
            _generate_volcengine_tts("Hello", str(tmp_path / "test.ogg"), {})

        payload = mock_post.call_args[1]["json"]
        assert payload["audio"]["encoding"] == "ogg_opus"
        assert payload["audio"]["compression_rate"] == 1

    def test_global_speed_maps_to_speed_ratio(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_volcengine_tts

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        monkeypatch.setenv("VOLC_APP_ID", "app-123")
        response = _MockJsonResponse({
            "code": 3000,
            "message": "Success",
            "data": base64.b64encode(b"audio").decode(),
        })

        with patch("requests.post", return_value=response) as mock_post:
            _generate_volcengine_tts("Hello", str(tmp_path / "test.mp3"), {"speed": 1.5})

        payload = mock_post.call_args[1]["json"]
        assert payload["audio"]["encoding"] == "mp3"
        assert payload["audio"]["speed_ratio"] == 1.5
        assert payload["audio"]["volume_ratio"] == 1.0
        assert payload["audio"]["pitch_ratio"] == 1.0

    def test_provider_speed_overrides_global_speed(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_volcengine_tts

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        monkeypatch.setenv("VOLC_APP_ID", "app-123")
        response = _MockJsonResponse({
            "code": 3000,
            "message": "Success",
            "data": base64.b64encode(b"audio").decode(),
        })

        with patch("requests.post", return_value=response) as mock_post:
            _generate_volcengine_tts(
                "Hello",
                str(tmp_path / "test.mp3"),
                {"speed": 1.5, "volcengine": {"speed": 0.8}},
            )

        payload = mock_post.call_args[1]["json"]
        assert payload["audio"]["speed_ratio"] == 0.8

    def test_api_error_code_raises_runtime_error(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_volcengine_tts

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        monkeypatch.setenv("VOLC_APP_ID", "app-123")
        response = _MockJsonResponse({"code": 3031, "message": "bad request", "data": None})

        with patch("requests.post", return_value=response):
            with pytest.raises(RuntimeError, match="bad request"):
                _generate_volcengine_tts("Hello", str(tmp_path / "test.mp3"), {})


class TestTtsDispatcherVolcengine:
    def test_dispatcher_routes_to_volcengine(self, tmp_path, monkeypatch):
        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        monkeypatch.setenv("VOLC_APP_ID", "app-123")
        response = _MockJsonResponse({
            "code": 3000,
            "message": "Success",
            "data": base64.b64encode(b"audio").decode(),
        })

        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "volcengine"}), \
             patch("requests.post", return_value=response):
            result = json.loads(text_to_speech_tool("Hello", output_path=str(tmp_path / "out.mp3")))

        assert result["success"] is True
        assert result["provider"] == "volcengine"


class TestCheckTtsRequirementsVolcengine:
    def test_volcengine_key_and_app_id_return_true(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        monkeypatch.setenv("VOLC_APP_ID", "app-123")
        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._import_mistral_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False):
            assert check_tts_requirements() is True

    def test_volcengine_key_without_app_id_returns_false(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "test-key")
        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._import_mistral_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False):
            assert check_tts_requirements() is False

    def test_volcengine_key_missing_returns_false(self):
        from tools.tts_tool import check_tts_requirements

        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._import_mistral_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False):
            assert check_tts_requirements() is False
