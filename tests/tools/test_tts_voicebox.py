import json
from unittest.mock import patch

import pytest


class TestGenerateVoiceboxTts:
    def test_missing_profile_id_raises(self, tmp_path):
        from tools.tts_tool import _generate_voicebox_tts

        with pytest.raises(ValueError, match="profile_id"):
            _generate_voicebox_tts("hello", str(tmp_path / "out.wav"), {"voicebox": {"base_url": "http://localhost:17493"}})

    def test_successful_generation_downloads_audio(self, tmp_path):
        from tools.tts_tool import _generate_voicebox_tts

        responses = [
            {"id": "gen-123"},
            {"id": "gen-123", "status": "completed", "audio_path": "/tmp/fake.wav", "error": None},
            b"FAKEAUDIO",
        ]

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                if isinstance(self.payload, bytes):
                    return self.payload
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            return FakeResponse(responses.pop(0))

        output_path = str(tmp_path / "out.wav")
        with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("tools.tts_tool.time.sleep", return_value=None):
            result = _generate_voicebox_tts(
                "hello world",
                output_path,
                {"voicebox": {"base_url": "http://localhost:17493", "profile_id": "profile-1", "language": "ko"}},
            )

        assert result == output_path
        assert (tmp_path / "out.wav").read_bytes() == b"FAKEAUDIO"


class TestVoiceboxTtsDispatcher:
    def test_dispatcher_routes_to_voicebox(self, tmp_path):
        from tools.tts_tool import text_to_speech_tool

        output_path = str(tmp_path / "out.wav")
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "voicebox", "voicebox": {"base_url": "http://localhost:17493", "profile_id": "profile-1"}}), \
             patch("tools.tts_tool._generate_voicebox_tts", side_effect=lambda text, path, cfg: open(path, "wb").write(b"VOICEBOX") or path):
            result = json.loads(text_to_speech_tool("hello", output_path=output_path))

        assert result["success"] is True
        assert result["provider"] == "voicebox"
        assert result["file_path"] == output_path


class TestVoiceboxTtsRequirements:
    def test_voicebox_configured_returns_true(self):
        from tools.tts_tool import check_tts_requirements

        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._import_mistral_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "voicebox", "voicebox": {"base_url": "http://localhost:17493", "profile_id": "profile-1"}}):
            assert check_tts_requirements() is True
