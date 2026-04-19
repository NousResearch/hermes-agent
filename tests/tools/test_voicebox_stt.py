import json
from unittest.mock import patch

import pytest


@pytest.fixture
def sample_wav(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF0000WAVEfmt ")
    return str(path)


class TestVoiceboxProviderSelection:
    def test_explicit_voicebox_provider_selected(self):
        from tools.transcription_tools import _get_provider

        assert _get_provider({"provider": "voicebox", "voicebox": {"base_url": "http://localhost:17493"}}) == "voicebox"


class TestTranscribeVoicebox:
    def test_missing_base_url_raises(self, sample_wav):
        from tools.transcription_tools import _transcribe_voicebox

        with pytest.raises(ValueError, match="base_url"):
            _transcribe_voicebox(sample_wav, "base", {"voicebox": {"base_url": ""}})

    def test_successful_transcription(self, sample_wav):
        from tools.transcription_tools import _transcribe_voicebox

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"text": "hello voicebox", "duration": 1.2}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = _transcribe_voicebox(
                sample_wav,
                "turbo",
                {"voicebox": {"base_url": "http://localhost:17493", "language": "ko"}},
            )

        assert result["success"] is True
        assert result["transcript"] == "hello voicebox"
        assert result["provider"] == "voicebox"


class TestTranscribeAudioDispatchVoicebox:
    def test_transcribe_audio_dispatches_to_voicebox(self, sample_wav):
        from tools.transcription_tools import transcribe_audio

        with patch("tools.transcription_tools._load_stt_config", return_value={"provider": "voicebox", "voicebox": {"base_url": "http://localhost:17493", "model": "turbo"}}), \
             patch("tools.transcription_tools._transcribe_voicebox", return_value={"success": True, "transcript": "from dispatch", "provider": "voicebox"}):
            result = transcribe_audio(sample_wav)

        assert result["success"] is True
        assert result["transcript"] == "from dispatch"
        assert result["provider"] == "voicebox"
