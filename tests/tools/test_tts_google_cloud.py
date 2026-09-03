"""Tests for the Google Cloud Text-to-Speech provider in tools/tts_tool.py."""

import os
from unittest.mock import MagicMock, patch
import pytest

from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_GOOGLE_CLOUD_TTS_LANGUAGE,
    DEFAULT_GOOGLE_CLOUD_TTS_VOICE,
    PROVIDER_MAX_TEXT_LENGTH,
    _generate_google_cloud_tts,
    _list_google_cloud_voices,
    _resolve_google_cloud_credentials,
    check_tts_requirements,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_audio_bytes():
    return b"FAKE_MP3_AUDIO_DATA"


@pytest.fixture
def mock_tts_client(fake_audio_bytes):
    client = MagicMock()
    response = MagicMock()
    response.audio_content = fake_audio_bytes
    client.synthesize_speech.return_value = response

    voice_mock = MagicMock()
    voice_mock.name = "en-US-Chirp3-HD-Charon"
    voice_mock.language_codes = ["en-US"]
    voice_mock.ssml_gender = 1
    voice_mock.natural_sample_rate_hertz = 24000

    list_voices_resp = MagicMock()
    list_voices_resp.voices = [voice_mock]
    client.list_voices.return_value = list_voices_resp

    return client


class TestGoogleCloudProviderRegistration:
    def test_in_builtin_providers(self):
        assert "google_cloud" in BUILTIN_TTS_PROVIDERS

    def test_max_text_length_defined(self):
        assert "google_cloud" in PROVIDER_MAX_TEXT_LENGTH
        assert PROVIDER_MAX_TEXT_LENGTH["google_cloud"] == 5000


class TestResolveGoogleCloudCredentials:
    def test_missing_credentials_raises_value_error(self):
        with patch("google.auth.default", side_effect=Exception("No ADC")):
            with pytest.raises(ValueError, match="Google Cloud TTS authentication failed"):
                _resolve_google_cloud_credentials({})

    def test_explicit_credentials_file_not_found_raises(self):
        config = {"credentials_file": "/nonexistent/path/sa.json"}
        with pytest.raises(ValueError, match="credentials file not found"):
            _resolve_google_cloud_credentials(config)

    def test_env_credentials_file_not_found_raises(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/path/sa.json")
        with pytest.raises(ValueError, match="credentials file not found"):
            _resolve_google_cloud_credentials({})

    def test_adc_success(self):
        mock_creds = MagicMock()
        with patch("google.auth.default", return_value=(mock_creds, "test-project")):
            creds = _resolve_google_cloud_credentials({})
            assert creds == mock_creds

    def test_valid_service_account_file(self, tmp_path):
        key_file = tmp_path / "sa.json"
        key_file.write_text('{"type": "service_account"}', encoding="utf-8")
        mock_creds = MagicMock()
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            creds = _resolve_google_cloud_credentials({"credentials_file": str(key_file)})
            assert creds == mock_creds


class TestGenerateGoogleCloudTts:
    def test_default_synthesis_mp3(self, tmp_path, mock_tts_client, fake_audio_bytes):
        output_path = str(tmp_path / "output.mp3")
        mock_creds = MagicMock()

        with patch("tools.tts_tool._resolve_google_cloud_credentials", return_value=mock_creds), \
             patch("tools.tts_tool._import_google_cloud_tts") as mock_import:
            mock_tts_module = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_tts_client
            mock_import.return_value = mock_tts_module

            result = _generate_google_cloud_tts("Hello world", output_path, {})

        assert result == output_path
        assert os.path.exists(output_path)
        assert (tmp_path / "output.mp3").read_bytes() == fake_audio_bytes

        # Verify API arguments
        mock_tts_client.synthesize_speech.assert_called_once()
        mock_tts_module.VoiceSelectionParams.assert_called_once_with(
            language_code=DEFAULT_GOOGLE_CLOUD_TTS_LANGUAGE,
            name=DEFAULT_GOOGLE_CLOUD_TTS_VOICE,
        )

    def test_custom_voice_and_language(self, tmp_path, mock_tts_client):
        output_path = str(tmp_path / "output.mp3")
        mock_creds = MagicMock()
        config = {
            "google_cloud": {
                "voice": "en-US-Journey-F",
                "language_code": "en-US",
                "speaking_rate": 1.25,
                "pitch": 2.0,
            }
        }

        with patch("tools.tts_tool._resolve_google_cloud_credentials", return_value=mock_creds), \
             patch("tools.tts_tool._import_google_cloud_tts") as mock_import:
            mock_tts_module = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_tts_client
            mock_import.return_value = mock_tts_module

            _generate_google_cloud_tts("Testing custom voice", output_path, config)

        mock_tts_module.VoiceSelectionParams.assert_called_once_with(
            language_code="en-US",
            name="en-US-Journey-F",
        )

    def test_ogg_opus_encoding_for_ogg_output(self, tmp_path, mock_tts_client):
        output_path = str(tmp_path / "output.ogg")
        mock_creds = MagicMock()

        with patch("tools.tts_tool._resolve_google_cloud_credentials", return_value=mock_creds), \
             patch("tools.tts_tool._import_google_cloud_tts") as mock_import:
            mock_tts_module = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_tts_client
            mock_import.return_value = mock_tts_module

            _generate_google_cloud_tts("Testing OGG", output_path, {})

        mock_tts_module.AudioConfig.assert_called()

    def test_project_id_client_option(self, tmp_path, mock_tts_client):
        output_path = str(tmp_path / "output.mp3")
        mock_creds = MagicMock()
        config = {"google_cloud": {"project_id": "my-custom-project"}}

        with patch("tools.tts_tool._resolve_google_cloud_credentials", return_value=mock_creds), \
             patch("tools.tts_tool._import_google_cloud_tts") as mock_import:
            mock_tts_module = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_tts_client
            mock_import.return_value = mock_tts_module

            _generate_google_cloud_tts("Testing project", output_path, config)

        mock_tts_module.TextToSpeechClient.assert_called_once_with(
            credentials=mock_creds,
            client_options={"quota_project_id": "my-custom-project"},
        )

    def test_permission_denied_error_handling(self, tmp_path):
        output_path = str(tmp_path / "output.mp3")
        mock_creds = MagicMock()
        mock_client = MagicMock()
        mock_client.synthesize_speech.side_effect = Exception("403 Cloud Text-to-Speech API PERMISSION_DENIED")

        with patch("tools.tts_tool._resolve_google_cloud_credentials", return_value=mock_creds), \
             patch("tools.tts_tool._import_google_cloud_tts") as mock_import:
            mock_tts_module = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_client
            mock_import.return_value = mock_tts_module

            with pytest.raises(RuntimeError, match="Google Cloud TTS permission denied"):
                _generate_google_cloud_tts("Hello", output_path, {})

    def test_invalid_argument_error_handling(self, tmp_path):
        output_path = str(tmp_path / "output.mp3")
        mock_creds = MagicMock()
        mock_client = MagicMock()
        mock_client.synthesize_speech.side_effect = Exception("INVALID_ARGUMENT: voice not found")

        with patch("tools.tts_tool._resolve_google_cloud_credentials", return_value=mock_creds), \
             patch("tools.tts_tool._import_google_cloud_tts") as mock_import:
            mock_tts_module = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_client
            mock_import.return_value = mock_tts_module

            with pytest.raises(RuntimeError, match="Google Cloud TTS invalid argument"):
                _generate_google_cloud_tts("Hello", output_path, {})

    def test_empty_audio_content_raises(self, tmp_path):
        output_path = str(tmp_path / "output.mp3")
        mock_creds = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.audio_content = b""
        mock_client.synthesize_speech.return_value = mock_resp

        with patch("tools.tts_tool._resolve_google_cloud_credentials", return_value=mock_creds), \
             patch("tools.tts_tool._import_google_cloud_tts") as mock_import:
            mock_tts_module = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_client
            mock_import.return_value = mock_tts_module

            with pytest.raises(RuntimeError, match="empty audio data"):
                _generate_google_cloud_tts("Hello", output_path, {})


class TestListGoogleCloudVoices:
    def test_list_voices_returns_structured_list(self, mock_tts_client):
        mock_creds = MagicMock()

        with patch("tools.tts_tool._resolve_google_cloud_credentials", return_value=mock_creds), \
             patch("tools.tts_tool._import_google_cloud_tts") as mock_import:
            mock_tts_module = MagicMock()
            mock_tts_module.TextToSpeechClient.return_value = mock_tts_client
            mock_tts_module.SsmlVoiceGender.return_value.name = "MALE"
            mock_import.return_value = mock_tts_module

            voices = _list_google_cloud_voices({}, language_code="en-US")

        assert len(voices) == 1
        assert voices[0]["name"] == "en-US-Chirp3-HD-Charon"
        assert voices[0]["language_codes"] == ["en-US"]
        assert voices[0]["gender"] == "MALE"


class TestGoogleCloudInCheckRequirements:
    def test_adc_available_satisfies_requirements(self):
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "google_cloud"}), \
             patch("tools.tts_tool._import_google_cloud_tts"), \
             patch("google.auth.default", return_value=(MagicMock(), "project")):
            assert check_tts_requirements() is True

    def test_no_credentials_fails(self):
        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "google_cloud"}), \
             patch("tools.tts_tool._import_google_cloud_tts"), \
             patch("google.auth.default", side_effect=Exception("No ADC")):
            assert check_tts_requirements() is False
