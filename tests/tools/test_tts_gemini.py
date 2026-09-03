"""Tests for the Google Gemini TTS provider in tools/tts_tool.py."""

import base64
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "HERMES_SESSION_PLATFORM",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_pcm_bytes():
    # 0.1s of silence at 24kHz mono 16-bit = 4800 bytes
    return b"\x00" * 4800


@pytest.fixture
def mock_gemini_response(fake_pcm_bytes):
    """A successful Gemini generateContent response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": base64.b64encode(fake_pcm_bytes).decode(),
                            }
                        }
                    ]
                }
            }
        ]
    }
    return resp


class TestWrapPcmAsWav:
    def test_riff_header_structure(self):
        from tools.tts_tool import _wrap_pcm_as_wav

        pcm = b"\x01\x02\x03\x04" * 10
        wav = _wrap_pcm_as_wav(pcm, sample_rate=24000, channels=1, sample_width=2)

        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        # Audio format (PCM=1)
        assert struct.unpack("<H", wav[20:22])[0] == 1
        # Channels
        assert struct.unpack("<H", wav[22:24])[0] == 1
        # Sample rate
        assert struct.unpack("<I", wav[24:28])[0] == 24000
        # Bits per sample
        assert struct.unpack("<H", wav[34:36])[0] == 16
        assert wav[36:40] == b"data"
        assert wav[44:] == pcm

    def test_header_size_is_44(self):
        from tools.tts_tool import _wrap_pcm_as_wav

        pcm = b"\xff" * 100
        wav = _wrap_pcm_as_wav(pcm)
        assert len(wav) == 44 + len(pcm)


class TestGenerateGeminiTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_gemini_tts

        output_path = str(tmp_path / "test.wav")
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            _generate_gemini_tts("Hello", output_path, {})

    def test_google_api_key_fallback(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GOOGLE_API_KEY", "from-google-env")
        output_path = str(tmp_path / "test.wav")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", output_path, {})

        # Confirm it used the GOOGLE_API_KEY as the query parameter
        _, kwargs = mock_post.call_args
        assert kwargs["params"]["key"] == "from-google-env"

    def test_wav_output_fast_path(self, tmp_path, monkeypatch, mock_gemini_response, fake_pcm_bytes):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        output_path = str(tmp_path / "test.wav")

        with patch("requests.post", return_value=mock_gemini_response):
            result = _generate_gemini_tts("Hi", output_path, {})

        assert result == output_path
        data = (tmp_path / "test.wav").read_bytes()
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"
        # Audio payload should match the PCM we put in
        assert data[44:] == fake_pcm_bytes

    def test_x_goog_api_client_header_is_set(self, tmp_path, monkeypatch, mock_gemini_response):
        """Gemini TTS requests should include Hermes client context."""
        from hermes_cli import __version__
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

        headers = mock_post.call_args[1]["headers"]
        assert headers["X-Goog-Api-Client"] == f"hermes-agent/{__version__}"

    def test_default_voice_and_model(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import (
            DEFAULT_GEMINI_TTS_MODEL,
            DEFAULT_GEMINI_TTS_VOICE,
            _generate_gemini_tts,
        )

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

        args, kwargs = mock_post.call_args
        assert DEFAULT_GEMINI_TTS_MODEL in args[0]
        payload = kwargs["json"]
        voice = (
            payload["generationConfig"]["speechConfig"]["voiceConfig"]
            ["prebuiltVoiceConfig"]["voiceName"]
        )
        assert voice == DEFAULT_GEMINI_TTS_VOICE

    def test_custom_voice(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {"gemini": {"voice": "Puck"}}

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), config)

        payload = mock_post.call_args[1]["json"]
        voice = (
            payload["generationConfig"]["speechConfig"]["voiceConfig"]
            ["prebuiltVoiceConfig"]["voiceName"]
        )
        assert voice == "Puck"


    def test_audio_tag_rewrite_failure_falls_back_to_original_text(
        self, tmp_path, monkeypatch, mock_gemini_response, caplog
    ):
        from tools.tts_tool import _generate_gemini_tts

        config = {
            "gemini": {
                "model": "gemini-3.1-flash-tts-preview",
                "audio_tags": True,
            }
        }
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")), \
             patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi there.", str(tmp_path / "test.wav"), config)

        prompt_text = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert prompt_text == "Hi there."
        assert "audio tag rewrite failed" in caplog.text


class TestGeminiInCheckRequirements:
    def test_gemini_api_key_satisfies_requirements(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        # Strip everything else
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
        monkeypatch.setenv("GEMINI_API_KEY", "k")

        # Force edge_tts import to fail so we actually hit the gemini check
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "edge_tts":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch(
            "tools.tts_tool._load_tts_config",
            return_value={"provider": "gemini"},
        ), patch("builtins.__import__", side_effect=fake_import):
            assert check_tts_requirements() is True


class TestGenerateVertexGeminiTts:
    def test_vertex_ai_mode_uses_bearer_token_and_project_endpoint(
        self, tmp_path, mock_gemini_response, fake_pcm_bytes
    ):
        from tools.tts_tool import _generate_gemini_tts

        output_path = str(tmp_path / "test.wav")
        config = {
            "provider": "vertex",
            "vertex": {
                "project_id": "test-project-123",
                "location": "global",
                "model": "gemini-2.5-flash-preview-tts",
                "voice": "Kore",
            },
        }

        mock_creds = MagicMock()
        mock_creds.token = "ya29.test-bearer-token"
        mock_creds.project_id = "test-project-123"

        with patch("google.auth.default", return_value=(mock_creds, "test-project-123")), \
             patch("google.auth.transport.requests.Request"), \
             patch("requests.post", return_value=mock_gemini_response) as mock_post:
            result = _generate_gemini_tts("Hello Vertex", output_path, config, provider="vertex")

        assert result == output_path
        args, kwargs = mock_post.call_args
        endpoint = args[0]
        assert "aiplatform.googleapis.com" in endpoint
        assert "projects/test-project-123" in endpoint
        assert "gemini-2.5-flash-preview-tts:generateContent" in endpoint
        assert kwargs["headers"]["Authorization"] == "Bearer ya29.test-bearer-token"
        assert kwargs["params"] is None
        payload = kwargs["json"]
        assert payload["contents"][0]["role"] == "user"
        assert payload["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"

    def test_vertex_regional_location_endpoint(
        self, tmp_path, mock_gemini_response
    ):
        from tools.tts_tool import _generate_gemini_tts

        output_path = str(tmp_path / "test.wav")
        config = {
            "provider": "vertex",
            "vertex": {
                "project_id": "test-project-123",
                "location": "us-central1",
                "model": "gemini-2.5-flash-preview-tts",
                "voice": "Puck",
            },
        }

        mock_creds = MagicMock()
        mock_creds.token = "ya29.test-bearer-token"

        with patch("google.auth.default", return_value=(mock_creds, "test-project-123")), \
             patch("google.auth.transport.requests.Request"), \
             patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hello Regional", output_path, config, provider="vertex")

        endpoint = mock_post.call_args[0][0]
        assert "us-central1-aiplatform.googleapis.com" in endpoint
        assert "locations/us-central1" in endpoint

    def test_vertex_missing_credentials_raises_error(self, tmp_path):
        from tools.tts_tool import _generate_gemini_tts

        output_path = str(tmp_path / "test.wav")
        config = {
            "provider": "vertex",
            "vertex": {"project_id": "test-project"},
        }

        with patch("google.auth.default", side_effect=Exception("No ADC")):
            with pytest.raises(ValueError, match="Vertex AI Gemini TTS authentication failed"):
                _generate_gemini_tts("Hello", output_path, config, provider="vertex")

    def test_vertex_boolean_config_does_not_crash(self, tmp_path, mock_gemini_response):
        """tts.vertex: true boolean config must normalize gracefully without AttributeError."""
        from tools.tts_tool import _generate_gemini_tts

        output_path = str(tmp_path / "test.wav")
        config = {
            "provider": "vertex",
            "vertex": True,
        }

        mock_creds = MagicMock()
        mock_creds.token = "ya29.test-bearer-token"
        mock_creds.project_id = "test-project-from-adc"

        with patch("google.auth.default", return_value=(mock_creds, "test-project-from-adc")), \
             patch("google.auth.transport.requests.Request"), \
             patch("requests.post", return_value=mock_gemini_response) as mock_post:
            result = _generate_gemini_tts("Hello Boolean", output_path, config, provider="vertex")

        assert result == output_path
        assert "projects/test-project-from-adc" in mock_post.call_args[0][0]

    def test_vertex_invalid_credentials_file_raises_file_not_found(self, tmp_path):
        """Explicit credentials_file that does not exist must fail fast instead of falling back to ADC."""
        from tools.tts_tool import _generate_gemini_tts

        output_path = str(tmp_path / "test.wav")
        config = {
            "provider": "vertex",
            "vertex": {
                "credentials_file": str(tmp_path / "nonexistent_sa.json"),
                "project_id": "test-project",
            },
        }

        with pytest.raises(ValueError, match="credentials file not found"):
            _generate_gemini_tts("Hello", output_path, config, provider="vertex")

    def test_check_tts_requirements_with_existing_credentials_file(self, tmp_path, monkeypatch):
        """check_tts_requirements should return True when an explicitly configured credentials_file exists on disk."""
        from tools.tts_tool import check_tts_requirements

        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        cred_file = tmp_path / "sa.json"
        cred_file.write_text("{}")

        with patch(
            "tools.tts_tool._load_tts_config",
            return_value={"provider": "vertex", "vertex": {"credentials_file": str(cred_file)}},
        ):
            assert check_tts_requirements() is True

    def test_check_tts_requirements_with_tilde_expanded_credentials_file(self, tmp_path, monkeypatch):
        """check_tts_requirements should expand tilde (~) paths before checking file existence."""
        from tools.tts_tool import check_tts_requirements

        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        real_file = tmp_path / "sa.json"
        real_file.write_text("{}")

        tilde_path = "~/credentials/sa.json"

        def fake_expanduser(path):
            if path == tilde_path:
                return str(real_file)
            return path

        with patch(
            "tools.tts_tool._load_tts_config",
            return_value={"provider": "vertex", "vertex": {"credentials_file": tilde_path}},
        ), patch("os.path.expanduser", side_effect=fake_expanduser):
            assert check_tts_requirements() is True

    def test_check_tts_requirements_with_invalid_credentials_file(self, tmp_path, monkeypatch):
        """check_tts_requirements should return False when credentials_file does not exist."""
        from tools.tts_tool import check_tts_requirements

        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        with patch(
            "tools.tts_tool._load_tts_config",
            return_value={"provider": "vertex", "vertex": {"credentials_file": str(tmp_path / "missing.json")}},
        ):
            assert check_tts_requirements() is False


class TestVertexCredentialPrecedence:
    """Verify explicit credentials_file always takes precedence and invalid
    explicit credentials never silently fall back to ADC."""

    def test_explicit_cred_file_used_over_adc(
        self, tmp_path, mock_gemini_response, monkeypatch
    ):
        """Case B: credentials_file valid + ADC available -> explicit file used, ADC not called."""
        from tools.tts_tool import _generate_gemini_tts

        key_file = tmp_path / "sa.json"
        key_file.write_text('{"type": "service_account"}', encoding="utf-8")

        config = {
            "provider": "vertex",
            "vertex": {
                "credentials_file": str(key_file),
                "project_id": "explicit-project",
            },
        }

        mock_sa_creds = MagicMock()
        mock_sa_creds.token = "ya29.from-sa-file"
        mock_sa_creds.project_id = "explicit-project"

        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_sa_creds,
        ) as mock_from_file, \
             patch("google.auth.default") as mock_adc, \
             patch("google.auth.transport.requests.Request"), \
             patch("requests.post", return_value=mock_gemini_response):
            _generate_gemini_tts("Test", str(tmp_path / "out.wav"), config, provider="vertex")

        # SA file was used
        mock_from_file.assert_called_once()
        # Compare the actual positional arg, not the stringified call_args
        # (str(call_args) double-escapes backslashes on Windows).
        assert mock_from_file.call_args[0][0] == str(key_file)
        # ADC was NOT called
        mock_adc.assert_not_called()

    def test_invalid_cred_file_does_not_fall_back_to_adc(self, tmp_path):
        """Case C: credentials_file nonexistent + ADC available -> FAIL, ADC NOT used."""
        from tools.tts_tool import _generate_gemini_tts

        config = {
            "provider": "vertex",
            "vertex": {
                "credentials_file": str(tmp_path / "typo_sa.json"),
                "project_id": "some-project",
            },
        }

        # ADC is available but should never be reached
        mock_adc_creds = MagicMock()
        mock_adc_creds.token = "ya29.from-adc"

        with patch("google.auth.default", return_value=(mock_adc_creds, "adc-project")) as mock_adc:
            with pytest.raises(ValueError, match="credentials file not found"):
                _generate_gemini_tts("Test", str(tmp_path / "out.wav"), config, provider="vertex")

        # ADC must NOT have been called
        mock_adc.assert_not_called()

    def test_malformed_cred_file_does_not_fall_back_to_adc(self, tmp_path):
        """Case E: credentials_file exists but invalid + ADC available -> explicit error, no ADC fallback."""
        from tools.tts_tool import _generate_gemini_tts

        bad_file = tmp_path / "bad_sa.json"
        bad_file.write_text("not valid json", encoding="utf-8")

        config = {
            "provider": "vertex",
            "vertex": {
                "credentials_file": str(bad_file),
                "project_id": "some-project",
            },
        }

        mock_adc_creds = MagicMock()
        mock_adc_creds.token = "ya29.from-adc"

        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            side_effect=ValueError("could not deserialize key data"),
        ), \
             patch("google.auth.default", return_value=(mock_adc_creds, "adc-project")) as mock_adc:
            with pytest.raises(ValueError, match="Failed to load Vertex AI service account"):
                _generate_gemini_tts("Test", str(tmp_path / "out.wav"), config, provider="vertex")

        # ADC must NOT have been called
        mock_adc.assert_not_called()

    def test_directory_path_as_credentials_file_raises(self, tmp_path):
        """A directory path should fail the isfile check with a clear error."""
        from tools.tts_tool import _generate_gemini_tts

        config = {
            "provider": "vertex",
            "vertex": {
                "credentials_file": str(tmp_path),  # directory, not a file
                "project_id": "test-project",
            },
        }

        with pytest.raises(ValueError, match="credentials file not found"):
            _generate_gemini_tts("Hello", str(tmp_path / "out.wav"), config, provider="vertex")


class TestVertexMalformedConfigTypes:
    """Verify non-dict configuration values for vertex/google_cloud don't crash."""

    @pytest.mark.parametrize("vertex_val", [
        True,
        False,
        [],
        "true",
        123,
        None,
    ])
    def test_vertex_non_dict_config_does_not_crash(
        self, tmp_path, vertex_val, mock_gemini_response
    ):
        """Various non-dict vertex config values must be handled safely."""
        from tools.tts_tool import _generate_gemini_tts

        config = {"provider": "vertex", "vertex": vertex_val}

        # vertex: False and vertex: None won't trigger use_vertex via bool(),
        # but provider_name == "vertex" from the explicit provider arg will.
        mock_creds = MagicMock()
        mock_creds.token = "ya29.test"
        mock_creds.project_id = "test-project"

        with patch("google.auth.default", return_value=(mock_creds, "test-project")), \
             patch("google.auth.transport.requests.Request"), \
             patch("requests.post", return_value=mock_gemini_response):
            result = _generate_gemini_tts(
                "Test", str(tmp_path / "out.wav"), config, provider="vertex"
            )
        assert result == str(tmp_path / "out.wav")

    def test_vertex_false_does_not_activate_vertex_when_provider_is_gemini(
        self, tmp_path, monkeypatch, mock_gemini_response
    ):
        """vertex: false with provider: gemini should NOT activate Vertex mode."""
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {"provider": "gemini", "vertex": False}

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "out.wav"), config)

        # Should use AI Studio (API key), not Vertex (Bearer token)
        kwargs = mock_post.call_args[1]
        assert kwargs["params"] is not None  # API key in params
        assert kwargs["params"]["key"] == "test-key"
        assert "Authorization" not in kwargs["headers"]


class TestVertexSecretLeakage:
    """Verify that sensitive credential content is never exposed in error messages."""

    def test_sa_file_error_does_not_leak_private_key_or_exception_details(self, tmp_path):
        """Credential loading errors must be safe and concise without leaking secret key material or arbitrary third-party exception text."""
        from tools.tts_tool import _generate_gemini_tts

        key_file = tmp_path / "sa.json"
        fake_secret_key = "FAKE_PRIVATE_KEY_DATA_12345"
        key_file.write_text(
            f'{{"type": "service_account", "private_key": "{fake_secret_key}"}}',
            encoding="utf-8",
        )

        config = {
            "provider": "vertex",
            "vertex": {
                "credentials_file": str(key_file),
                "project_id": "test",
            },
        }

        fake_exception_detail = "SENSITIVE_INTERNAL_EXCEPTION_DETAIL_999"
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            side_effect=ValueError(fake_exception_detail),
        ):
            with pytest.raises(ValueError) as exc_info:
                _generate_gemini_tts("Test", str(tmp_path / "out.wav"), config, provider="vertex")

        error_text = str(exc_info.value)
        assert fake_secret_key not in error_text
        assert fake_exception_detail not in error_text
        assert "Failed to load Vertex AI service account credentials from" in error_text

    def test_bearer_token_not_in_error_on_api_failure(self, tmp_path):
        """Bearer tokens must not appear in HTTP error messages."""
        from tools.tts_tool import _generate_gemini_tts

        config = {
            "provider": "vertex",
            "vertex": {"project_id": "test-project"},
        }

        mock_creds = MagicMock()
        fake_access_token = "ya29.FAKE_ACCESS_TOKEN_VALUE_SECRET_987"
        mock_creds.token = fake_access_token

        error_response = MagicMock()
        error_response.status_code = 403
        error_response.iter_content = MagicMock(return_value=[b'{"error": {"message": "Permission denied"}}'])
        error_response.headers = {"content-type": "application/json"}

        with patch("google.auth.default", return_value=(mock_creds, "test-project")), \
             patch("google.auth.transport.requests.Request"), \
             patch("requests.post", return_value=error_response):
            with pytest.raises(RuntimeError) as exc_info:
                _generate_gemini_tts("Test", str(tmp_path / "out.wav"), config, provider="vertex")

        error_text = str(exc_info.value)
        assert fake_access_token not in error_text

