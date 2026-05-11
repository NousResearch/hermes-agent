"""Tests for the OpenRouter STT provider in tools/transcription_tools.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "VOICE_TOOLS_OPENAI_KEY",
        "STT_OPENROUTER_BASE_URL",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_audio_file(tmp_path):
    """A tiny fake audio file that passes os.path.exists."""
    f = tmp_path / "test.mp3"
    f.write_bytes(b"\xff\xfb\x90\x00" + b"x" * 1000)
    return str(f)


@pytest.fixture
def mock_openai_audio_response():
    """A successful OpenRouter audio/transcriptions JSON response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"text": "Hello world, this is a test."}
    return resp


# =============================================================================
# _get_provider — OpenRouter in the provider selection logic
# =============================================================================


class TestGetProvider:
    def test_openrouter_selected_when_key_available(self, monkeypatch):
        from tools.transcription_tools import _get_provider

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setattr(
            "tools.transcription_tools._HAS_OPENAI", True, raising=False
        )

        cfg = {"provider": "openrouter"}
        with patch("tools.transcription_tools._load_stt_config", return_value=cfg):
            result = _get_provider(cfg)
        assert result == "openrouter"

    def test_openrouter_returns_none_when_no_key(self, monkeypatch):
        from tools.transcription_tools import _get_provider

        monkeypatch.setattr(
            "tools.transcription_tools._HAS_OPENAI", True, raising=False
        )

        cfg = {"provider": "openrouter"}
        with patch("tools.transcription_tools._load_stt_config", return_value=cfg):
            result = _get_provider(cfg)
        assert result == "none"

    def test_openrouter_in_autodetect_fallback_order(self, monkeypatch):
        """OpenRouter STT appears in the auto-detect fallback after mistral, before xai."""
        from tools.transcription_tools import _get_provider

        monkeypatch.setattr(
            "tools.transcription_tools._HAS_OPENAI", True, raising=False
        )
        monkeypatch.setattr(
            "tools.transcription_tools._HAS_FASTER_WHISPER", False, raising=False
        )
        monkeypatch.setattr(
            "tools.transcription_tools._has_local_command",
            lambda: False, raising=False,
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        cfg = {}  # no explicit provider
        with patch("tools.transcription_tools._load_stt_config", return_value=cfg):
            result = _get_provider(cfg)
        assert result == "openrouter"


# =============================================================================
# _transcribe_openrouter — the core transcription function.
# requests is imported inside the function, so we patch at the requests module level.
# =============================================================================


class TestTranscribeOpenRouter:
    def test_missing_api_key_returns_error(self, fake_audio_file):
        from tools.transcription_tools import _transcribe_openrouter

        result = _transcribe_openrouter(fake_audio_file, "openai/whisper-1")
        assert result["success"] is False
        assert "OPENROUTER_API_KEY" in result["error"]

    def test_successful_transcription(self, fake_audio_file, monkeypatch, mock_openai_audio_response):
        from tools.transcription_tools import _transcribe_openrouter

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("requests.post", return_value=mock_openai_audio_response) as mock_post:
            result = _transcribe_openrouter(fake_audio_file, "openai/whisper-1")

        assert result["success"] is True
        assert result["transcript"] == "Hello world, this is a test."
        assert result["provider"] == "openrouter"

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "audio/transcriptions" in args[0]  # positional URL arg
        assert kwargs["headers"]["Authorization"] == "Bearer sk-or-test"
        assert kwargs["data"]["model"] == "openai/whisper-1"

    def test_http_error_returns_failure(self, fake_audio_file, monkeypatch):
        from tools.transcription_tools import _transcribe_openrouter

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        err_resp = MagicMock()
        err_resp.status_code = 401
        err_resp.json.return_value = {"error": {"message": "Invalid API key"}}
        err_resp.text = json.dumps(err_resp.json.return_value)

        with patch("requests.post", return_value=err_resp):
            result = _transcribe_openrouter(fake_audio_file, "openai/whisper-1")

        assert result["success"] is False
        assert "401" in result["error"]
        assert "Invalid API key" in result["error"]

    def test_default_base_url_is_openrouter_ai(self, fake_audio_file, monkeypatch, mock_openai_audio_response):
        """Default base URL points to openrouter.ai."""
        from tools.transcription_tools import _transcribe_openrouter

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("requests.post", return_value=mock_openai_audio_response) as mock_post:
            _transcribe_openrouter(fake_audio_file, "openai/whisper-1")

        args, _ = mock_post.call_args  # positional URL arg
        assert "openrouter.ai" in args[0]

    def test_custom_base_url_from_env(self, fake_audio_file, monkeypatch, mock_openai_audio_response):
        """STT_OPENROUTER_BASE_URL env var overrides the default base URL."""
        from tools.transcription_tools import _transcribe_openrouter

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setenv("STT_OPENROUTER_BASE_URL", "https://custom.example.com/v1")

        with patch("requests.post", return_value=mock_openai_audio_response) as mock_post:
            _transcribe_openrouter(fake_audio_file, "openai/whisper-1")

        url = mock_post.call_args[0][0]
        assert "custom.example.com" in url

    def test_permission_error_handled(self, fake_audio_file, monkeypatch):
        from tools.transcription_tools import _transcribe_openrouter

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("requests.post", side_effect=PermissionError("read denied")):
            result = _transcribe_openrouter(fake_audio_file, "openai/whisper-1")

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    def test_generic_exception_returns_failure(self, fake_audio_file, monkeypatch):
        from tools.transcription_tools import _transcribe_openrouter

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("requests.post", side_effect=RuntimeError("connection reset")):
            result = _transcribe_openrouter(fake_audio_file, "openai/whisper-1")

        assert result["success"] is False
        assert "connection reset" in result["error"]

    def test_request_timeout_is_120s(self, fake_audio_file, monkeypatch, mock_openai_audio_response):
        from tools.transcription_tools import _transcribe_openrouter

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("requests.post", return_value=mock_openai_audio_response) as mock_post:
            _transcribe_openrouter(fake_audio_file, "openai/whisper-1")

        assert mock_post.call_args.kwargs["timeout"] == 120

    def test_correct_content_type_in_file_upload(self, fake_audio_file, monkeypatch, mock_openai_audio_response):
        from tools.transcription_tools import _transcribe_openrouter

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("requests.post", return_value=mock_openai_audio_response) as mock_post:
            _transcribe_openrouter(fake_audio_file, "openai/whisper-1")

        files_arg = mock_post.call_args.kwargs["files"]
        file_tuple = files_arg["file"]
        assert file_tuple[2] == "audio/mp3"  # content-type from .mp3 suffix


# =============================================================================
# transcribe_audio — dispatch to _transcribe_openrouter
# =============================================================================


class TestTranscribeAudioDispatch:
    def test_dispatches_to_openrouter_when_configured(
        self, fake_audio_file, monkeypatch, mock_openai_audio_response
    ):
        """transcribe_audio() with provider=openrouter calls _transcribe_openrouter."""
        from tools.transcription_tools import transcribe_audio

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("tools.transcription_tools._load_stt_config") as mock_load:
            mock_load.return_value = {
                "provider": "openrouter",
                "openrouter": {"model": "openai/whisper-1"},
            }
            with patch("requests.post", return_value=mock_openai_audio_response):
                result = transcribe_audio(fake_audio_file)

        assert result["success"] is True
        assert result["provider"] == "openrouter"
        assert result["transcript"] == "Hello world, this is a test."


# =============================================================================
# DEFAULT_CONFIG wiring
# =============================================================================


class TestDefaultConfig:
    def test_openrouter_block_in_stt_defaults(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert "openrouter" in DEFAULT_CONFIG["stt"]
        stt_or = DEFAULT_CONFIG["stt"]["openrouter"]
        assert "model" in stt_or
        assert stt_or["model"] == "openai/whisper-1"

    def test_openrouter_stt_in_provider_option_list(self):
        """Verify openrouter is listed as a valid STT provider option in config comments.
        
        The stt provider comment in config.py lists all supported providers
        including openrouter. We verify this by reading the config module source.
        """
        import inspect, hermes_cli.config
        source = inspect.getsource(hermes_cli.config)
        # find the stt provider line
        lines = source.split('\n')
        in_stt = False
        found = False
        for line in lines:
            if '"stt"' in line and ':' in line and '{' in line:
                in_stt = True
            if in_stt and '"provider"' in line and '#' in line:
                found = True
                assert 'openrouter' in line.lower(), f"openrouter not found in provider comment: {line}"
                break
        assert found, "Could not find stt provider comment line in config.py"

    def test_openrouter_tts_in_provider_option_list(self):
        """Verify openrouter is listed as a valid TTS provider option in config comments."""
        import inspect, hermes_cli.config
        source = inspect.getsource(hermes_cli.config)
        lines = source.split('\n')
        in_tts = False
        found = False
        for line in lines:
            if '"tts"' in line and ':' in line and '{' in line:
                in_tts = True
            if in_tts and '"provider"' in line and '#' in line:
                found = True
                assert 'openrouter' in line.lower(), f"openrouter not found in provider comment: {line}"
                break
        assert found, "Could not find tts provider comment line in config.py"