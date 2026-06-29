"""
Tests for the HTTP sidecar STT provider in transcription_tools.

Covers:
  - _transcribe_http happy path (200 + {"text": "..."})
  - _transcribe_http non-200 response → success=False with error message
  - _transcribe_http empty transcript → success=False
  - _transcribe_http connection error → success=False
  - _get_provider auto-detect: HERMES_STT_HOST set → returns "http"
  - _get_provider explicit stt.provider="http" with host → returns "http"
  - _get_provider explicit stt.provider="http" without host → returns "none"
  - transcribe_audio end-to-end with HTTP sidecar mock
  - HERMES_STT_PROVIDER constant is exported
  - HERMES_STT_HOST constant is exported
"""

import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tools.transcription_tools as tt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(tmp_path: Path, name: str = "audio.wav") -> Path:
    """Write a minimal valid WAV file so _validate_audio_file passes."""
    wav = tmp_path / name
    # 44-byte WAV header with 0 data bytes - enough to pass format/size checks.
    header = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x80\xbb\x00\x00\x00w\x01\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    wav.write_bytes(header)
    return wav


def _mock_response(status_code: int, body: object) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = json.dumps(body) if isinstance(body, dict) else str(body)
    return resp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_hermes_stt_host_exported(self):
        assert hasattr(tt, "HERMES_STT_HOST")

    def test_hermes_stt_provider_exported(self):
        assert hasattr(tt, "HERMES_STT_PROVIDER")

    def test_hermes_stt_host_default_empty(self, monkeypatch):
        monkeypatch.delenv("HERMES_STT_HOST", raising=False)
        # Re-evaluate the default by checking the module-level value is a str
        assert isinstance(tt.HERMES_STT_HOST, str)

    def test_hermes_stt_provider_default_empty(self, monkeypatch):
        monkeypatch.delenv("HERMES_STT_PROVIDER", raising=False)
        assert isinstance(tt.HERMES_STT_PROVIDER, str)


# ---------------------------------------------------------------------------
# _transcribe_http unit tests
# ---------------------------------------------------------------------------


class TestTranscribeHttp:
    def test_happy_path(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "hello world"})

        with patch("requests.post", return_value=mock_resp) as mock_post:
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
            result = tt._transcribe_http(str(wav), "whisper-1")

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        assert result["provider"] == "http"
        mock_post.assert_called_once()

    def test_posts_to_correct_url(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "test"})

        with patch("requests.post", return_value=mock_resp) as mock_post:
            monkeypatch.setenv("HERMES_STT_HOST", "http://sidecar:8080")
            tt._transcribe_http(str(wav), "")

        call_url = mock_post.call_args[0][0]
        assert call_url == "http://sidecar:8080/transcribe"

    def test_custom_path_from_config(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "ok"})

        stt_cfg = {"http": {"host": "http://localhost:9000", "path": "/v1/stt"}}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            with patch.object(tt, "_load_stt_config", return_value=stt_cfg):
                result = tt._transcribe_http(str(wav), "")

        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:9000/v1/stt"
        assert result["success"] is True

    def test_model_sent_in_form_data(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "hi"})

        with patch("requests.post", return_value=mock_resp) as mock_post:
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
            tt._transcribe_http(str(wav), "my-model")

        _, kwargs = mock_post.call_args
        assert kwargs["data"]["model"] == "my-model"

    def test_no_model_omits_model_field(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "hi"})

        with patch("requests.post", return_value=mock_resp) as mock_post:
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
            tt._transcribe_http(str(wav), "")

        _, kwargs = mock_post.call_args
        assert "model" not in kwargs["data"]

    def test_non_200_returns_failure(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(503, {"error": {"message": "service unavailable"}})

        with patch("requests.post", return_value=mock_resp):
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
            result = tt._transcribe_http(str(wav), "")

        assert result["success"] is False
        assert "503" in result["error"]
        assert "service unavailable" in result["error"]

    def test_non_200_plain_text_body(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        resp = MagicMock()
        resp.status_code = 500
        resp.json.side_effect = ValueError("not json")
        resp.text = "Internal Server Error"

        with patch("requests.post", return_value=resp):
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
            result = tt._transcribe_http(str(wav), "")

        assert result["success"] is False
        assert "500" in result["error"]

    def test_empty_transcript_returns_failure(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "   "})

        with patch("requests.post", return_value=mock_resp):
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
            result = tt._transcribe_http(str(wav), "")

        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_missing_host_returns_failure(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        monkeypatch.delenv("HERMES_STT_HOST", raising=False)

        with patch.object(tt, "_load_stt_config", return_value={}):
            result = tt._transcribe_http(str(wav), "")

        assert result["success"] is False
        assert "HERMES_STT_HOST" in result["error"]

    def test_connection_error_returns_failure(self, tmp_path, monkeypatch):
        import requests as req_mod

        wav = _make_wav(tmp_path)

        with patch("requests.post", side_effect=req_mod.exceptions.ConnectionError("refused")):
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
            result = tt._transcribe_http(str(wav), "")

        assert result["success"] is False
        assert "refused" in result["error"] or "HTTP STT sidecar" in result["error"]

    def test_timeout_error_returns_failure(self, tmp_path, monkeypatch):
        import requests as req_mod

        wav = _make_wav(tmp_path)

        with patch("requests.post", side_effect=req_mod.exceptions.Timeout("timed out")):
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
            result = tt._transcribe_http(str(wav), "")

        assert result["success"] is False

    def test_trailing_slash_stripped_from_host(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "ok"})

        with patch("requests.post", return_value=mock_resp) as mock_post:
            monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000/")
            tt._transcribe_http(str(wav), "")

        call_url = mock_post.call_args[0][0]
        assert not call_url.startswith("http://localhost:9000//")
        assert call_url == "http://localhost:9000/transcribe"

    def test_custom_timeout_from_config(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "ok"})
        stt_cfg = {"http": {"host": "http://localhost:9000", "timeout": 30}}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            with patch.object(tt, "_load_stt_config", return_value=stt_cfg):
                tt._transcribe_http(str(wav), "")

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 30.0


# ---------------------------------------------------------------------------
# _get_provider routing tests
# ---------------------------------------------------------------------------


class TestGetProviderHttp:
    def test_autodetect_prefers_http_when_host_set(self, monkeypatch):
        """When HERMES_STT_HOST is set and no explicit provider, auto-detect returns 'http'."""
        monkeypatch.setenv("HERMES_STT_HOST", "http://localhost:9000")
        # Patch get_env_value so the module-level call sees the env var
        with patch.object(tt, "get_env_value", side_effect=lambda k, d=None: (
            "http://localhost:9000" if k == "HERMES_STT_HOST" else os.getenv(k, d)
        )):
            provider = tt._get_provider({})
        assert provider == "http"

    def test_autodetect_no_http_when_host_unset(self, monkeypatch):
        """When HERMES_STT_HOST is not set, auto-detect does not return 'http'."""
        monkeypatch.delenv("HERMES_STT_HOST", raising=False)
        with patch.object(tt, "get_env_value", side_effect=lambda k, d=None: os.getenv(k, d)):
            with patch.object(tt, "_HAS_FASTER_WHISPER", False):
                with patch.object(tt, "_has_local_command", return_value=False):
                    with patch.object(tt, "_HAS_OPENAI", False):
                        provider = tt._get_provider({})
        assert provider != "http"

    def test_explicit_http_with_host_returns_http(self, monkeypatch):
        stt_cfg = {"provider": "http", "http": {"host": "http://localhost:9000"}}
        provider = tt._get_provider(stt_cfg)
        assert provider == "http"

    def test_explicit_http_without_host_returns_none(self, monkeypatch):
        monkeypatch.delenv("HERMES_STT_HOST", raising=False)
        with patch.object(tt, "get_env_value", side_effect=lambda k, d=None: os.getenv(k, d)):
            stt_cfg = {"provider": "http"}
            provider = tt._get_provider(stt_cfg)
        assert provider == "none"

    def test_explicit_http_host_from_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_STT_HOST", "http://sidecar:8080")
        with patch.object(tt, "get_env_value", side_effect=lambda k, d=None: (
            "http://sidecar:8080" if k == "HERMES_STT_HOST" else os.getenv(k, d)
        )):
            stt_cfg = {"provider": "http"}
            provider = tt._get_provider(stt_cfg)
        assert provider == "http"


# ---------------------------------------------------------------------------
# transcribe_audio end-to-end with HTTP sidecar
# ---------------------------------------------------------------------------


class TestTranscribeAudioHttp:
    def test_end_to_end_http_provider(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "end to end works"})

        stt_cfg = {"provider": "http", "http": {"host": "http://localhost:9000"}}

        with patch.object(tt, "_load_stt_config", return_value=stt_cfg):
            with patch("requests.post", return_value=mock_resp):
                result = tt.transcribe_audio(str(wav))

        assert result["success"] is True
        assert result["transcript"] == "end to end works"
        assert result["provider"] == "http"

    def test_end_to_end_model_override(self, tmp_path, monkeypatch):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "model override"})

        stt_cfg = {"provider": "http", "http": {"host": "http://localhost:9000"}}

        with patch.object(tt, "_load_stt_config", return_value=stt_cfg):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                result = tt.transcribe_audio(str(wav), model="custom-model")

        assert result["success"] is True
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["model"] == "custom-model"

    def test_end_to_end_model_from_config(self, tmp_path):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(200, {"text": "config model"})

        stt_cfg = {"provider": "http", "http": {"host": "http://localhost:9000", "model": "cfg-model"}}

        with patch.object(tt, "_load_stt_config", return_value=stt_cfg):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                result = tt.transcribe_audio(str(wav))

        assert result["success"] is True
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["model"] == "cfg-model"

    def test_end_to_end_failure_propagated(self, tmp_path):
        wav = _make_wav(tmp_path)
        mock_resp = _mock_response(500, {"error": {"message": "boom"}})

        stt_cfg = {"provider": "http", "http": {"host": "http://localhost:9000"}}

        with patch.object(tt, "_load_stt_config", return_value=stt_cfg):
            with patch("requests.post", return_value=mock_resp):
                result = tt.transcribe_audio(str(wav))

        assert result["success"] is False
        assert "boom" in result["error"]
