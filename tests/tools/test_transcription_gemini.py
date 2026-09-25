import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.transcription_tools import transcribe_audio
from tools.transcription_cloud import _transcribe_gemini


@pytest.fixture
def fake_wav(tmp_path):
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    return str(wav)


class TestGeminiSTT:
    def test_missing_api_key_returns_error(self, fake_wav, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch("tools.transcription_tools._load_stt_config", return_value={"gemini": {}}):
            res = _transcribe_gemini(fake_wav, "gemini-3.5-transcribe")
        assert res["success"] is False
        assert "GEMINI_API_KEY not set" in res["error"]

    def test_empty_audio_file_returns_no_speech(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        empty_wav = tmp_path / "empty.wav"
        empty_wav.write_bytes(b"")
        res = _transcribe_gemini(str(empty_wav), "gemini-3.5-transcribe")
        assert res["success"] is False
        assert res.get("no_speech") is True

    def test_interactions_api_transcribe_success(self, fake_wav, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "output_text": "Hello world from Gemini 3.5 Transcribe."
        }

        stt_cfg = {
            "gemini": {
                "custom_vocabulary": ["Hermes", "QQBot"],
                "mode": "smart",
            }
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=stt_cfg), \
             patch("requests.post", return_value=mock_response) as mock_post:
            res = _transcribe_gemini(fake_wav, "gemini-3.5-transcribe", prompt="CPA, v2rayA")

        assert res["success"] is True
        assert res["transcript"] == "Hello world from Gemini 3.5 Transcribe."
        assert res["provider"] == "gemini"

        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0].endswith("/interactions")
        assert call_kwargs["headers"]["x-goog-api-key"] == "test-api-key"
        payload = call_kwargs["json"]
        assert payload["model"] == "gemini-3.5-transcribe"
        assert payload["input"][0]["mime_type"] == "audio/wav"
        vocab = payload["generation_config"]["transcription_config"]["custom_vocabulary"]
        assert "Hermes" in vocab
        assert "QQBot" in vocab
        assert "CPA" in vocab
        assert "v2rayA" in vocab
        assert payload["generation_config"]["transcription_config"]["mode"] == "smart"

    def test_multimodal_generate_content_success(self, fake_wav, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Recognized speech via Gemini 2.5 Flash."}]
                    }
                }
            ]
        }

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response) as mock_post:
            res = _transcribe_gemini(fake_wav, "gemini-2.5-flash", language="zh")

        assert res["success"] is True
        assert res["transcript"] == "Recognized speech via Gemini 2.5 Flash."
        assert res["provider"] == "gemini"

        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        assert "/models/gemini-2.5-flash:generateContent" in call_args[0]
        assert call_kwargs["params"]["key"] == "test-api-key"
        payload = call_kwargs["json"]
        parts = payload["contents"][0]["parts"]
        assert "Transcribe the following audio in zh" in parts[0]["text"]
        assert parts[1]["inline_data"]["mime_type"] == "audio/wav"

    def test_gemini_proxy_configuration(self, fake_wav, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"output_text": "Proxied transcript."}

        stt_cfg = {
            "gemini": {
                "proxy": "http://127.0.0.1:30172",
            }
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=stt_cfg), \
             patch("requests.post", return_value=mock_response) as mock_post:
            res = _transcribe_gemini(fake_wav, "gemini-3.5-transcribe")

        assert res["success"] is True
        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["proxies"] == {
            "http": "http://127.0.0.1:30172",
            "https": "http://127.0.0.1:30172",
        }

    def test_gemini_api_http_error(self, fake_wav, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_response = MagicMock(status_code=400)
        mock_response.json.return_value = {
            "error": {"message": "Invalid audio encoding"}
        }

        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("requests.post", return_value=mock_response):
            res = _transcribe_gemini(fake_wav, "gemini-3.5-transcribe")

        assert res["success"] is False
        assert "Gemini STT API error (HTTP 400): Invalid audio encoding" in res["error"]

    def test_transcribe_audio_dispatcher_integration(self, fake_wav, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"output_text": "Dispatched transcription."}

        stt_cfg = {
            "enabled": True,
            "provider": "gemini",
            "gemini": {
                "model": "gemini-3.5-transcribe"
            }
        }
        with patch("tools.transcription_tools._load_stt_config", return_value=stt_cfg), \
             patch("requests.post", return_value=mock_response):
            res = transcribe_audio(fake_wav)

        assert res["success"] is True
        assert res["transcript"] == "Dispatched transcription."
        assert res["provider"] == "gemini"
