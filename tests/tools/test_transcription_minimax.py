"""Tests for MiniMax native STT provider."""

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools import transcription_tools
from tools.transcription_cloud import _transcribe_minimax


@pytest.fixture
def fake_wav(tmp_path):
    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 3200)
    return str(wav_path)


class TestMiniMaxTranscription:
    def test_missing_api_key_returns_error(self, fake_wav, monkeypatch):
        monkeypatch.setattr("tools.transcription_tools._load_stt_config", lambda: {})
        monkeypatch.setattr("tools.tool_backend_helpers.resolve_provider_secret", lambda env_var, prov, **kw: "")

        result = _transcribe_minimax(fake_wav, "asr-1.0")
        assert result["success"] is False
        assert "MINIMAX" in result["error"]

    def test_transcribe_minimax_cn_success(self, fake_wav, monkeypatch):
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config",
            lambda: {"minimax": {"region": "cn", "model": "asr-1.0"}},
        )
        monkeypatch.setattr(
            "tools.tool_backend_helpers.resolve_provider_secret",
            lambda env_var, prov, **kw: "test-cn-key" if "CN" in env_var else "",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "text": "你好，这是MiniMax语音识别测试",
            "duration": 2.5,
            "base_resp": {"status_code": 0},
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = _transcribe_minimax(fake_wav, "asr-1.0")

        assert result["success"] is True
        assert result["transcript"] == "你好，这是MiniMax语音识别测试"
        assert result["provider"] == "minimax"

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "api.minimax.cn" in call_url or "api.minimaxi.com" in call_url
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-cn-key"

    def test_transcribe_minimax_global_success(self, fake_wav, monkeypatch):
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config",
            lambda: {"minimax": {"region": "global", "model": "asr-1.0"}},
        )
        monkeypatch.setattr(
            "tools.tool_backend_helpers.resolve_provider_secret",
            lambda env_var, prov, **kw: "test-global-key" if "CN" not in env_var else "",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "text": "Hello global test",
            "duration": 1.2,
            "base_resp": {"status_code": 0},
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = _transcribe_minimax(fake_wav, "asr-1.0")

        assert result["success"] is True
        assert result["transcript"] == "Hello global test"
        call_url = mock_post.call_args[0][0]
        assert "api.minimax.io" in call_url

    def test_transcribe_minimax_api_error(self, fake_wav, monkeypatch):
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config",
            lambda: {"minimax": {"region": "cn"}},
        )
        monkeypatch.setattr(
            "tools.tool_backend_helpers.resolve_provider_secret",
            lambda env_var, prov, **kw: "test-key",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"base_resp": {"status_code": 1004, "status_msg": "Invalid file format"}}
        mock_resp.text = "Invalid file format"

        with patch("requests.post", return_value=mock_resp):
            result = _transcribe_minimax(fake_wav, "asr-1.0")

        assert result["success"] is False
        assert "Invalid file format" in result["error"]

    def test_transcribe_audio_dispatcher_integration(self, fake_wav, monkeypatch):
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config",
            lambda: {"provider": "minimax", "minimax": {"region": "cn", "model": "asr-1.0"}},
        )
        monkeypatch.setattr("tools.transcription_tools._get_provider", lambda cfg: "minimax")
        monkeypatch.setattr(
            "tools.transcription_tools._transcribe_minimax",
            lambda path, model, **kw: {"success": True, "transcript": "dispatched to minimax", "provider": "minimax"},
        )

        result = transcription_tools.transcribe_audio(fake_wav)
        assert result["success"] is True
        assert result["transcript"] == "dispatched to minimax"
        assert result["provider"] == "minimax"
