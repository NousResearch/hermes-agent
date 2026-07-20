"""
Tests for the native VOICEVOX TTS provider.

These tests pin the registration / dispatch / error paths for VOICEVOX
without requiring a running VOICEVOX engine (HTTP calls are monkey-patched).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools import tts_tool
from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_VOICEVOX_BASE_URL,
    DEFAULT_VOICEVOX_SPEAKER,
    PROVIDER_MAX_TEXT_LENGTH,
    _check_voicevox_available,
    _generate_voicevox_tts,
    check_tts_requirements,
    text_to_speech_tool,
)


# ---------------------------------------------------------------------------
# Registry / constants
# ---------------------------------------------------------------------------

class TestVoicevoxRegistration:
    def test_voicevox_is_a_builtin_provider(self):
        assert "voicevox" in BUILTIN_TTS_PROVIDERS

    def test_voicevox_has_a_text_length_cap(self):
        assert PROVIDER_MAX_TEXT_LENGTH.get("voicevox", 0) > 0

    def test_voicevox_in_registry_builtin_names(self):
        from agent.tts_registry import _BUILTIN_NAMES

        assert "voicevox" in _BUILTIN_NAMES


# ---------------------------------------------------------------------------
# _check_voicevox_available
# ---------------------------------------------------------------------------

class TestCheckVoicevoxAvailable:
    def test_returns_true_when_engine_responds(self, monkeypatch):
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {
            "voicevox": {"base_url": "http://localhost:50021"}
        })
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _check_voicevox_available() is True

    def test_returns_false_when_engine_offline(self, monkeypatch):
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {
            "voicevox": {"base_url": "http://localhost:59999"}
        })
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            assert _check_voicevox_available() is False

    def test_returns_bool_without_raising(self):
        assert isinstance(_check_voicevox_available(), bool)


# ---------------------------------------------------------------------------
# _generate_voicevox_tts
# ---------------------------------------------------------------------------

class TestGenerateVoicevoxTts:
    def _mock_urlopen(self, responses):
        """Create a mock urlopen that returns different responses per call."""
        call_count = [0]

        def side_effect(req, **kwargs):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            resp = MagicMock()
            resp.read.return_value = responses[idx]
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        return side_effect

    def test_synthesizes_and_writes_wav(self, tmp_path, monkeypatch):
        fake_query = b'{"accent_phrases": []}'
        fake_wav = b"RIFF" + b"\x00" * 100  # minimal WAV-like bytes

        config = {"voicevox": {"base_url": "http://localhost:50021", "speaker": 3}}
        out_path = str(tmp_path / "out.wav")

        with patch("urllib.request.urlopen", side_effect=self._mock_urlopen([fake_query, fake_wav])):
            result = _generate_voicevox_tts("こんにちは", out_path, config)

        assert result == out_path
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0

    def test_converts_wav_to_mp3_via_ffmpeg(self, tmp_path, monkeypatch):
        fake_query = b'{"accent_phrases": []}'
        fake_wav = b"RIFF" + b"\x00" * 100

        config = {"voicevox": {"base_url": "http://localhost:50021", "speaker": 0}}
        out_path = str(tmp_path / "out.mp3")

        # Mock ffmpeg conversion
        def fake_run(cmd, **kwargs):
            # Simulate ffmpeg writing the output file
            Path(out_path).write_bytes(b"\xff\xfb" + b"\x00" * 50)
            return MagicMock(returncode=0)

        with patch("urllib.request.urlopen", side_effect=self._mock_urlopen([fake_query, fake_wav])):
            with patch("tools.tts_tool.subprocess.run", side_effect=fake_run):
                with patch("tools.tts_tool.shutil.which", return_value="/usr/bin/ffmpeg"):
                    result = _generate_voicevox_tts("テスト", out_path, config)

        assert result == out_path

    def test_connection_refused_raises_runtime_error(self, tmp_path):
        import urllib.error

        config = {"voicevox": {"base_url": "http://localhost:59999", "speaker": 0}}
        out_path = str(tmp_path / "out.wav")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(RuntimeError, match="Cannot connect to VOICEVOX"):
                _generate_voicevox_tts("hello", out_path, config)

    def test_invalid_speaker_raises_value_error(self, tmp_path):
        import urllib.error

        config = {"voicevox": {"base_url": "http://localhost:50021", "speaker": 9999}}
        out_path = str(tmp_path / "out.wav")

        error_resp = MagicMock()
        error_resp.read.return_value = b"speaker not found"
        http_error = urllib.error.HTTPError(
            "http://localhost:50021/audio_query", 422, "Unprocessable", {}, error_resp
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(ValueError, match="speaker ID 9999 not found"):
                _generate_voicevox_tts("hello", out_path, config)

    def test_default_speaker_used_for_invalid_config(self, tmp_path):
        """Non-integer speaker values fall back to DEFAULT_VOICEVOX_SPEAKER."""
        fake_query = b'{"accent_phrases": []}'
        fake_wav = b"RIFF" + b"\x00" * 50

        config = {"voicevox": {"base_url": "http://localhost:50021", "speaker": "invalid"}}
        out_path = str(tmp_path / "out.wav")

        with patch(
            "urllib.request.urlopen",
            side_effect=self._mock_urlopen([fake_query, fake_wav]),
        ) as mock_open:
            _generate_voicevox_tts("hi", out_path, config)

        # Verify the speaker in the URL is the default (0), not "invalid"
        first_call_url = mock_open.call_args_list[0][0][0].full_url
        assert f"speaker={DEFAULT_VOICEVOX_SPEAKER}" in first_call_url

    def test_empty_synthesis_raises_runtime_error(self, tmp_path):
        fake_query = b'{"accent_phrases": []}'
        fake_wav = b""  # empty

        config = {"voicevox": {"base_url": "http://localhost:50021", "speaker": 0}}
        out_path = str(tmp_path / "out.wav")

        with patch("urllib.request.urlopen", side_effect=self._mock_urlopen([fake_query, fake_wav])):
            with pytest.raises(RuntimeError, match="empty audio"):
                _generate_voicevox_tts("hello", out_path, config)


# ---------------------------------------------------------------------------
# text_to_speech_tool end-to-end (provider == "voicevox")
# ---------------------------------------------------------------------------

class TestTextToSpeechToolWithVoicevox:
    def test_dispatches_to_voicevox(self, tmp_path, monkeypatch):
        fake_query = b'{"accent_phrases": []}'
        fake_wav = b"RIFF" + b"\x00" * 100

        monkeypatch.setattr(tts_tool, "_check_voicevox_available", lambda: True)

        cfg = {
            "provider": "voicevox",
            "voicevox": {"base_url": "http://localhost:50021", "speaker": 3},
        }
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        with patch("urllib.request.urlopen") as mock_open:
            call_count = [0]

            def side_effect(req, **kwargs):
                idx = min(call_count[0], 1)
                call_count[0] += 1
                resp = MagicMock()
                resp.read.return_value = [fake_query, fake_wav][idx]
                resp.status = 200
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp

            mock_open.side_effect = side_effect

            result = text_to_speech_tool(
                text="こんにちは", output_path=str(tmp_path / "clip.wav")
            )

        data = json.loads(result)
        assert data["success"] is True, data
        assert data["provider"] == "voicevox"
        assert Path(data["file_path"]).exists()

    def test_engine_offline_surfaces_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_tool, "_check_voicevox_available", lambda: False)

        cfg = {
            "provider": "voicevox",
            "voicevox": {"base_url": "http://localhost:59999"},
        }
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)

        result = text_to_speech_tool(
            text="hello", output_path=str(tmp_path / "clip.wav")
        )
        data = json.loads(result)

        assert data["success"] is False
        assert "VOICEVOX" in data["error"]
        assert "running" in data["error"].lower()


# ---------------------------------------------------------------------------
# check_tts_requirements
# ---------------------------------------------------------------------------

class TestCheckTtsRequirementsVoicevox:
    def test_voicevox_available_satisfies_requirements(self, monkeypatch):
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "voicevox"})
        monkeypatch.setattr(tts_tool, "_has_any_command_tts_provider", lambda: False)

        monkeypatch.setattr(tts_tool, "_check_voicevox_available", lambda: True)
        assert check_tts_requirements() is True

        monkeypatch.setattr(tts_tool, "_check_voicevox_available", lambda: False)
        assert check_tts_requirements() is False
