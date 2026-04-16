"""Tests for the TADA TTS provider in tools/tts_tool.py."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERMES_SESSION_PLATFORM",):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# _generate_tada unit tests
# ---------------------------------------------------------------------------

class TestGenerateTada:
    def _make_response(self, content=b"fake-wav-bytes", status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.content = content
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_successful_generation_wav(self, tmp_path):
        from tools.tts_tool import _generate_tada

        output_path = str(tmp_path / "out.wav")
        mock_resp = self._make_response(b"wav-audio")

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = _generate_tada("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "out.wav").read_bytes() == b"wav-audio"
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "http://localhost:8050/tts"
        assert call_kwargs[1]["json"] == {"text": "Hello world"}

    def test_custom_endpoint_from_config(self, tmp_path):
        from tools.tts_tool import _generate_tada

        output_path = str(tmp_path / "out.wav")
        mock_resp = self._make_response(b"audio")
        config = {"tada": {"endpoint": "http://192.168.1.50:8050/tts"}}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _generate_tada("Hi", output_path, config)

        assert mock_post.call_args[0][0] == "http://192.168.1.50:8050/tts"

    def test_default_endpoint_when_config_empty(self, tmp_path):
        from tools.tts_tool import DEFAULT_TADA_ENDPOINT, _generate_tada

        output_path = str(tmp_path / "out.wav")
        mock_resp = self._make_response(b"audio")

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _generate_tada("Hi", output_path, {})

        assert mock_post.call_args[0][0] == DEFAULT_TADA_ENDPOINT

    def test_default_endpoint_when_tada_key_missing(self, tmp_path):
        from tools.tts_tool import DEFAULT_TADA_ENDPOINT, _generate_tada

        output_path = str(tmp_path / "out.wav")
        mock_resp = self._make_response(b"audio")

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _generate_tada("Hi", output_path, {"tada": {}})

        assert mock_post.call_args[0][0] == DEFAULT_TADA_ENDPOINT

    def test_connection_error_raises_runtime_error(self, tmp_path):
        import requests as req_module
        from tools.tts_tool import _generate_tada

        output_path = str(tmp_path / "out.wav")

        with patch("requests.post", side_effect=req_module.exceptions.ConnectionError("refused")):
            with pytest.raises(RuntimeError, match="unreachable"):
                _generate_tada("Hi", output_path, {})

    def test_http_error_raises_runtime_error(self, tmp_path):
        import requests as req_module
        from tools.tts_tool import _generate_tada

        output_path = str(tmp_path / "out.wav")
        mock_resp = self._make_response(b"", status_code=500)
        mock_resp.raise_for_status.side_effect = req_module.exceptions.HTTPError("500")

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="error"):
                _generate_tada("Hi", output_path, {})

    def test_timeout_raises_runtime_error(self, tmp_path):
        import requests as req_module
        from tools.tts_tool import _generate_tada

        output_path = str(tmp_path / "out.wav")

        with patch("requests.post", side_effect=req_module.exceptions.Timeout("timed out")):
            with pytest.raises(RuntimeError, match="timed out"):
                _generate_tada("Hi", output_path, {})

    def test_non_wav_output_path_writes_wav_first(self, tmp_path):
        """Output path .mp3 — TADA writes WAV then renames (no ffmpeg)."""
        from tools.tts_tool import _generate_tada

        output_path = str(tmp_path / "out.mp3")
        mock_resp = self._make_response(b"wav-audio")

        with patch("requests.post", return_value=mock_resp), \
             patch("shutil.which", return_value=None):  # no ffmpeg
            result = _generate_tada("Hi", output_path, {})

        # Without ffmpeg, the WAV is renamed to the requested path
        assert result == output_path

    def test_non_wav_with_ffmpeg_converts(self, tmp_path):
        """Output path .mp3 — ffmpeg present, WAV converted and original removed."""
        from tools.tts_tool import _generate_tada

        output_path = str(tmp_path / "out.mp3")
        mock_resp = self._make_response(b"wav-audio")

        # Write the WAV so os.remove doesn't fail
        wav_path = str(tmp_path / "out.wav")

        def fake_post(*args, **kwargs):
            return mock_resp

        import subprocess as sp
        def fake_run(cmd, **kwargs):
            # Simulate ffmpeg: write output file
            out = cmd[-1]
            with open(out, "wb") as f:
                f.write(b"mp3-audio")
            return MagicMock(returncode=0)

        with patch("requests.post", side_effect=fake_post), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", side_effect=fake_run):
            result = _generate_tada("Hi", output_path, {})

        assert result == output_path


# ---------------------------------------------------------------------------
# Dispatcher integration tests
# ---------------------------------------------------------------------------

class TestTtsDispatcherTada:
    def test_dispatcher_routes_to_tada(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = b"wav-audio"
        mock_resp.raise_for_status = MagicMock()

        output_path = str(tmp_path / "out.wav")
        with patch("requests.post", return_value=mock_resp), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "tada"}):
            result = json.loads(
                __import__("tools.tts_tool", fromlist=["text_to_speech_tool"])
                .text_to_speech_tool("Hello", output_path=output_path)
            )

        assert result["success"] is True
        assert result["provider"] == "tada"

    def test_dispatcher_tada_connection_error_returns_error_json(self, tmp_path):
        import requests as req_module

        output_path = str(tmp_path / "out.wav")
        with patch("requests.post", side_effect=req_module.exceptions.ConnectionError("refused")), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "tada"}):
            result = json.loads(
                __import__("tools.tts_tool", fromlist=["text_to_speech_tool"])
                .text_to_speech_tool("Hello", output_path=output_path)
            )

        assert result["success"] is False
        assert "unreachable" in result["error"].lower()

    def test_tada_telegram_produces_voice_compatible(self, tmp_path, monkeypatch):
        """On Telegram, TADA WAV output triggers Opus conversion path."""
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        mock_resp = MagicMock()
        mock_resp.content = b"wav-audio"
        mock_resp.raise_for_status = MagicMock()

        # Provide a .wav output_path so _generate_tada writes directly without
        # invoking ffmpeg on fake bytes, then _convert_to_opus is called by the
        # dispatcher's Telegram voice-note path.
        wav_path = str(tmp_path / "out.wav")
        ogg_path = str(tmp_path / "out.ogg")
        with patch("requests.post", return_value=mock_resp), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "tada"}), \
             patch("tools.tts_tool._convert_to_opus", return_value=ogg_path) as mock_opus:
            result = json.loads(
                __import__("tools.tts_tool", fromlist=["text_to_speech_tool"])
                .text_to_speech_tool("Hello", output_path=wav_path)
            )

        # _convert_to_opus must be called — tada is in the WAV-output providers list
        mock_opus.assert_called_once()


# ---------------------------------------------------------------------------
# check_tts_requirements with tada provider configured
# ---------------------------------------------------------------------------

class TestCheckTtsRequirementsTada:
    def test_tada_provider_configured_returns_true(self):
        from tools.tts_tool import check_tts_requirements

        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "tada"}):
            assert check_tts_requirements() is True

    def test_no_tada_config_does_not_affect_other_providers(self):
        from tools.tts_tool import check_tts_requirements

        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), \
             patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError), \
             patch("tools.tts_tool._import_openai_client", side_effect=ImportError), \
             patch("tools.tts_tool._check_neutts_available", return_value=False), \
             patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}):
            assert check_tts_requirements() is False
