"""Tests for the Google Gemini TTS provider in tools/tts_tool.py."""

import base64
import struct
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "HERMES_SESSION_PLATFORM",
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

    def test_custom_model(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {"gemini": {"model": "gemini-2.5-pro-preview-tts"}}

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), config)

        endpoint = mock_post.call_args[0][0]
        assert "gemini-2.5-pro-preview-tts" in endpoint

    def test_response_modality_is_audio(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

        payload = mock_post.call_args[1]["json"]
        assert payload["generationConfig"]["responseModalities"] == ["AUDIO"]

    def test_http_error_raises_runtime_error(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        err_resp = MagicMock()
        err_resp.status_code = 400
        err_resp.json.return_value = {"error": {"message": "Invalid voice"}}

        with patch("requests.post", return_value=err_resp):
            with pytest.raises(RuntimeError, match="HTTP 400.*Invalid voice"):
                _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

    def test_empty_audio_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"inlineData": {"data": ""}}]}}
            ]
        }

        with patch("requests.post", return_value=resp):
            with pytest.raises(RuntimeError, match="empty audio"):
                _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

    def test_malformed_response_raises(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"candidates": []}  # no content

        with patch("requests.post", return_value=resp):
            with pytest.raises(RuntimeError, match="malformed"):
                _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

    def test_snake_case_inline_data_accepted(self, tmp_path, monkeypatch, fake_pcm_bytes):
        """Some Gemini SDK versions return inline_data instead of inlineData."""
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "data": base64.b64encode(fake_pcm_bytes).decode()
                                }
                            }
                        ]
                    }
                }
            ]
        }

        output_path = str(tmp_path / "test.wav")
        with patch("requests.post", return_value=resp):
            _generate_gemini_tts("Hi", output_path, {})

        data = (tmp_path / "test.wav").read_bytes()
        assert data[:4] == b"RIFF"

    def test_custom_base_url_env(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_BASE_URL", "https://custom-gemini.example.com/v1beta")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

        assert mock_post.call_args[0][0].startswith("https://custom-gemini.example.com/v1beta/")


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

        with patch("builtins.__import__", side_effect=fake_import):
            assert check_tts_requirements() is True


class TestResolveGeminiModels:
    def test_default_is_single_primary(self):
        from tools.tts_tool import DEFAULT_GEMINI_TTS_MODEL, _resolve_gemini_models

        assert _resolve_gemini_models({}) == [DEFAULT_GEMINI_TTS_MODEL]

    def test_primary_then_fallbacks_in_order(self):
        from tools.tts_tool import _resolve_gemini_models

        cfg = {"model": "m-primary", "fallback_models": ["m-a", "m-b"]}
        assert _resolve_gemini_models(cfg) == ["m-primary", "m-a", "m-b"]

    def test_dedup_primary_and_duplicates(self):
        from tools.tts_tool import _resolve_gemini_models

        cfg = {"model": "m-primary", "fallback_models": ["m-primary", "m-a", "m-a", "m-b"]}
        assert _resolve_gemini_models(cfg) == ["m-primary", "m-a", "m-b"]

    def test_blank_entries_dropped(self):
        from tools.tts_tool import _resolve_gemini_models

        cfg = {"model": "m-primary", "fallback_models": ["", "   ", "m-a"]}
        assert _resolve_gemini_models(cfg) == ["m-primary", "m-a"]

    def test_non_list_fallback_ignored(self):
        from tools.tts_tool import DEFAULT_GEMINI_TTS_MODEL, _resolve_gemini_models

        assert _resolve_gemini_models({"fallback_models": "nope"}) == [DEFAULT_GEMINI_TTS_MODEL]


class TestGeminiFallbackModels:
    def _err_resp(self, status=429, message="rate limited"):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = {"error": {"message": message}}
        return resp

    def test_single_model_makes_one_request(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})
        assert mock_post.call_count == 1

    def test_falls_back_on_http_error(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {"gemini": {"model": "m-primary", "fallback_models": ["m-fallback"]}}
        with patch(
            "requests.post",
            side_effect=[self._err_resp(429), mock_gemini_response],
        ) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "out.wav"), config)

        assert mock_post.call_count == 2
        # Primary is tried first, fallback second.
        assert "m-primary" in mock_post.call_args_list[0][0][0]
        assert "m-fallback" in mock_post.call_args_list[1][0][0]
        assert (tmp_path / "out.wav").read_bytes()[:4] == b"RIFF"

    def test_falls_back_on_empty_audio(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        empty = MagicMock()
        empty.status_code = 200
        empty.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": {"data": ""}}]}}]
        }
        config = {"gemini": {"model": "m-primary", "fallback_models": ["m-fallback"]}}
        with patch(
            "requests.post",
            side_effect=[empty, mock_gemini_response],
        ) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "out.wav"), config)

        assert mock_post.call_count == 2

    def test_all_models_fail_raises_last_error(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {"gemini": {"model": "m-primary", "fallback_models": ["m-fallback"]}}
        with patch(
            "requests.post",
            side_effect=[self._err_resp(500, "first"), self._err_resp(503, "last")],
        ) as mock_post:
            with pytest.raises(RuntimeError, match="HTTP 503.*last"):
                _generate_gemini_tts("Hi", str(tmp_path / "out.wav"), config)
        assert mock_post.call_count == 2

    def test_no_fallback_does_not_retry(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with patch(
            "requests.post",
            side_effect=[self._err_resp(400, "bad")],
        ) as mock_post:
            with pytest.raises(RuntimeError, match="HTTP 400.*bad"):
                _generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {})
        assert mock_post.call_count == 1
