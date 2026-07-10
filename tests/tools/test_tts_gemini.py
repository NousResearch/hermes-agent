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
        assert "X-Goog-Api-Client" not in mock_post.call_args[1]["headers"]

    def test_lookalike_base_url_omits_client_context(
        self, tmp_path, monkeypatch, mock_gemini_response
    ):
        from tools.tts_tool import _generate_gemini_tts

        lookalike = "https://generativelanguage.googleapis.com.evil.example/v1beta"
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_BASE_URL", lookalike)

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

        assert mock_post.call_args[0][0].startswith(f"{lookalike}/")
        assert "X-Goog-Api-Client" not in mock_post.call_args[1]["headers"]

    def test_persona_prompt_file_appends_labeled_transcript(
        self, tmp_path, monkeypatch, mock_gemini_response
    ):
        from tools.tts_tool import _generate_gemini_tts

        persona_file = tmp_path / "voice-persona.md"
        persona_file.write_text(
            "# AUDIO PROFILE: Dry Butler\n\n### DIRECTOR'S NOTES\nStyle: Understated.",
            encoding="utf-8",
        )
        config = {"gemini": {"persona_prompt_file": str(persona_file)}}
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), config)

        prompt_text = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert "Synthesize speech from the TRANSCRIPT only" in prompt_text
        assert "# AUDIO PROFILE: Dry Butler" in prompt_text
        assert "### DIRECTOR'S NOTES\nStyle: Understated." in prompt_text
        assert "#### TRANSCRIPT\nHi" in prompt_text

    def test_persona_prompt_file_supports_transcript_placeholder(
        self, tmp_path, monkeypatch, mock_gemini_response
    ):
        from tools.tts_tool import _generate_gemini_tts

        persona_file = tmp_path / "voice-persona.md"
        persona_file.write_text(
            "### DIRECTOR'S NOTES\nPacing: Slow.\n\n#### TRANSCRIPT\n{{ transcript }}",
            encoding="utf-8",
        )
        config = {"gemini": {"persona_prompt_file": str(persona_file)}}
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Read this.", str(tmp_path / "test.wav"), config)

        prompt_text = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert "{{ transcript }}" not in prompt_text
        assert "#### TRANSCRIPT\nRead this." in prompt_text

    def test_missing_persona_prompt_file_warns_and_continues(
        self, tmp_path, monkeypatch, caplog, mock_gemini_response
    ):
        from tools.tts_tool import _generate_gemini_tts

        config = {"gemini": {"persona_prompt_file": str(tmp_path / "missing.md")}}
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), config)

        prompt_text = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert prompt_text == "Hi"
        assert "persona prompt file unavailable" in caplog.text

    def test_audio_tags_disabled_does_not_call_rewriter(
        self, tmp_path, monkeypatch, mock_gemini_response
    ):
        from tools.tts_tool import _generate_gemini_tts

        config = {
            "gemini": {
                "model": "gemini-3.1-flash-tts-preview",
                "audio_tags": False,
            }
        }
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("agent.auxiliary_client.call_llm") as mock_call_llm, \
             patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi there.", str(tmp_path / "test.wav"), config)

        mock_call_llm.assert_not_called()
        prompt_text = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert prompt_text == "Hi there."

    def test_audio_tags_enabled_rewrites_hidden_tts_script(
        self, tmp_path, monkeypatch, mock_gemini_response
    ):
        from tools.tts_tool import _generate_gemini_tts

        persona_file = tmp_path / "voice-persona.md"
        persona_file.write_text(
            "### DIRECTOR'S NOTES\nStyle: Warm and amused.",
            encoding="utf-8",
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="[warmly] Hi there. [soft laugh]")
                )
            ]
        )
        config = {
            "gemini": {
                "model": "gemini-3.1-flash-tts-preview",
                "audio_tags": True,
                "persona_prompt_file": str(persona_file),
            }
        }
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("agent.auxiliary_client.call_llm", return_value=response) as mock_call_llm, \
             patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi there.", str(tmp_path / "test.wav"), config)

        mock_call_llm.assert_called_once()
        call_kwargs = mock_call_llm.call_args.kwargs
        assert call_kwargs["task"] == "tts_audio_tags"
        assert "Audio tags are inline square-bracket modifiers" in call_kwargs["messages"][0]["content"]
        assert "Style: Warm and amused." in call_kwargs["messages"][1]["content"]
        assert "Hi there." in call_kwargs["messages"][1]["content"]

        prompt_text = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert "Synthesize speech from the TRANSCRIPT only" in prompt_text
        assert "### DIRECTOR'S NOTES\nStyle: Warm and amused." in prompt_text
        assert "#### TRANSCRIPT\n[warmly] Hi there. [soft laugh]" in prompt_text

    def test_audio_tags_enabled_skips_non_tag_capable_model(
        self, tmp_path, monkeypatch, mock_gemini_response, caplog
    ):
        from tools.tts_tool import _generate_gemini_tts

        config = {
            "gemini": {
                "model": "gemini-2.5-flash-preview-tts",
                "audio_tags": True,
            }
        }
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("agent.auxiliary_client.call_llm") as mock_call_llm, \
             patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi there.", str(tmp_path / "test.wav"), config)

        mock_call_llm.assert_not_called()
        prompt_text = mock_post.call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert prompt_text == "Hi there."
        assert "not known to support Gemini audio tags" in caplog.text

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

        with patch("builtins.__import__", side_effect=fake_import):
            assert check_tts_requirements() is True
