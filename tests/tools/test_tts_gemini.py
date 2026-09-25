"""Tests for the Google Gemini TTS provider in tools/tts_tool.py."""

import base64
import struct
import wave
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
        from tools.tts_tool_delivery import _wrap_pcm_as_wav

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



class TestGenerateGeminiTts:
    def test_public_tool_forwards_instructions_to_gemini(self, tmp_path):
        import json
        from tools.tts_tool import text_to_speech_tool

        output = tmp_path / "out.mp3"
        def generate(text, path, config, instructions=None):
            assert instructions == "Speak warmly"
            assert text == "Hello world."
            output.write_bytes(b"ID3fakeaudio")
            return path

        with patch("tools.tts_tool._load_tts_config", return_value={"provider": "gemini"}), \
             patch("tools.tts_tool._generate_gemini_tts", side_effect=generate):
            result = json.loads(text_to_speech_tool("Hello world.", output_path=str(output),
                                                    instructions="Speak warmly"))
        assert result["success"] is True

    @pytest.mark.parametrize("model", ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"])
    def test_38_wire_shape_and_wav(self, model, tmp_path, monkeypatch, fake_pcm_bytes):
        from tools.tts_tool import _generate_gemini_tts
        from tools.tts_tool_delivery import _wrap_pcm_as_wav

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        persona = tmp_path / "persona.md"
        persona.write_text("THE SCENE\nQuiet study.\nDIRECTOR'S NOTES\nWarm and measured.")
        wav = _wrap_pcm_as_wav(fake_pcm_bytes)
        response = MagicMock(status_code=200)
        response.json.return_value = {"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav).decode()}}
        ]}}]}
        config = {"gemini": {"model": model, "voice": "voice_custom", "audio_tags": False,
                             "persona_prompt_file": str(persona)}}
        with patch("requests.post", return_value=response) as post, \
             patch("agent.auxiliary_client.call_llm") as rewrite:
            _generate_gemini_tts("Hello <laugh> world", str(tmp_path / "out.wav"), config)
        rewrite.assert_not_called()
        assert post.call_args.args[0].endswith(f"/models/{model}:generateContent")
        assert post.call_args.kwargs["json"] == {
            "contents": [{"role": "user", "parts": [{"text": "Hello <laugh> world",
                "speech_metadata": {"style": persona.read_text()}}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"voice": "voice_custom"}}}}
        assert (tmp_path / "out.wav").read_bytes() == wav

    def test_38_audio_tags_only_insert_supported_events(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {"gemini": {"model": "gemini-3.8-flash-tts", "audio_tags": True}}
        with patch("tools.tts_tool_providers._rewrite_with_auxiliary_model",
                   return_value="Hello <laugh> world") as rewrite, \
             patch("requests.post", return_value=mock_gemini_response) as post:
            _generate_gemini_tts("Hello  world", str(tmp_path / "out.wav"), config)
        assert rewrite.called
        assert post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"] == "Hello <laugh> world"

        for unsafe in ("Hello [whispers] world", "Hello <whisper> world", "Different text"):
            with patch("tools.tts_tool_providers._rewrite_with_auxiliary_model", return_value=unsafe), \
                 patch("requests.post", return_value=mock_gemini_response) as post:
                _generate_gemini_tts("Hello  world", str(tmp_path / "out.wav"), config)
            assert post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"] == "Hello  world"

    def test_38_style_precedence_and_legacy_placeholder(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        persona = tmp_path / "persona.md"
        persona.write_text("Say this: {{transcript}}")
        config = {"gemini": {"model": "gemini-3.8-flash-tts", "persona_prompt_file": str(persona)}}
        with pytest.raises(ValueError, match="migrate"):
            _generate_gemini_tts("Hello", str(tmp_path / "out.wav"), config)
        for style, call in [("configured", None), ("configured", "per-call"), ("", "per-call")]:
            config["gemini"]["style"] = style
            with patch("requests.post", return_value=mock_gemini_response) as post:
                _generate_gemini_tts("Hello", str(tmp_path / "out.wav"), config, instructions=call)
            part = post.call_args.kwargs["json"]["contents"][0]["parts"][0]
            assert part == {"text": "Hello", "speech_metadata": {"style": call or style}}

    def test_38_l16_respects_mime_rate_and_rejects_unknown(self, tmp_path, monkeypatch, fake_pcm_bytes):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        response = MagicMock(status_code=200)
        inline = {"mimeType": "audio/L16;codec=pcm;rate=16000",
                  "data": base64.b64encode(fake_pcm_bytes).decode()}
        response.json.return_value = {"candidates": [{"content": {"parts": [{"inlineData": inline}]}}]}
        with patch("requests.post", return_value=response):
            _generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {"gemini": {"model": "gemini-3.8-flash-tts"}})
        with wave.open(str(tmp_path / "out.wav"), "rb") as audio:
            assert audio.getframerate() == 16000
            assert audio.readframes(audio.getnframes()) == fake_pcm_bytes
        inline["mimeType"] = "audio/mp3"
        with patch("requests.post", return_value=response), pytest.raises(RuntimeError, match="unsupported audio MIME"):
            _generate_gemini_tts("Hi", str(tmp_path / "out.wav"), {"gemini": {"model": "gemini-3.8-flash-tts"}})

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
        from hermes_cli.version_info import get_version_info
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("requests.post", return_value=mock_gemini_response) as mock_post:
            _generate_gemini_tts("Hi", str(tmp_path / "test.wav"), {})

        headers = mock_post.call_args[1]["headers"]
        assert headers["X-Goog-Api-Client"] == f"hermes-agent/{get_version_info().base_version}"

    def test_default_voice_and_model(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts
        from tools.tts_tool_providers import DEFAULT_GEMINI_TTS_MODEL, DEFAULT_GEMINI_TTS_VOICE

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


class TestGeminiInteractionsProtocol:
    def test_interactions_protocol_wire_shape_and_speakers(self, tmp_path, monkeypatch, fake_pcm_bytes):
        from tools.tts_tool import _generate_gemini_tts
        from tools.tts_tool_delivery import _wrap_pcm_as_wav

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        wav = _wrap_pcm_as_wav(fake_pcm_bytes)
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "audio",
                            "data": base64.b64encode(wav).decode(),
                            "mime_type": "audio/wav",
                        }
                    ],
                }
            ]
        }
        config = {
            "gemini": {
                "model": "gemini-3.8-flash-tts",
                "voice": "Kore",
                "protocol": "interactions",
            }
        }
        with patch("requests.post", return_value=response) as post:
            _generate_gemini_tts("Hello world", str(tmp_path / "out.wav"), config)

        assert post.call_args.args[0].endswith("/interactions")
        payload = post.call_args.kwargs["json"]
        assert payload["model"] == "gemini-3.8-flash-tts"
        assert payload["input"] == [{"type": "text", "text": "Hello world"}]
        assert payload["generation_config"]["speech_config"]["speakers"] == [{"voice": "Kore"}]
        assert (tmp_path / "out.wav").read_bytes() == wav

    def test_interactions_auto_detects_custom_base_url(self, tmp_path, monkeypatch, fake_pcm_bytes):
        from tools.tts_tool import _generate_gemini_tts
        from tools.tts_tool_delivery import _wrap_pcm_as_wav

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        wav = _wrap_pcm_as_wav(fake_pcm_bytes)
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "steps": [
                {
                    "content": [
                        {
                            "type": "audio",
                            "data": base64.b64encode(wav).decode(),
                            "mime_type": "audio/wav",
                        }
                    ]
                }
            ]
        }
        config = {
            "gemini": {
                "base_url": "http://192.168.21.6:8317/v1beta",
                "model": "Gemini 3.8 Flash TTS",
            }
        }
        with patch("requests.post", return_value=response) as post:
            _generate_gemini_tts("Custom proxy test", str(tmp_path / "out.wav"), config)

        assert post.call_args.args[0] == "http://192.168.21.6:8317/v1beta/interactions"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "test-key"

    def test_proxy_configuration_forwarded_to_requests(self, tmp_path, monkeypatch, mock_gemini_response):
        from tools.tts_tool import _generate_gemini_tts

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {
            "gemini": {
                "model": "gemini-3.8-flash-tts",
                "proxy": "http://192.168.21.6:17893",
            }
        }
        with patch("requests.post", return_value=mock_gemini_response) as post:
            _generate_gemini_tts("Proxy test", str(tmp_path / "out.wav"), config)

        assert post.call_args.kwargs["proxies"] == {
            "http": "http://192.168.21.6:17893",
            "https": "http://192.168.21.6:17893",
        }

