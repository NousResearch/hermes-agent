import json
from pathlib import Path

import tools.transcription_tools as stt
import tools.tts_tool as tts_tool
class TestTranscriptionTools:
    def test_auto_falls_back_to_local_without_openai_key(self, tmp_path, monkeypatch):
        audio = tmp_path / "note.ogg"
        audio.write_bytes(b"fake audio")

        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setattr(stt, "_load_stt_config", lambda: {"enabled": True, "provider": "auto", "local": {"backend": "auto", "model": "base"}})
        monkeypatch.setattr(stt, "_transcribe_with_local", lambda file_path, cfg: {"success": True, "transcript": "hello from local", "provider": "whisper"})
        monkeypatch.setattr(stt, "_transcribe_with_openai", lambda *args, **kwargs: {"success": False, "transcript": "", "error": "should not be called"})

        result = stt.transcribe_audio(str(audio))
        assert result["success"] is True
        assert result["transcript"] == "hello from local"
        assert result["provider"] == "whisper"

    def test_auto_prefers_openai_when_key_present(self, tmp_path, monkeypatch):
        audio = tmp_path / "note.ogg"
        audio.write_bytes(b"fake audio")

        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "test-key")
        monkeypatch.setattr(stt, "_load_stt_config", lambda: {"enabled": True, "provider": "auto", "model": "whisper-1"})
        monkeypatch.setattr(stt, "_transcribe_with_openai", lambda file_path, model: {"success": True, "transcript": "hello from openai", "provider": "openai"})
        monkeypatch.setattr(stt, "_transcribe_with_local", lambda *args, **kwargs: {"success": False, "transcript": "", "error": "should not be called"})

        result = stt.transcribe_audio(str(audio))
        assert result["success"] is True
        assert result["transcript"] == "hello from openai"
        assert result["provider"] == "openai"

    def test_auto_falls_back_to_local_when_openai_fails(self, tmp_path, monkeypatch):
        audio = tmp_path / "note.ogg"
        audio.write_bytes(b"fake audio")

        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "test-key")
        monkeypatch.setattr(stt, "_load_stt_config", lambda: {"enabled": True, "provider": "auto", "local": {"backend": "auto", "model": "base"}})
        monkeypatch.setattr(stt, "_transcribe_with_openai", lambda file_path, model: {"success": False, "transcript": "", "error": "API down"})
        monkeypatch.setattr(stt, "_transcribe_with_local", lambda file_path, cfg: {"success": True, "transcript": "hello from fallback", "provider": "whisper-cli"})

        result = stt.transcribe_audio(str(audio))
        assert result["success"] is True
        assert result["transcript"] == "hello from fallback"
        assert result["provider"] == "whisper-cli"

    def test_returns_generic_error_when_no_backend_available(self, tmp_path, monkeypatch):
        audio = tmp_path / "note.ogg"
        audio.write_bytes(b"fake audio")

        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setattr(stt, "_load_stt_config", lambda: {"enabled": True, "provider": "auto", "local": {"backend": "auto", "model": "base"}})
        monkeypatch.setattr(stt, "_transcribe_with_local", lambda file_path, cfg: {"success": False, "transcript": "", "error": "whisper not installed"})

        result = stt.transcribe_audio(str(audio))
        assert result["success"] is False
        assert "No speech-to-text backend available" in result["error"]

    def test_respects_disabled_config(self, tmp_path, monkeypatch):
        audio = tmp_path / "note.ogg"
        audio.write_bytes(b"fake audio")

        monkeypatch.setattr(stt, "_load_stt_config", lambda: {"enabled": False})

        result = stt.transcribe_audio(str(audio))
        assert result["success"] is False
        assert result["error"] == "Speech transcription is disabled in config"


class TestMiMoTTSProvider:
    """Tests for the Xiaomi MiMo TTS provider integration."""

    def test_generate_mimo_tts_writes_wav(self, tmp_path, monkeypatch):
        """_generate_mimo_tts decodes base64 audio and writes a file."""
        import base64

        fake_audio = b"RIFF\x00\x00\x00\x00WAVEfmt fake-wav-data"
        b64_audio = base64.b64encode(fake_audio).decode()

        # Build a mock that mimics OpenAI SDK's ChatCompletionAudio
        class FakeAudio:
            data = b64_audio
            id = "audio_123"
            expires_at = 9999999999
            transcript = "hello"

        class FakeMessage:
            audio = FakeAudio()
            content = None

        class FakeChoice:
            message = FakeMessage()

        class FakeCompletion:
            choices = [FakeChoice()]

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        # Verify the audio parameter is passed correctly
                        assert "audio" in kwargs
                        assert kwargs["audio"]["voice"] == "default_en"
                        return FakeCompletion()

            def __init__(self, **kwargs):
                pass

        monkeypatch.setenv("MIMO_API_KEY", "test-key-123")
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: FakeClient)

        out_file = str(tmp_path / "test_output.wav")
        config = {"mimo": {"voice": "default_en"}}
        result = tts_tool._generate_mimo_tts("Hello world", out_file, config)

        assert result == out_file
        assert Path(out_file).exists()
        assert Path(out_file).read_bytes() == fake_audio

    def test_mimo_tts_missing_api_key_raises(self, monkeypatch):
        """_generate_mimo_tts raises ValueError when MIMO_API_KEY is unset."""
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        import pytest
        with pytest.raises(ValueError, match="MIMO_API_KEY not set"):
            tts_tool._generate_mimo_tts("Hello", "/tmp/test.wav", {})

    def test_mimo_style_tag_prepended(self, tmp_path, monkeypatch):
        """When style is configured, a <style> tag is prepended to the text."""
        import base64

        captured_messages = []

        class FakeAudio:
            data = base64.b64encode(b"wav-bytes").decode()
        class FakeMessage:
            audio = FakeAudio()
            content = None
        class FakeChoice:
            message = FakeMessage()
        class FakeCompletion:
            choices = [FakeChoice()]

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured_messages.extend(kwargs.get("messages", []))
                        return FakeCompletion()
            def __init__(self, **kwargs):
                pass

        monkeypatch.setenv("MIMO_API_KEY", "test-key-123")
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: FakeClient)

        out_file = str(tmp_path / "styled.wav")
        config = {"mimo": {"voice": "mimo_default", "style": "Happy"}}
        tts_tool._generate_mimo_tts("Hello", out_file, config)

        # The assistant message should have the style tag prepended
        assistant_msg = [m for m in captured_messages if m["role"] == "assistant"][0]
        assert assistant_msg["content"] == "<style>Happy</style>Hello"

    def test_check_tts_requirements_detects_mimo(self, monkeypatch):
        """check_tts_requirements returns True when MIMO_API_KEY is set."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setenv("MIMO_API_KEY", "test-key")
        monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)

        # Make edge_tts import fail so we test the MiMo detection path.
        def _edge_unavailable():
            raise ImportError("no edge_tts")
        monkeypatch.setattr(tts_tool, "_import_edge_tts", _edge_unavailable)

        assert tts_tool.check_tts_requirements() is True

    def test_telegram_mimo_tts_converts_to_opus(self, tmp_path, monkeypatch):
        """MiMo TTS output is converted to Opus for Telegram voice bubbles."""
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "mimo"})

        def fake_generate_mimo_tts(text, output_path, tts_config):
            Path(output_path).write_bytes(b"wav-bytes")
            return output_path

        def fake_convert_to_opus(path):
            ogg_path = str(Path(path).with_suffix(".ogg"))
            Path(ogg_path).write_bytes(b"ogg-bytes")
            return ogg_path

        monkeypatch.setattr(tts_tool, "_generate_mimo_tts", fake_generate_mimo_tts)
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: None)
        monkeypatch.setattr(tts_tool, "_convert_to_opus", fake_convert_to_opus)
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path / "audio"))

        result = json.loads(tts_tool.text_to_speech_tool("Hello MiMo"))
        assert result["success"] is True
        assert result["voice_compatible"] is True
        assert result["file_path"].endswith(".ogg")


class TestTextToSpeechTool:
    def test_telegram_edge_tts_returns_voice_directive(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "audio"
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge", "edge": {"voice": "en-US-AriaNeural"}})
        monkeypatch.setattr(tts_tool, "_HAS_EDGE_TTS", True)

        async def fake_generate_edge_tts(text, output_path, tts_config):
            Path(output_path).write_bytes(b"mp3-bytes")
            return output_path

        def fake_convert_to_opus(mp3_path):
            ogg_path = str(Path(mp3_path).with_suffix(".ogg"))
            Path(ogg_path).write_bytes(b"ogg-bytes")
            return ogg_path

        monkeypatch.setattr(tts_tool, "_generate_edge_tts", fake_generate_edge_tts)
        monkeypatch.setattr(tts_tool, "_convert_to_opus", fake_convert_to_opus)
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(out_dir))

        result = json.loads(tts_tool.text_to_speech_tool("Hello voice"))
        assert result["success"] is True
        assert result["voice_compatible"] is True
        assert result["file_path"].endswith(".ogg")
        assert result["media_tag"].startswith("[[audio_as_voice]]\nMEDIA:")
