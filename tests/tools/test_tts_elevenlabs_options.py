"""ElevenLabs TTS configuration passthrough tests."""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest


class TestElevenLabsOptions:
    def test_passes_language_voice_settings_and_convert_options(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            tts_tool._generate_elevenlabs(
                "Hello",
                str(tmp_path / "out.mp3"),
                {
                    "elevenlabs": {
                        "voice_id": "voice-123",
                        "model_id": "eleven_v3",
                        "language_code": "en",
                        "voice_settings": {
                            "stability": 0.55,
                            "similarity_boost": 0.8,
                            "style": 0.25,
                            "use_speaker_boost": False,
                            "speed": 1.1,
                        },
                        "convert_options": {
                            "seed": 1234,
                            "previous_text": "Previous sentence.",
                            "next_text": "Next sentence.",
                            "apply_text_normalization": "on",
                            "apply_language_text_normalization": True,
                            "enable_logging": False,
                            "optimize_streaming_latency": 2,
                        },
                    }
                },
            )

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["text"] == "Hello"
        assert kwargs["voice_id"] == "voice-123"
        assert kwargs["model_id"] == "eleven_v3"
        assert kwargs["language_code"] == "en"
        assert kwargs["seed"] == 1234
        assert kwargs["previous_text"] == "Previous sentence."
        assert kwargs["next_text"] == "Next sentence."
        assert kwargs["apply_text_normalization"] == "on"
        assert kwargs["apply_language_text_normalization"] is True
        assert kwargs["enable_logging"] is False
        assert kwargs["optimize_streaming_latency"] == 2
        assert _voice_settings_value(kwargs["voice_settings"], "stability") == 0.55
        assert _voice_settings_value(kwargs["voice_settings"], "similarity_boost") == 0.8
        assert _voice_settings_value(kwargs["voice_settings"], "style") == 0.25
        assert _voice_settings_value(kwargs["voice_settings"], "use_speaker_boost") is False
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 1.1

    def test_accepts_legacy_top_level_voice_settings_keys(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            tts_tool._generate_elevenlabs(
                "hi",
                str(tmp_path / "out.mp3"),
                {
                    "elevenlabs": {
                        "stability": 0.4,
                        "similarity_boost": 0.7,
                        "style": 0.2,
                        "use_speaker_boost": True,
                        "speed": 0.95,
                    }
                },
            )

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert _voice_settings_value(kwargs["voice_settings"], "stability") == 0.4
        assert _voice_settings_value(kwargs["voice_settings"], "similarity_boost") == 0.7
        assert _voice_settings_value(kwargs["voice_settings"], "style") == 0.2
        assert _voice_settings_value(kwargs["voice_settings"], "use_speaker_boost") is True
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 0.95

    def test_uses_global_speed_fallback_when_provider_speed_is_absent(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            tts_tool._generate_elevenlabs(
                "hi",
                str(tmp_path / "out.mp3"),
                {
                    "speed": 1.2,
                    "elevenlabs": {
                        "voice_id": "voice-123",
                        "voice_settings": {"stability": 0.4},
                    },
                },
            )

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert _voice_settings_value(kwargs["voice_settings"], "stability") == 0.4
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 1.2

    def test_provider_speed_overrides_global_speed(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            tts_tool._generate_elevenlabs(
                "hi",
                str(tmp_path / "out.mp3"),
                {
                    "speed": 1.2,
                    "elevenlabs": {
                        "voice_id": "voice-123",
                        "voice_settings": {"speed": 0.9},
                    },
                },
            )

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 0.9

    def test_rejects_unknown_convert_options_before_api_call(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            with pytest.raises(ValueError, match="Unknown ElevenLabs convert_options"):
                tts_tool._generate_elevenlabs(
                    "hi",
                    str(tmp_path / "out.mp3"),
                    {"elevenlabs": {"convert_options": {"not_a_real_option": True}}},
                )

        mock_client.text_to_speech.convert.assert_not_called()

    def test_rejects_non_mapping_voice_settings_before_api_call(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            with pytest.raises(ValueError, match="voice_settings must be a mapping"):
                tts_tool._generate_elevenlabs(
                    "hi",
                    str(tmp_path / "out.mp3"),
                    {"elevenlabs": {"voice_settings": "fast"}},
                )

        mock_client.text_to_speech.convert.assert_not_called()

    def test_rejects_non_mapping_convert_options_before_api_call(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            with pytest.raises(ValueError, match="convert_options must be a mapping"):
                tts_tool._generate_elevenlabs(
                    "hi",
                    str(tmp_path / "out.mp3"),
                    {"elevenlabs": {"convert_options": "seed=1234"}},
                )

        mock_client.text_to_speech.convert.assert_not_called()

    def test_rejects_unknown_voice_settings_before_api_call(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            with pytest.raises(ValueError, match="Unknown ElevenLabs voice_settings"):
                tts_tool._generate_elevenlabs(
                    "hi",
                    str(tmp_path / "out.mp3"),
                    {"elevenlabs": {"voice_settings": {"not_a_real_setting": 1}}},
                )

        mock_client.text_to_speech.convert.assert_not_called()

    def test_uses_mapping_voice_settings_when_sdk_type_is_unavailable(self, tmp_path, monkeypatch):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "elevenlabs":
                raise ImportError("no VoiceSettings type")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            tts_tool._generate_elevenlabs(
                "hi",
                str(tmp_path / "out.mp3"),
                {"elevenlabs": {"voice_settings": {"stability": 0.5}}},
            )

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["voice_settings"] == {"stability": 0.5}


class TestStreamingElevenLabsOptions:
    """stream_tts_to_speaker must forward the same option surfaces as
    _generate_elevenlabs, resolved once per invocation rather than per
    sentence."""

    def test_forwards_language_voice_settings_and_convert_options(self):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"\x00\x00"])

        fake_sd = MagicMock()
        fake_output_stream = MagicMock()
        fake_sd.OutputStream.return_value = fake_output_stream

        tts_config = {
            "speed": 1.2,
            "elevenlabs": {
                "voice_id": "voice-123",
                "model_id": "eleven_v3",
                "streaming_model_id": "eleven_flash_v2_5",
                "language_code": "en",
                "voice_settings": {"stability": 0.5},
                "convert_options": {"seed": 42},
            },
        }

        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()

        text_q.put("This is a long enough sentence to speak. ")
        text_q.put(None)

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)), \
             patch.object(tts_tool, "_import_sounddevice", return_value=fake_sd), \
             patch.object(tts_tool, "_load_tts_config", return_value=tts_config):
            tts_tool.stream_tts_to_speaker(text_q, stop_evt, done_evt)

        assert done_evt.is_set()
        mock_client.text_to_speech.convert.assert_called_once()
        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["voice_id"] == "voice-123"
        assert kwargs["model_id"] == "eleven_flash_v2_5"
        assert kwargs["output_format"] == "pcm_24000"
        assert kwargs["language_code"] == "en"
        assert kwargs["seed"] == 42
        assert _voice_settings_value(kwargs["voice_settings"], "stability") == 0.5
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 1.2


def _voice_settings_value(voice_settings, key):
    if isinstance(voice_settings, dict):
        return voice_settings[key]
    return getattr(voice_settings, key)
