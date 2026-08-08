"""ElevenLabs TTS configuration passthrough tests.

Covers both request-building surfaces that must accept the same
``tts.elevenlabs`` option shape:

* the synchronous file-generation path (``tools.tts_tool._generate_elevenlabs``)
* the live-streaming path (``tools.tts_streaming.ElevenLabsStreamer.stream``)
"""

from unittest.mock import MagicMock, patch

import pytest


def _voice_settings_value(voice_settings, key):
    if isinstance(voice_settings, dict):
        return voice_settings[key]
    return getattr(voice_settings, key)


# ── Synchronous file-generation path ─────────────────────────────────────


class TestElevenLabsOptionsSync:
    def test_default_config_keeps_request_shape_unchanged(self, tmp_path):
        """No options set anywhere -> only the four managed kwargs are sent."""
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            tts_tool._generate_elevenlabs("Hello", str(tmp_path / "out.mp3"), {})

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert set(kwargs) == {"text", "voice_id", "model_id", "output_format"}

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
                            "apply_text_normalization": "on",
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
        assert kwargs["apply_text_normalization"] == "on"
        assert _voice_settings_value(kwargs["voice_settings"], "stability") == 0.55
        assert _voice_settings_value(kwargs["voice_settings"], "similarity_boost") == 0.8
        assert _voice_settings_value(kwargs["voice_settings"], "style") == 0.25
        assert _voice_settings_value(kwargs["voice_settings"], "use_speaker_boost") is False
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 1.1

    def test_base_url_environment_coexists_with_convert_options(self, tmp_path):
        from tools import tts_tool

        environment = object()
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])
        factory = MagicMock(return_value=mock_client)

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=factory), \
             patch.object(
                 tts_tool,
                 "_elevenlabs_environment_kwargs",
                 return_value={"environment": environment},
             ) as environment_kwargs:
            tts_tool._generate_elevenlabs(
                "hi",
                str(tmp_path / "out.mp3"),
                {
                    "elevenlabs": {
                        "base_url": "https://el-proxy.example",
                        "language_code": "en",
                    }
                },
            )

        environment_kwargs.assert_called_once_with(
            {"base_url": "https://el-proxy.example", "language_code": "en"}
        )
        factory.assert_called_once_with(api_key="el-key", environment=environment)
        assert mock_client.text_to_speech.convert.call_args.kwargs["language_code"] == "en"

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
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 0.95

    def test_nested_voice_settings_wins_over_legacy_top_level_keys(self, tmp_path):
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
                        "stability": 0.1,
                        "voice_settings": {"stability": 0.9},
                    }
                },
            )

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert _voice_settings_value(kwargs["voice_settings"], "stability") == 0.9

    def test_options_alias_forwards_convert_options(self, tmp_path):
        """``options`` remains a compatibility alias for ``convert_options``."""
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            tts_tool._generate_elevenlabs(
                "hi",
                str(tmp_path / "out.mp3"),
                {"elevenlabs": {"options": {"seed": 99}}},
            )

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["seed"] == 99

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

    def test_rejects_controlled_keys_in_convert_options_before_api_call(self, tmp_path):
        from tools import tts_tool

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch.object(tts_tool, "get_env_value", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            with pytest.raises(ValueError, match="Hermes-managed option"):
                tts_tool._generate_elevenlabs(
                    "hi",
                    str(tmp_path / "out.mp3"),
                    {
                        "elevenlabs": {
                            "convert_options": {"voice_id": "someone-elses-voice"},
                        }
                    },
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


# ── Live-streaming path ──────────────────────────────────────────────────


class TestElevenLabsStreamerOptions:
    """Direct coverage of ``ElevenLabsStreamer.stream`` — the provider-agnostic
    streaming architecture introduced after the hard-coded ElevenLabs branch
    was removed from ``stream_tts_to_speaker``.
    """

    def _stream(self, tts_config, section, text="Hello", mock_client=None):
        from tools import tts_streaming as ts

        if mock_client is None:
            mock_client = MagicMock()
            mock_client.text_to_speech.convert.return_value = iter([b"\x00\x00"])

        with patch.object(ts, "get_env_value", lambda k, *a: "el-key" if k == "ELEVENLABS_API_KEY" else None), \
             patch("tools.tts_tool._import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            streamer = ts.ElevenLabsStreamer(tts_config, section)
            list(streamer.stream(text))

        return mock_client

    def test_default_config_keeps_request_shape_unchanged(self):
        mock_client = self._stream({}, {})
        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert set(kwargs) == {"text", "voice_id", "model_id", "output_format"}
        assert kwargs["output_format"] == "pcm_24000"

    def test_forwards_language_voice_settings_and_convert_options(self):
        mock_client = self._stream(
            {"speed": 1.2},
            {
                "voice_id": "voice-123",
                "streaming_model_id": "eleven_flash_v2_5",
                "language_code": "en",
                "voice_settings": {"stability": 0.5},
                "convert_options": {"seed": 42},
            },
            text="Hello there",
        )

        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["text"] == "Hello there"
        assert kwargs["voice_id"] == "voice-123"
        assert kwargs["model_id"] == "eleven_flash_v2_5"
        assert kwargs["output_format"] == "pcm_24000"
        assert kwargs["language_code"] == "en"
        assert kwargs["seed"] == 42
        assert _voice_settings_value(kwargs["voice_settings"], "stability") == 0.5
        # No provider-specific speed was set -> global tts.speed fills in.
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 1.2

    def test_base_url_environment_coexists_with_convert_options(self):
        from tools import tts_streaming as ts
        from tools import tts_tool

        environment = object()
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"\x00\x00"])
        factory = MagicMock(return_value=mock_client)

        with patch.object(ts, "_resolve_key", return_value="el-key"), \
             patch.object(tts_tool, "_import_elevenlabs", return_value=factory), \
             patch.object(
                 tts_tool,
                 "_elevenlabs_environment_kwargs",
                 return_value={"environment": environment},
             ) as environment_kwargs:
            streamer = ts.ElevenLabsStreamer(
                {}, {"base_url": "https://el-proxy.example", "language_code": "en"}
            )
            list(streamer.stream("hi"))

        environment_kwargs.assert_called_once_with(
            {"base_url": "https://el-proxy.example", "language_code": "en"}
        )
        factory.assert_called_once_with(api_key="el-key", environment=environment)
        assert mock_client.text_to_speech.convert.call_args.kwargs["language_code"] == "en"

    def test_accepts_legacy_top_level_voice_settings_keys(self):
        mock_client = self._stream({}, {"stability": 0.4, "speed": 0.8})
        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert _voice_settings_value(kwargs["voice_settings"], "stability") == 0.4
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 0.8

    def test_provider_speed_overrides_global_speed(self):
        mock_client = self._stream(
            {"speed": 1.2},
            {"voice_settings": {"speed": 0.9}},
        )
        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert _voice_settings_value(kwargs["voice_settings"], "speed") == 0.9

    def test_rejects_unknown_convert_options_before_api_call(self):
        from tools import tts_streaming as ts

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"\x00\x00"])

        with patch.object(ts, "get_env_value", lambda k, *a: "el-key" if k == "ELEVENLABS_API_KEY" else None), \
             patch("tools.tts_tool._import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            streamer = ts.ElevenLabsStreamer({}, {"convert_options": {"not_a_real_option": True}})
            with pytest.raises(ValueError, match="Unknown ElevenLabs convert_options"):
                list(streamer.stream("hi"))

        mock_client.text_to_speech.convert.assert_not_called()

    def test_rejects_controlled_keys_in_convert_options_before_api_call(self):
        from tools import tts_streaming as ts

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"\x00\x00"])

        with patch.object(ts, "get_env_value", lambda k, *a: "el-key" if k == "ELEVENLABS_API_KEY" else None), \
             patch("tools.tts_tool._import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            streamer = ts.ElevenLabsStreamer({}, {"convert_options": {"model_id": "not-allowed"}})
            with pytest.raises(ValueError, match="Hermes-managed option"):
                list(streamer.stream("hi"))

        mock_client.text_to_speech.convert.assert_not_called()

    def test_rejects_unknown_voice_settings_before_api_call(self):
        from tools import tts_streaming as ts

        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"\x00\x00"])

        with patch.object(ts, "get_env_value", lambda k, *a: "el-key" if k == "ELEVENLABS_API_KEY" else None), \
             patch("tools.tts_tool._import_elevenlabs", return_value=MagicMock(return_value=mock_client)):
            streamer = ts.ElevenLabsStreamer({}, {"voice_settings": {"not_a_real_setting": 1}})
            with pytest.raises(ValueError, match="Unknown ElevenLabs voice_settings"):
                list(streamer.stream("hi"))

        mock_client.text_to_speech.convert.assert_not_called()

    def test_streaming_model_resolution_unaffected_by_options(self):
        """model resolution order (streaming_model_id > model_id > default)
        must survive the option-forwarding change untouched."""
        mock_client = self._stream(
            {},
            {"model_id": "eleven_v3", "language_code": "de"},
        )
        kwargs = mock_client.text_to_speech.convert.call_args.kwargs
        assert kwargs["model_id"] == "eleven_v3"
        assert kwargs["language_code"] == "de"
