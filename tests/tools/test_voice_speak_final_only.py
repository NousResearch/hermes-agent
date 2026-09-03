"""Tests for voice.speak_final_only (speak only the final response).

Contract:
  - `should_stream_tts()` returns True by default (streaming enabled).
  - With `voice.speak_final_only: True` it returns False, so the CLI skips
    the streaming TTS feed and only the turn's final response is spoken.
  - A malformed ``voice:`` config block and config-load errors fall back to
    streaming enabled (the pre-existing default behaviour).
"""

from unittest.mock import patch

from tools.tts_tool import should_stream_tts


class TestShouldStreamTts:
    def test_default_streaming_enabled(self):
        assert should_stream_tts({}) is True

    def test_speak_final_only_disables_streaming(self):
        assert should_stream_tts({"speak_final_only": True}) is False

    def test_false_explicitly_keeps_streaming(self):
        assert should_stream_tts({"speak_final_only": False}) is True

    def test_other_voice_keys_keep_streaming(self):
        assert should_stream_tts({"silence_duration": 1.5,
                                  "barge_in": True}) is True

    def test_loads_from_active_config_when_omitted(self):
        with patch("hermes_cli.config.load_config",
                   return_value={"voice": {"speak_final_only": True}}):
            assert should_stream_tts() is False

    def test_defaults_when_voice_block_missing(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert should_stream_tts() is True

    def test_config_error_falls_back_to_streaming(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError):
            assert should_stream_tts() is True

    def test_malformed_voice_block_falls_back_to_streaming(self):
        # A hand-edited ``voice: true`` leaves load_config()['voice'] as a
        # non-dict; coerce to {} instead of crashing.
        with patch("hermes_cli.config.load_config",
                   return_value={"voice": True}):
            assert should_stream_tts() is True
