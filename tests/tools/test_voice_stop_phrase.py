"""Tests for the voice-chat stop phrase (say "stop" and nothing else to end).

Contract:
  - `is_voice_stop_phrase` matches ONLY when the whole utterance equals a
    configured phrase (case-insensitive, surrounding punctuation stripped).
  - Default phrase list is ("stop", "cancel"); `voice.stop_phrases` in config.yaml
    customizes it; `[]` disables the feature.
  - In the shared continuous loop, a stop phrase halts the loop (like the
    silent-cycle limit) and is NEVER delivered to the agent.
"""

from unittest.mock import patch

import pytest

from tools.voice_mode_transcript import (
    DEFAULT_VOICE_END_PHRASES,
    DEFAULT_VOICE_STOP_PHRASES,
    _load_voice_stop_phrases,
    is_voice_end_phrase,
    is_voice_stop_phrase,
    strip_voice_end_phrase,
    voice_stop_hint,
)


class TestVoiceStopHint:
    """The 'Say "stop" to end the voice chat.' hint shown on voice-mode start."""

    def test_default_phrase(self):
        with patch("tools.voice_mode_transcript._load_voice_stop_phrases", return_value=("stop",)):
            assert voice_stop_hint() == 'Say "stop" to end the voice chat.'


    def test_disabled_phrases_show_no_hint(self):
        with patch("tools.voice_mode_transcript._load_voice_stop_phrases", return_value=()):
            assert voice_stop_hint() == ""


class TestIsVoiceStopPhrase:
    @pytest.mark.parametrize("utterance", [
        "stop", "Stop", "STOP", "stop.", "Stop!", " stop ", '"Stop."', "stop?",
    ])
    def test_bare_stop_matches(self, utterance):
        assert is_voice_stop_phrase(utterance, ("stop",)) is True


    def test_uses_config_when_phrases_omitted(self):
        with patch("tools.voice_mode_transcript._load_voice_stop_phrases", return_value=("halt",)):
            assert is_voice_stop_phrase("halt") is True
            assert is_voice_stop_phrase("stop") is False


class TestCancelIsADefaultStopPhrase:
    """"cancel" ends the exchange by default, alongside "stop"."""

    def test_default_list_is_stop_and_cancel(self):
        assert DEFAULT_VOICE_STOP_PHRASES == ("stop", "cancel")

    @pytest.mark.parametrize("utterance", [
        "cancel", "Cancel", "CANCEL", "cancel.", "Cancel!", " cancel ", '"Cancel."', "cancel?",
    ])
    def test_cancel_matches(self, utterance):
        assert is_voice_stop_phrase(utterance, DEFAULT_VOICE_STOP_PHRASES) is True

    def test_stop_still_matches_under_the_default(self):
        assert is_voice_stop_phrase("stop", DEFAULT_VOICE_STOP_PHRASES) is True

    def test_cancel_inside_a_sentence_is_not_a_stop(self):
        # Whole-utterance match only: a real request that merely contains "cancel" reaches the agent.
        assert is_voice_stop_phrase("cancel my 3pm meeting", DEFAULT_VOICE_STOP_PHRASES) is False


class TestIsVoiceEndPhrase:
    """The assistant's sign-off: the model's reply ENDS with an end phrase to close the session."""

    def test_default_is_over_and_out(self):
        assert DEFAULT_VOICE_END_PHRASES == ("over and out",)

    @pytest.mark.parametrize("reply", [
        "Over and out.", "over and out", "OVER AND OUT!",
        "Your lights are on. Over and out.", "Done — over and out.",
    ])
    def test_reply_ending_with_signoff_matches(self, reply):
        assert is_voice_end_phrase(reply, DEFAULT_VOICE_END_PHRASES) is True

    @pytest.mark.parametrize("reply", [
        "It's 72 degrees out.", "Over and out is a phrase pilots use.",  # only a TRAILING match ends
        "", "   ",
    ])
    def test_non_signoff_reply_does_not_match(self, reply):
        assert is_voice_end_phrase(reply, DEFAULT_VOICE_END_PHRASES) is False

    def test_disabled_when_no_end_phrases(self):
        assert is_voice_end_phrase("Over and out.", ()) is False


class TestStripVoiceEndPhrase:
    """The sign-off is a control marker, not speech: it's stripped from the TTS text."""

    @pytest.mark.parametrize("reply,spoken", [
        ("Give me a shout. Over and out.", "Give me a shout."),
        ("All done — over and out!", "All done"),
        ("Over and out.", ""),
        ("Talk soon, over and out", "Talk soon"),
    ])
    def test_trailing_signoff_is_removed(self, reply, spoken):
        assert strip_voice_end_phrase(reply, DEFAULT_VOICE_END_PHRASES) == spoken

    def test_non_trailing_mention_is_untouched(self):
        # "over and out" only in the middle → left alone (only a TRAILING sign-off is a marker).
        text = "Over and out is radio slang for goodbye."
        assert strip_voice_end_phrase(text, DEFAULT_VOICE_END_PHRASES) == text

    def test_plain_reply_untouched(self):
        assert strip_voice_end_phrase("It's sunny and 72.", DEFAULT_VOICE_END_PHRASES) == "It's sunny and 72."


class TestLoadVoiceStopPhrases:
    def _with_cfg(self, voice_cfg):
        return patch(
            "hermes_cli.config.load_config",
            return_value={"voice": voice_cfg},
        )

    def test_default(self):
        with self._with_cfg({}):
            assert _load_voice_stop_phrases() == DEFAULT_VOICE_STOP_PHRASES


    def test_config_error_falls_back(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError):
            assert _load_voice_stop_phrases() == DEFAULT_VOICE_STOP_PHRASES


class TestContinuousLoopStopPhrase:
    """The shared continuous-loop silence callback ends the loop on a stop
    phrase and never forwards it to on_transcript."""

    def _run_silence_cycle(self, transcript_text):
        import hermes_cli.voice as v

        delivered = []
        silent_limit_fired = []

        class _FakeRecorder:
            _peak_rms = 500

            def stop(self):
                return "/tmp/fake.wav"

            def cancel(self):
                pass

            def start(self, on_silence_stop=None):
                pass

        fake_result = {"success": True, "transcript": transcript_text}
        with patch.object(v, "_continuous_active", True), \
             patch.object(v, "_continuous_recorder", _FakeRecorder()), \
             patch.object(v, "_continuous_callbacks",
                          (delivered.append, None, lambda: silent_limit_fired.append(True), None)), \
             patch.object(v, "_continuous_no_speech_count", 0), \
             patch.object(v, "transcribe_recording", return_value=fake_result), \
             patch.object(v, "_play_beep", lambda **kw: None), \
             patch.object(v.os.path, "isfile", return_value=False):
            v._continuous_on_silence()
            still_active = v._continuous_active
        return delivered, silent_limit_fired, still_active

    def test_stop_phrase_halts_loop_and_is_not_delivered(self):
        delivered, silent_limit, still_active = self._run_silence_cycle("Stop.")
        assert delivered == []
        assert silent_limit == [True]
        assert still_active is False

    def test_normal_transcript_is_delivered(self):
        delivered, silent_limit, _ = self._run_silence_cycle("stop the build and rerun")
        assert delivered == ["stop the build and rerun"]
        assert silent_limit == []


class TestContinuousLoopStopPhraseSignal:
    """The explicit on_stop_phrase signal: fired on a spoken stop phrase so
    consumers (TUI, desktop) end the conversation as user intent, with
    on_silent_limit as the legacy fallback when it isn't wired."""

    class _FakeRecorder:
        _peak_rms = 500

        def stop(self):
            return "/tmp/fake.wav"

        def cancel(self):
            pass

        def start(self, on_silence_stop=None):
            pass

    def _run_silence_cycle(self, transcript_text, on_stop_phrase):
        import hermes_cli.voice as v

        delivered = []
        silent_limit_fired = []

        fake_result = {"success": True, "transcript": transcript_text}
        with patch.object(v, "_continuous_active", True), \
             patch.object(v, "_continuous_recorder", self._FakeRecorder()), \
             patch.object(v, "_continuous_callbacks",
                          (delivered.append, None, lambda: silent_limit_fired.append(True), on_stop_phrase)), \
             patch.object(v, "_continuous_no_speech_count", 0), \
             patch.object(v, "transcribe_recording", return_value=fake_result), \
             patch.object(v, "_play_beep", lambda **kw: None), \
             patch.object(v.os.path, "isfile", return_value=False):
            v._continuous_on_silence()
            still_active = v._continuous_active
        return delivered, silent_limit_fired, still_active

    def test_stop_phrase_fires_dedicated_signal_not_silent_limit(self):
        stop_fired = []
        delivered, silent_limit, still_active = self._run_silence_cycle(
            "Stop.", stop_fired.append
        )
        assert stop_fired == ["Stop."]
        assert silent_limit == []
        assert delivered == []
        assert still_active is False


    def test_start_continuous_accepts_on_stop_phrase_kwarg(self):
        import inspect

        import hermes_cli.voice as v

        assert "on_stop_phrase" in inspect.signature(v.start_continuous).parameters


class _ImmediateThread:
    """Thread stand-in that runs the target synchronously on start()."""

    def __init__(self, target):
        self._target = target

    def start(self):
        self._target()


class TestStopPhraseSurvivesHallucinationFilter:
    """Ordering contract: a configured stop phrase must never be eaten by the
    Whisper hallucination filter inside transcribe_recording. "bye" is BOTH a
    known hallucination and a plausible stop phrase — when configured as a
    stop phrase it must come through so the stop check can end the chat."""

    def _transcribe(self, text, phrases):
        import tools.voice_mode as vm
        import tools.voice_mode_transcript as vmt

        with patch.object(
            vmt, "_load_voice_stop_phrases", return_value=tuple(phrases)
        ), patch(
            "tools.transcription_tools.transcribe_audio",
            return_value={"success": True, "transcript": text},
        ):
            return vm.transcribe_recording("/tmp/fake.wav")

    def test_configured_stop_phrase_survives_blocklist(self):
        result = self._transcribe("Bye.", ["stop", "bye"])
        assert result["success"] is True
        assert result["transcript"] == "Bye."
        assert not result.get("filtered")

    def test_unconfigured_hallucination_still_filtered(self):
        result = self._transcribe("Bye.", ["stop"])
        assert result["success"] is True
        assert result["transcript"] == ""
        assert result.get("filtered") is True

    def test_default_stop_survives_repeat_regex_adjacent_phrases(self):
        # "stop." is not in the blocklist/repeat regex today; this pins the
        # contract so a future blocklist addition can't swallow it.
        result = self._transcribe("Stop.", ["stop"])
        assert result["transcript"] == "Stop."
        assert not result.get("filtered")
