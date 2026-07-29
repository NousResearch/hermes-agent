"""Tests for the voice-chat stop phrase (say "stop" and nothing else to end).

Contract:
  - `is_voice_stop_phrase` matches ONLY when the whole utterance equals a
    configured phrase (case-insensitive, surrounding punctuation stripped).
  - Default phrase list is ("stop",); `voice.stop_phrases` in config.yaml
    customizes it; `[]` disables the feature.
  - In the shared continuous loop, a stop phrase halts the loop (like the
    silent-cycle limit) and is NEVER delivered to the agent.
"""

from unittest.mock import patch

import pytest

from tools.voice_mode import (
    DEFAULT_VOICE_STOP_PHRASES,
    _load_voice_stop_phrases,
    is_voice_stop_phrase,
    voice_stop_hint,
)


class TestVoiceStopHint:
    """The 'Say "stop" to end the voice chat.' hint shown on voice-mode start."""

    def test_default_phrase(self):
        with patch("tools.voice_mode._load_voice_stop_phrases", return_value=("stop",)):
            assert voice_stop_hint() == 'Say "stop" to end the voice chat.'

    def test_custom_phrase_uses_first_entry(self):
        with patch(
            "tools.voice_mode._load_voice_stop_phrases",
            return_value=("goodbye hermes", "stop"),
        ):
            assert voice_stop_hint() == 'Say "goodbye hermes" to end the voice chat.'

    def test_disabled_phrases_show_no_hint(self):
        with patch("tools.voice_mode._load_voice_stop_phrases", return_value=()):
            assert voice_stop_hint() == ""


class TestIsVoiceStopPhrase:
    @pytest.mark.parametrize("utterance", [
        "stop", "Stop", "STOP", "stop.", "Stop!", " stop ", '"Stop."', "stop?",
    ])
    def test_bare_stop_matches(self, utterance):
        assert is_voice_stop_phrase(utterance, ("stop",)) is True

    @pytest.mark.parametrize("utterance", [
        "stop doing that",
        "please stop",
        "stop the build and rerun tests",
        "don't stop",
        "stopwatch",
        "",
        "   ",
        "ok",
    ])
    def test_longer_utterances_pass_through(self, utterance):
        assert is_voice_stop_phrase(utterance, ("stop",)) is False

    def test_custom_phrases(self):
        phrases = ("stop", "goodbye hermes")
        assert is_voice_stop_phrase("Goodbye Hermes!", phrases) is True
        assert is_voice_stop_phrase("goodbye hermes, one more thing", phrases) is False

    def test_empty_phrase_list_disables(self):
        assert is_voice_stop_phrase("stop", ()) is False

    def test_uses_config_when_phrases_omitted(self):
        with patch("tools.voice_mode._load_voice_stop_phrases", return_value=("halt",)):
            assert is_voice_stop_phrase("halt") is True
            assert is_voice_stop_phrase("stop") is False


class TestLoadVoiceStopPhrases:
    def _with_cfg(self, voice_cfg):
        return patch(
            "hermes_cli.config.load_config",
            return_value={"voice": voice_cfg},
        )

    def test_default(self):
        with self._with_cfg({}):
            assert _load_voice_stop_phrases() == DEFAULT_VOICE_STOP_PHRASES

    def test_custom_list(self):
        with self._with_cfg({"stop_phrases": ["Stop", "  Goodbye Hermes "]}):
            assert _load_voice_stop_phrases() == ("stop", "goodbye hermes")

    def test_empty_list_disables(self):
        with self._with_cfg({"stop_phrases": []}):
            assert _load_voice_stop_phrases() == ()

    def test_bare_string_coerced(self):
        with self._with_cfg({"stop_phrases": "halt"}):
            assert _load_voice_stop_phrases() == ("halt",)

    def test_malformed_falls_back(self):
        with self._with_cfg({"stop_phrases": {"bad": "shape"}}):
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
             patch.object(v, "_continuous_on_transcript", delivered.append), \
             patch.object(v, "_continuous_on_status", None), \
             patch.object(v, "_continuous_on_silent_limit",
                          lambda: silent_limit_fired.append(True)), \
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
             patch.object(v, "_continuous_on_transcript", delivered.append), \
             patch.object(v, "_continuous_on_status", None), \
             patch.object(v, "_continuous_on_silent_limit",
                          lambda: silent_limit_fired.append(True)), \
             patch.object(v, "_continuous_on_stop_phrase", on_stop_phrase), \
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

    def test_normal_transcript_never_fires_stop_signal(self):
        stop_fired = []
        delivered, silent_limit, _ = self._run_silence_cycle(
            "stop the build and rerun", stop_fired.append
        )
        assert stop_fired == []
        assert delivered == ["stop the build and rerun"]
        assert silent_limit == []

    def test_force_transcribe_stop_phrase_fires_signal(self):
        """stop_continuous(force_transcribe=True) — the auto_restart=False
        client-driven path (TUI/desktop voice.record stop) — must fire the
        stop signal instead of silently discarding the transcript, or the
        client re-arms the next capture and the conversation never ends."""
        import hermes_cli.voice as v

        delivered = []
        stop_fired = []
        threads = []

        fake_result = {"success": True, "transcript": "stop"}
        with patch.object(v, "_continuous_active", True), \
             patch.object(v, "_continuous_auto_restart", False), \
             patch.object(v, "_continuous_recorder", self._FakeRecorder()), \
             patch.object(v, "_continuous_on_transcript", delivered.append), \
             patch.object(v, "_continuous_on_status", None), \
             patch.object(v, "_continuous_on_silent_limit", None), \
             patch.object(v, "_continuous_on_stop_phrase", stop_fired.append), \
             patch.object(v, "_continuous_no_speech_count", 0), \
             patch.object(v, "transcribe_recording", return_value=fake_result), \
             patch.object(v, "_play_beep", lambda **kw: None), \
             patch.object(v.threading, "Thread",
                          side_effect=lambda target, daemon=None: threads.append(target)
                          or _ImmediateThread(target)), \
             patch.object(v.os.path, "isfile", return_value=False):
            v.stop_continuous(force_transcribe=True)

        assert stop_fired == ["stop"]
        assert delivered == []

    def test_force_transcribe_stop_phrase_falls_back_to_silent_limit(self):
        """Legacy consumers that never wired on_stop_phrase still get the
        voice-off signal via on_silent_limit."""
        import hermes_cli.voice as v

        silent_limit_fired = []

        fake_result = {"success": True, "transcript": "stop"}
        with patch.object(v, "_continuous_active", True), \
             patch.object(v, "_continuous_auto_restart", False), \
             patch.object(v, "_continuous_recorder", self._FakeRecorder()), \
             patch.object(v, "_continuous_on_transcript", lambda t: None), \
             patch.object(v, "_continuous_on_status", None), \
             patch.object(v, "_continuous_on_silent_limit",
                          lambda: silent_limit_fired.append(True)), \
             patch.object(v, "_continuous_on_stop_phrase", None), \
             patch.object(v, "_continuous_no_speech_count", 0), \
             patch.object(v, "transcribe_recording", return_value=fake_result), \
             patch.object(v, "_play_beep", lambda **kw: None), \
             patch.object(v.threading, "Thread",
                          side_effect=lambda target, daemon=None: _ImmediateThread(target)), \
             patch.object(v.os.path, "isfile", return_value=False):
            v.stop_continuous(force_transcribe=True)

        assert silent_limit_fired == [True]

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

        with patch.object(
            vm, "_load_voice_stop_phrases", return_value=tuple(phrases)
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
