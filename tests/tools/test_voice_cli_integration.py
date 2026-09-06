"""Tests for CLI voice mode integration -- markdown stripping, voice state
management, TTS/STT wiring, barge-in and the full-duplex listener."""

import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_voice_cli(**overrides):
    """Create a minimal HermesCLI with only voice-related attrs initialized.

    Uses ``__new__()`` to bypass ``__init__`` so no config/env/API setup is
    needed.  Only the voice state attributes (from __init__ lines 3749-3758)
    are populated.
    """
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._voice_lock = threading.Lock()
    cli._voice_mode = False
    cli._voice_tts = False
    cli._voice_recorder = None
    cli._voice_recording = False
    cli._voice_processing = False
    cli._voice_continuous = False
    cli._voice_tts_done = threading.Event()
    cli._voice_tts_done.set()
    cli._voice_tts_stop = None
    cli._voice_barge_capture = threading.Event()
    cli._voice_last_tts_text = ""
    cli._voice_barge_phase = None
    cli._pending_input = queue.Queue()
    cli._app = None
    cli._attached_images = []
    cli.console = SimpleNamespace(width=80)
    for k, v in overrides.items():
        setattr(cli, k, v)
    return cli


# ============================================================================
# Markdown stripping — import real function from tts_tool
# ============================================================================

from tools.tts_text_normalize import _strip_markdown_for_tts


class TestMarkdownStripping:
    def test_empty_after_stripping_returns_empty(self):
        text = "```python\nprint('hello')\n```"
        result = _strip_markdown_for_tts(text)
        assert result == ""


    def test_complex_response(self):
        text = (
            "## Answer\n\n"
            "Here's how to do it:\n\n"
            "```python\ndef hello():\n    print('hi')\n```\n\n"
            "Run it with `python main.py`. "
            "See [docs](https://example.com) for more.\n\n"
            "- Step one\n- Step two\n\n"
            "---\n\n"
            "**Good luck!**"
        )
        result = _strip_markdown_for_tts(text)
        assert "```" not in result
        assert "https://" not in result
        assert "**" not in result
        assert "---" not in result
        assert "Answer" in result
        assert "Good luck!" in result
        assert "docs" in result


# ============================================================================
# Real behavior tests — CLI voice methods via _make_voice_cli()
# ============================================================================

class TestHandleVoiceCommandReal:
    """Tests _handle_voice_command routing with real CLI instance."""

    def _cli(self):
        cli = _make_voice_cli()
        cli._enable_voice_mode = MagicMock()
        cli._disable_voice_mode = MagicMock()
        cli._toggle_voice_tts = MagicMock()
        cli._show_voice_status = MagicMock()
        return cli

    @patch("cli._cprint")
    def test_on_calls_enable(self, _cp):
        cli = self._cli()
        cli._handle_voice_command("/voice on")
        cli._enable_voice_mode.assert_called_once()


    @patch("cli._cprint")
    def test_unknown_subcommand(self, mock_cp):
        cli = self._cli()
        cli._handle_voice_command("/voice foobar")
        cli._enable_voice_mode.assert_not_called()
        cli._disable_voice_mode.assert_not_called()
        # Should print usage via _cprint
        assert any("Unknown" in str(c) or "unknown" in str(c)
                    for c in mock_cp.call_args_list)


class TestEnableVoiceModeReal:
    """Tests _enable_voice_mode with real CLI instance."""

    @patch("cli._cprint")
    @patch("hermes_cli.config.load_config", return_value={"voice": {}})
    @patch("tools.voice_mode.check_voice_requirements",
           return_value={"available": True, "details": "OK"})
    @patch("tools.voice_mode.detect_audio_environment",
           return_value={"available": True, "warnings": []})
    def test_success_sets_voice_mode(self, _env, _req, _cfg, _cp):
        cli = _make_voice_cli()
        cli._enable_voice_mode()
        assert cli._voice_mode is True


    @patch("cli._cprint")
    @patch("hermes_cli.config.load_config", side_effect=Exception("broken config"))
    @patch("tools.voice_mode.check_voice_requirements",
           return_value={"available": True, "details": "OK"})
    @patch("tools.voice_mode.detect_audio_environment",
           return_value={"available": True, "warnings": []})
    def test_config_exception_still_enables(self, _env, _req, _cfg, _cp):
        cli = _make_voice_cli()
        cli._enable_voice_mode()
        assert cli._voice_mode is True


class TestVoiceBeepConfigReal:
    """Tests the CLI voice beep toggle."""

    @patch("hermes_cli.config.load_config", return_value={"voice": {"beep_enabled": False}})
    def test_beeps_can_be_disabled(self, _cfg):
        cli = _make_voice_cli()
        assert cli._voice_beeps_enabled() is False

    @patch("cli._cprint")
    @patch("cli.threading.Thread")
    @patch("tools.voice_mode.play_beep")
    @patch("tools.voice_mode.create_audio_recorder")
    @patch(
        "tools.voice_mode.check_voice_requirements",
        return_value={
            "available": True,
            "audio_available": True,
            "stt_available": True,
            "details": "OK",
            "missing_packages": [],
        },
    )
    @patch(
        "hermes_cli.config.load_config",
        return_value={
            "voice": {
                "beep_enabled": False,
                "silence_threshold": 200,
                "silence_duration": 3.0,
            }
        },
    )
    def test_start_recording_skips_beep_when_disabled(
        self, _cfg, _req, mock_create, mock_beep, mock_thread, _cp
    ):
        recorder = MagicMock()
        recorder.supports_silence_autostop = True
        mock_create.return_value = recorder
        mock_thread.return_value = MagicMock(start=MagicMock())

        cli = _make_voice_cli()
        cli._voice_start_recording()

        recorder.start.assert_called_once()
        mock_beep.assert_not_called()


class TestMaxRecordingSecondsConfigReal:
    """voice.max_recording_seconds must reach the recorder from config.

    Regression for the dead-config fix: the predicate alone can stay green
    while the CLI wiring regresses, so pin the actual assignment made by
    ``_voice_start_recording`` for the valid / disabled / corrupted cases.
    """

    def _start_with_voice_cfg(self, voice_cfg):
        with patch("cli._cprint"), \
             patch("cli.threading.Thread", return_value=MagicMock(start=MagicMock())), \
             patch("tools.voice_mode.play_beep"), \
             patch("tools.voice_mode.create_audio_recorder") as mock_create, \
             patch(
                 "tools.voice_mode.check_voice_requirements",
                 return_value={
                     "available": True,
                     "audio_available": True,
                     "stt_available": True,
                     "details": "OK",
                     "missing_packages": [],
                 },
             ), \
             patch("hermes_cli.config.load_config", return_value={"voice": voice_cfg}):
            recorder = MagicMock()
            recorder.supports_silence_autostop = True
            mock_create.return_value = recorder

            cli = _make_voice_cli()
            cli._voice_start_recording()

        return recorder

    def test_configured_cap_reaches_recorder(self):
        recorder = self._start_with_voice_cfg({"max_recording_seconds": 45})
        assert recorder._max_recording_seconds == 45


    def test_bool_falls_back_to_documented_default(self):
        # bool is a subclass of int — ``max_recording_seconds: true`` must not
        # become a 1-second cap; it falls back to the documented 120 default,
        # mirroring the silence-param corruption handling.
        recorder = self._start_with_voice_cfg({"max_recording_seconds": True})
        assert recorder._max_recording_seconds == 120.0

class TestDisableVoiceModeReal:
    """Tests _disable_voice_mode with real CLI instance."""

    @patch("cli._cprint")
    @patch("tools.voice_mode.stop_playback")
    def test_all_flags_reset(self, _sp, _cp):
        cli = _make_voice_cli(_voice_mode=True, _voice_tts=True,
                              _voice_continuous=True)
        cli._disable_voice_mode()
        assert cli._voice_mode is False
        assert cli._voice_tts is False
        assert cli._voice_continuous is False


    @patch("cli._cprint")
    @patch("tools.voice_mode.stop_playback", side_effect=RuntimeError("boom"))
    def test_stop_playback_exception_swallowed(self, _sp, _cp):
        cli = _make_voice_cli(_voice_mode=True)
        cli._disable_voice_mode()
        assert cli._voice_mode is False


class TestVoiceSpeakResponseReal:
    """Tests _voice_speak_response with real CLI instance."""

    def test_async_scheduling_clears_done_before_thread_start(self):
        cli = _make_voice_cli(_voice_tts=True)
        starts = []

        class FakeThread:
            def __init__(self, target=None, args=(), daemon=None):
                self.target = target
                self.args = args
                self.daemon = daemon

            def start(self):
                starts.append(cli._voice_tts_done.is_set())

        with patch("cli.threading.Thread", FakeThread):
            cli._voice_speak_response_async("Hello")

        assert starts == [False]
        assert not cli._voice_tts_done.is_set()

    @patch("cli._cprint")
    def test_early_return_when_tts_off(self, _cp):
        cli = _make_voice_cli(_voice_tts=False)
        with patch("tools.tts_tool.text_to_speech_tool") as mock_tts:
            cli._voice_speak_response("Hello")
            mock_tts.assert_not_called()


    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.getsize", return_value=1000)
    @patch("cli.os.path.isfile", return_value=True)
    @patch("cli.os.makedirs")
    @patch("tools.voice_mode.play_audio_file")
    @patch("tools.tts_tool.text_to_speech_tool")
    def test_play_audio_uses_returned_file_paths(
        self, mock_tts, mock_play, _mkd, _isf, _gsz, _unl, _cp
    ):
        def fake_tts(**kwargs):
            mp3_path = kwargs["output_path"]
            ogg_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
            # The tool result is authoritative — file_paths drives playback
            return json.dumps({
                "success": True,
                "file_path": ogg_path,
                "file_paths": [ogg_path],
            })

        mock_tts.side_effect = fake_tts

        cli = _make_voice_cli(_voice_tts=True)
        cli._voice_speak_response("Hello world")

        # Should play the returned OGG path, not the requested MP3 path
        mock_play.assert_called_once_with(
            mock_tts.call_args.kwargs["output_path"].rsplit(".", 1)[0] + ".ogg"
        )


class TestVoiceStopAndTranscribeReal:
    """Tests _voice_stop_and_transcribe with real CLI instance."""

    @patch("cli._cprint")
    def test_guard_not_recording(self, _cp):
        cli = _make_voice_cli(_voice_recording=False)
        with patch("tools.voice_mode.transcribe_recording") as mock_tr:
            cli._voice_stop_and_transcribe()
            mock_tr.assert_not_called()

    @patch("cli._cprint")
    @patch("tools.voice_mode.play_beep")
    def test_no_speech_detected(self, _beep, _cp):
        recorder = MagicMock()
        recorder.stop.return_value = None
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()
        assert cli._pending_input.empty()

    @patch("cli._cprint")
    @patch("cli.os.unlink")
    @patch("cli.os.path.isfile", return_value=True)
    @patch("hermes_cli.config.load_config", return_value={"stt": {}})
    @patch("tools.voice_mode.transcribe_recording",
           return_value={"success": True, "transcript": "hello world"})
    @patch("tools.voice_mode.play_beep")
    def test_successful_transcription_queues_input(
        self, _beep, _tr, _cfg, _isf, _unl, _cp
    ):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/test.wav"
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)
        cli._voice_stop_and_transcribe()
        queued = cli._pending_input.get_nowait()
        # Voice transcripts are wrapped in the _VoiceInputMessage sentinel so
        # only genuine STT output gets the voice prefix (#65827).
        from cli import _VoiceInputMessage
        assert isinstance(queued, _VoiceInputMessage)
        assert str(queued) == "hello world"


    def test_non_local_stt_keeps_generic_transcribing_status(self):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/test.wav"
        cli = _make_voice_cli(_voice_recording=True, _voice_recorder=recorder)

        with patch("cli._cprint") as mock_print, \
             patch("cli.os.path.isfile", return_value=False), \
             patch(
                 "hermes_cli.config.load_config",
                 return_value={"stt": {"provider": "openai", "model": "whisper-1"}},
             ), \
             patch("tools.voice_mode.transcribe_recording",
                   return_value={"success": True, "transcript": "hello"}) as mock_transcribe, \
             patch("tools.voice_mode.play_beep"):
            cli._voice_stop_and_transcribe()

        messages = [call.args[0] for call in mock_print.call_args_list]
        assert any("Transcribing..." in message for message in messages)
        assert all("Hugging Face" not in message for message in messages)
        mock_transcribe.assert_called_once_with("/tmp/test.wav", model="whisper-1")


# ---------------------------------------------------------------------------
# Barge-in capture — the interruption is transcribed and queued directly
# ---------------------------------------------------------------------------


class TestVoiceBargeCaptureSubmit:
    """_voice_submit_barge_utterance: the barge monitor's captured WAV becomes
    the next turn without a re-record round trip."""

    def test_transcript_is_queued_and_wav_removed(self, tmp_path, monkeypatch):
        cli = _make_voice_cli()
        cli._voice_barge_capture.set()
        wav = tmp_path / "barge.wav"
        wav.write_bytes(b"RIFF")

        monkeypatch.setattr(
            "tools.voice_mode.transcribe_recording",
            lambda path, model=None: {"success": True, "transcript": "stop, do it differently"},
        )

        cli._voice_submit_barge_utterance(str(wav))

        queued = cli._pending_input.get_nowait()
        from cli import _VoiceInputMessage
        assert isinstance(queued, _VoiceInputMessage)
        assert str(queued) == "stop, do it differently"
        assert not cli._voice_barge_capture.is_set()
        assert not wav.exists()

    def test_no_speech_hands_mic_back_without_queueing(self, tmp_path, monkeypatch):
        cli = _make_voice_cli(_voice_mode=True, _voice_continuous=True)
        cli._voice_barge_capture.set()
        wav = tmp_path / "barge.wav"
        wav.write_bytes(b"RIFF")
        restarted = threading.Event()
        cli._voice_start_recording = lambda: restarted.set()

        monkeypatch.setattr(
            "tools.voice_mode.transcribe_recording",
            lambda path, model=None: {"success": True, "transcript": "", "no_speech": True},
        )

        cli._voice_submit_barge_utterance(str(wav))

        assert cli._pending_input.empty()
        assert not cli._voice_barge_capture.is_set()
        assert restarted.wait(2.0)  # continuous mode resumes listening

    def test_playback_phase_echo_of_own_tts_is_dropped(self, tmp_path, monkeypatch):
        """#75780: a playback-phase capture that closely matches the TTS
        text Hermes just spoke is speaker bleed, not real user speech --
        it must be dropped instead of queued as the next turn, and the mic
        handed back so continuous mode keeps listening."""
        cli = _make_voice_cli(_voice_mode=True, _voice_continuous=True)
        cli._voice_barge_capture.set()
        cli._voice_barge_phase = "playback"
        cli._voice_last_tts_text = "네, 방금도 제 답변이 그대로 다시 입력됐어요."
        wav = tmp_path / "barge.wav"
        wav.write_bytes(b"RIFF")
        restarted = threading.Event()
        cli._voice_start_recording = lambda: restarted.set()

        monkeypatch.setattr(
            "tools.voice_mode.transcribe_recording",
            lambda path, model=None: {
                "success": True,
                "transcript": "네 방금 네 방금도 제 답변이 그대로 다시 입력됐어요.",
            },
        )

        cli._voice_submit_barge_utterance(str(wav))

        assert cli._pending_input.empty()  # not queued as a user turn
        assert not cli._voice_barge_capture.is_set()
        assert restarted.wait(2.0)  # mic handed back instead of self-triggering another turn

    def test_playback_phase_genuine_interjection_is_still_queued(self, tmp_path, monkeypatch):
        """A real user interjection during playback -- unrelated to the TTS
        text -- must still reach the agent."""
        cli = _make_voice_cli()
        cli._voice_barge_capture.set()
        cli._voice_barge_phase = "playback"
        cli._voice_last_tts_text = "The weather today is sunny with a light breeze."
        wav = tmp_path / "barge.wav"
        wav.write_bytes(b"RIFF")

        monkeypatch.setattr(
            "tools.voice_mode.transcribe_recording",
            lambda path, model=None: {
                "success": True,
                "transcript": "actually can you check my calendar for tomorrow",
            },
        )

        cli._voice_submit_barge_utterance(str(wav))

        queued = cli._pending_input.get_nowait()
        from cli import _VoiceInputMessage
        assert str(queued) == "actually can you check my calendar for tomorrow"

    def test_generation_phase_transcript_not_echo_checked(self, tmp_path, monkeypatch):
        """Generation-phase barges (no TTS playing) are never treated as
        echo, even if the transcript happens to match old TTS text."""
        cli = _make_voice_cli()
        cli._voice_barge_capture.set()
        cli._voice_barge_phase = "generation"
        cli._voice_last_tts_text = "stop, do it differently"
        wav = tmp_path / "barge.wav"
        wav.write_bytes(b"RIFF")

        monkeypatch.setattr(
            "tools.voice_mode.transcribe_recording",
            lambda path, model=None: {"success": True, "transcript": "stop, do it differently"},
        )

        cli._voice_submit_barge_utterance(str(wav))

        queued = cli._pending_input.get_nowait()
        from cli import _VoiceInputMessage
        assert str(queued) == "stop, do it differently"


# ============================================================================
# Full-duplex agent-turn listener — CLI phase behaviour
# ============================================================================


class TestVoiceFullDuplexListener:
    """_voice_full_duplex_listener: one mic for the whole turn. Generation-
    phase speech interrupts the in-flight agent turn; playback-phase speech
    cuts TTS; the capture is submitted either way."""

    def _cli(self, monkeypatch, *, listen, voice_cfg=None, **overrides):
        cli = _make_voice_cli(
            _voice_mode=True, _voice_continuous=True, **overrides
        )
        cli.agent = None
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"voice": dict(voice_cfg or {"barge_in": True})},
        )
        monkeypatch.setattr("tools.voice_mode.full_duplex_listen", listen)
        monkeypatch.setattr("tools.voice_mode.is_audio_output_active", lambda: False)
        monkeypatch.setattr("tools.voice_mode.stop_playback", lambda: None)
        return cli

    def test_generation_trip_interrupts_agent_and_submits(self, monkeypatch, tmp_path):
        """Speech during generation → agent.interrupt() (the same seam the
        typed interrupt uses) + pending TTS pipeline cut + capture queued."""
        wav = tmp_path / "fd.wav"
        wav.write_bytes(b"RIFF")

        def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
            on_trigger("generation")
            return str(wav)

        cli = self._cli(monkeypatch, listen=fake_listen, _agent_running=True)
        interrupted = threading.Event()
        cli.agent = SimpleNamespace(interrupt=lambda: interrupted.set())
        pipe_stop = threading.Event()
        cli._voice_tts_stop = pipe_stop
        monkeypatch.setattr(
            "tools.voice_mode.transcribe_recording",
            lambda path, model=None: {"success": True, "transcript": "actually wait"},
        )

        cli._voice_full_duplex_listener()

        assert interrupted.is_set()
        assert pipe_stop.is_set()  # stale reply's TTS can never play
        from cli import _VoiceInputMessage
        queued = cli._pending_input.get_nowait()
        assert isinstance(queued, _VoiceInputMessage)
        assert str(queued) == "actually wait"
        assert not cli._voice_barge_capture.is_set()


    def test_listener_arms_at_submit_and_survives_into_playback(self, monkeypatch):
        """Lifecycle: should_stop is False during generation AND during
        pending TTS (survives the phase transition — no re-arm race), and
        True once the turn is fully done."""
        probes = {}

        def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
            # generation: agent running, TTS not started
            probes["generation"] = should_stop()
            # transition: agent done, TTS still pending
            cli._agent_running = False
            cli._voice_tts_done.clear()
            probes["playback_pending"] = should_stop()
            # turn fully done
            cli._voice_tts_done.set()
            probes["done"] = should_stop()
            return None

        cli = self._cli(monkeypatch, listen=fake_listen, _agent_running=True)
        cli._voice_tts_done.set()

        cli._voice_full_duplex_listener()

        assert probes["generation"] is False
        assert probes["playback_pending"] is False  # same listener spans phases
        assert probes["done"] is True


    def test_stop_phrase_mid_generation_interrupts_and_ends_chat(self, monkeypatch, tmp_path):
        """Bare 'stop' during generation = stop everything: the turn is
        interrupted at trip time AND the voice chat is disabled."""
        wav = tmp_path / "fd.wav"
        wav.write_bytes(b"RIFF")

        def fake_listen(should_stop, is_playing=None, on_trigger=None, **_kw):
            on_trigger("generation")
            return str(wav)

        cli = self._cli(monkeypatch, listen=fake_listen, _agent_running=True)
        interrupted = threading.Event()
        cli.agent = SimpleNamespace(interrupt=lambda: interrupted.set())
        disabled = []
        cli._disable_voice_mode = lambda: disabled.append(True)
        monkeypatch.setattr(
            "tools.voice_mode.transcribe_recording",
            lambda path, model=None: {"success": True, "transcript": "stop"},
        )
        monkeypatch.setattr(
            "tools.voice_mode.is_voice_stop_phrase",
            lambda text: text.strip().lower() == "stop",
        )

        cli._voice_full_duplex_listener()

        assert interrupted.is_set()   # turn interrupted at trip
        assert disabled == [True]     # chat ended by the stop phrase
        assert cli._pending_input.empty()  # stop phrase never reaches the agent


# ============================================================================
# Typed stop phrase — typing "stop" during a voice chat ends it
# ============================================================================
class TestTypedVoiceStop:
    """_typed_voice_stop: a TYPED bare stop phrase during an active voice chat
    ends the chat (same as saying "stop"); outside voice mode it passes
    through to the agent untouched."""

    def _cli(self, **overrides):
        cli = _make_voice_cli(**overrides)
        cli._disable_calls = []
        cli._disable_voice_mode = lambda: cli._disable_calls.append(True)
        return cli

    @pytest.fixture(autouse=True)
    def _pin_stop_phrases(self, monkeypatch):
        # Hermetic: don't let a dev machine's voice.stop_phrases config
        # change which utterances count as a stop phrase.
        monkeypatch.setattr(
            "tools.voice_mode_transcript._load_voice_stop_phrases", lambda: ("stop",)
        )

    def test_typed_stop_ends_voice_chat_when_voice_on(self):
        cli = self._cli(_voice_mode=True)
        assert cli._typed_voice_stop("stop") is True
        assert cli._disable_calls == [True]


    def test_longer_typed_message_passes_through_in_voice_mode(self):
        cli = self._cli(_voice_mode=True)
        assert cli._typed_voice_stop("stop the docker container") is False
        assert cli._disable_calls == []


# ============================================================================
# Fallback (whole-file) TTS path arms the full-duplex listener
# ============================================================================

class TestFallbackSpeakArmsBargeMonitor:
    """_voice_speak_response_async must arm _voice_full_duplex_listener in
    continuous voice mode. This is the safety net for speak calls outside a
    chat turn — the primary arm happens at utterance-submit in chat()."""

    def _cli(self, **overrides):
        cli = _make_voice_cli(**overrides)
        cli._monitor_calls = []
        cli._monitor_armed = threading.Event()

        def _armed():
            cli._monitor_calls.append(True)
            cli._monitor_armed.set()

        cli._voice_full_duplex_listener = _armed
        cli._voice_speak_response = lambda text: None
        return cli

    def test_monitor_armed_in_continuous_voice_mode(self):
        cli = self._cli(_voice_mode=True, _voice_tts=True, _voice_continuous=True)
        cli._voice_speak_response_async("a reply")
        assert cli._monitor_armed.wait(5.0), "listener was never armed"
        assert len(cli._monitor_calls) == 1

    def test_no_monitor_outside_continuous_mode(self):
        cli = self._cli(_voice_mode=True, _voice_tts=True, _voice_continuous=False)
        cli._voice_speak_response_async("a reply")
        # Nothing to wait for — a short negative window is enough to prove the
        # speak thread came and went without arming the mic.
        assert not cli._monitor_armed.wait(0.05)
        assert cli._monitor_calls == []

