"""Tests for CLI voice mode integration -- markdown stripping, voice state
management, TTS/STT wiring, barge-in and the full-duplex listener."""

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
    cli._voice_stt_stream = None
    cli._voice_recording = False
    cli._voice_recording_generation = 0
    cli._voice_capture_started_generation = 0
    cli._voice_processing = False
    cli._voice_continuous = False
    cli._voice_tts_done = threading.Event()
    cli._voice_tts_done.set()
    cli._voice_tts_stop = None
    cli._voice_barge_capture = threading.Event()
    cli._voice_stream_barge_release = threading.Event()
    cli._voice_stream_audio_sink = None
    cli._voice_stream_pending_finals = []
    cli._voice_stream_final_sequence = 0
    cli._voice_stream_drain_after = None
    cli._voice_stream_barge_candidate_after = None
    cli._voice_stream_seen_final_ids = set()
    cli._voice_stream_endpoint_timer = None
    cli._voice_stream_endpoint_token = 0
    cli._voice_stream_endpoint_segment_id = None
    cli._voice_stream_last_submitted_segment_id = None
    cli._voice_stream_partial_barge_segment_id = None
    cli._voice_stream_manual_commit = False
    cli._voice_stream_dispatch_pending = False
    cli._voice_stream_dispatch_barge = False
    cli._pending_input = queue.Queue()
    cli._app = None
    cli._attached_images = []
    cli.console = SimpleNamespace(width=80)
    for k, v in overrides.items():
        setattr(cli, k, v)
    return cli


def _expire_streaming_endpoint(cli):
    timer = cli._voice_stream_endpoint_timer
    assert timer is not None
    timer.cancel()
    timer.function()


# ============================================================================
# Markdown stripping — import real function from tts_tool
# ============================================================================

from tools.tts_tool import _strip_markdown_for_tts


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

    def test_start_cue_follows_input_preparation(self):
        order = []
        recorder = MagicMock()
        recorder.supports_silence_autostop = True
        recorder.prepare.side_effect = lambda: order.append("prepare")
        recorder.start.side_effect = lambda **_kwargs: (
            order.append("start") or True
        )

        cli = _make_voice_cli(_voice_recorder=recorder)
        cli._voice_get_streaming_stt = MagicMock(return_value=None)

        with (
            patch("cli._cprint"),
            patch(
                "tools.voice_mode.check_voice_requirements",
                return_value={
                    "available": True,
                    "audio_available": True,
                    "stt_available": True,
                    "details": "OK",
                    "missing_packages": [],
                },
            ),
            patch(
                "hermes_cli.config.load_config",
                return_value={"voice": {"beep_enabled": True}},
            ),
            patch(
                "tools.voice_mode.play_beep",
                side_effect=lambda **_kwargs: order.append("beep"),
            ),
        ):
            cli._voice_start_recording()

        assert order == ["prepare", "beep", "start"]


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
    def test_play_audio_prefers_requested_mp3_over_returned_ogg(
        self, mock_tts, mock_play, _mkd, _isf, _gsz, _unl, _cp
    ):
        def fake_tts(**kwargs):
            mp3_path = kwargs["output_path"]
            ogg_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
            return f'{{"success": true, "file_path": "{ogg_path}"}}'

        mock_tts.side_effect = fake_tts

        cli = _make_voice_cli(_voice_tts=True)
        cli._voice_speak_response("Hello world")

        requested_path = mock_tts.call_args.kwargs["output_path"]
        mock_play.assert_called_once_with(requested_path)


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


class TestStreamingVoiceInputReal:
    """Classic CLI streaming STT reuses the established voice lifecycle."""

    def test_streaming_start_disables_local_silence_endpointing(self):
        recorder = MagicMock()
        recorder.supports_streaming_frames = True
        recorder._max_recording_seconds = 0
        coordinator = MagicMock()
        coordinator.start_utterance.return_value = True
        cli = _make_voice_cli(
            _voice_recorder=recorder,
            _voice_stream_pending_finals=["stale TTS transcript"],
            _voice_stream_drain_after=2,
            _voice_stream_barge_candidate_after=1,
        )
        cli._voice_get_streaming_stt = MagicMock(return_value=coordinator)

        class _Timer:
            daemon = True

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        with patch("cli._cprint"), \
             patch("cli.threading.Timer", _Timer), \
             patch("cli.threading.Thread", return_value=MagicMock(start=MagicMock())), \
             patch("tools.voice_mode.play_beep"), \
             patch(
                 "tools.voice_mode.check_voice_requirements",
                 return_value={"audio_available": True, "stt_available": True},
             ), \
             patch(
                 "hermes_cli.config.load_config",
                 return_value={"voice": {"beep_enabled": False}},
             ):
            cli._voice_start_recording()

        coordinator.start_utterance.assert_called_once_with()
        start_kwargs = recorder.start.call_args.kwargs
        assert start_kwargs["on_silence_stop"] is None
        assert "frame_sink" not in start_kwargs
        recorder.add_continuous_frame_sink.assert_called_once()
        assert start_kwargs["on_no_speech"] == cli._voice_streaming_no_speech
        assert start_kwargs["on_max_duration"] == cli._voice_streaming_commit
        assert callable(start_kwargs["start_guard"])
        assert "on_frame_sink_error" not in start_kwargs
        assert cli._voice_stream_pending_finals == []
        assert cli._voice_stream_drain_after is None
        assert cli._voice_stream_barge_candidate_after is None
        assert cli._voice_capture_started_generation == 1

    def test_initial_stream_connection_failure_uses_local_endpointing(self):
        recorder = MagicMock()
        recorder.supports_streaming_frames = True
        coordinator = MagicMock()
        coordinator.start_utterance.return_value = False
        cli = _make_voice_cli(_voice_recorder=recorder)
        cli._voice_get_streaming_stt = MagicMock(return_value=coordinator)

        with patch("cli._cprint"), \
             patch("cli.threading.Thread", return_value=MagicMock(start=MagicMock())), \
             patch("tools.voice_mode.play_beep"), \
             patch(
                 "tools.voice_mode.check_voice_requirements",
                 return_value={"audio_available": True, "stt_available": True},
             ), \
             patch(
                 "hermes_cli.config.load_config",
                 return_value={"voice": {"beep_enabled": False}},
             ):
            cli._voice_start_recording()

        start_kwargs = recorder.start.call_args.kwargs
        assert callable(start_kwargs["on_silence_stop"])
        assert "frame_sink" not in start_kwargs

    def test_cancel_during_stream_handshake_never_starts_recorder(self):
        recorder = MagicMock(supports_streaming_frames=True)
        coordinator = MagicMock()
        cli = _make_voice_cli(_voice_recorder=recorder)

        def _finish_after_cancel():
            with cli._voice_lock:
                cli._voice_recording = False
            return True

        coordinator.start_utterance.side_effect = _finish_after_cancel
        cli._voice_get_streaming_stt = MagicMock(return_value=coordinator)

        with patch("cli._cprint"), patch(
            "tools.voice_mode.check_voice_requirements",
            return_value={"audio_available": True, "stt_available": True},
        ), patch(
            "hermes_cli.config.load_config",
            return_value={"voice": {"beep_enabled": False}},
        ):
            cli._voice_start_recording()

        recorder.start.assert_not_called()
        coordinator.pause.assert_called_once_with()

    def test_real_turn_controller_is_reused_across_two_cli_turns(self):
        provider = MagicMock()
        provider.name = "openai"
        recorder = MagicMock(supports_streaming_frames=True)
        cli = _make_voice_cli(_voice_recorder=recorder)

        with patch(
            "hermes_cli.config.load_config",
            return_value={"stt": {"provider": "openai"}},
        ), patch(
            "tools.stt_streaming.resolve_streaming_stt_provider",
            return_value=provider,
        ):
            first = cli._voice_get_streaming_stt()
            first._coordinator.start_utterance = MagicMock(return_value=True)
            first._coordinator.pause = MagicMock()
            assert first.start_utterance() is True
            first.pause()

            second = cli._voice_get_streaming_stt()

        assert second is first
        assert second.provider_name == "openai"

    def test_same_provider_with_changed_config_rebuilds_controller(self):
        recorder = MagicMock(supports_streaming_frames=True)
        cli = _make_voice_cli(_voice_recorder=recorder)
        first_provider = SimpleNamespace(
            name="openai",
            configuration_key=("openai", "key", "https://one.example/v1"),
        )
        second_provider = SimpleNamespace(
            name="openai",
            configuration_key=("openai", "key", "https://two.example/v1"),
        )

        with patch(
            "hermes_cli.config.load_config",
            return_value={"stt": {"provider": "openai"}},
        ), patch(
            "tools.stt_streaming.resolve_streaming_stt_provider",
            side_effect=[first_provider, second_provider],
        ):
            first = cli._voice_get_streaming_stt()
            second = cli._voice_get_streaming_stt()

        assert second is not first
        assert second.provider_configuration_key == second_provider.configuration_key

    def test_committed_final_bypasses_wav_and_batch_transcription(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        coordinator = MagicMock()
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=coordinator,
        )

        with patch("cli._cprint"), \
             patch("tools.voice_mode.play_beep"), \
             patch("tools.voice_mode.transcribe_recording") as transcribe:
            cli._voice_streaming_final(
                StreamingTranscriptEvent(
                    "hello from stream", True, "final-1", "segment-1"
                )
            )
            _expire_streaming_endpoint(cli)

        assert str(cli._pending_input.get_nowait()) == "hello from stream"
        recorder.cancel.assert_called_once()
        recorder.stop.assert_not_called()
        transcribe.assert_not_called()

    def test_late_tts_final_cannot_end_new_recording_before_speech(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        recorder.has_spoken = False
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(),
        )

        cli._voice_streaming_final(
            StreamingTranscriptEvent("late text from TTS", True, "tts-final")
        )

        assert cli._voice_recording is True
        assert cli._voice_processing is False
        assert cli._voice_stream_pending_finals == []
        assert cli._pending_input.empty()
        recorder.cancel.assert_not_called()

        recorder.has_spoken = True
        with patch("cli._cprint"), patch("tools.voice_mode.play_beep"):
            cli._voice_streaming_final(
                StreamingTranscriptEvent(
                    "actual user request",
                    True,
                    "user-final",
                    "user-segment",
                )
            )
            _expire_streaming_endpoint(cli)

        assert str(cli._pending_input.get_nowait()) == "actual user request"
        recorder.cancel.assert_called_once_with()

    def test_late_final_cannot_claim_capture_before_recorder_starts(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        recorder.has_spoken = True
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_recording=True,
            _voice_recording_generation=2,
            _voice_capture_started_generation=1,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(),
        )

        cli._voice_streaming_final(
            StreamingTranscriptEvent("late prior turn", True, "late-final")
        )

        assert cli._voice_recording is True
        assert cli._voice_processing is False
        assert cli._voice_stream_pending_finals == []
        assert cli._pending_input.empty()
        recorder.cancel.assert_not_called()

    def test_manual_commit_finishes_capture_immediately_and_only_once(self):
        recorder = MagicMock()
        coordinator = MagicMock()
        coordinator.commit.return_value = True
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=coordinator,
        )

        cli._voice_streaming_commit()
        cli._voice_streaming_commit()

        assert cli._voice_recording is False
        assert cli._voice_processing is True
        recorder.finish_capture.assert_called_once_with()
        coordinator.commit.assert_called_once_with()

    def test_manual_stop_reuses_pending_boundary_flush(self):
        recorder = MagicMock()
        coordinator = MagicMock()
        coordinator.commit.return_value = False
        coordinator.delivering = False
        coordinator.commit_pending = True
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=coordinator,
        )
        cli._voice_streaming_error = MagicMock()

        cli._voice_streaming_commit()

        assert cli._voice_recording is False
        assert cli._voice_processing is True
        assert cli._voice_stream_drain_after == 0
        recorder.finish_capture.assert_called_once_with()
        cli._voice_streaming_error.assert_not_called()

    def test_manual_commit_final_discards_retained_fallback_audio(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        coordinator = MagicMock()
        coordinator.commit.return_value = True
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=coordinator,
        )

        with patch("cli._cprint"), patch("tools.voice_mode.play_beep"):
            cli._voice_streaming_commit()
            cli._voice_streaming_final(
                StreamingTranscriptEvent("manual transcript", True)
            )

        assert str(cli._pending_input.get_nowait()) == "manual transcript"
        recorder.finish_capture.assert_called_once_with()
        recorder.cancel.assert_called_once_with()
        recorder.stop.assert_not_called()
        assert cli._voice_processing is False

    def test_manual_commit_timeout_falls_back_with_retained_audio(self):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/retained.wav"
        coordinator = MagicMock()
        coordinator.commit.return_value = True
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=coordinator,
        )

        with patch("cli._cprint"), \
             patch("cli.os.path.isfile", return_value=False), \
             patch("tools.voice_mode.play_beep"), \
             patch(
                 "hermes_cli.config.load_config",
                 return_value={"stt": {"provider": "openai"}},
             ), \
             patch(
                 "tools.voice_mode.transcribe_recording",
                 return_value={"success": True, "transcript": "recovered"},
             ) as transcribe:
            cli._voice_streaming_commit()
            cli._voice_streaming_error(
                RuntimeError("Streaming STT commit timed out")
            )

        recorder.finish_capture.assert_called_once_with()
        recorder.stop.assert_called_once_with()
        transcribe.assert_called_once_with("/tmp/retained.wav", model=None)
        assert str(cli._pending_input.get_nowait()) == "recovered"

    def test_provider_final_delivery_cannot_race_into_batch_path(self):
        recorder = MagicMock()
        coordinator = MagicMock()
        coordinator.active = False
        coordinator.delivering = True
        coordinator.commit.return_value = False
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=coordinator,
        )

        with patch("tools.voice_mode.transcribe_recording") as transcribe:
            cli._voice_stop_and_transcribe()

        recorder.finish_capture.assert_called_once_with()
        recorder.stop.assert_not_called()
        transcribe.assert_not_called()
        assert cli._voice_processing is True

    def test_empty_committed_final_finishes_as_no_speech(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(),
        )

        with patch("cli._cprint") as output, patch("tools.voice_mode.play_beep"):
            cli._voice_streaming_final(
                StreamingTranscriptEvent("", True, "empty", "empty-segment")
            )
            _expire_streaming_endpoint(cli)

        assert cli._pending_input.empty()
        assert cli._voice_processing is False
        assert any("No speech detected" in call.args[0] for call in output.call_args_list)

    def test_stream_error_uses_retained_audio_batch_fallback(self):
        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/retained.wav"
        coordinator = MagicMock()
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=coordinator,
        )

        with patch("cli._cprint"), \
             patch("cli.os.path.isfile", return_value=False), \
             patch("tools.voice_mode.play_beep"), \
             patch(
                 "hermes_cli.config.load_config",
                 return_value={"stt": {"provider": "openai"}},
             ), \
             patch(
                 "tools.voice_mode.transcribe_recording",
                 return_value={"success": True, "transcript": "recovered"},
             ) as transcribe:
            cli._voice_streaming_error(RuntimeError("socket closed"))

        transcribe.assert_called_once_with("/tmp/retained.wav", model=None)
        assert str(cli._pending_input.get_nowait()) == "recovered"

    def test_no_speech_guard_keeps_stream_and_discards_empty_capture(self):
        recorder = MagicMock()
        coordinator = MagicMock()
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=coordinator,
        )

        with patch("cli._cprint"):
            cli._voice_streaming_no_speech()

        coordinator.pause.assert_not_called()
        recorder.cancel.assert_called_once_with()
        assert cli._voice_recording is False
        assert cli._voice_processing is False

    def test_stale_and_duplicate_finals_are_ignored(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(),
        )
        event = StreamingTranscriptEvent(
            "one transcript", True, "one-final", "one-segment"
        )

        with patch("cli._cprint"), patch("tools.voice_mode.play_beep"):
            cli._voice_streaming_final(event)
            cli._voice_streaming_final(event)
            _expire_streaming_endpoint(cli)

        assert cli._pending_input.qsize() == 1
        recorder.cancel.assert_called_once()

    def test_provider_final_is_authoritative_over_local_silence_state(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = SimpleNamespace(
            has_spoken=True,
            speech_silence_seconds=0.1,
            cancel=MagicMock(),
            clear_continuous_fallback_buffer=MagicMock(),
        )
        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(active=True),
        )

        with patch("cli._cprint"), patch("tools.voice_mode.play_beep"):
            cli._voice_streaming_final(
                StreamingTranscriptEvent(
                    "provider committed transcript",
                    True,
                    "provider-final",
                    "provider-segment",
                )
            )
            _expire_streaming_endpoint(cli)

        assert cli._voice_recording is False
        assert str(cli._pending_input.get_nowait()) == "provider committed transcript"
        assert cli._pending_input.empty()
        recorder.cancel.assert_called_once_with()

    def test_partial_during_endpoint_grace_keeps_logical_turn_open(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(active=True),
        )

        with patch("cli._cprint"), patch("tools.voice_mode.play_beep"):
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "first clause", True, "final-1", "segment-1"
                )
            )
            first_timer = cli._voice_stream_endpoint_timer
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "continuing", False, "partial-2", "segment-2"
                )
            )

            assert cli._voice_stream_endpoint_timer is None
            assert cli._voice_recording is True
            assert cli._voice_stream_pending_finals == ["first clause"]
            assert cli._pending_input.empty()

            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "second clause", True, "final-2", "segment-2"
                )
            )
            _expire_streaming_endpoint(cli)

        first_timer.cancel()
        assert str(cli._pending_input.get_nowait()) == (
            "first clause second clause"
        )
        recorder.cancel.assert_called_once_with()

    def test_partial_after_endpoint_grace_confirms_and_submits_barge(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        agent = MagicMock()
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_continuous=True,
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(active=True),
            _agent_running=False,
            agent=agent,
        )

        with (
            patch("cli._cprint"),
            patch("tools.voice_mode.play_beep"),
            patch("tools.voice_mode.is_audio_output_active", return_value=False),
        ):
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "initial request", True, "final-1", "segment-1"
                )
            )
            _expire_streaming_endpoint(cli)
            assert str(cli._pending_input.get_nowait()) == "initial request"

            cli._agent_running = True
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "interrupting", False, "partial-2", "segment-2"
                )
            )
            assert cli._voice_barge_capture.is_set()
            assert cli._voice_stream_drain_after == 1
            agent.interrupt.assert_called_once_with()

            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "interrupting now", True, "final-2", "segment-2"
                )
            )
            _expire_streaming_endpoint(cli)

        assert str(cli._pending_input.get_nowait()) == "interrupting now"
        assert cli._pending_input.empty()

    def test_partial_inside_grace_confirms_barge_once_during_agent_turn(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        agent = MagicMock()
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_continuous=True,
            _voice_recording=False,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(active=True),
            _agent_running=True,
            agent=agent,
        )

        with patch("cli._cprint"):
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "earlier words", True, "final-1", "segment-1"
                )
            )
            endpoint_timer = cli._voice_stream_endpoint_timer
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "more words", False, "partial-2", "segment-2"
                )
            )
            cli._voice_handle_barge_trigger("generation")

        endpoint_timer.cancel()
        assert cli._voice_stream_endpoint_timer is None
        assert cli._voice_stream_pending_finals == ["earlier words"]
        assert cli._voice_stream_drain_after == 1
        agent.interrupt.assert_called_once_with()

    def test_streaming_partial_preserves_playback_only_barge_semantics(self):
        agent = MagicMock()
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_continuous=True,
            _voice_tts=True,
            _voice_stt_stream=MagicMock(active=True),
            _agent_running=True,
            agent=agent,
        )
        cli._voice_tts_done.clear()

        with (
            patch("cli._cprint"),
            patch("tools.voice_mode.is_audio_output_active", return_value=True),
            patch("tools.voice_mode.stop_playback") as stop_playback,
            patch("tools.tts_streaming.mark_speech_interrupted"),
        ):
            cli._voice_handle_barge_trigger()
            cli._voice_handle_barge_trigger("generation")

        stop_playback.assert_called_once_with()
        agent.interrupt.assert_not_called()

    def test_endpoint_beep_continuation_interrupts_queued_voice_turn(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        agent = MagicMock()
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_continuous=True,
            _voice_recording=True,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(active=True),
            _agent_running=False,
            agent=agent,
        )

        with (
            patch("cli._cprint"),
            patch("tools.voice_mode.play_beep"),
            patch(
                "tools.voice_mode.is_audio_output_active",
                return_value=True,
            ),
            patch("tools.voice_mode.stop_playback") as stop_playback,
        ):
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "first part", True, "final-1", "segment-1"
                )
            )
            _expire_streaming_endpoint(cli)

            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "still talking", False, "partial-2", "segment-2"
                )
            )

        assert cli._voice_stream_dispatch_barge is True
        agent.interrupt.assert_not_called()
        stop_playback.assert_not_called()

        cli._voice_apply_pending_streaming_barge(
            agent,
            voice_input=True,
        )

        agent.interrupt.assert_called_once_with()
        assert cli._voice_stream_dispatch_pending is False
        assert cli._voice_stream_dispatch_barge is False

    def test_barge_drains_prior_finals_and_the_next_final_once(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_recording=False,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(active=True),
            _agent_running=True,
        )

        with patch("cli._cprint"), \
             patch("tools.voice_mode.play_beep"), \
             patch("tools.voice_mode.transcribe_recording") as transcribe:
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "first interruption", True, "final-1", "segment-1"
                )
            )
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "first interruption", True, "final-1", "segment-1"
                )
            )
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "second thought", True, "final-2", "segment-2"
                )
            )
            assert cli._pending_input.empty()

            cli._voice_streaming_barge_confirmed()
            assert cli._voice_stream_barge_release.is_set()
            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "finish this request", True, "final-3", "segment-3"
                )
            )
            _expire_streaming_endpoint(cli)

        assert str(cli._pending_input.get_nowait()) == (
            "first interruption second thought finish this request"
        )
        assert cli._pending_input.empty()
        assert not cli._voice_barge_capture.is_set()
        recorder.cancel.assert_called_once_with()
        transcribe.assert_not_called()

    def test_confirmed_streaming_barge_releases_tts_wait(self):
        class StuckThread:
            def __init__(self):
                self.join_calls = []

            def is_alive(self):
                return True

            def join(self, timeout=None):
                self.join_calls.append(timeout)

        cli = _make_voice_cli()
        thread = StuckThread()
        cli._voice_stream_barge_release.set()

        assert cli._voice_wait_for_streaming_tts(thread, timeout=120) is False
        assert thread.join_calls == []

    def test_uninterrupted_streaming_tts_waits_for_worker(self):
        class FinishingThread:
            def __init__(self):
                self.alive = True
                self.join_calls = []

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                self.join_calls.append(timeout)
                self.alive = False

        cli = _make_voice_cli()
        thread = FinishingThread()

        assert cli._voice_wait_for_streaming_tts(thread, timeout=120) is True
        assert thread.join_calls == [0.1]

    def test_barge_stream_failure_preserves_prior_finals_in_batch_fallback(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        recorder = MagicMock()
        recorder.stop.return_value = "/tmp/barge-retained.wav"
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_recording=False,
            _voice_recorder=recorder,
            _voice_stt_stream=MagicMock(active=True),
            _agent_running=True,
        )

        with patch("cli._cprint"), \
             patch("cli.os.path.isfile", return_value=False), \
             patch("tools.voice_mode.play_beep"), \
             patch(
                 "hermes_cli.config.load_config",
                 return_value={"stt": {"provider": "openai"}},
             ), \
             patch(
                 "tools.voice_mode.transcribe_recording",
                 return_value={"success": True, "transcript": "current words"},
             ):
            cli._voice_streaming_event(
                StreamingTranscriptEvent("earlier words", True)
            )
            cli._voice_streaming_barge_confirmed()
            cli._voice_streaming_error(RuntimeError("socket closed"))

        recorder.begin_continuous_fallback_capture.assert_called_once_with()
        assert str(cli._pending_input.get_nowait()) == (
            "earlier words current words"
        )

    def test_new_barge_candidate_replaces_a_decayed_noise_candidate(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_recording=False,
            _voice_recorder=MagicMock(),
            _voice_stt_stream=MagicMock(active=True),
            _agent_running=True,
        )

        with patch("cli._cprint"), patch("tools.voice_mode.play_beep"):
            cli._voice_streaming_barge_candidate()
            cli._voice_streaming_event(
                StreamingTranscriptEvent("earlier unsent words", True)
            )
            cli._voice_streaming_barge_candidate()
            cli._voice_streaming_barge_confirmed()
            assert cli._pending_input.empty()

            cli._voice_streaming_event(
                StreamingTranscriptEvent(
                    "actual interruption",
                    True,
                    "actual-final",
                    "actual-segment",
                )
            )
            _expire_streaming_endpoint(cli)

        assert str(cli._pending_input.get_nowait()) == (
            "earlier unsent words actual interruption"
        )

    def test_streaming_stop_phrase_uses_existing_voice_shutdown(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        cli = _make_voice_cli(
            _voice_recording=True,
            _voice_recorder=MagicMock(),
            _voice_stt_stream=MagicMock(),
        )
        cli._disable_voice_mode = MagicMock()

        with patch("cli._cprint"), \
             patch("tools.voice_mode.play_beep"), \
             patch("tools.voice_mode.is_voice_stop_phrase", return_value=True):
            cli._voice_streaming_final(
                StreamingTranscriptEvent("stop", True, "stop-final", "stop-segment")
            )
            _expire_streaming_endpoint(cli)

        cli._disable_voice_mode.assert_called_once()
        assert cli._pending_input.empty()

    def test_disable_invalidates_turn_and_closes_persistent_session(self):
        from tools.stt_streaming import StreamingTranscriptEvent

        coordinator = MagicMock()
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_processing=True,
            _voice_stt_stream=coordinator,
        )

        class _Thread:
            def __init__(self, target=None, **kwargs):
                self.target = target

            def start(self):
                self.target()

        with patch("cli._cprint"), \
             patch("cli.threading.Thread", _Thread), \
             patch("tools.voice_mode.stop_playback"):
            cli._disable_voice_mode()
            cli._voice_streaming_final(
                StreamingTranscriptEvent("late transcript", True)
            )

        coordinator.close.assert_called_once()
        assert cli._voice_stt_stream is None
        assert cli._voice_processing is False
        assert cli._pending_input.empty()


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

    def test_streaming_barge_reuses_recorder_and_drains_provider_final(
        self, monkeypatch
    ):
        from tools.stt_streaming import StreamingTranscriptEvent

        observed = {}
        interrupted = threading.Event()
        recorder = MagicMock()
        stream = MagicMock(active=True)
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_continuous=True,
            _voice_recorder=recorder,
            _voice_stt_stream=stream,
            _agent_running=True,
        )
        cli.agent = SimpleNamespace(interrupt=lambda: interrupted.set())

        def fake_listen(should_stop, is_playing=None, on_trigger=None, **kwargs):
            observed.update(kwargs)
            kwargs["on_candidate"]()
            cli._voice_streaming_event(
                StreamingTranscriptEvent("streamed interruption", True)
            )
            assert cli._pending_input.empty()
            on_trigger("generation")
            return None

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"voice": {"barge_in": True}},
        )
        monkeypatch.setattr("tools.voice_mode.full_duplex_listen", fake_listen)
        monkeypatch.setattr("tools.voice_mode.is_audio_output_active", lambda: False)
        monkeypatch.setattr("tools.voice_mode.stop_playback", lambda: None)

        with patch("cli._cprint"), \
             patch("tools.voice_mode.play_beep"), \
             patch("tools.voice_mode.transcribe_recording") as transcribe:
            cli._voice_full_duplex_listener()

        assert interrupted.is_set()
        assert observed["frame_queue"] is not None
        assert observed["capture"]() is False
        recorder.add_continuous_frame_sink.assert_called_once_with(
            observed["frame_queue"].enqueue
        )
        recorder.remove_continuous_frame_sink.assert_called_once()
        assert str(cli._pending_input.get_nowait()) == "streamed interruption"
        transcribe.assert_not_called()

    def test_batch_barge_reuses_persistent_recorder(self, monkeypatch):
        """Batch STT must not open a second mic while the recorder stays open."""
        observed = {}
        recorder = MagicMock()
        cli = _make_voice_cli(
            _voice_mode=True,
            _voice_continuous=True,
            _voice_recorder=recorder,
            _voice_stt_stream=None,
            _agent_running=True,
        )
        cli.agent = SimpleNamespace(interrupt=lambda: None)

        def fake_listen(should_stop, is_playing=None, on_trigger=None, **kwargs):
            observed.update(kwargs)
            return None

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"voice": {"barge_in": True}},
        )
        monkeypatch.setattr("tools.voice_mode.full_duplex_listen", fake_listen)
        monkeypatch.setattr("tools.voice_mode.is_audio_output_active", lambda: False)
        monkeypatch.setattr("tools.voice_mode.stop_playback", lambda: None)

        cli._voice_full_duplex_listener()

        assert observed["frame_queue"] is not None
        assert observed["capture"]() is True
        recorder.add_continuous_frame_sink.assert_called_once_with(
            observed["frame_queue"].enqueue
        )
        recorder.remove_continuous_frame_sink.assert_called_once()


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
            "tools.voice_mode._load_voice_stop_phrases", lambda: ("stop",)
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
