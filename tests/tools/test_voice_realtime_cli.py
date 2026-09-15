"""Tests for the CLI glue around the realtime (xAI S2S) voice input backend.

Covers the HermesCLI-side callbacks and gates in cli.py: transcript submit,
stop-phrase handling, the input gate's barge/half-duplex rules, speech_started
barge-in actions, and the classic-recorder branch in _voice_start_recording.
"""

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_rt_cli(**overrides):
    """Minimal HermesCLI with voice + realtime attrs (bypasses __init__)."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._voice_lock = threading.Lock()
    cli._voice_mode = True
    cli._voice_tts = False
    cli._voice_recorder = None
    cli._voice_recording = False
    cli._voice_processing = False
    cli._voice_continuous = True
    cli._voice_tts_done = threading.Event()
    cli._voice_tts_done.set()
    cli._voice_tts_stop = None
    cli._voice_barge_capture = threading.Event()
    cli._voice_rt_session = None
    cli._voice_rt_lock = threading.Lock()
    cli._voice_rt_failed = False
    cli._voice_rt_playback_t0 = None
    cli._voice_rt_reconnect_notified = False
    cli._voice_rt_barge_enabled = True
    cli._voice_rt_barge_grace_s = 0.5
    cli._voice_rt_supervisor = False
    cli._voice_rt_ctrl = None
    cli._voice_rt_narrate = True
    cli._voice_rt_full_duplex = False
    cli._voice_rt_output_ended_at = None
    cli._pending_input = queue.Queue()
    cli._app = None
    cli._attached_images = []
    cli._agent_running = False
    cli._clarify_state = None
    cli._sudo_state = None
    cli._approval_state = None
    cli._slash_confirm_state = None
    cli._secret_state = None
    cli.agent = None
    cli.console = SimpleNamespace(width=80)
    for k, v in overrides.items():
        setattr(cli, k, v)
    return cli


class TestOnTranscript:
    def test_transcript_queued_as_voice_input(self):
        from cli import _VoiceInputMessage

        cli = _make_rt_cli()
        cli._voice_realtime_on_transcript("  what's in this repo?  ")
        msg = cli._pending_input.get_nowait()
        assert isinstance(msg, _VoiceInputMessage)
        assert msg.text == "what's in this repo?"
        assert cli._no_speech_count == 0

    def test_empty_transcript_ignored(self):
        cli = _make_rt_cli()
        cli._voice_realtime_on_transcript("   ")
        assert cli._pending_input.empty()

    def test_stop_phrase_ends_voice_chat(self):
        cli = _make_rt_cli()
        with patch.object(cli, "_disable_voice_mode") as disable:
            with patch("tools.voice_mode.is_voice_stop_phrase", return_value=True):
                cli._voice_realtime_on_transcript("stop")
        disable.assert_called_once()
        assert cli._pending_input.empty()

    def test_transcript_clears_attached_images(self):
        cli = _make_rt_cli()
        cli._attached_images = [object()]
        cli._voice_realtime_on_transcript("hello")
        assert cli._attached_images == []


class TestInputGate:
    def test_open_when_idle_listening(self):
        cli = _make_rt_cli()
        assert cli._voice_realtime_gate() is True

    def test_closed_when_voice_mode_off(self):
        cli = _make_rt_cli(_voice_mode=False)
        assert cli._voice_realtime_gate() is False

    def test_closed_when_not_continuous(self):
        cli = _make_rt_cli(_voice_continuous=False)
        assert cli._voice_realtime_gate() is False

    def test_closed_during_modal_prompts(self):
        cli = _make_rt_cli(_approval_state={"cmd": "rm -rf"})
        assert cli._voice_realtime_gate() is False

    def test_half_duplex_when_barge_disabled_during_playback(self):
        cli = _make_rt_cli(_voice_rt_barge_enabled=False)
        cli._voice_tts_done.clear()
        assert cli._voice_realtime_gate() is False

    def test_half_duplex_when_barge_disabled_during_agent_turn(self):
        cli = _make_rt_cli(_voice_rt_barge_enabled=False, _agent_running=True)
        assert cli._voice_realtime_gate() is False

    def test_open_during_agent_turn_when_barge_enabled(self):
        cli = _make_rt_cli(_agent_running=True)
        assert cli._voice_realtime_gate() is True

    def test_playback_grace_window_suppresses_then_opens(self):
        cli = _make_rt_cli(_voice_rt_barge_grace_s=0.5)
        cli._voice_tts_done.clear()
        with patch("tools.voice_mode.is_audio_output_active", return_value=True):
            # First call anchors playback onset — inside grace → closed.
            assert cli._voice_realtime_gate() is False
            # Simulate the grace window having elapsed.
            cli._voice_rt_playback_t0 = time.monotonic() - 1.0
            assert cli._voice_realtime_gate() is True

    def test_playback_anchor_resets_when_output_ends(self):
        cli = _make_rt_cli()
        cli._voice_rt_playback_t0 = time.monotonic() - 99
        with patch("tools.voice_mode.is_audio_output_active", return_value=False):
            assert cli._voice_realtime_gate() is True
        assert cli._voice_rt_playback_t0 is None


class TestOnSpeechStarted:
    def test_barge_during_playback_cuts_tts(self):
        cli = _make_rt_cli()
        cli._voice_tts_done.clear()
        pipe_stop = threading.Event()
        cli._voice_tts_stop = pipe_stop
        with patch("tools.voice_mode.stop_playback") as stop_pb, \
             patch("tools.voice_mode.is_audio_output_active", return_value=True), \
             patch("tools.tts_streaming.mark_speech_interrupted") as mark:
            cli._voice_realtime_on_speech_started()
        mark.assert_called_once()
        stop_pb.assert_called_once()
        assert pipe_stop.is_set()

    def test_barge_during_generation_interrupts_agent(self):
        agent = MagicMock()
        cli = _make_rt_cli(_agent_running=True, agent=agent)
        with patch("tools.voice_mode.is_audio_output_active", return_value=False), \
             patch("tools.voice_mode.stop_playback"):
            cli._voice_realtime_on_speech_started()
        agent.interrupt.assert_called_once()

    def test_idle_speech_does_nothing_destructive(self):
        agent = MagicMock()
        cli = _make_rt_cli(agent=agent)
        with patch("tools.voice_mode.is_audio_output_active", return_value=False), \
             patch("tools.voice_mode.stop_playback") as stop_pb:
            cli._voice_realtime_on_speech_started()
        agent.interrupt.assert_not_called()
        stop_pb.assert_not_called()


class TestActivityHold:
    def test_holds_while_agent_running(self):
        cli = _make_rt_cli(_agent_running=True)
        assert cli._voice_realtime_activity_hold() is True

    def test_holds_while_tts_pending(self):
        cli = _make_rt_cli()
        cli._voice_tts_done.clear()
        assert cli._voice_realtime_activity_hold() is True

    def test_released_when_idle(self):
        cli = _make_rt_cli()
        with patch("tools.voice_mode.is_audio_output_active", return_value=False):
            assert cli._voice_realtime_activity_hold() is False


class TestStartRecordingBranch:
    def test_realtime_branch_short_circuits_classic_recorder(self):
        cli = _make_rt_cli()
        with patch.object(cli, "_voice_realtime_config_enabled", return_value=True), \
             patch.object(cli, "_voice_realtime_start", return_value=True) as rt_start, \
             patch("tools.voice_mode.create_audio_recorder") as create_rec:
            cli._voice_start_recording()
        rt_start.assert_called_once()
        create_rec.assert_not_called()
        assert cli._voice_recorder is None

    def test_failed_realtime_falls_through_to_classic(self):
        cli = _make_rt_cli(_voice_rt_failed=True, _should_exit=False)
        with patch.object(cli, "_voice_realtime_config_enabled", return_value=True), \
             patch.object(cli, "_voice_realtime_start") as rt_start, \
             patch("tools.voice_mode.check_voice_requirements",
                   return_value={"audio_available": False, "stt_available": False,
                                 "available": False, "missing_packages": [],
                                 "details": ""}):
            # Classic path raises on missing audio — the point is that it was
            # REACHED (realtime start was never attempted).
            with pytest.raises(RuntimeError):
                cli._voice_start_recording()
        rt_start.assert_not_called()


class TestOnStateDead:
    def test_dead_state_marks_failed_and_falls_back(self):
        sess = MagicMock()
        cli = _make_rt_cli(_voice_rt_session=sess, _voice_recording=True)
        with patch.object(cli, "_voice_realtime_fallback_to_classic") as fallback:
            cli._voice_realtime_on_state("dead", "socket refused")
            # Fallback runs on a spawned thread — wait for it.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not fallback.called:
                time.sleep(0.02)
        assert cli._voice_rt_failed is True
        assert cli._voice_rt_session is None
        assert cli._voice_recording is False
        sess.stop.assert_called_once()
        fallback.assert_called_once()

    def test_connected_state_resets_reconnect_notice(self):
        cli = _make_rt_cli(_voice_rt_reconnect_notified=True)
        cli._voice_realtime_on_state("connected", "")
        assert cli._voice_rt_reconnect_notified is False


class TestClassicFallback:
    def test_fallback_starts_classic_recorder(self):
        cli = _make_rt_cli()
        with patch.object(cli, "_voice_start_recording") as start:
            cli._voice_realtime_fallback_to_classic()
        start.assert_called_once()

    def test_fallback_noops_when_voice_chat_not_live(self):
        cli = _make_rt_cli(_voice_continuous=False)
        with patch.object(cli, "_voice_start_recording") as start:
            cli._voice_realtime_fallback_to_classic()
        start.assert_not_called()

    def test_fallback_disables_voice_mode_when_classic_unavailable(self):
        """Realtime-only setup (no STT): a dead session must not leave a
        deaf live chat — voice mode ends with a clear message."""
        cli = _make_rt_cli()
        with patch.object(
            cli, "_voice_start_recording", side_effect=RuntimeError("no STT")
        ), patch.object(cli, "_disable_voice_mode") as disable:
            cli._voice_realtime_fallback_to_classic()
        disable.assert_called_once()

    def test_start_or_fallback_falls_back_on_failure(self):
        cli = _make_rt_cli()
        with patch.object(cli, "_voice_realtime_start", return_value=False), \
             patch.object(cli, "_voice_realtime_fallback_to_classic") as fallback:
            cli._voice_realtime_start_or_fallback()
        fallback.assert_called_once()

    def test_start_or_fallback_skips_fallback_on_success(self):
        cli = _make_rt_cli()
        with patch.object(cli, "_voice_realtime_start", return_value=True), \
             patch.object(cli, "_voice_realtime_fallback_to_classic") as fallback:
            cli._voice_realtime_start_or_fallback()
        fallback.assert_not_called()


def _make_supervisor_cli(**overrides):
    sess = MagicMock()
    sess.alive = True
    sess.barge_active = False
    sess.speaking = False
    cli = _make_rt_cli(
        _voice_rt_session=sess, _voice_rt_supervisor=True, **overrides
    )
    return cli, sess


class TestSupervisorGlue:
    """cli.py is thin wiring now — the lifecycle logic lives in
    agent.voice_supervisor (tests/agent/test_voice_supervisor.py)."""

    def test_function_call_delegates_to_controller(self):
        ctrl = MagicMock()
        cli, sess = _make_supervisor_cli(_voice_rt_ctrl=ctrl)
        cli._voice_realtime_on_function_call("consult_hermes", "c1", "{}")
        ctrl.on_function_call.assert_called_once_with("consult_hermes", "c1", "{}")

    def test_function_call_noop_without_controller(self):
        cli, sess = _make_supervisor_cli(_voice_rt_ctrl=None)
        cli._voice_realtime_on_function_call("consult_hermes", "c1", "{}")  # no raise

    def test_ctrl_c_ends_idle_voice_chat_but_yields_to_a_running_turn(self):
        """With a live realtime session Ctrl+C ends the voice chat only when
        no agent turn is running; otherwise the press must fall through to
        the agent-interrupt path (the composer hint promises 'Ctrl+C to
        cancel' the turn)."""
        app = MagicMock()
        cli, _ = _make_supervisor_cli(_agent_running=True, agent=MagicMock())
        with patch.object(cli, "_disable_voice_mode") as disable:
            assert cli._voice_realtime_handle_ctrl_c(app) is False
            disable.assert_not_called()
        cli._agent_running = False
        with patch.object(cli, "_disable_voice_mode") as disable:
            assert cli._voice_realtime_handle_ctrl_c(app) is True
            for _ in range(50):
                if disable.called:
                    break
                time.sleep(0.01)
            disable.assert_called_once()

    def test_consult_complete_delegates_and_returns_verdict(self):
        ctrl = MagicMock()
        ctrl.on_turn_complete.return_value = True
        cli, sess = _make_supervisor_cli(_voice_rt_ctrl=ctrl)
        assert cli._voice_realtime_consult_complete("task", "reply") is True
        ctrl.on_turn_complete.assert_called_once_with("task", "reply", interrupted=False)
        ctrl.on_turn_complete.reset_mock()
        # An interrupted turn (steer) is reported as such so the partial
        # output never completes the consult.
        cli._voice_realtime_consult_complete("task", "partial", interrupted=True)
        ctrl.on_turn_complete.assert_called_once_with("task", "partial", interrupted=True)
        cli2, _ = _make_supervisor_cli(_voice_rt_ctrl=None)
        assert cli2._voice_realtime_consult_complete("task", "reply") is False

    def test_narrate_delegates_to_controller(self):
        ctrl = MagicMock()
        cli, sess = _make_supervisor_cli(_voice_rt_ctrl=ctrl)
        cli._voice_realtime_narrate_tool("terminal")
        ctrl.narrate_tool.assert_called_once_with("terminal")

    def test_cli_turn_runner_contract(self):
        from hermes_cli.cli_voice_realtime_mixin import _CLIVoiceTurnRunner

        agent = MagicMock()
        cli = _make_rt_cli(agent=agent, _agent_running=True)
        runner = _CLIVoiceTurnRunner(cli)
        runner.submit("!rm -rf /tmp/x")
        queued = cli._pending_input.get_nowait()
        # Literal turn (full toolset, no concise-voice prefix): the model's
        # restated task is never keyboard input, so it must not reach the
        # `!` shell dispatcher, slash routing or file-drop detection.
        from cli import _SeededQueryMessage, _VoiceInputMessage
        assert isinstance(queued, _SeededQueryMessage)
        assert not isinstance(queued, _VoiceInputMessage)
        assert queued.text == "!rm -rf /tmp/x"
        assert runner.is_busy() is True
        assert runner.is_queue_empty() is True
        runner.interrupt()
        agent.interrupt.assert_called_once()

    def test_seeded_consult_text_is_a_turn_not_a_typed_stop_phrase(self):
        """A consult the voice model restates as exactly ``stop`` (\"stop the dev server\" →
        steer text ``stop``) is literal agent input, like every seeded query; only KEYBOARD input
        may end the voice chat via the typed stop phrase."""
        from cli import _SeededQueryMessage

        cli = _make_rt_cli()
        cli._pending_resume_sessions = []
        cli._status_bar_suppressed_after_resize = False
        cli._app = SimpleNamespace(invalidate=lambda: None)
        cli._turn_summary_begin = lambda: None
        cli._tui_after_turn = lambda: None
        cli._print_user_message_preview = lambda *_a, **_k: None
        cli.chat = MagicMock()
        stop_calls = []
        cli._typed_voice_stop = lambda text: stop_calls.append(text) or True

        cli._tui_process_one_input(_SeededQueryMessage("stop"))
        assert stop_calls == []
        assert cli.chat.call_args.args[0] == "stop"

        cli.chat.reset_mock()
        cli._tui_process_one_input("stop")
        assert stop_calls == ["stop"]
        cli.chat.assert_not_called()

    def test_idle_pause_clears_continuous(self):
        """Idle pause must behave like manual pause — otherwise the next
        completed turn's auto-restart re-arms the mic (billing guard)."""
        cli, sess = _make_supervisor_cli(_voice_recording=True)
        with patch.object(cli, "_voice_record_key_label", return_value="Ctrl+B"):
            cli._voice_realtime_on_idle_pause()
        assert cli._voice_continuous is False
        assert cli._voice_recording is False

    def test_supervisor_transcript_never_enqueues_or_echoes(self):
        cli, sess = _make_supervisor_cli()
        with patch("cli._cprint") as cprint:
            cli._voice_realtime_on_transcript("what's the weather like")
        assert cli._pending_input.empty()
        # Sidecar ASR is approximate and unused by the model — never shown.
        cprint.assert_not_called()

    def test_supervisor_stop_phrase_still_ends_chat(self):
        cli, sess = _make_supervisor_cli()
        with patch.object(cli, "_disable_voice_mode") as disable, \
             patch("tools.voice_mode.is_voice_stop_phrase", return_value=True):
            cli._voice_realtime_on_transcript("stop")
        disable.assert_called_once()

    def test_supervisor_speech_started_never_interrupts_agent(self):
        agent = MagicMock()
        cli, sess = _make_supervisor_cli(_agent_running=True, agent=agent)
        cli._voice_realtime_on_speech_started()
        agent.interrupt.assert_not_called()


    def test_steer_flows_through_real_controller_and_cli_runner(self):
        """End-to-end wiring: session callback → controller → CLI runner."""
        import json as _json

        from agent.voice_supervisor import VoiceSupervisorController
        from hermes_cli.cli_voice_realtime_mixin import _CLIVoiceTurnRunner

        agent = MagicMock()
        cli, sess = _make_supervisor_cli(_agent_running=True, agent=agent)
        ctrl = VoiceSupervisorController(sess, _CLIVoiceTurnRunner(cli))
        cli._voice_rt_ctrl = ctrl
        sess.last_response_had_audio = True
        cli._voice_realtime_on_function_call(
            "consult_hermes", "c1", _json.dumps({"task": "original"})
        )
        cli._voice_realtime_on_function_call(
            "steer_hermes", "s1", _json.dumps({"instruction": "also check logs"})
        )
        assert cli._pending_input.get_nowait().text == "original"
        assert cli._pending_input.get_nowait().text == "also check logs"
        agent.interrupt.assert_called_once()
        assert sess.send_function_output.call_args[0][0] == "s1"
        # The steered continuation completes the consult with c1's call id.
        assert cli._voice_realtime_consult_complete("also check logs", "done") is True
        assert sess.send_function_output.call_args[0] == ("c1", "done")

    def test_gate_mutes_mic_while_speech_plays_by_default(self):
        """Half-duplex default: open speakers must never feed the assistant
        its own voice (the self-conversation loop)."""
        cli, sess = _make_supervisor_cli()
        sess.barge_active = False
        with patch("tools.voice_mode.is_audio_output_active", return_value=True):
            assert cli._voice_realtime_gate() is False

    def test_gate_opens_during_playback_when_barge_window_active(self):
        cli, sess = _make_supervisor_cli()
        sess.barge_active = True  # user talked over the speech loudly
        with patch("tools.voice_mode.is_audio_output_active", return_value=True):
            assert cli._voice_realtime_gate() is True

    def test_gate_holds_briefly_after_speech_ends(self):
        cli, sess = _make_supervisor_cli()
        with patch("tools.voice_mode.is_audio_output_active", return_value=False):
            # First quiet poll starts the hangover window → still closed.
            assert cli._voice_realtime_gate() is False
            # After the tail window the mic reopens.
            cli._voice_rt_output_ended_at = time.monotonic() - 1.0
            assert cli._voice_realtime_gate() is True

    def test_gate_open_while_agent_works_in_supervisor(self):
        """Chatting during a background consult is the point — a running
        agent must never mute the supervisor mic."""
        cli, sess = _make_supervisor_cli(_agent_running=True)
        cli._voice_rt_output_ended_at = time.monotonic() - 1.0
        with patch("tools.voice_mode.is_audio_output_active", return_value=False):
            assert cli._voice_realtime_gate() is True

    def test_gate_full_duplex_uses_grace_window(self):
        cli, sess = _make_supervisor_cli(_voice_rt_full_duplex=True)
        with patch("tools.voice_mode.is_audio_output_active", return_value=True):
            assert cli._voice_realtime_gate() is False  # inside onset grace
            cli._voice_rt_playback_t0 = time.monotonic() - 1.0
            assert cli._voice_realtime_gate() is True

    def test_stop_clears_supervisor_state(self):
        ctrl = MagicMock()
        cli, sess = _make_supervisor_cli(_voice_rt_ctrl=ctrl)
        cli._voice_realtime_stop()
        assert cli._voice_rt_ctrl is None
        ctrl.reset.assert_called_once()
        assert cli._voice_rt_supervisor is False


class TestPauseAndStop:
    def test_pause_disarms_and_clears_flags(self):
        sess = MagicMock()
        sess.alive = True
        cli = _make_rt_cli(_voice_rt_session=sess, _voice_recording=True)
        cli._voice_realtime_pause(announce=False)
        assert cli._voice_continuous is False
        assert cli._voice_recording is False
        sess.set_armed.assert_called_once_with(False)

    def test_stop_tears_down_session_and_resets_failure(self):
        sess = MagicMock()
        cli = _make_rt_cli(
            _voice_rt_session=sess, _voice_rt_failed=True, _voice_recording=True
        )
        cli._voice_realtime_stop()
        assert cli._voice_rt_session is None
        assert cli._voice_rt_failed is False
        assert cli._voice_recording is False
        # Session teardown happens off-thread; wait for it.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not sess.stop.called:
            time.sleep(0.02)
        assert sess.stop.called


class TestStatusLine:
    """``/voice status`` before ``/voice on``: the brain shown is what config WOULD start, so the
    line must agree with ``load_realtime_config`` (supervisor default, ``ears`` opt-in) instead of
    the not-yet-started session's zeroed flag."""

    @pytest.mark.parametrize("realtime_cfg", [{"enabled": True}, {"enabled": True, "brain": "ears"}])
    def test_idle_status_reports_the_configured_brain(self, realtime_cfg):
        from tools.voice_realtime_config import load_realtime_config

        cli = _make_rt_cli(_voice_rt_session=None)
        voice_cfg = {"realtime": realtime_cfg}
        with patch("hermes_cli.config.load_config", return_value={"voice": voice_cfg}):
            line = cli._voice_realtime_status_line()
        assert line is not None and "enabled (not active)" in line
        expected = "supervisor" if load_realtime_config(voice_cfg).supervisor else "ears"
        assert f"brain: {expected}" in line

    def test_live_status_reports_the_running_sessions_brain(self):
        sess = MagicMock()
        sess.alive = True
        sess.connected = True
        sess.armed = True
        cli = _make_rt_cli(_voice_rt_session=sess, _voice_rt_supervisor=False)
        with patch("hermes_cli.config.load_config", return_value={"voice": {"realtime": {"enabled": True}}}):
            line = cli._voice_realtime_status_line()
        assert "connected, listening" in line
        assert "brain: ears" in line  # the live session, not the (supervisor) config default
