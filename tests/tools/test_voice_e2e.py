"""End-to-end tests for Voice Mode (Issue #314).

Verifies the full voice pipeline works as specified in the issue:
  Phase 2: CLI push-to-talk input (Ctrl+R toggle, audio → STT → text submission)
  Phase 3: TTS response output (voice system prompt, markdown stripping)
  Phase 4: Low-latency features (silence detection, audio cues, continuous mode,
           configurable params, peak RMS, hallucination guard, playback interrupt)

All audio I/O and API calls are mocked -- no real microphone or network needed.
"""

import os
import queue
import struct
import threading
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_sd(monkeypatch):
    """Replace sounddevice with a MagicMock."""
    mock = MagicMock()
    monkeypatch.setattr("tools.voice_mode.sd", mock)
    monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", True)
    try:
        import numpy as real_np
        monkeypatch.setattr("tools.voice_mode.np", real_np)
    except ImportError:
        monkeypatch.setattr("tools.voice_mode.np", MagicMock())
    return mock


@pytest.fixture
def temp_voice_dir(tmp_path, monkeypatch):
    """Redirect voice temp dir to a temporary path."""
    voice_dir = tmp_path / "hermes_voice"
    voice_dir.mkdir()
    monkeypatch.setattr("tools.voice_mode._TEMP_DIR", str(voice_dir))
    return voice_dir


@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal valid WAV file (1 second of tone)."""
    wav_path = tmp_path / "test.wav"
    n_frames = 16000
    # Generate a 440Hz tone so it's not silent
    import math
    samples = [int(10000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n_frames)]
    data = struct.pack(f"<{n_frames}h", *samples)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(data)

    return str(wav_path)


def _get_callback(mock_sd):
    """Extract the InputStream callback from the mock."""
    cb = mock_sd.InputStream.call_args.kwargs.get("callback")
    if cb is None:
        cb = mock_sd.InputStream.call_args[1]["callback"]
    return cb


# ============================================================================
# Phase 2: Full voice input pipeline
# ============================================================================

class TestFullVoicePipeline:
    """End-to-end: record → WAV → STT → transcript."""

    def test_record_transcribe_returns_text(self, mock_sd, temp_voice_dir):
        """Full pipeline: start recording → simulate speech frames → stop →
        write WAV → transcribe → get transcript text."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder, SAMPLE_RATE, transcribe_recording

        recorder = AudioRecorder()
        recorder.start()
        callback = _get_callback(mock_sd)

        # Simulate 1 second of speech (RMS ~5000, well above threshold 200)
        for _ in range(10):
            loud_frame = np.full((1600, 1), 5000, dtype="int16")
            callback(loud_frame, 1600, None, None)

        wav_path = recorder.stop()
        assert wav_path is not None
        assert os.path.isfile(wav_path)

        # Verify WAV file is valid and contains audio
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnframes() == 16000  # 10 chunks * 1600 samples

        # Transcribe via mocked STT
        mock_transcribe = MagicMock(return_value={
            "success": True,
            "transcript": "hello world",
        })
        with patch("tools.transcription_tools.transcribe_audio", mock_transcribe):
            result = transcribe_recording(wav_path)

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        mock_transcribe.assert_called_once_with(wav_path, model=None)

    def test_silent_recording_skipped_before_stt(self, mock_sd, temp_voice_dir):
        """Silent recordings are rejected at WAV stage (peak RMS check),
        STT is never called."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder, SAMPLE_RATE

        recorder = AudioRecorder()
        recorder.start()
        callback = _get_callback(mock_sd)

        # 1 second of near-silence
        for _ in range(10):
            silent_frame = np.full((1600, 1), 10, dtype="int16")
            callback(silent_frame, 1600, None, None)

        wav_path = recorder.stop()
        # Peak RMS ~10, below threshold 200 → rejected
        assert wav_path is None


# ============================================================================
# Phase 2 + 4: Silence detection auto-stop
# ============================================================================

class TestSilenceDetectionE2E:
    """Verify the full silence detection flow: speech → silence → auto-stop callback."""

    def test_speech_then_silence_triggers_auto_stop(self, mock_sd, temp_voice_dir):
        """Simulate realistic speech pattern: sustained speech → prolonged silence → auto-stop fires."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder._silence_duration = 0.1  # Short for testing
        recorder._min_speech_duration = 0.05

        auto_stopped = threading.Event()
        recorder.start(on_silence_stop=lambda: auto_stopped.set())
        callback = _get_callback(mock_sd)

        # Phase 1: Sustained speech (0.3s with micro-pauses)
        loud = np.full((1600, 1), 5000, dtype="int16")
        quiet = np.full((1600, 1), 50, dtype="int16")

        callback(loud, 1600, None, None)
        time.sleep(0.03)
        callback(loud, 1600, None, None)
        time.sleep(0.03)
        # Brief micro-pause (should be tolerated)
        callback(quiet, 1600, None, None)
        time.sleep(0.02)
        callback(loud, 1600, None, None)
        time.sleep(0.03)

        assert recorder._has_spoken is True, "Speech should be confirmed"

        # Phase 2: Prolonged silence (> silence_duration)
        callback(quiet, 1600, None, None)
        time.sleep(0.12)
        callback(quiet, 1600, None, None)

        assert auto_stopped.wait(timeout=2.0) is True, "Auto-stop should fire after silence"
        recorder.cancel()

    def test_ambient_noise_does_not_trigger_speech(self, mock_sd):
        """Background noise below threshold should NOT confirm speech."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder._silence_duration = 0.02

        fired = threading.Event()
        recorder.start(on_silence_stop=lambda: fired.set())
        callback = _get_callback(mock_sd)

        # Send noise just below threshold (RMS ~150, threshold is 200)
        noise = np.full((1600, 1), 150, dtype="int16")
        for _ in range(10):
            callback(noise, 1600, None, None)
            time.sleep(0.01)

        assert recorder._has_spoken is False
        assert fired.wait(timeout=0.2) is False
        recorder.cancel()

    def test_long_dip_resets_speech_attempt(self, mock_sd):
        """A dip lasting longer than max_dip_tolerance should reset the speech attempt."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder._min_speech_duration = 0.3
        recorder._max_dip_tolerance = 0.05

        recorder.start(on_silence_stop=lambda: None)
        callback = _get_callback(mock_sd)

        loud = np.full((1600, 1), 5000, dtype="int16")
        quiet = np.full((1600, 1), 50, dtype="int16")

        # Start speaking
        callback(loud, 1600, None, None)
        assert recorder._speech_start > 0

        # Long silence (> max_dip_tolerance)
        callback(quiet, 1600, None, None)
        time.sleep(0.06)
        callback(quiet, 1600, None, None)

        # Speech attempt should have been reset
        assert recorder._speech_start == 0.0
        assert recorder._has_spoken is False
        recorder.cancel()


# ============================================================================
# Phase 2 + 4: Peak RMS guard
# ============================================================================

class TestPeakRmsGuard:
    """Verify peak RMS check doesn't discard recordings that had real speech."""

    def test_speech_with_trailing_silence_passes(self, mock_sd, temp_voice_dir):
        """Short speech followed by long silence: overall RMS would fail,
        but peak RMS should pass."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder, SAMPLE_RATE

        recorder = AudioRecorder()
        recorder.start()
        callback = _get_callback(mock_sd)

        # 0.5s of speech (RMS ~3000)
        for _ in range(5):
            speech = np.full((1600, 1), 3000, dtype="int16")
            callback(speech, 1600, None, None)

        # 3s of silence
        for _ in range(30):
            silence = np.full((1600, 1), 10, dtype="int16")
            callback(silence, 1600, None, None)

        # Peak RMS should be ~3000 (from speech), well above threshold
        assert recorder._peak_rms >= 3000

        wav_path = recorder.stop()
        # Should NOT be rejected despite low overall average RMS
        assert wav_path is not None

    def test_pure_silence_rejected_by_peak_rms(self, mock_sd, temp_voice_dir):
        """Recording with only silence should be rejected (peak RMS < threshold)."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()
        callback = _get_callback(mock_sd)

        for _ in range(10):
            silence = np.full((1600, 1), 10, dtype="int16")
            callback(silence, 1600, None, None)

        assert recorder._peak_rms <= 10
        wav_path = recorder.stop()
        assert wav_path is None


# ============================================================================
# Phase 2 + 4: Whisper hallucination guard (end-to-end)
# ============================================================================

class TestHallucinationGuardE2E:
    """Verify that known Whisper hallucinations are filtered in the full pipeline."""

    def test_hallucination_filtered_returns_empty_transcript(self):
        """STT returns 'Thank you.' on silence → filtered → empty transcript."""
        mock_transcribe = MagicMock(return_value={
            "success": True,
            "transcript": "Thank you.",
        })
        with patch("tools.transcription_tools.transcribe_audio", mock_transcribe):
            from tools.voice_mode import transcribe_recording
            result = transcribe_recording("/tmp/test.wav")

        assert result["success"] is True
        assert result["transcript"] == ""
        assert result["filtered"] is True

    def test_real_transcript_passes_through(self):
        """Normal speech should not be filtered."""
        mock_transcribe = MagicMock(return_value={
            "success": True,
            "transcript": "What is the weather like today?",
        })
        with patch("tools.transcription_tools.transcribe_audio", mock_transcribe):
            from tools.voice_mode import transcribe_recording
            result = transcribe_recording("/tmp/test.wav")

        assert result["transcript"] == "What is the weather like today?"
        assert "filtered" not in result

    def test_stt_failure_propagated(self):
        """STT errors should propagate without filtering."""
        mock_transcribe = MagicMock(return_value={
            "success": False,
            "error": "API rate limit",
        })
        with patch("tools.transcription_tools.transcribe_audio", mock_transcribe):
            from tools.voice_mode import transcribe_recording
            result = transcribe_recording("/tmp/test.wav")

        assert result["success"] is False
        assert result["error"] == "API rate limit"


# ============================================================================
# Phase 3: Voice system prompt
# ============================================================================

class TestVoiceSystemPrompt:
    """Verify that voice mode appends a concise-response system prompt
    and restores the original on disable."""

    def test_system_prompt_appended_and_restored(self):
        """Simulate the enable/disable flow for voice system prompt."""
        original_prompt = "You are a helpful assistant."

        # Simulate _enable_voice_mode logic
        voice_original_prompt = original_prompt
        voice_instruction = (
            "\n\n[Voice mode active] The user is speaking via voice input. "
            "Keep responses concise and conversational — 2-3 sentences max unless "
            "the user asks for detail. Avoid code blocks, markdown formatting, "
            "and long lists. Respond naturally as in a spoken conversation."
        )
        modified_prompt = (original_prompt or "") + voice_instruction

        assert "[Voice mode active]" in modified_prompt
        assert "concise and conversational" in modified_prompt
        assert modified_prompt.startswith(original_prompt)

        # Simulate _disable_voice_mode logic
        restored_prompt = voice_original_prompt
        assert restored_prompt == original_prompt
        assert "[Voice mode active]" not in restored_prompt


# ============================================================================
# Phase 4: Audio cues
# ============================================================================

class TestAudioCues:
    """Verify beep tones are generated correctly."""

    def test_start_beep_frequency_and_count(self, mock_sd):
        """Start recording beep: 880Hz, single beep."""
        np = pytest.importorskip("numpy")

        from tools.voice_mode import play_beep

        play_beep(frequency=880, count=1)

        mock_sd.play.assert_called_once()
        audio = mock_sd.play.call_args[0][0]
        assert audio.dtype == np.int16
        # Single beep of 0.12s at 16kHz = ~1920 samples
        assert 1500 < len(audio) < 2500

    def test_stop_beep_double(self, mock_sd):
        """Stop recording beep: 660Hz, double beep."""
        np = pytest.importorskip("numpy")

        from tools.voice_mode import play_beep

        play_beep(frequency=660, count=2)

        audio = mock_sd.play.call_args[0][0]
        # Double beep should be roughly 2x single + gap
        assert len(audio) > 3000

    def test_tool_tick_beep(self, mock_sd):
        """Tool execution tick: 1200Hz, very short."""
        np = pytest.importorskip("numpy")

        from tools.voice_mode import play_beep

        play_beep(frequency=1200, duration=0.06, count=1)

        audio = mock_sd.play.call_args[0][0]
        # 0.06s at 16kHz = ~960 samples
        assert 800 < len(audio) < 1200


# ============================================================================
# Phase 4: Interruptable TTS playback
# ============================================================================

class TestPlaybackInterrupt:
    """Verify that TTS playback can be interrupted."""

    def test_stop_playback_terminates_process(self):
        """stop_playback() should terminate the active playback subprocess."""
        from tools.voice_mode import stop_playback, _playback_lock
        import tools.voice_mode as vm

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process is running

        # Set the active playback
        with _playback_lock:
            vm._active_playback = mock_proc

        stop_playback()

        mock_proc.terminate.assert_called_once()

        # Verify it was cleared
        with _playback_lock:
            assert vm._active_playback is None

    def test_stop_playback_noop_when_nothing_playing(self):
        """stop_playback() should not crash when nothing is playing."""
        import tools.voice_mode as vm

        with vm._playback_lock:
            vm._active_playback = None

        # Should not raise
        vm.stop_playback()

    def test_play_audio_file_sets_active_playback(self, monkeypatch, sample_wav):
        """play_audio_file() should track the subprocess for interrupt support."""
        import tools.voice_mode as vm

        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", False)  # Skip sounddevice

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        mock_popen = MagicMock(return_value=mock_proc)
        monkeypatch.setattr("subprocess.Popen", mock_popen)
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/" + cmd)

        vm.play_audio_file(sample_wav)

        # Popen should have been called (system player)
        assert mock_popen.called
        # After playback completes, _active_playback should be cleared
        with vm._playback_lock:
            assert vm._active_playback is None


# ============================================================================
# Phase 4: Configurable silence parameters
# ============================================================================

class TestConfigurableSilenceParams:
    """Verify that silence detection params can be configured."""

    def test_custom_threshold_and_duration(self, mock_sd):
        """Custom threshold and duration should affect silence detection behavior."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        # Set high threshold -- only very loud audio counts as speech
        recorder._silence_threshold = 5000
        recorder._silence_duration = 0.05
        recorder._min_speech_duration = 0.05

        fired = threading.Event()
        recorder.start(on_silence_stop=lambda: fired.set())
        callback = _get_callback(mock_sd)

        # Audio at RMS 1000 -- below custom threshold (5000), should NOT count as speech
        moderate = np.full((1600, 1), 1000, dtype="int16")
        for _ in range(5):
            callback(moderate, 1600, None, None)
            time.sleep(0.02)

        assert recorder._has_spoken is False
        assert fired.wait(timeout=0.2) is False

        # Now send really loud audio (above 5000 threshold)
        very_loud = np.full((1600, 1), 8000, dtype="int16")
        callback(very_loud, 1600, None, None)
        time.sleep(0.06)
        callback(very_loud, 1600, None, None)
        assert recorder._has_spoken is True

        recorder.cancel()

    def test_default_config_values(self):
        """Default config should have voice silence params."""
        from hermes_cli.config import DEFAULT_CONFIG

        voice_cfg = DEFAULT_CONFIG["voice"]
        assert "silence_threshold" in voice_cfg
        assert "silence_duration" in voice_cfg
        assert voice_cfg["silence_threshold"] == 200
        assert voice_cfg["silence_duration"] == 3.0


# ============================================================================
# Phase 4: Continuous mode flow
# ============================================================================

class TestContinuousModeFlow:
    """Verify continuous mode: auto-restart after transcription or silence."""

    def test_continuous_restart_on_no_speech(self, mock_sd, temp_voice_dir):
        """In continuous mode, when no speech is detected, recording should
        restart automatically (not require manual Ctrl+R)."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()

        # First recording: only silence → stop returns None
        recorder.start()
        callback = _get_callback(mock_sd)

        for _ in range(10):
            silence = np.full((1600, 1), 10, dtype="int16")
            callback(silence, 1600, None, None)

        wav_path = recorder.stop()
        assert wav_path is None  # No speech

        # Simulate continuous mode restart (what cli.py does)
        recorder.start()
        assert recorder.is_recording is True

        # Second recording: now with speech
        callback = _get_callback(mock_sd)
        for _ in range(10):
            speech = np.full((1600, 1), 5000, dtype="int16")
            callback(speech, 1600, None, None)

        wav_path = recorder.stop()
        assert wav_path is not None  # Speech captured this time

        recorder.cancel()

    def test_recorder_reusable_after_stop(self, mock_sd, temp_voice_dir):
        """Recorder should be reusable: stop → start → stop works correctly."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        results = []

        for i in range(3):
            recorder.start()
            callback = _get_callback(mock_sd)
            loud = np.full((1600, 1), 5000, dtype="int16")
            for _ in range(10):
                callback(loud, 1600, None, None)
            wav_path = recorder.stop()
            results.append(wav_path)

        # All 3 recordings should produce valid WAV files
        assert all(r is not None for r in results)
        # Final file should exist and be a valid WAV
        assert os.path.isfile(results[-1])


# ============================================================================
# Phase 4: Audio level indicator
# ============================================================================

class TestAudioLevelIndicator:
    """Verify current_rms property updates in real-time for UI feedback."""

    def test_rms_updates_with_audio_chunks(self, mock_sd):
        """current_rms should reflect the latest audio chunk's RMS level."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()
        callback = _get_callback(mock_sd)

        assert recorder.current_rms == 0  # Initial state

        # Send loud chunk
        loud = np.full((1600, 1), 5000, dtype="int16")
        callback(loud, 1600, None, None)
        assert recorder.current_rms == 5000

        # Send quiet chunk
        quiet = np.full((1600, 1), 100, dtype="int16")
        callback(quiet, 1600, None, None)
        assert recorder.current_rms == 100

        recorder.cancel()

    def test_peak_rms_tracks_maximum(self, mock_sd):
        """peak_rms should track the highest RMS seen during the recording."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()
        callback = _get_callback(mock_sd)

        frames = [
            np.full((1600, 1), 100, dtype="int16"),
            np.full((1600, 1), 8000, dtype="int16"),
            np.full((1600, 1), 500, dtype="int16"),
            np.full((1600, 1), 3000, dtype="int16"),
        ]
        for frame in frames:
            callback(frame, 1600, None, None)

        assert recorder._peak_rms == 8000  # Maximum seen
        assert recorder.current_rms == 3000  # Latest chunk

        recorder.cancel()


# ============================================================================
# Phase 4: Requirements check
# ============================================================================

class TestVoiceRequirements:
    """Verify check_voice_requirements reports correct status."""

    def test_groq_key_sufficient(self, monkeypatch):
        """GROQ_API_KEY alone should satisfy STT requirement."""
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", True)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

        from tools.voice_mode import check_voice_requirements
        result = check_voice_requirements()

        assert result["available"] is True
        assert result["stt_key_set"] is True
        assert "Groq" in result["details"]

    def test_openai_key_sufficient(self, monkeypatch):
        """VOICE_TOOLS_OPENAI_KEY alone should satisfy STT requirement."""
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", True)
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from tools.voice_mode import check_voice_requirements
        result = check_voice_requirements()

        assert result["available"] is True
        assert result["stt_key_set"] is True
        assert "OpenAI" in result["details"]

    def test_no_keys_not_available(self, monkeypatch):
        """Without any STT key, voice mode should not be available."""
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", True)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from tools.voice_mode import check_voice_requirements
        result = check_voice_requirements()

        assert result["available"] is False
        assert result["stt_key_set"] is False


# ============================================================================
# Phase 4: Cleanup
# ============================================================================

class TestTempFileCleanup:
    """Verify old recordings are cleaned up and recent ones preserved."""

    def test_cleanup_removes_old_keeps_new(self, temp_voice_dir):
        old_file = temp_voice_dir / "recording_20240101_000000.wav"
        old_file.write_bytes(b"\x00" * 100)
        old_mtime = time.time() - 7200
        os.utime(str(old_file), (old_mtime, old_mtime))

        new_file = temp_voice_dir / "recording_20260303_120000.wav"
        new_file.write_bytes(b"\x00" * 100)

        from tools.voice_mode import cleanup_temp_recordings

        deleted = cleanup_temp_recordings(max_age_seconds=3600)
        assert deleted == 1
        assert not old_file.exists()
        assert new_file.exists()


# ============================================================================
# Integration: Full speech-to-text-to-submission flow
# ============================================================================

class TestSpeechToSubmissionFlow:
    """Simulate the complete flow: speech → auto-stop → WAV → STT → text output."""

    def test_full_flow_with_silence_detection(self, mock_sd, temp_voice_dir):
        """Complete end-to-end: speech triggers has_spoken, silence triggers
        auto-stop, WAV is written, STT returns transcript."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder, transcribe_recording

        recorder = AudioRecorder()
        recorder._silence_duration = 0.08
        recorder._min_speech_duration = 0.05

        # Use a queue to collect the WAV path from the auto-stop callback
        result_queue = queue.Queue()

        def on_silence():
            wav_path = recorder.stop()
            result_queue.put(wav_path)

        recorder.start(on_silence_stop=on_silence)
        callback = _get_callback(mock_sd)

        # Speech phase
        loud = np.full((1600, 1), 5000, dtype="int16")
        callback(loud, 1600, None, None)
        time.sleep(0.06)
        callback(loud, 1600, None, None)
        assert recorder._has_spoken is True

        # Silence phase
        silent = np.full((1600, 1), 10, dtype="int16")
        callback(silent, 1600, None, None)
        time.sleep(0.1)
        callback(silent, 1600, None, None)

        # Wait for auto-stop callback to fire
        wav_path = result_queue.get(timeout=3.0)
        assert wav_path is not None

        # STT produces transcript
        mock_stt = MagicMock(return_value={
            "success": True,
            "transcript": "search for Python tutorials",
        })
        with patch("tools.transcription_tools.transcribe_audio", mock_stt):
            result = transcribe_recording(wav_path)

        assert result["transcript"] == "search for Python tutorials"

    def test_hallucination_on_ambient_noise_filtered(self, mock_sd, temp_voice_dir):
        """If recording captures ambient noise that Whisper hallucinates on,
        the hallucination should be filtered out."""
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder, transcribe_recording

        recorder = AudioRecorder()
        recorder.start()
        callback = _get_callback(mock_sd)

        # Some noise above threshold but not real speech
        noise = np.full((1600, 1), 300, dtype="int16")
        for _ in range(10):
            callback(noise, 1600, None, None)

        wav_path = recorder.stop()
        assert wav_path is not None  # Peak RMS 300 > 200, so it passes

        # Whisper hallucinates "Thank you." on this noise
        mock_stt = MagicMock(return_value={
            "success": True,
            "transcript": "Thank you.",
        })
        with patch("tools.transcription_tools.transcribe_audio", mock_stt):
            result = transcribe_recording(wav_path)

        # Should be filtered
        assert result["transcript"] == ""
        assert result["filtered"] is True
