"""Tests for tools.voice_mode -- all mocked, no real microphone or API calls."""

import os
import struct
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal valid WAV file (1 second of silence at 16kHz)."""
    wav_path = tmp_path / "test.wav"
    n_frames = 16000  # 1 second at 16kHz
    silence = struct.pack(f"<{n_frames}h", *([0] * n_frames))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(silence)

    return str(wav_path)


@pytest.fixture
def temp_voice_dir(tmp_path, monkeypatch):
    """Redirect _TEMP_DIR to a temporary path."""
    voice_dir = tmp_path / "hermes_voice"
    voice_dir.mkdir()
    monkeypatch.setattr("tools.voice_mode._TEMP_DIR", str(voice_dir))
    return voice_dir


@pytest.fixture
def mock_sd(monkeypatch):
    """Replace tools.voice_mode.sd with a MagicMock (sounddevice may not be installed)."""
    mock = MagicMock()
    monkeypatch.setattr("tools.voice_mode.sd", mock)
    monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", True)
    # Also ensure numpy is available (use real numpy if installed, else mock)
    try:
        import numpy as real_np
        monkeypatch.setattr("tools.voice_mode.np", real_np)
    except ImportError:
        monkeypatch.setattr("tools.voice_mode.np", MagicMock())
    return mock


# ============================================================================
# check_voice_requirements
# ============================================================================

class TestCheckVoiceRequirements:
    def test_all_requirements_met(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", True)
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test-key")

        from tools.voice_mode import check_voice_requirements

        result = check_voice_requirements()
        assert result["available"] is True
        assert result["audio_available"] is True
        assert result["stt_key_set"] is True
        assert result["missing_packages"] == []

    def test_missing_audio_packages(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", False)
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test-key")

        from tools.voice_mode import check_voice_requirements

        result = check_voice_requirements()
        assert result["available"] is False
        assert result["audio_available"] is False
        assert "sounddevice" in result["missing_packages"]
        assert "numpy" in result["missing_packages"]

    def test_missing_stt_key(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", True)
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from tools.voice_mode import check_voice_requirements

        result = check_voice_requirements()
        assert result["available"] is False
        assert result["stt_key_set"] is False
        assert "STT API key: MISSING" in result["details"]


# ============================================================================
# AudioRecorder
# ============================================================================

class TestAudioRecorderStart:
    def test_start_raises_without_audio(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", False)

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        with pytest.raises(RuntimeError, match="sounddevice and numpy"):
            recorder.start()

    def test_start_creates_and_starts_stream(self, mock_sd):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()

        assert recorder.is_recording is True
        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_double_start_is_noop(self, mock_sd):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()
        recorder.start()  # second call should be noop

        assert mock_sd.InputStream.call_count == 1


class TestAudioRecorderStop:
    def test_stop_returns_none_when_not_recording(self):
        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        assert recorder.stop() is None

    def test_stop_writes_wav_file(self, mock_sd, temp_voice_dir):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder, SAMPLE_RATE

        recorder = AudioRecorder()
        recorder.start()

        # Simulate captured audio frames (1 second of loud audio above RMS threshold)
        frame = np.full((SAMPLE_RATE, 1), 1000, dtype="int16")
        recorder._frames = [frame]

        wav_path = recorder.stop()

        assert wav_path is not None
        assert os.path.isfile(wav_path)
        assert wav_path.endswith(".wav")
        assert recorder.is_recording is False

        # Verify it is a valid WAV
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE

    def test_stop_returns_none_for_very_short_recording(self, mock_sd, temp_voice_dir):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()

        # Very short recording (100 samples = ~6ms at 16kHz)
        frame = np.zeros((100, 1), dtype="int16")
        recorder._frames = [frame]

        wav_path = recorder.stop()
        assert wav_path is None

    def test_stop_returns_none_for_silent_recording(self, mock_sd, temp_voice_dir):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder, SAMPLE_RATE

        recorder = AudioRecorder()
        recorder.start()

        # 1 second of near-silence (RMS well below threshold)
        frame = np.full((SAMPLE_RATE, 1), 10, dtype="int16")
        recorder._frames = [frame]

        wav_path = recorder.stop()
        assert wav_path is None


class TestAudioRecorderCancel:
    def test_cancel_discards_frames(self, mock_sd):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()
        recorder._frames = [MagicMock()]  # simulate captured data

        recorder.cancel()

        assert recorder.is_recording is False
        assert recorder._frames == []
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_cancel_when_not_recording_is_safe(self):
        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.cancel()  # should not raise
        assert recorder.is_recording is False


class TestAudioRecorderProperties:
    def test_elapsed_seconds_when_not_recording(self):
        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        assert recorder.elapsed_seconds == 0.0

    def test_elapsed_seconds_when_recording(self, mock_sd):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()

        # Force start time to 1 second ago
        recorder._start_time = time.monotonic() - 1.0
        elapsed = recorder.elapsed_seconds
        assert 0.9 < elapsed < 2.0

        recorder.cancel()


# ============================================================================
# transcribe_recording
# ============================================================================

class TestTranscribeRecording:
    def test_delegates_to_transcribe_audio(self):
        mock_transcribe = MagicMock(return_value={
            "success": True,
            "transcript": "hello world",
        })

        with patch("tools.transcription_tools.transcribe_audio", mock_transcribe):
            from tools.voice_mode import transcribe_recording
            result = transcribe_recording("/tmp/test.wav", model="whisper-1")

        assert result["success"] is True
        assert result["transcript"] == "hello world"
        mock_transcribe.assert_called_once_with("/tmp/test.wav", model="whisper-1")

    def test_filters_whisper_hallucination(self):
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

    def test_does_not_filter_real_speech(self):
        mock_transcribe = MagicMock(return_value={
            "success": True,
            "transcript": "Thank you for helping me with this code.",
        })

        with patch("tools.transcription_tools.transcribe_audio", mock_transcribe):
            from tools.voice_mode import transcribe_recording
            result = transcribe_recording("/tmp/test.wav")

        assert result["transcript"] == "Thank you for helping me with this code."
        assert "filtered" not in result


class TestWhisperHallucinationFilter:
    def test_known_hallucinations(self):
        from tools.voice_mode import is_whisper_hallucination

        assert is_whisper_hallucination("Thank you.") is True
        assert is_whisper_hallucination("thank you") is True
        assert is_whisper_hallucination("Thanks for watching.") is True
        assert is_whisper_hallucination("Bye.") is True
        assert is_whisper_hallucination("  Thank you.  ") is True  # with whitespace
        assert is_whisper_hallucination("you") is True

    def test_real_speech_not_filtered(self):
        from tools.voice_mode import is_whisper_hallucination

        assert is_whisper_hallucination("Hello, how are you?") is False
        assert is_whisper_hallucination("Thank you for your help with the project.") is False
        assert is_whisper_hallucination("Can you explain this code?") is False


# ============================================================================
# play_audio_file
# ============================================================================

class TestPlayAudioFile:
    def test_play_wav_via_sounddevice(self, monkeypatch, sample_wav):
        np = pytest.importorskip("numpy")

        mock_sd = MagicMock()
        monkeypatch.setattr("tools.voice_mode.sd", mock_sd)
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", True)
        monkeypatch.setattr("tools.voice_mode.np", np)

        from tools.voice_mode import play_audio_file

        result = play_audio_file(sample_wav)

        assert result is True
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

    def test_returns_false_when_no_player(self, monkeypatch, sample_wav):
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", False)
        monkeypatch.setattr("shutil.which", lambda _: None)

        from tools.voice_mode import play_audio_file

        result = play_audio_file(sample_wav)
        assert result is False

    def test_returns_false_for_missing_file(self):
        from tools.voice_mode import play_audio_file

        result = play_audio_file("/nonexistent/file.wav")
        assert result is False


# ============================================================================
# cleanup_temp_recordings
# ============================================================================

class TestCleanupTempRecordings:
    def test_old_files_deleted(self, temp_voice_dir):
        # Create an "old" file
        old_file = temp_voice_dir / "recording_20240101_000000.wav"
        old_file.write_bytes(b"\x00" * 100)
        # Set mtime to 2 hours ago
        old_mtime = time.time() - 7200
        os.utime(str(old_file), (old_mtime, old_mtime))

        from tools.voice_mode import cleanup_temp_recordings

        deleted = cleanup_temp_recordings(max_age_seconds=3600)
        assert deleted == 1
        assert not old_file.exists()

    def test_recent_files_preserved(self, temp_voice_dir):
        # Create a "recent" file
        recent_file = temp_voice_dir / "recording_20260303_120000.wav"
        recent_file.write_bytes(b"\x00" * 100)

        from tools.voice_mode import cleanup_temp_recordings

        deleted = cleanup_temp_recordings(max_age_seconds=3600)
        assert deleted == 0
        assert recent_file.exists()

    def test_nonexistent_dir_returns_zero(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._TEMP_DIR", "/nonexistent/dir")

        from tools.voice_mode import cleanup_temp_recordings

        assert cleanup_temp_recordings() == 0

    def test_non_recording_files_ignored(self, temp_voice_dir):
        # Create a file that doesn't match the pattern
        other_file = temp_voice_dir / "other_file.txt"
        other_file.write_bytes(b"\x00" * 100)
        old_mtime = time.time() - 7200
        os.utime(str(other_file), (old_mtime, old_mtime))

        from tools.voice_mode import cleanup_temp_recordings

        deleted = cleanup_temp_recordings(max_age_seconds=3600)
        assert deleted == 0
        assert other_file.exists()


# ============================================================================
# play_beep
# ============================================================================

class TestPlayBeep:
    def test_beep_calls_sounddevice_play(self, mock_sd):
        np = pytest.importorskip("numpy")

        from tools.voice_mode import play_beep

        play_beep(frequency=880, duration=0.1, count=1)

        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()
        # Verify audio data is int16 numpy array
        audio_arg = mock_sd.play.call_args[0][0]
        assert audio_arg.dtype == np.int16
        assert len(audio_arg) > 0

    def test_beep_double_produces_longer_audio(self, mock_sd):
        np = pytest.importorskip("numpy")

        from tools.voice_mode import play_beep

        play_beep(frequency=660, duration=0.1, count=2)

        audio_arg = mock_sd.play.call_args[0][0]
        single_beep_samples = int(16000 * 0.1)
        # Double beep should be longer than a single beep
        assert len(audio_arg) > single_beep_samples

    def test_beep_noop_without_audio(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._HAS_AUDIO", False)

        from tools.voice_mode import play_beep

        # Should not raise
        play_beep()

    def test_beep_handles_playback_error(self, mock_sd):
        mock_sd.play.side_effect = Exception("device error")

        from tools.voice_mode import play_beep

        # Should not raise
        play_beep()


# ============================================================================
# Silence detection
# ============================================================================

class TestSilenceDetection:
    def test_silence_callback_fires_after_speech_then_silence(self, mock_sd):
        np = pytest.importorskip("numpy")
        import threading

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder, SAMPLE_RATE

        recorder = AudioRecorder()
        # Use very short silence duration for testing
        recorder._silence_duration = 0.05

        fired = threading.Event()

        def on_silence():
            fired.set()

        recorder.start(on_silence_stop=on_silence)

        # Get the callback function from InputStream constructor
        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        # Simulate loud audio (speech) -- RMS well above threshold
        loud_frame = np.full((1600, 1), 5000, dtype="int16")
        callback(loud_frame, 1600, None, None)
        assert recorder._has_spoken is True

        # Simulate silence
        silent_frame = np.zeros((1600, 1), dtype="int16")
        callback(silent_frame, 1600, None, None)

        # Wait a bit past the silence duration, then send another silent frame
        time.sleep(0.06)
        callback(silent_frame, 1600, None, None)

        # The callback should have been fired
        assert fired.wait(timeout=1.0) is True

        recorder.cancel()

    def test_silence_without_speech_does_not_fire(self, mock_sd):
        np = pytest.importorskip("numpy")
        import threading

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder._silence_duration = 0.02

        fired = threading.Event()
        recorder.start(on_silence_stop=lambda: fired.set())

        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        # Only silence -- no speech detected, so callback should NOT fire
        silent_frame = np.zeros((1600, 1), dtype="int16")
        for _ in range(5):
            callback(silent_frame, 1600, None, None)
            time.sleep(0.01)

        assert fired.wait(timeout=0.2) is False

        recorder.cancel()

    def test_no_callback_means_no_silence_detection(self, mock_sd):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()  # no on_silence_stop

        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        # Even with speech then silence, nothing should happen
        loud_frame = np.full((1600, 1), 5000, dtype="int16")
        silent_frame = np.zeros((1600, 1), dtype="int16")
        callback(loud_frame, 1600, None, None)
        callback(silent_frame, 1600, None, None)

        # No crash, no callback
        assert recorder._on_silence_stop is None
        recorder.cancel()
