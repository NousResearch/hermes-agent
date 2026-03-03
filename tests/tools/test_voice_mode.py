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

        # Simulate captured audio frames (1 second of silence)
        frame = np.zeros((SAMPLE_RATE, 1), dtype="int16")
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
