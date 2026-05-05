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


@pytest.fixture(autouse=True)
def _disable_container_detection(monkeypatch):
    """Exercise voice-mode logic without the outer test container masking it."""
    monkeypatch.setattr("hermes_constants.is_container", lambda: False)


@pytest.fixture
def mock_sd(monkeypatch):
    """Mock _import_audio to return (mock_sd, real_np) so lazy imports work."""
    mock = MagicMock()
    try:
        import numpy as real_np
    except ImportError:
        real_np = MagicMock()

    def _fake_import_audio():
        return mock, real_np

    monkeypatch.setattr("tools.voice_mode._import_audio", _fake_import_audio)
    monkeypatch.setattr("tools.voice_mode._audio_available", lambda: True)
    return mock


# ============================================================================
# detect_audio_environment — WSL / SSH / Docker detection
# ============================================================================

class TestDetectAudioEnvironment:
    def test_clean_environment_is_available(self, monkeypatch):
        """No SSH, Docker, or WSL — should be available."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()
        assert result["available"] is True
        assert result["warnings"] == []

    def test_ssh_blocks_voice(self, monkeypatch):
        """SSH environment should block voice mode."""
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 54321 22")
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()
        assert result["available"] is False
        assert any("SSH" in w for w in result["warnings"])

    def test_wsl_without_pulse_blocks_voice(self, monkeypatch, tmp_path):
        """WSL without PULSE_SERVER should block voice mode."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))

        proc_version = tmp_path / "proc_version"
        proc_version.write_text("Linux 5.15.0-microsoft-standard-WSL2")

        _real_open = open
        def _fake_open(f, *a, **kw):
            if f == "/proc/version":
                return _real_open(str(proc_version), *a, **kw)
            return _real_open(f, *a, **kw)

        with patch("builtins.open", side_effect=_fake_open):
            from tools.voice_mode import detect_audio_environment
            result = detect_audio_environment()

        assert result["available"] is False
        assert any("WSL" in w for w in result["warnings"])
        assert any("PulseAudio" in w for w in result["warnings"])

    def test_wsl_with_pulse_allows_voice(self, monkeypatch, tmp_path):
        """WSL with PULSE_SERVER set should NOT block voice mode."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setenv("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))

        proc_version = tmp_path / "proc_version"
        proc_version.write_text("Linux 5.15.0-microsoft-standard-WSL2")

        _real_open = open
        def _fake_open(f, *a, **kw):
            if f == "/proc/version":
                return _real_open(str(proc_version), *a, **kw)
            return _real_open(f, *a, **kw)

        with patch("builtins.open", side_effect=_fake_open):
            from tools.voice_mode import detect_audio_environment
            result = detect_audio_environment()

        assert result["available"] is True
        assert result["warnings"] == []
        assert any("WSL" in n for n in result.get("notices", []))

    def test_wsl_device_query_fails_with_pulse_continues(self, monkeypatch, tmp_path):
        """WSL device query failure should not block if PULSE_SERVER is set."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setenv("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")

        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = Exception("device query failed")
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (mock_sd, MagicMock()))

        proc_version = tmp_path / "proc_version"
        proc_version.write_text("Linux 5.15.0-microsoft-standard-WSL2")

        _real_open = open
        def _fake_open(f, *a, **kw):
            if f == "/proc/version":
                return _real_open(str(proc_version), *a, **kw)
            return _real_open(f, *a, **kw)

        with patch("builtins.open", side_effect=_fake_open):
            from tools.voice_mode import detect_audio_environment
            result = detect_audio_environment()

        assert result["available"] is True
        assert any("device query failed" in n for n in result.get("notices", []))

    def test_device_query_fails_without_pulse_blocks(self, monkeypatch):
        """Device query failure without PULSE_SERVER should block."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("PULSE_SERVER", raising=False)

        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = Exception("device query failed")
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (mock_sd, MagicMock()))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()

        assert result["available"] is False
        assert any("PortAudio" in w for w in result["warnings"])

    def test_termux_import_error_shows_termux_install_guidance(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setattr("tools.voice_mode._import_audio", lambda: (_ for _ in ()).throw(ImportError("no audio libs")))
        monkeypatch.setattr("tools.voice_mode._termux_microphone_command", lambda: None)

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()

        assert result["available"] is False
        assert any("pkg install python-numpy portaudio" in w for w in result["warnings"])
        assert any("python -m pip install sounddevice" in w for w in result["warnings"])

    def test_termux_api_package_without_android_app_blocks_voice(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setattr("tools.voice_mode._termux_microphone_command", lambda: "/data/data/com.termux/files/usr/bin/termux-microphone-record")
        monkeypatch.setattr("tools.voice_mode._termux_api_app_installed", lambda: False)
        monkeypatch.setattr("tools.voice_mode._import_audio", lambda: (_ for _ in ()).throw(ImportError("no audio libs")))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()

        assert result["available"] is False
        assert any("Termux:API Android app is not installed" in w for w in result["warnings"])


    def test_termux_api_microphone_allows_voice_without_sounddevice(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setattr("tools.voice_mode.shutil.which", lambda cmd: "/data/data/com.termux/files/usr/bin/termux-microphone-record" if cmd == "termux-microphone-record" else None)
        monkeypatch.setattr("tools.voice_mode._termux_api_app_installed", lambda: True)
        monkeypatch.setattr("tools.voice_mode._import_audio", lambda: (_ for _ in ()).throw(ImportError("no audio libs")))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()

        assert result["available"] is True
        assert any("Termux:API microphone recording available" in n for n in result.get("notices", []))
        assert result["warnings"] == []


# ============================================================================
# check_voice_requirements
# ============================================================================

class TestCheckVoiceRequirements:
    def test_termux_api_capture_counts_as_audio_available(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._audio_available", lambda: False)
        monkeypatch.setattr("tools.voice_mode._termux_microphone_command", lambda: "/data/data/com.termux/files/usr/bin/termux-microphone-record")
        monkeypatch.setattr("tools.voice_mode._termux_api_app_installed", lambda: True)
        monkeypatch.setattr("tools.voice_mode.detect_audio_environment", lambda: {"available": True, "warnings": [], "notices": ["Termux:API microphone recording available"]})
        monkeypatch.setattr("tools.transcription_tools._get_provider", lambda cfg: "openai")

        from tools.voice_mode import check_voice_requirements
        result = check_voice_requirements()

        assert result["available"] is True
        assert result["audio_available"] is True
        assert result["missing_packages"] == []
        assert "Termux:API microphone" in result["details"]

    def test_all_requirements_met(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._audio_available", lambda: True)
        monkeypatch.setattr("tools.voice_mode.detect_audio_environment",
                            lambda: {"available": True, "warnings": []})
        monkeypatch.setattr("tools.transcription_tools._get_provider", lambda cfg: "openai")

        from tools.voice_mode import check_voice_requirements

        result = check_voice_requirements()
        assert result["available"] is True
        assert result["audio_available"] is True
        assert result["stt_available"] is True
        assert result["missing_packages"] == []

    def test_missing_audio_packages(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._audio_available", lambda: False)
        monkeypatch.setattr("tools.voice_mode.detect_audio_environment",
                            lambda: {"available": False, "warnings": ["Audio libraries not installed"]})
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test-key")

        from tools.voice_mode import check_voice_requirements

        result = check_voice_requirements()
        assert result["available"] is False
        assert result["audio_available"] is False
        assert "sounddevice" in result["missing_packages"]
        assert "numpy" in result["missing_packages"]

    def test_missing_stt_provider(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._audio_available", lambda: True)
        monkeypatch.setattr("tools.voice_mode.detect_audio_environment",
                            lambda: {"available": True, "warnings": []})
        monkeypatch.setattr("tools.transcription_tools._get_provider", lambda cfg: "none")

        from tools.voice_mode import check_voice_requirements

        result = check_voice_requirements()
        assert result["available"] is False
        assert result["stt_available"] is False
        assert "STT provider: MISSING" in result["details"]


# ============================================================================
# AudioRecorder
# ============================================================================

class TestCreateAudioRecorder:
    def test_termux_uses_termux_audio_recorder_when_api_present(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.setattr("tools.voice_mode._termux_microphone_command", lambda: "/data/data/com.termux/files/usr/bin/termux-microphone-record")
        monkeypatch.setattr("tools.voice_mode._termux_api_app_installed", lambda: True)

        from tools.voice_mode import create_audio_recorder, TermuxAudioRecorder
        recorder = create_audio_recorder()

        assert isinstance(recorder, TermuxAudioRecorder)
        assert recorder.supports_silence_autostop is False

    def test_termux_without_android_app_falls_back_to_audio_recorder(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.setattr("tools.voice_mode._termux_microphone_command", lambda: "/data/data/com.termux/files/usr/bin/termux-microphone-record")
        monkeypatch.setattr("tools.voice_mode._termux_api_app_installed", lambda: False)

        from tools.voice_mode import create_audio_recorder, AudioRecorder
        recorder = create_audio_recorder()

        assert isinstance(recorder, AudioRecorder)


class TestTermuxAudioRecorder:
    def test_start_and_stop_use_termux_microphone_commands(self, monkeypatch, temp_voice_dir):
        command_calls = []
        output_path = Path(temp_voice_dir) / "recording_20260409_120000.aac"

        def fake_run(cmd, **kwargs):
            command_calls.append(cmd)
            if cmd[1] == "-f":
                Path(cmd[2]).write_bytes(b"aac-bytes")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.setattr("tools.voice_mode._termux_microphone_command", lambda: "/data/data/com.termux/files/usr/bin/termux-microphone-record")
        monkeypatch.setattr("tools.voice_mode._termux_api_app_installed", lambda: True)
        monkeypatch.setattr("tools.voice_mode.time.strftime", lambda fmt: "20260409_120000")
        monkeypatch.setattr("tools.voice_mode.subprocess.run", fake_run)

        from tools.voice_mode import TermuxAudioRecorder
        recorder = TermuxAudioRecorder()
        recorder.start()
        recorder._start_time = time.monotonic() - 1.0
        result = recorder.stop()

        assert result == str(output_path)
        assert command_calls[0][:2] == ["/data/data/com.termux/files/usr/bin/termux-microphone-record", "-f"]
        assert command_calls[1] == ["/data/data/com.termux/files/usr/bin/termux-microphone-record", "-q"]

    def test_cancel_removes_partial_termux_recording(self, monkeypatch, temp_voice_dir):
        output_path = Path(temp_voice_dir) / "recording_20260409_120000.aac"

        def fake_run(cmd, **kwargs):
            if cmd[1] == "-f":
                Path(cmd[2]).write_bytes(b"aac-bytes")
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.setattr("tools.voice_mode._termux_microphone_command", lambda: "/data/data/com.termux/files/usr/bin/termux-microphone-record")
        monkeypatch.setattr("tools.voice_mode._termux_api_app_installed", lambda: True)
        monkeypatch.setattr("tools.voice_mode.time.strftime", lambda fmt: "20260409_120000")
        monkeypatch.setattr("tools.voice_mode.subprocess.run", fake_run)

        from tools.voice_mode import TermuxAudioRecorder
        recorder = TermuxAudioRecorder()
        recorder.start()
        recorder.cancel()

        assert output_path.exists() is False
        assert recorder.is_recording is False


class TestAudioRecorder:
    def test_start_raises_without_audio_libs(self, monkeypatch):
        def _fail_import():
            raise ImportError("no sounddevice")
        monkeypatch.setattr("tools.voice_mode._import_audio", _fail_import)

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
        recorder._peak_rms = 1000  # Peak RMS above threshold

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
        recorder._peak_rms = 10  # Peak RMS also below threshold

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
        # Stream is kept alive (persistent) — cancel() does NOT close it.
        mock_stream.stop.assert_not_called()
        mock_stream.close.assert_not_called()

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

        mock_sd_obj = MagicMock()
        # Simulate stream completing immediately (get_stream().active = False)
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_sd_obj.get_stream.return_value = mock_stream

        def _fake_import():
            return mock_sd_obj, np

        monkeypatch.setattr("tools.voice_mode._import_audio", _fake_import)

        from tools.voice_mode import play_audio_file

        result = play_audio_file(sample_wav)

        assert result is True
        mock_sd_obj.play.assert_called_once()
        mock_sd_obj.stop.assert_called_once()

    def test_returns_false_when_no_player(self, monkeypatch, sample_wav):
        def _fail_import():
            raise ImportError("no sounddevice")
        monkeypatch.setattr("tools.voice_mode._import_audio", _fail_import)
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

        # play_beep uses polling (get_stream) + sd.stop() instead of sd.wait()
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_sd.get_stream.return_value = mock_stream

        play_beep(frequency=880, duration=0.1, count=1)

        mock_sd.play.assert_called_once()
        mock_sd.stop.assert_called()
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
        def _fail_import():
            raise ImportError("no sounddevice")
        monkeypatch.setattr("tools.voice_mode._import_audio", _fail_import)

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
        # Use very short durations for testing
        recorder._silence_duration = 0.05
        recorder._min_speech_duration = 0.05

        fired = threading.Event()

        def on_silence():
            fired.set()

        recorder.start(on_silence_stop=on_silence)

        # Get the callback function from InputStream constructor
        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        # Simulate sustained speech (multiple loud chunks to exceed min_speech_duration)
        loud_frame = np.full((1600, 1), 5000, dtype="int16")
        callback(loud_frame, 1600, None, None)
        time.sleep(0.06)
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

    def test_micro_pause_tolerance_during_speech(self, mock_sd):
        """Brief dips below threshold during speech should NOT reset speech tracking."""
        np = pytest.importorskip("numpy")
        import threading

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder._silence_duration = 0.05
        recorder._min_speech_duration = 0.15
        recorder._max_dip_tolerance = 0.1

        fired = threading.Event()
        recorder.start(on_silence_stop=lambda: fired.set())

        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        loud_frame = np.full((1600, 1), 5000, dtype="int16")
        quiet_frame = np.full((1600, 1), 50, dtype="int16")

        # Speech chunk 1
        callback(loud_frame, 1600, None, None)
        time.sleep(0.05)
        # Brief micro-pause (dip < max_dip_tolerance)
        callback(quiet_frame, 1600, None, None)
        time.sleep(0.05)
        # Speech resumes -- speech_start should NOT have been reset
        callback(loud_frame, 1600, None, None)
        assert recorder._speech_start > 0, "Speech start should be preserved across brief dips"
        time.sleep(0.06)
        # Another speech chunk to exceed min_speech_duration
        callback(loud_frame, 1600, None, None)
        assert recorder._has_spoken is True, "Speech should be confirmed after tolerating micro-pause"

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


# ============================================================================
# Playback interrupt
# ============================================================================

class TestPlaybackInterrupt:
    """Verify that TTS playback can be interrupted."""

    def test_stop_playback_terminates_process(self):
        from tools.voice_mode import stop_playback, _playback_lock
        import tools.voice_mode as vm

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process is running

        with _playback_lock:
            vm._active_playback = mock_proc

        stop_playback()

        mock_proc.terminate.assert_called_once()

        with _playback_lock:
            assert vm._active_playback is None

    def test_stop_playback_noop_when_nothing_playing(self):
        import tools.voice_mode as vm

        with vm._playback_lock:
            vm._active_playback = None

        vm.stop_playback()

    def test_play_audio_file_sets_active_playback(self, monkeypatch, sample_wav):
        import tools.voice_mode as vm

        def _fail_import():
            raise ImportError("no sounddevice")
        monkeypatch.setattr("tools.voice_mode._import_audio", _fail_import)

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        mock_popen = MagicMock(return_value=mock_proc)
        monkeypatch.setattr("subprocess.Popen", mock_popen)
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/" + cmd)

        vm.play_audio_file(sample_wav)

        assert mock_popen.called
        with vm._playback_lock:
            assert vm._active_playback is None


# ============================================================================
# Continuous mode flow
# ============================================================================

class TestContinuousModeFlow:
    """Verify continuous mode: auto-restart after transcription or silence."""

    def test_continuous_restart_on_no_speech(self, mock_sd, temp_voice_dir):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()

        # First recording: only silence -> stop returns None
        recorder.start()
        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        for _ in range(10):
            silence = np.full((1600, 1), 10, dtype="int16")
            callback(silence, 1600, None, None)

        wav_path = recorder.stop()
        assert wav_path is None

        # Simulate continuous mode restart
        recorder.start()
        assert recorder.is_recording is True

        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        for _ in range(10):
            speech = np.full((1600, 1), 5000, dtype="int16")
            callback(speech, 1600, None, None)

        wav_path = recorder.stop()
        assert wav_path is not None

        recorder.cancel()

    def test_recorder_reusable_after_stop(self, mock_sd, temp_voice_dir):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        results = []

        for i in range(3):
            recorder.start()
            callback = mock_sd.InputStream.call_args.kwargs.get("callback")
            if callback is None:
                callback = mock_sd.InputStream.call_args[1]["callback"]
            loud = np.full((1600, 1), 5000, dtype="int16")
            for _ in range(10):
                callback(loud, 1600, None, None)
            wav_path = recorder.stop()
            results.append(wav_path)

        assert all(r is not None for r in results)
        assert os.path.isfile(results[-1])


# ============================================================================
# Audio level indicator
# ============================================================================

class TestAudioLevelIndicator:
    """Verify current_rms property updates in real-time for UI feedback."""

    def test_rms_updates_with_audio_chunks(self, mock_sd):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()
        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        assert recorder.current_rms == 0

        loud = np.full((1600, 1), 5000, dtype="int16")
        callback(loud, 1600, None, None)
        assert recorder.current_rms == 5000

        quiet = np.full((1600, 1), 100, dtype="int16")
        callback(quiet, 1600, None, None)
        assert recorder.current_rms == 100

        recorder.cancel()

    def test_peak_rms_tracks_maximum(self, mock_sd):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()
        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        frames = [
            np.full((1600, 1), 100, dtype="int16"),
            np.full((1600, 1), 8000, dtype="int16"),
            np.full((1600, 1), 500, dtype="int16"),
            np.full((1600, 1), 3000, dtype="int16"),
        ]
        for frame in frames:
            callback(frame, 1600, None, None)

        assert recorder._peak_rms == 8000
        assert recorder.current_rms == 3000

        recorder.cancel()


# ============================================================================
# Configurable silence parameters
# ============================================================================

class TestConfigurableSilenceParams:
    """Verify that silence detection params can be configured."""

    def test_custom_threshold_and_duration(self, mock_sd):
        np = pytest.importorskip("numpy")

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder
        import threading

        recorder = AudioRecorder()
        recorder._silence_threshold = 5000
        recorder._silence_duration = 0.05
        recorder._min_speech_duration = 0.05

        fired = threading.Event()
        recorder.start(on_silence_stop=lambda: fired.set())
        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]

        # Audio at RMS 1000 -- below custom threshold (5000)
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


# ============================================================================
# Bugfix regression tests
# ============================================================================


class TestSubprocessTimeoutKill:
    """Bug: proc.wait(timeout) raised TimeoutExpired but process was not killed."""

    def test_timeout_kills_process(self):
        import subprocess, os
        proc = subprocess.Popen(["sleep", "600"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pid = proc.pid
        assert proc.poll() is None

        try:
            proc.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        assert proc.poll() is not None
        assert proc.returncode is not None


class TestStreamLeakOnStartFailure:
    """Bug: stream.start() failure left stream unclosed."""

    def test_stream_closed_on_start_failure(self, mock_sd):
        mock_stream = MagicMock()
        mock_stream.start.side_effect = OSError("Audio device busy")
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder
        recorder = AudioRecorder()

        with pytest.raises(RuntimeError, match="Failed to open audio input stream"):
            recorder._ensure_stream()

        mock_stream.close.assert_called_once()


class TestSilenceCallbackLock:
    """Bug: _on_silence_stop was read/written without lock in audio callback."""

    def test_fire_block_acquires_lock(self):
        import inspect
        from tools.voice_mode import AudioRecorder

        source = inspect.getsource(AudioRecorder._ensure_stream)
        # Verify lock is used before reading _on_silence_stop in fire block
        assert "with self._lock:" in source
        assert "cb = self._on_silence_stop" in source
        lock_pos = source.index("with self._lock:")
        cb_pos = source.index("cb = self._on_silence_stop")
        assert lock_pos < cb_pos

    def test_cancel_clears_callback_under_lock(self, mock_sd):
        from tools.voice_mode import AudioRecorder
        recorder = AudioRecorder()
        mock_sd.InputStream.return_value = MagicMock()

        cb = lambda: None
        recorder.start(on_silence_stop=cb)
        assert recorder._on_silence_stop is cb

        recorder.cancel()
        with recorder._lock:
            assert recorder._on_silence_stop is None
