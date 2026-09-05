"""Tests for tools.voice_mode -- all mocked, no real microphone or API calls."""

import os
import struct
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _non_wsl_proc_version(real_open):
    """Return an open() shim that makes host WSL detection deterministic."""
    def _fake_open(file, *args, **kwargs):
        if file == "/proc/version":
            from io import StringIO

            return StringIO("Linux test-kernel")
        return real_open(file, *args, **kwargs)

    return _fake_open


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


class _FakeTime:
    """Stand-in for the ``time`` module with a monotonic clock the test drives.

    Silence detection compares ``time.monotonic()`` deltas against thresholds
    of a few dozen milliseconds.  Driving those deltas with real ``sleep()``
    calls only works when the platform clock is finer-grained than the margin
    the test leaves: ``time.monotonic()`` is ``GetTickCount64()`` (15.625 ms
    resolution) on Windows until CPython 3.13 moved it to
    ``QueryPerformanceCounter()``, so a 60 ms sleep can legitimately measure
    as 46 ms and land under a 50 ms threshold.  Advancing an explicit clock
    keeps the arithmetic exact on every platform.

    Everything other than ``monotonic`` delegates to the real module, so
    ``time.sleep``/``time.strftime`` in the code under test keep working.
    """

    def __init__(self, real_time, start: float = 1000.0) -> None:
        self._real = real_time
        self._now = start

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def fake_clock(monkeypatch):
    """Give voice_mode a hand-driven clock.

    Patches the name ``time`` inside ``tools.voice_mode`` rather than
    ``time.monotonic`` itself -- ``voice_mode.time`` *is* the stdlib module,
    so setting the attribute on it would swap the clock out from under every
    other importer for the duration of the test.
    """
    import time as real_time

    import tools.voice_mode as voice_mode

    clock = _FakeTime(real_time)
    monkeypatch.setattr(voice_mode, "time", clock)
    return clock


# ============================================================================
# detect_audio_environment — WSL / SSH / Docker detection
# ============================================================================

class TestPulseSocketReachable:
    def test_stale_socket_file_not_reachable(self, monkeypatch, tmp_path):
        """A socket file with no listener should not count as reachable."""
        import socket as _socket
        sock_path = tmp_path / "pulse" / "native"
        sock_path.parent.mkdir(parents=True)
        # Create + bind, then close so the path is a stale socket file.
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.bind(str(sock_path))
        s.close()
        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.delenv("PULSE_RUNTIME_PATH", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        from tools.voice_mode import _pulse_socket_reachable
        assert _pulse_socket_reachable() is False

    def test_listening_socket_reachable_via_xdg_runtime(self, monkeypatch, tmp_path):
        """A live PulseAudio-style socket under XDG_RUNTIME_DIR is reachable (#35622)."""
        import socket as _socket
        sock_path = tmp_path / "pulse" / "native"
        sock_path.parent.mkdir(parents=True)
        server = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(1)
        try:
            monkeypatch.delenv("PULSE_SERVER", raising=False)
            monkeypatch.delenv("PULSE_RUNTIME_PATH", raising=False)
            monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
            from tools.voice_mode import _pulse_socket_reachable
            assert _pulse_socket_reachable() is True
        finally:
            server.close()

class TestDetectAudioEnvironment:
    def test_clean_environment_is_available(self, monkeypatch):
        """No SSH, Docker, or WSL — should be available."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setattr("hermes_constants.is_container", lambda: False)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))
        monkeypatch.setattr("builtins.open", _non_wsl_proc_version(open))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()
        assert result["available"] is True
        assert result["warnings"] == []

    def test_ssh_blocks_voice(self, monkeypatch):
        """SSH environment without a reachable sound server should block voice mode."""
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 54321 22")
        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.delenv("PIPEWIRE_REMOTE", raising=False)
        monkeypatch.setattr("tools.voice_mode._pulse_socket_reachable", lambda: False)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()
        assert result["available"] is False
        assert any("SSH" in w for w in result["warnings"])

    def test_ssh_with_pulse_server_allows_voice(self, monkeypatch):
        """SSH with PULSE_SERVER set should NOT block voice mode (#35622)."""
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 54321 22")
        monkeypatch.setenv("PULSE_SERVER", "unix:/run/user/1002/pulse/native")
        monkeypatch.delenv("PIPEWIRE_REMOTE", raising=False)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))
        monkeypatch.setattr("builtins.open", _non_wsl_proc_version(open))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()
        assert result["available"] is True
        assert result["warnings"] == []
        assert any("SSH" in n for n in result.get("notices", []))

    def test_wsl_without_pulse_blocks_voice(self, monkeypatch, tmp_path):
        """WSL without PULSE_SERVER should block voice mode."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.setattr("tools.voice_mode._pulse_socket_reachable", lambda: False)
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

    def test_docker_with_pipewire_remote_and_no_devices_allows_voice(self, monkeypatch):
        """PIPEWIRE_REMOTE should bypass empty PortAudio device lists in Docker."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.setenv("PIPEWIRE_REMOTE", "/run/user/1000/pipewire-0")
        monkeypatch.setattr("hermes_constants.is_container", lambda: True)

        sd = MagicMock()
        sd.query_devices.return_value = []
        monkeypatch.setattr("tools.voice_mode._import_audio", lambda: (sd, MagicMock()))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()

        assert result["available"] is True
        assert result["warnings"] == []
        assert any("host audio forwarding" in n.lower() for n in result.get("notices", []))

    def test_docker_without_audio_forwarding_blocks_voice(self, monkeypatch):
        """Docker without PULSE_SERVER/PIPEWIRE_REMOTE keeps blocking voice mode."""
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.delenv("PIPEWIRE_REMOTE", raising=False)
        monkeypatch.setattr("tools.voice_mode._pulse_socket_reachable", lambda: False)
        monkeypatch.setattr("hermes_constants.is_container", lambda: True)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))

        from tools.voice_mode import detect_audio_environment
        result = detect_audio_environment()

        assert result["available"] is False
        assert any("container" in w.lower() for w in result["warnings"])
        assert any("PULSE_SERVER" in w or "PIPEWIRE_REMOTE" in w for w in result["warnings"])

# ============================================================================
# check_voice_requirements
# ============================================================================

class TestCheckVoiceRequirements:
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


    def test_plugin_stt_provider(self, monkeypatch):
        """Plugin STT provider is recognized."""
        monkeypatch.setattr("tools.voice_mode._audio_available", lambda: True)
        monkeypatch.setattr("tools.voice_mode.detect_audio_environment",
                            lambda: {"available": True, "warnings": []})
        monkeypatch.setattr(
            "tools.transcription_tools._load_stt_config",
            lambda: {"enabled": True, "provider": "my-plugin-stt"},
        )
        plugin_provider = MagicMock()
        plugin_provider.is_available.return_value = True
        monkeypatch.setattr(
            "agent.transcription_registry.get_provider",
            lambda p: plugin_provider if p == "my-plugin-stt" else None,
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._ensure_plugins_discovered",
            lambda force=False: None,
        )

        from tools.voice_mode import check_voice_requirements

        result = check_voice_requirements()
        assert result["available"] is True
        assert result["stt_available"] is True
        assert "STT provider: OK (plugin: my-plugin-stt)" in result["details"]

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

    def test_start_oserror_points_at_portaudio_not_pip(self, monkeypatch):
        """OSError from _import_audio means PortAudio's shared library is
        missing — pip can't fix that. The error must point at the system
        package, not 'pip install sounddevice numpy' (#18432)."""
        def _fail_import():
            raise OSError("PortAudio library not found")
        monkeypatch.setattr("tools.voice_mode._import_audio", _fail_import)
        monkeypatch.setattr("tools.voice_mode._is_termux_environment", lambda: False)

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        with pytest.raises(RuntimeError) as exc_info:
            recorder.start()
        msg = str(exc_info.value)
        assert "PortAudio system library not found" in msg
        assert "libportaudio2" in msg
        assert "pip install" not in msg

    def test_start_creates_and_starts_stream(self, mock_sd):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()

        assert recorder.is_recording is True
        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()

class TestAudioRecorderStop:
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

# ============================================================================
# transcribe_recording
# ============================================================================

class TestTranscribeRecording:
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


    def test_other_error_does_not_trigger_chunk(self, tmp_path, monkeypatch):
        """Non-size errors from transcribe_audio are returned as-is."""
        wav_path = tmp_path / "record.wav"
        n_frames = 50000
        audio = struct.pack(f"<{n_frames}h", *([1000] * n_frames))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio)

        mock_transcribe = MagicMock(return_value={
            "success": False,
            "transcript": "",
            "error": "STT is disabled in config.yaml",
        })

        with patch("tools.transcription_tools.transcribe_audio", mock_transcribe):
            from tools.voice_mode import transcribe_recording
            result = transcribe_recording(str(wav_path), model="base")

        assert result["success"] is False
        assert "STT is disabled" in result["error"]
        mock_transcribe.assert_called_once()


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
    @pytest.mark.linux_only
    def test_play_wav_via_sounddevice(self, monkeypatch, sample_wav):
        np = pytest.importorskip("numpy")
        # Linux-gated rather than faking a non-macOS platform: on macOS WAV
        # output deliberately skips sounddevice (see
        # TestMacOSAudioOutputPolicy), so this path is only exercised off
        # Darwin and the host now selects it by itself.

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

# ============================================================================
# macOS output policy (no sounddevice for OUTPUT -> avoids TCC prompt)
# ============================================================================

class TestMacOSAudioOutputPolicy:
    """macOS-gated: the policy exists because PortAudio/CoreAudio init raises
    a TCC media-library prompt, which no faked platform on Linux reproduces —
    and `afplay` only resolves on a real macOS host."""

    @pytest.mark.macos_only
    def test_play_audio_file_skips_sounddevice_on_macos(self, monkeypatch, sample_wav):
        """On macOS, WAV playback must not import sounddevice; it routes to afplay."""

        def _forbidden_import():
            raise AssertionError("sounddevice must not be imported for output on macOS")

        monkeypatch.setattr("tools.voice_mode._import_audio", _forbidden_import)

        popen_cmds = []

        class _FakeProc:
            returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        def _fake_popen(cmd, **kwargs):
            popen_cmds.append(cmd)
            return _FakeProc()

        # Only Popen is stubbed: the host resolves afplay for real, so the
        # argv assertion below reflects real player selection.
        monkeypatch.setattr("subprocess.Popen", _fake_popen)

        from tools.voice_mode import play_audio_file

        result = play_audio_file(sample_wav)

        assert result is True
        assert popen_cmds, "expected a system player to be invoked"
        assert popen_cmds[0][0] == "afplay"

    @pytest.mark.macos_only
    def test_play_beep_routes_through_afplay_on_macos(self, monkeypatch):
        """On macOS, beeps synthesize with numpy but play via the tempfile/afplay path."""
        pytest.importorskip("numpy")

        def _forbidden_import():
            raise AssertionError("sounddevice must not be imported for beeps on macOS")

        monkeypatch.setattr("tools.voice_mode._import_audio", _forbidden_import)

        calls = []
        monkeypatch.setattr(
            "tools.voice_mode._play_int16_via_tempfile",
            lambda audio, sample_rate: calls.append((len(audio), sample_rate)),
        )

        import tools.voice_mode as vm

        vm.play_beep(frequency=880, count=1)

        assert len(calls) == 1
        n_samples, sample_rate = calls[0]
        assert n_samples > 0
        assert sample_rate == vm.SAMPLE_RATE

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

# ============================================================================
# Silence detection
# ============================================================================

class TestSilenceDetection:
    def test_silence_callback_fires_after_speech_then_silence(self, mock_sd, fake_clock):
        np = pytest.importorskip("numpy")
        import threading

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

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
        fake_clock.advance(0.06)
        callback(loud_frame, 1600, None, None)
        assert recorder._has_spoken is True

        # Simulate silence
        silent_frame = np.zeros((1600, 1), dtype="int16")
        callback(silent_frame, 1600, None, None)

        # Move past the silence duration, then send another silent frame
        fake_clock.advance(0.06)
        callback(silent_frame, 1600, None, None)

        # The callback should have been fired (it runs on a real thread, so
        # this wait is the one place real time is still involved)
        assert fired.wait(timeout=5.0) is True

        recorder.cancel()

    def test_micro_pause_tolerance_during_speech(self, mock_sd, fake_clock):
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
        fake_clock.advance(0.05)
        # Brief micro-pause (dip < max_dip_tolerance)
        callback(quiet_frame, 1600, None, None)
        fake_clock.advance(0.05)
        # Speech resumes -- speech_start should NOT have been reset
        callback(loud_frame, 1600, None, None)
        assert recorder._speech_start > 0, "Speech start should be preserved across brief dips"
        fake_clock.advance(0.06)
        # Another speech chunk to exceed min_speech_duration
        callback(loud_frame, 1600, None, None)
        assert recorder._has_spoken is True, "Speech should be confirmed after tolerating micro-pause"

        recorder.cancel()

# ============================================================================
# Max recording length cap (voice.max_recording_seconds)
# ============================================================================

class TestMaxRecordingCap:
    """The hard cap must auto-stop through the real InputStream-callback
    path — not just the predicate — and fire the one-shot callback exactly
    once, independent of the silence-detection branches."""

    def _get_stream_callback(self, mock_sd):
        callback = mock_sd.InputStream.call_args.kwargs.get("callback")
        if callback is None:
            callback = mock_sd.InputStream.call_args[1]["callback"]
        return callback

    def test_cap_fires_one_shot_callback_during_continuous_speech(self, mock_sd, fake_clock):
        np = pytest.importorskip("numpy")
        import threading

        mock_sd.InputStream.return_value = MagicMock()

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder._max_recording_seconds = 0.1
        # Park the other auto-stop branches far away so only the cap can fire:
        # loud frames keep the silence branch off, and max_wait covers the
        # no-speech branch.
        recorder._silence_duration = 60.0
        recorder._max_wait = 60.0

        fires = []
        fired = threading.Event()

        def on_stop():
            fires.append(1)
            fired.set()

        recorder.start(on_silence_stop=on_stop)
        callback = self._get_stream_callback(mock_sd)

        loud_frame = np.full((1600, 1), 5000, dtype="int16")
        callback(loud_frame, 1600, None, None)
        assert not fired.is_set(), "cap must not fire before the limit elapses"

        # Cross the cap while the user is STILL speaking — the silence branch
        # can never fire here, so a hit proves the cap path.
        fake_clock.advance(0.12)
        callback(loud_frame, 1600, None, None)
        assert fired.wait(timeout=5.0) is True

        # One-shot: the handler cleared _on_silence_stop, further frames past
        # the cap must not fire again — with the callback cleared, no further
        # notifier thread is even spawned, so this needs no settle time.
        assert recorder._on_silence_stop is None
        callback(loud_frame, 1600, None, None)
        assert len(fires) == 1

        recorder.cancel()

    def test_disabled_cap_never_fires_on_duration(self, mock_sd, fake_clock):
        np = pytest.importorskip("numpy")
        import threading

        mock_sd.InputStream.return_value = MagicMock()

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder._max_recording_seconds = 0.0  # disabled (previous behaviour)
        recorder._silence_duration = 60.0
        recorder._max_wait = 60.0

        fired = threading.Event()
        recorder.start(on_silence_stop=lambda: fired.set())
        callback = self._get_stream_callback(mock_sd)

        loud_frame = np.full((1600, 1), 5000, dtype="int16")
        callback(loud_frame, 1600, None, None)
        fake_clock.advance(0.12)
        callback(loud_frame, 1600, None, None)

        assert fired.wait(timeout=0.1) is False
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

# ============================================================================
# Audio level indicator
# ============================================================================

class TestAudioLevelIndicator:
    """Verify current_rms property updates in real-time for UI feedback."""

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

    def test_custom_threshold_and_duration(self, mock_sd, fake_clock):
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
            fake_clock.advance(0.02)

        assert recorder._has_spoken is False
        assert fired.wait(timeout=0.2) is False

        # Now send really loud audio (above 5000 threshold)
        very_loud = np.full((1600, 1), 8000, dtype="int16")
        callback(very_loud, 1600, None, None)
        fake_clock.advance(0.06)
        callback(very_loud, 1600, None, None)
        assert recorder._has_spoken is True

        recorder.cancel()


# ============================================================================
# Bugfix regression tests
# ============================================================================


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


# ============================================================================
# listen_for_speech — VAD barge-in monitor
# ============================================================================

class _FakeInputStream:
    """Context-manager InputStream serving a fixed sequence of RMS levels."""

    def __init__(self, np, levels):
        self._np = np
        self._levels = list(levels)
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, frames):
        level = self._levels[min(self.reads, len(self._levels) - 1)]
        self.reads += 1
        return self._np.full((frames, 1), level, dtype=self._np.int16), False


class TestListenForSpeech:
    """listen_for_speech: calibration → sustained-speech trigger → barge-in."""

    CALIB_BLOCKS = 14   # 400ms / 30ms
    TRIP_BLOCKS = 10    # 300ms / 30ms

    def _run(self, mock_sd, levels, should_stop=None, **kwargs):
        np = pytest.importorskip("numpy")
        stream = _FakeInputStream(np, levels)
        mock_sd.InputStream.return_value = stream
        from tools.voice_mode import listen_for_speech
        stops = iter([False] * 200 + [True] * 10_000)
        return listen_for_speech(should_stop or (lambda: next(stops)), **kwargs), stream

    def test_sustained_speech_triggers(self, mock_sd):
        levels = [0] * self.CALIB_BLOCKS + [5000] * 50
        heard, _ = self._run(mock_sd, levels)
        assert heard is True


    def test_returns_false_when_audio_unavailable(self, monkeypatch):
        monkeypatch.setattr("tools.voice_mode._import_audio", MagicMock(side_effect=OSError("no audio")))
        from tools.voice_mode import listen_for_speech
        assert listen_for_speech(lambda: False) is False

    def test_quiet_then_loud_playback_does_not_trip(self, mock_sd):
        """TTS that starts quiet and gets louder must NOT trip barge-in.

        This is the core regression: a one-shot calibration freezes the
        floor from the quiet opening, then louder TTS exceeds the stale
        floor and false-triggers.  The rolling window keeps the floor
        current so the louder passage is absorbed into the floor.
        """
        levels = [100] * self.CALIB_BLOCKS + [200] * 30 + [500] * 30 + [1000] * 30
        heard, _ = self._run(mock_sd, levels)
        assert heard is False

    def test_silence_calibration_does_not_false_trip_on_tts(self, mock_sd):
        """Calibration during an inter-sentence gap must NOT false-trip.

        If the grace period ends during a pause between TTS sentences, the
        calibration window samples near-silence.  Without the min_floor clamp,
        min_floor locks near zero, the trigger drops to 400 RMS (SILENCE_RMS_THRESHOLD
        * 2), and the next TTS sentence at 800 RMS exceeds it — those blocks are
        excluded from the rolling window (rms >= trigger), the floor freezes, and
        after sustained_ms the VAD false-triggers and cuts playback mid-sentence.

        With the clamp, min_floor stays at SILENCE_RMS_THRESHOLD * 2 = 400, the
        trigger is max(400, 400 * 8.0) = 3200, and 800-RMS TTS stays below it and
        feeds the rolling floor.  No false trip.
        """
        # calibration_ms=800 → CALIB_BLOCKS = 800/30 ≈ 26 blocks of silence
        # Then TTS resumes at 800 RMS — must NOT trip (below 3200 trigger).
        calib = 800 // 30
        levels = [0] * calib + [800] * 100
        heard, _ = self._run(
            mock_sd, levels,
            sustained_ms=1000,
            calibration_ms=800,
        )
        assert heard is False


class TestListenForSpeechCapture:
    """capture=True: the barge monitor records the interruption with pre-roll,
    so the utterance is complete from its first syllable — nothing is lost
    between detection and a recorder restart."""

    CALIB_BLOCKS = 14   # 400ms / 30ms
    LOUD_BLOCKS = 30    # speech: trips after 10, keeps talking
    BLOCK = 480         # 16000 * 0.03

    def _run(self, mock_sd, monkeypatch, levels, should_stop=None, **kwargs):
        np = pytest.importorskip("numpy")
        stream = _FakeInputStream(np, levels)
        mock_sd.InputStream.return_value = stream
        written = {}
        monkeypatch.setattr(
            "tools.voice_mode.AudioRecorder._write_wav",
            staticmethod(lambda audio, **kw: written.update(audio=audio, **kw) or "/tmp/barge.wav"),
        )
        from tools.voice_mode import listen_for_speech
        stops = iter([False] * 200 + [True] * 10_000)
        path = listen_for_speech(
            should_stop or (lambda: next(stops)), capture=True, **kwargs
        )
        return path, written.get("audio"), stream

    def test_captured_utterance_includes_speech_onset(self, mock_sd, monkeypatch):
        """Every loud block — including the ones BEFORE detection tripped —
        must land in the WAV. That pre-roll is the whole point."""
        triggered = []
        levels = [0] * self.CALIB_BLOCKS + [5000] * self.LOUD_BLOCKS + [0] * 500
        path, audio, _ = self._run(
            mock_sd, monkeypatch, levels,
            should_stop=lambda: False,
            on_trigger=lambda: triggered.append(True),
        )
        assert path == "/tmp/barge.wav"
        assert triggered == [True]
        assert int((audio == 5000).sum()) == self.LOUD_BLOCKS * self.BLOCK

    def test_no_trip_returns_none(self, mock_sd, monkeypatch):
        triggered = []
        path, audio, _ = self._run(
            mock_sd, monkeypatch, [0] * 500,
            on_trigger=lambda: triggered.append(True),
        )
        assert path is None
        assert audio is None
        assert triggered == []

class TestFullDuplexListen:
    """full_duplex_listen: one agent-turn listener spanning generation and
    playback — pre-playback calibration, phase-aware trigger, grace window."""

    CALIB = 15       # 450ms / 30ms — pre-playback quiet calibration
    TRIP = 10        # 300ms / 30ms window; trip needs >=8 above
    GRACE = 16       # 500ms / 30ms
    BLOCK = 480      # 30ms at 16 kHz

    def _run(self, mock_sd, monkeypatch, levels, playing_from=None,
             playing_until=None, should_stop=None, on_trigger=None, **kwargs):
        np = pytest.importorskip("numpy")
        stream = _FakeInputStream(np, levels)
        mock_sd.InputStream.return_value = stream
        written = {}
        monkeypatch.setattr(
            "tools.voice_mode.AudioRecorder._write_wav",
            staticmethod(lambda audio, **kw: written.update(audio=audio, **kw) or "/tmp/fd.wav"),
        )
        from tools.voice_mode import full_duplex_listen

        def is_playing():
            if playing_from is None:
                return False
            if stream.reads < playing_from:
                return False
            if playing_until is not None and stream.reads >= playing_until:
                return False
            return True

        stops = iter([False] * len(levels) + [True] * 10_000)
        path = full_duplex_listen(
            should_stop or (lambda: next(stops)),
            is_playing=is_playing,
            on_trigger=on_trigger,
            **kwargs,
        )
        return path, written.get("audio"), stream

    def test_generation_phase_speech_trips_and_captures(self, mock_sd, monkeypatch):
        """Speech while the LLM generates (no playback) trips with
        phase='generation' and the utterance is captured with pre-roll."""
        phases = []
        levels = [100] * self.CALIB + [5000] * 30 + [0] * 500
        path, audio, _ = self._run(
            mock_sd, monkeypatch, levels,
            on_trigger=lambda phase: phases.append(phase),
        )
        assert path == "/tmp/fd.wav"
        assert phases == ["generation"]
        assert int((audio == 5000).sum()) > 0  # speech onset in the capture

    def test_playback_bleed_alone_does_not_trip(self, mock_sd, monkeypatch):
        """Speaker bleed (~1000 RMS) during playback stays below the playback
        trigger clamp — the old monitor's self-calibration deafness class."""
        phases = []
        levels = [100] * self.CALIB + [1000] * 300
        path, _, _ = self._run(
            mock_sd, monkeypatch, levels,
            playing_from=self.CALIB,
            on_trigger=lambda phase: phases.append(phase),
        )
        assert path is None
        assert phases == []

    def test_speech_over_bleed_trips_in_playback_phase(self, mock_sd, monkeypatch):
        """Real speech (5000 RMS) over playback trips with phase='playback'
        even though playback bleed was present — the trigger comes from the
        PRE-playback quiet floor, never from bleed self-calibration."""
        phases = []
        # quiet calib → playback bleed (past grace) → user speaks over it
        levels = (
            [100] * self.CALIB
            + [1000] * (self.GRACE + 20)
            + [5000] * 30
            + [1000] * 200
        )
        path, _, _ = self._run(
            mock_sd, monkeypatch, levels,
            playing_from=self.CALIB,
            on_trigger=lambda phase: phases.append(phase),
        )
        assert path == "/tmp/fd.wav"
        assert phases == ["playback"]

    def test_grace_window_suppresses_playback_onset(self, mock_sd, monkeypatch):
        """Loud blocks inside the 0.5s grace right after playback starts are
        suppressed (onset transient), but speech after grace still trips."""
        phases = []
        # loud transient fully inside grace, then quiet bleed, then speech
        levels = (
            [100] * self.CALIB
            + [5000] * 8            # onset transient (inside 16-block grace)
            + [800] * 40
            + [5000] * 30
            + [0] * 200
        )

        def trig(phase):
            phases.append(phase)

        path, _, stream = self._run(
            mock_sd, monkeypatch, levels,
            playing_from=self.CALIB,
            on_trigger=trig,
        )
        assert path == "/tmp/fd.wav"
        assert phases == ["playback"]
        # Trip must come from the post-grace speech, not the onset transient:
        # by the time capture starts, we're past calib+transient+bleed blocks.
        assert stream.reads > self.CALIB + 8 + 40

class TestGetBeepVolume:
    """Issue #55908: beep amplitude must come from config.yaml, with safe fallback."""

    def _get(self):
        from tools.voice_mode import _get_beep_volume
        return _get_beep_volume()

    @pytest.mark.parametrize("config,expected", [
        ({"voice": {}}, 0.3),                              # unset -> default
        ({"voice": {"beep_volume": 0.0}}, 0.0),            # boundary, honored
        ({"voice": {"beep_volume": 1.5}}, 0.3),            # out of range -> default
        ({"voice": {"beep_volume": "0.7"}}, 0.7),          # numeric string coerced
        ({"voice": {"beep_volume": True}}, 0.3),           # bool is not a volume
    ])
    def test_config_value_resolution(self, config, expected):
        with patch("hermes_cli.config.load_config", return_value=config):
            assert self._get() == expected

    def test_load_config_exception_falls_back(self):
        with patch("hermes_cli.config.load_config",
                   side_effect=RuntimeError("broken config")):
            assert self._get() == 0.3

# ============================================================================
# Device-native input sample rate — mics that reject 16 kHz capture
# ============================================================================

class TestDefaultInputSamplerate:
    def test_uses_device_default_rate(self):
        from tools.voice_mode import _default_input_samplerate

        sd = MagicMock()
        sd.query_devices.return_value = {"default_samplerate": 44100.0}
        assert _default_input_samplerate(sd) == 44100


    def test_wav_written_at_capture_rate(self, mock_sd, temp_voice_dir):
        np = pytest.importorskip("numpy")

        mock_sd.query_devices.return_value = {"default_samplerate": 48000.0}
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        from tools.voice_mode import AudioRecorder

        recorder = AudioRecorder()
        recorder.start()

        # 1 second of loud audio at the device rate (above RMS threshold)
        frame = np.full((48000, 1), 1000, dtype="int16")
        recorder._frames = [frame]
        recorder._peak_rms = 1000

        wav_path = recorder.stop()

        assert wav_path is not None
        with wave.open(wav_path, "rb") as wf:
            assert wf.getframerate() == 48000


class TestWSL2PowerShellFallback:
    """Regression tests for WSL2 PowerShell TTS fallback (issue #17608).

    On WSL2 without a PulseAudio bridge, ffplay/aplay have no audio device.
    play_audio_file() should insert a PowerShell-based player at the front
    of the player list when powershell.exe and ffmpeg are available.
    """

    def _fake_check_output(self, responses):
        """Build a subprocess.check_output side_effect from a list of responses."""
        it = iter(responses)
        def _side_effect(cmd, **kwargs):
            return next(it)
        return _side_effect

    def test_powershell_pipeline_preserves_real_exit_status(self, sample_wav):
        """Regression (review of #63768): the shell pipeline must preserve
        the (ffmpeg && powershell) exit status past the unconditional
        cleanup, so a real conversion/playback failure falls through to the
        next player instead of being masked by rm -f's always-zero exit."""
        from unittest.mock import patch, MagicMock
        from tools import voice_mode as vm

        captured_cmds = []

        def _capture_popen(cmd, **kw):
            captured_cmds.append(list(cmd))
            m = MagicMock()
            # Simulate the PowerShell pipeline failing (nonzero rc), and
            # the fallback ffplay succeeding.
            if cmd[0] in ("/bin/sh", "sh"):
                m.returncode = 1
            else:
                m.returncode = 0
            m.wait = MagicMock(return_value=m.returncode)
            return m

        with patch("tools.voice_mode._is_wsl2_env", return_value=True), \
             patch("tools.voice_mode._import_audio", side_effect=ImportError), \
             patch("tools.voice_mode.shutil.which",
                   side_effect=lambda x: f"/bin/{x}" if x in ("powershell.exe", "ffmpeg", "ffplay", "sh") else (x if x.startswith("/") else None)), \
             patch("tools.voice_mode.subprocess.check_output",
                   side_effect=self._fake_check_output([
                       b"C:/Temp\r\n",
                       b"/mnt/c/Temp\n",
                       b"C:/Temp/hermes.wav\n",
                   ])), \
             patch("tools.voice_mode.subprocess.Popen", side_effect=_capture_popen):
            result = vm.play_audio_file(str(sample_wav))

        assert result is True, "Must fall through to ffplay and succeed"
        assert len(captured_cmds) == 2, (
            f"Expected sh pipeline to be tried and fail, then ffplay to be "
            f"tried: {captured_cmds}"
        )
        assert captured_cmds[0][0] in ("/bin/sh", "sh")
        assert captured_cmds[1][0] == "ffplay"
        # The subshell command must capture and re-exit with $rc, not rely
        # on rm -f's exit status.
        sh_script = captured_cmds[0][2]
        assert "rc=$?" in sh_script and "exit $rc" in sh_script, (
            "Shell pipeline must preserve the real exit status past cleanup: " + sh_script
        )

    def test_wsl2_unique_temp_filename(self, monkeypatch, tmp_path, sample_wav):
        """Two concurrent calls must use different temp WAV filenames."""
        from unittest.mock import patch, MagicMock
        from tools import voice_mode as vm

        filenames = []

        def _capture_check_output(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "TEMP" in cmd_str:
                return b"C:\\Temp\r\n"
            if "wslpath" in cmd_str and "-u" in cmd_str:
                return b"/mnt/c/Temp\n"
            if "wslpath" in cmd_str and "-w" in cmd_str:
                wsl_path = cmd[-1] if isinstance(cmd[-1], str) else cmd[-1].decode()
                filenames.append(wsl_path.split("/")[-1])
                return f"C:\\Temp\\{wsl_path.split('/')[-1]}\n".encode()
            return b""

        def _fake_open(path, *args, **kwargs):
            if str(path) == "/proc/version":
                import io
                return io.StringIO("Linux Microsoft WSL2")
            return open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=_fake_open), \
             patch("shutil.which", side_effect=lambda x: f"/bin/{x}" if x in ("powershell.exe", "ffmpeg", "ffplay") else None), \
             patch("subprocess.check_output", side_effect=_capture_check_output), \
             patch("subprocess.Popen", return_value=MagicMock(returncode=0, wait=lambda **k: 0)), \
             patch("tools.voice_mode._playback_lock"), \
             patch("tools.voice_mode._active_playback", None):
            vm.play_audio_file(str(sample_wav))
            vm.play_audio_file(str(sample_wav))

        # Regression (review of #63768): the original test made this
        # assertion conditional on len(filenames) >= 2, so a broken
        # (zero-captured) run passed trivially. Require exactly two.
        assert len(filenames) == 2, (
            f"Expected exactly 2 captured temp filenames from 2 calls, got "
            f"{len(filenames)}: {filenames}"
        )
        assert filenames[0] != filenames[1], (
            "Concurrent TTS calls must use unique temp WAV filenames"
        )

    def test_non_wsl_skips_powershell_fallback(self, monkeypatch, sample_wav):
        """On non-WSL Linux, the PowerShell player must not be inserted."""
        from unittest.mock import patch, MagicMock
        from tools import voice_mode as vm

        captured_players = []

        def _capture_popen(cmd, **kw):
            captured_players.append(cmd)
            m = MagicMock()
            m.returncode = 0
            m.wait.return_value = 0
            return m

        def _fake_open(path, *args, **kwargs):
            if str(path) == "/proc/version":
                import io
                return io.StringIO("Linux version 5.15.0-generic #72-Ubuntu")
            return open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=_fake_open), \
             patch("tools.voice_mode._import_audio", side_effect=ImportError), \
             patch("shutil.which", side_effect=lambda x: f"/bin/{x}" if x in ("ffplay", "aplay") else None), \
             patch("subprocess.Popen", side_effect=_capture_popen), \
             patch("tools.voice_mode._playback_lock"), \
             patch("tools.voice_mode._active_playback", None):
            vm.play_audio_file(str(sample_wav))

        assert captured_players, "No players were tried"
        for cmd in captured_players:
            assert not (cmd[0] == "sh" and "powershell" in " ".join(str(c) for c in cmd)), (
                "PowerShell player must not appear on non-WSL Linux"
            )


class TestWSLAudioEnvironmentGate:
    """Regression tests (review of #63768) for detect_audio_environment()'s
    WSL gate: when the PowerShell TTS fallback is viable, voice mode must
    not be hard-blocked, but the recording/STT PulseAudio-bridge guidance
    must still be surfaced (as a non-blocking notice)."""

    def _fake_open_wsl(self, path, *args, **kwargs):
        if str(path) == "/proc/version":
            import io
            return io.StringIO("Linux version 5.15 Microsoft Standard WSL2")
        return open(path, *args, **kwargs)

    def test_wsl_no_pulse_but_powershell_available_not_hard_blocked(self, monkeypatch):
        from unittest.mock import patch
        from tools import voice_mode as vm

        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.delenv("PIPEWIRE_REMOTE", raising=False)
        for _ssh_var in ("SSH_CLIENT", "SSH_TTY", "SSH_CONNECTION"):
            monkeypatch.delenv(_ssh_var, raising=False)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))
        with patch("builtins.open", side_effect=self._fake_open_wsl), \
             patch("tools.voice_mode._wsl_powershell_tts_available", return_value=True), \
             patch("tools.voice_mode._pulse_socket_reachable", return_value=False), \
             patch("hermes_constants.is_container", return_value=False):
            result = vm.detect_audio_environment()

        assert result["available"] is True, (
            "PowerShell TTS fallback must keep voice mode enabled even "
            "without a PulseAudio bridge: " + str(result["warnings"])
        )
        assert any("PowerShell" in n or "Media.SoundPlayer" in n for n in result["notices"]), (
            "The PowerShell fallback path must be mentioned in notices"
        )
        assert any("recording" in n.lower() or "PulseAudio" in n for n in result["notices"]), (
            "The recording/STT PulseAudio caveat must still be surfaced"
        )

    def test_wsl_no_pulse_no_powershell_still_blocked(self, monkeypatch):
        from unittest.mock import patch
        from tools import voice_mode as vm

        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.delenv("PIPEWIRE_REMOTE", raising=False)
        for _ssh_var in ("SSH_CLIENT", "SSH_TTY", "SSH_CONNECTION"):
            monkeypatch.delenv(_ssh_var, raising=False)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))
        with patch("builtins.open", side_effect=self._fake_open_wsl), \
             patch("tools.voice_mode._wsl_powershell_tts_available", return_value=False), \
             patch("tools.voice_mode._pulse_socket_reachable", return_value=False), \
             patch("hermes_constants.is_container", return_value=False):
            result = vm.detect_audio_environment()

        assert result["available"] is False, (
            "Without PulseAudio AND without the PowerShell fallback, WSL "
            "must still be hard-blocked as before"
        )

    def test_wsl_with_pulse_server_unaffected(self, monkeypatch):
        """PULSE_SERVER already configured: existing behavior unchanged."""
        from unittest.mock import patch
        from tools import voice_mode as vm

        monkeypatch.setenv("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")
        for _ssh_var in ("SSH_CLIENT", "SSH_TTY", "SSH_CONNECTION"):
            monkeypatch.delenv(_ssh_var, raising=False)
        monkeypatch.setattr("tools.voice_mode._import_audio",
                            lambda: (MagicMock(), MagicMock()))
        with patch("builtins.open", side_effect=self._fake_open_wsl), \
             patch("hermes_constants.is_container", return_value=False):
            result = vm.detect_audio_environment()

        assert result["available"] is True
        # Merged with #37346: any forwarded sound server (PULSE_SERVER or
        # PIPEWIRE_REMOTE) yields the shared reachable-sound-server notice.
        assert any(
            "PulseAudio" in n and "WSL" in n for n in result["notices"]
        )
