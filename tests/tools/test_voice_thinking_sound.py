"""Tests for the ambient voice-chat "thinking" sound (tools/voice_mode.py).

Contract:
  - `voice.thinking_sound` config gates it (default True).
  - `start_thinking_sound()` is idempotent, returns False when disabled.
  - The loop synthesizes blips with numpy (no assets), scales volume by
    `voice.beep_volume`, and NEVER plays through sounddevice on macOS
    (_sounddevice_output_allowed → TCC-safe silent skip).
  - `stop_thinking_sound()` stops the loop instantly and is idempotent.
  - The should_play callback gates each blip (no blips while TTS speaks
    or the mic captures).
  - mark_audio_output_active / is_audio_output_active ref-count playback.
"""

import threading
import time
from unittest.mock import patch

import pytest

np = pytest.importorskip(
    "numpy", reason="numpy is a lazy voice dependency, absent in hermetic CI"
)

import tools.voice_mode as vm


class _FakeSD:
    def __init__(self):
        self.played = []
        self.output_active_during_play = []

    def play(self, audio, samplerate=None, blocksize=0):
        # blocksize: _sd_play_blocking passes the caller's block size through.
        self.output_active_during_play.append(vm.is_audio_output_active())
        self.played.append((audio, samplerate))

    def get_stream(self):
        # _sd_play_blocking polls get_stream() to know when playback is over.
        return None

    def stop(self):
        pass


def _reset():
    vm.stop_thinking_sound()
    # Drain any residual output ref-counts from prior tests.
    with vm._audio_output_lock:
        vm._audio_output_active_count = 0


class TestConfigGate:


    def test_start_refuses_when_disabled(self):
        _reset()
        with patch.object(vm, "thinking_sound_enabled", return_value=False):
            assert vm.start_thinking_sound() is False
        assert vm._thinking_stop is None


class TestBlipSynthesis:
    def test_blip_is_int16_low_volume(self):
        with patch.object(vm, "_get_beep_volume", return_value=0.3):
            blip = vm._synth_thinking_blip(np, 392.0)
        assert blip.dtype == np.int16
        assert len(blip) == int(vm.SAMPLE_RATE * 0.16)
        # Quieter than the beeps: 0.3 * 0.5 * 32767 ≈ 4915 peak ceiling.
        assert int(np.abs(blip).max()) <= int(0.3 * 0.5 * 32767) + 1
        assert int(np.abs(blip).max()) > 0


    def test_no_click_smooth_attack(self):
        blip = vm._synth_thinking_blip(np, 392.0)
        # First sample near zero (enveloped attack, no click).
        assert abs(int(blip[0])) < 200


class TestLoopLifecycle:
    def test_loop_plays_blips_and_stops_instantly(self):
        _reset()
        fake = _FakeSD()
        stop = threading.Event()
        with patch.object(vm, "_sounddevice_output_allowed", return_value=True), \
             patch.object(vm, "_import_audio", return_value=(fake, np)), \
             patch.object(vm, "_get_beep_volume", return_value=0.3):
            t = threading.Thread(
                target=vm._thinking_sound_loop, args=(stop, None), daemon=True
            )
            t.start()
            deadline = time.monotonic() + 3.0
            while not fake.played and time.monotonic() < deadline:
                time.sleep(0.01)
            stop.set()
            t.join(timeout=3.0)
        assert fake.played, "loop never played a blip"
        assert not t.is_alive()


    def test_start_is_idempotent_and_stop_clears(self):
        _reset()
        with patch.object(vm, "thinking_sound_enabled", return_value=True), \
             patch.object(vm, "_sounddevice_output_allowed", return_value=False):
            assert vm.start_thinking_sound() is True
            first_stop = vm._thinking_stop
            assert vm.start_thinking_sound() is True
            assert vm._thinking_stop is first_stop  # no second loop
            vm.stop_thinking_sound()
            assert vm._thinking_stop is None
            assert first_stop.is_set()
            vm.stop_thinking_sound()  # idempotent


class TestAudioOutputRefcount:
    def test_refcount_tracks_nested_playback(self):
        _reset()
        assert vm.is_audio_output_active() is False
        vm.mark_audio_output_active(True)
        vm.mark_audio_output_active(True)
        assert vm.is_audio_output_active() is True
        vm.mark_audio_output_active(False)
        assert vm.is_audio_output_active() is True
        vm.mark_audio_output_active(False)
        assert vm.is_audio_output_active() is False
        # Never goes negative.
        vm.mark_audio_output_active(False)
        assert vm.is_audio_output_active() is False

    def test_play_audio_file_wav_brackets_refcount(self, tmp_path):
        """WAV playback through play_audio_file -> _play_wav_via_sounddevice marks output."""
        _reset()
        wav = tmp_path / "speech.wav"
        with open(wav, "wb") as f:
            vm._write_wav_frames(f, np.zeros(16, dtype=np.int16).tobytes(), vm.SAMPLE_RATE)
        fake = _FakeSD()
        with patch.object(vm, "_sounddevice_output_allowed", return_value=True), \
             patch.object(vm, "_import_audio", return_value=(fake, np)):
            assert vm.play_audio_file(str(wav)) is True
        assert fake.output_active_during_play == [True]
        assert vm.is_audio_output_active() is False

    def test_play_audio_file_system_player_brackets_refcount(self, tmp_path, monkeypatch):
        """Subprocess playback through play_audio_file -> _run_system_player marks output."""
        _reset()
        audio = tmp_path / "speech.mp3"
        audio.write_bytes(b"ID3")
        seen = []

        class _Proc:
            returncode = 0

            def wait(self, timeout=None):
                seen.append(vm.is_audio_output_active())
                return 0

        monkeypatch.setattr(vm, "_system_player_candidates", lambda p: [["afplay", p]])
        monkeypatch.setattr(vm.shutil, "which", lambda _cmd: "/usr/bin/afplay")
        monkeypatch.setattr(vm.subprocess, "Popen", lambda *a, **k: _Proc())
        assert vm.play_audio_file(str(audio)) is True
        assert seen == [True]
        assert vm.is_audio_output_active() is False

    def test_sounddevice_beep_and_thinking_blip_do_not_claim_speaker(self):
        """Sounddevice beeps and thinking blips must not raise is_audio_output_active()."""
        _reset()
        fake = _FakeSD()
        with patch.object(vm, "_sounddevice_output_allowed", return_value=True), \
             patch.object(vm, "_import_audio", return_value=(fake, np)), \
             patch.object(vm, "_get_beep_volume", return_value=0.3):
            vm.play_beep(duration=0.02)
            stop = threading.Event()
            t = threading.Thread(
                target=vm._thinking_sound_loop, args=(stop, None), daemon=True
            )
            t.start()
            deadline = time.monotonic() + 3.0
            while len(fake.played) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            stop.set()
            t.join(timeout=3.0)
        assert len(fake.output_active_during_play) >= 2
        assert all(active is False for active in fake.output_active_during_play)
        assert vm.is_audio_output_active() is False
