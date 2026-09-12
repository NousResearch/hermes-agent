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
    def __init__(self, default_samplerate=None, raise_query=False):
        self.played = []
        self.default_samplerate = default_samplerate
        self.raise_query = raise_query

    def play(self, audio, samplerate=None, blocksize=0):
        self.played.append((audio, samplerate))

    def stop(self):
        pass

    def get_stream(self):
        return None

    def query_devices(self, device=None, kind=None):
        if self.raise_query:
            raise RuntimeError("query_devices unavailable")
        if self.default_samplerate is None:
            raise RuntimeError("no default output device")
        return {"default_samplerate": self.default_samplerate}


def _reset():
    vm.stop_thinking_sound()
    # Drain any residual output ref-counts from prior tests.
    with vm._audio_output_lock:
        vm._audio_output_active_count = 0


class TestConfigGate:
    def test_default_enabled(self):
        with patch("hermes_cli.config.load_config", return_value={"voice": {}}):
            assert vm.thinking_sound_enabled() is True


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

    def test_play_audio_file_brackets_refcount(self, tmp_path):
        """play_audio_file flags real speaker output for its whole duration,
        so the thinking loop knows audio is flowing."""
        _reset()
        seen = []

        def fake_impl(path):
            seen.append(vm.is_audio_output_active())
            return True

        with patch.object(vm, "_play_audio_file_impl", fake_impl):
            vm.play_audio_file(str(tmp_path / "x.wav"))
        assert seen == [True]
        assert vm.is_audio_output_active() is False


def _wait_for_play(fake, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not fake.played and time.monotonic() < deadline:
        time.sleep(0.01)


def _run_thinking_loop(fake, platform_name):
    stop = threading.Event()
    with patch.object(vm.platform, "system", return_value=platform_name), \
         patch.object(vm, "_sounddevice_output_allowed", return_value=True), \
         patch.object(vm, "_import_audio", return_value=(fake, np)), \
         patch.object(vm, "_get_beep_volume", return_value=0.3):
        t = threading.Thread(
            target=vm._thinking_sound_loop, args=(stop, None), daemon=True
        )
        t.start()
        _wait_for_play(fake)
        stop.set()
        t.join(timeout=3.0)
    return fake


class TestWindowsOutputResample:
    """Native Windows WASAPI shared-mode: 16 kHz PCM must be resampled to
    the default output device rate before PortAudio ``sd.play``.
    """

    def test_thinking_loop_resamples_to_device_rate_on_windows(self):
        _reset()
        fake = _run_thinking_loop(_FakeSD(default_samplerate=48000), "Windows")
        assert fake.played, "loop never played a blip"
        audio, rate = fake.played[0]
        src_len = int(vm.SAMPLE_RATE * 0.16)
        assert rate == 48000
        assert len(audio) == int(round(src_len * 48000 / 16000))

    def test_thinking_loop_fail_open_when_query_devices_raises(self):
        _reset()
        fake = _run_thinking_loop(_FakeSD(raise_query=True), "Windows")
        assert fake.played, "loop never played a blip"
        audio, rate = fake.played[0]
        assert rate == vm.SAMPLE_RATE
        assert len(audio) == int(vm.SAMPLE_RATE * 0.16)

    def test_thinking_loop_no_resample_on_linux_even_if_device_is_48k(self):
        _reset()
        fake = _run_thinking_loop(_FakeSD(default_samplerate=48000), "Linux")
        assert fake.played, "loop never played a blip"
        audio, rate = fake.played[0]
        assert rate == vm.SAMPLE_RATE
        assert len(audio) == int(vm.SAMPLE_RATE * 0.16)

    def test_thinking_sound_starts_when_beep_disabled(self):
        _reset()
        with patch("hermes_cli.config.load_config",
                   return_value={"voice": {"beep_enabled": False}}), \
             patch.object(vm, "_sounddevice_output_allowed", return_value=False):
            assert vm.thinking_sound_enabled() is True
            assert vm.start_thinking_sound() is True
            assert vm._thinking_stop is not None
            vm.stop_thinking_sound()

    def test_play_beep_resamples_to_device_rate_on_windows(self):
        fake = _FakeSD(default_samplerate=48000)
        duration = 0.1
        with patch.object(vm.platform, "system", return_value="Windows"), \
             patch.object(vm, "_import_audio", return_value=(fake, np)):
            vm.play_beep(frequency=880, duration=duration, count=1)
        assert fake.played, "play_beep never submitted audio"
        audio, rate = fake.played[0]
        src_len = int(vm.SAMPLE_RATE * duration)
        assert rate == 48000
        assert len(audio) == int(round(src_len * 48000 / 16000))

    def test_play_wav_resamples_to_device_rate_on_windows(self, tmp_path):
        import struct
        import wave

        n_frames = 1600  # 0.1 s at 16 kHz
        wav_path = tmp_path / "src.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack(f"<{n_frames}h", *([1000] * n_frames)))

        fake = _FakeSD(default_samplerate=48000)
        with patch.object(vm.platform, "system", return_value="Windows"), \
             patch.object(vm, "_import_audio", return_value=(fake, np)), \
             patch.object(vm, "_is_wsl2_env", return_value=False):
            assert vm._play_wav_via_sounddevice(str(wav_path)) is True
        assert fake.played, "wav playback never submitted audio"
        audio, rate = fake.played[0]
        assert rate == 48000
        assert len(audio) == int(round(n_frames * 48000 / 16000))
