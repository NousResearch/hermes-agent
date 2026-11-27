"""Contract tests for the non-WAV playback path: convert via ffmpeg to a temp WAV and play
through sounddevice (ffplay, the first system-player fallback, can hang indefinitely on
SDL2 video init even with -nodisp). Regression coverage for the local patch documented in
~/hermes-projects/hermes-local-patches/."""

import os

from tools import voice_mode as vm


def test_convert_helper_declines_without_ffmpeg_or_sounddevice(monkeypatch):
    monkeypatch.setattr(vm.shutil, "which", lambda _name: None)
    assert vm._convert_to_wav_for_sounddevice("/x/audio.mp3") is None
    monkeypatch.setattr(vm.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(vm, "_sounddevice_output_allowed", lambda: False)
    assert vm._convert_to_wav_for_sounddevice("/x/audio.mp3") is None


def test_nonwav_playback_cleans_up_temp_wav_and_falls_through(monkeypatch, tmp_path):
    src = tmp_path / "tts.mp3"
    src.write_bytes(b"mp3")
    converted = tmp_path / "tts.wav"
    converted.write_bytes(b"wav")
    wav_calls = []
    monkeypatch.setattr(vm, "_convert_to_wav_for_sounddevice", lambda _p: str(converted))
    monkeypatch.setattr(vm, "_play_wav_via_sounddevice",
                        lambda _p: wav_calls.append(_p) or False)
    monkeypatch.setattr(vm, "_system_player_candidates", lambda _p: [])
    assert vm._play_audio_file_impl(str(src)) is False
    assert wav_calls == [str(converted)]
    assert not os.path.exists(converted)  # temp WAV removed even when playback failed
