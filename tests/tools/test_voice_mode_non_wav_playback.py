"""Regression for #159359 — non-WAV TTS audio must not end in ``ffplay -autoexit``.

TTS providers hand back MP3/OGG. ``_play_audio_file_impl`` only sent ``.wav`` through the
blocking sounddevice path, so MP3 fell to the system player, whose ``-autoexit`` returns at
end-of-stream and drops whatever the output device still had buffered — the tail of the last
sentence. These tests pin the contract: with sounddevice output allowed and ffmpeg present, a
non-WAV clip is decoded and played through the same blocking path; every other case keeps the
system-player fallback.
"""

import pathlib

import pytest

import tools.voice_mode as vm


@pytest.fixture
def mp3(tmp_path):
    path = tmp_path / "reply.mp3"
    path.write_bytes(b"ID3fake")
    return str(path)


@pytest.fixture
def calls(monkeypatch):
    """Record every hop of the playback path; nothing real runs."""
    recorded = {"converted": [], "played": [], "players": []}
    monkeypatch.setattr(vm, "_play_wav_via_sounddevice",
                        lambda path: recorded["played"].append(path) or True)
    monkeypatch.setattr(vm, "_run_system_player",
                        lambda cmd: recorded["players"].append(list(cmd)) or True)
    return recorded


def _stub_tools(monkeypatch, *, ffmpeg: str | None = "/usr/bin/ffmpeg",
                decode_error=None, recorded=None):
    def which(name):
        return {"/usr/bin/ffmpeg": ffmpeg, "ffmpeg": ffmpeg,
                "ffplay": "/usr/bin/ffplay", "afplay": "/usr/bin/afplay",
                "aplay": "/usr/bin/aplay"}.get(name)

    def fake_run(cmd, **kwargs):
        if recorded is not None:
            recorded["converted"].append(list(cmd))
        if decode_error is not None:
            raise decode_error
        pathlib.Path(cmd[-1]).write_bytes(b"RIFFfake")

    monkeypatch.setattr(vm.shutil, "which", which)
    monkeypatch.setattr(vm.subprocess, "run", fake_run)


def test_non_wav_plays_through_temp_wav_when_sounddevice_allowed(monkeypatch, mp3, calls):
    monkeypatch.setattr(vm, "_sounddevice_output_allowed", lambda: True)
    _stub_tools(monkeypatch, recorded=calls)

    assert vm.play_audio_file(mp3) is True

    assert calls["players"] == [], "a system player must not be used when sounddevice can play"
    assert len(calls["converted"]) == 1
    argv = calls["converted"][0]
    assert argv[0] == "/usr/bin/ffmpeg"
    assert mp3 in argv
    assert argv[-1].endswith(".wav")

    assert len(calls["played"]) == 1
    played = calls["played"][0]
    assert played.endswith(".wav") and played != mp3
    assert not pathlib.Path(played).exists(), "the temporary WAV must be removed"


def test_non_wav_uses_system_player_when_sounddevice_not_allowed(monkeypatch, mp3, calls):
    monkeypatch.setattr(vm, "_sounddevice_output_allowed", lambda: False)
    _stub_tools(monkeypatch, recorded=calls)

    assert vm.play_audio_file(mp3) is True

    assert calls["converted"] == [], "no decode when sounddevice output is disallowed (macOS)"
    assert calls["played"] == []
    assert len(calls["players"]) == 1


def test_non_wav_uses_system_player_when_ffmpeg_missing(monkeypatch, mp3, calls):
    monkeypatch.setattr(vm, "_sounddevice_output_allowed", lambda: True)
    _stub_tools(monkeypatch, ffmpeg=None, recorded=calls)

    assert vm.play_audio_file(mp3) is True

    assert calls["converted"] == [], "no decode without ffmpeg"
    assert calls["played"] == []
    assert len(calls["players"]) == 1


def test_failed_decode_falls_back_to_system_player(monkeypatch, mp3, calls):
    monkeypatch.setattr(vm, "_sounddevice_output_allowed", lambda: True)
    _stub_tools(monkeypatch,
                decode_error=vm.subprocess.CalledProcessError(1, "ffmpeg"),
                recorded=calls)

    assert vm.play_audio_file(mp3) is True

    assert calls["played"] == []
    assert len(calls["players"]) == 1


def test_wav_still_bypasses_the_decoder(monkeypatch, tmp_path, calls):
    wav = tmp_path / "tone.wav"
    wav.write_bytes(b"RIFFfake")
    monkeypatch.setattr(vm, "_sounddevice_output_allowed", lambda: True)
    _stub_tools(monkeypatch, recorded=calls)

    assert vm.play_audio_file(str(wav)) is True

    assert calls["converted"] == [], "a WAV must not be transcoded"
    assert calls["played"] == [str(wav)]
    assert calls["players"] == []
