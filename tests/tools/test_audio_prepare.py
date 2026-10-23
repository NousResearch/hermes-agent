"""Native audio keeps acoustic fidelity and never sends a truncated decoded prefix."""

import io
import subprocess
import wave

import pytest

from hermes_platform.resolver import locate_command
from tools.audio_prepare import prepare_audio
from tools.environments.local import build_subprocess_env


def stereo_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\x01\x00\x02\x00" * 4410)
    return output.getvalue()


def test_wav_is_sent_byte_for_byte():
    data = stereo_wav()
    assert prepare_audio(data) == (data, "wav")


def test_real_m4a_decode_preserves_stereo_and_sample_rate(tmp_path, monkeypatch):
    from tools import audio_prepare
    ffmpeg = locate_command("ffmpeg").command
    if not ffmpeg:
        pytest.skip("ffmpeg is required for the real M4A conversion test")
    source, encoded = tmp_path / "source.wav", tmp_path / "song.m4a"
    source.write_bytes(stereo_wav())
    subprocess.run([*ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(source), str(encoded)],
                   check=True, timeout=20, env=build_subprocess_env())
    original = encoded.read_bytes()
    assert prepare_audio(original, supported_formats=("m4a",)) == (original, "m4a")
    data, fmt = prepare_audio(original)
    with wave.open(io.BytesIO(data)) as decoded:
        assert (fmt, decoded.getnchannels(), decoded.getframerate()) == ("wav", 2, 44100)
        assert decoded.getnframes() >= 4410
    assert encoded.read_bytes() == original
    monkeypatch.setattr(audio_prepare, "MAX_AUDIO_BYTES", 1024)
    with pytest.raises(ValueError, match="no partial analysis"):
        prepare_audio(original)
