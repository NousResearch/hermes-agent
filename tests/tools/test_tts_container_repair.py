"""Tests for the class-level TTS .ogg container repair.

Root cause class (#57048, #54589, #57213, #58845, #14841, #45557, #57049):
several TTS backends silently write MP3/WAV bytes into a ``.ogg`` output
path (Edge only emits MP3; Piper writes WAV; xAI writes MP3; some
OpenAI-compatible servers ignore ``response_format="opus"``). Platforms
that require real Ogg/Opus for native voice bubbles (Telegram, Matrix,
Feishu, WhatsApp, Signal) then render a broken 0-second bubble.

Instead of per-provider fixes, ``text_to_speech_tool`` sniffs the magic
bytes once after synthesis and repairs the container centrally.

The container alone was not the whole story: the codec inside the
Ogg matters too. ffmpeg's ``.ogg`` muxer defaults to Vorbis, and a build
without libvorbis silently writes Ogg/FLAC — real Ogg, but not the Opus a
voice bubble is expected to carry, and the tool then failed to advertise
the file as a voice note at all. The repair is now codec-aware, and the
WAV→``.ogg`` conversion forces Opus explicitly.
"""

import shutil
import struct
import subprocess

import pytest

from tools.tts_tool import OPUS_VOICE_PLATFORMS, _repair_ogg_container
from tools.tts_tool_delivery import (
    _finalize_wav_output, _sniff_audio_container, _sniff_ogg_codec)

MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64
MP3_FRAME = b"\xff\xfb\x90\x00" + b"\x00" * 64
OGG = b"OggS\x00\x02" + b"\x00" * 64
FLAC = b"fLaC" + b"\x00" * 64


def _ffmpeg_has(encoder: str) -> bool:
    """The binary must exist AND carry the encoder: CI runners have shipped an ffmpeg that
    exits 127, and a Homebrew build without libvorbis is exactly what the bug below needs."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    probe = subprocess.run([ffmpeg, "-hide_banner", "-h", f"encoder={encoder}"],
                           capture_output=True, check=False)
    return probe.returncode == 0 and encoder.encode() in probe.stdout


def _sine_wav(path) -> bool:
    """A real 0.3 s WAV (a WAV-native engine's product); False when ffmpeg can't make one."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run(
        [ffmpeg, "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
         "-acodec", "pcm_s16le", "-y", str(path)],
        capture_output=True, check=False)
    return result.returncode == 0 and path.exists() and path.stat().st_size > 0


def _ogg_flac(wav_path, out_path) -> bool:
    """Ogg/FLAC — what a bare ``ffmpeg -i in.wav out.ogg`` writes when libvorbis is absent."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not _ffmpeg_has("flac"):
        return False
    result = subprocess.run(
        [ffmpeg, "-i", str(wav_path), "-acodec", "flac", "-f", "ogg", "-y", str(out_path)],
        capture_output=True, check=False)
    return result.returncode == 0 and _sniff_ogg_codec(str(out_path)) == "flac"


def _wav_bytes() -> bytes:
    return b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"\x00" * 64


class TestSniffAudioContainer:
    @pytest.mark.parametrize(
        "data,expected",
        [
            (MP3_ID3, "mp3"),
            (MP3_FRAME, "mp3"),
            (OGG, "ogg"),
            (FLAC, "flac"),
        ],
    )
    def test_magic_bytes(self, tmp_path, data, expected):
        p = tmp_path / "a.bin"
        p.write_bytes(data)
        assert _sniff_audio_container(str(p)) == expected


    def test_unknown_and_missing(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"\x00\x01\x02\x03" * 8)
        assert _sniff_audio_container(str(p)) == "unknown"
        assert _sniff_audio_container(str(tmp_path / "missing")) == "unknown"


class TestRepairOggContainer:
    def test_real_ogg_untouched(self, tmp_path):
        p = tmp_path / "v.ogg"
        p.write_bytes(OGG)
        assert _repair_ogg_container(str(p)) == str(p)
        assert p.read_bytes() == OGG


    def test_ffmpeg_real_transcode_if_available(self, tmp_path):
        """Live ffmpeg round-trip when the binary exists (skipped otherwise)."""
        import shutil as _shutil
        import subprocess as _sp

        if not _shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")
        # Synthesize a real tiny mp3 with ffmpeg, misname it .ogg. "Available" means the binary
        # runs AND carries the encoder: CI runners have shipped an ffmpeg on PATH that exits 127.
        p = tmp_path / "v.ogg"
        synth = _sp.run(
            ["ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
             "-acodec", "libmp3lame", "-f", "mp3", str(p), "-y"],
            capture_output=True, check=False,
        )
        if synth.returncode != 0:
            pytest.skip(f"ffmpeg on PATH cannot synthesize mp3 (exit {synth.returncode})")
        assert _sniff_audio_container(str(p)) == "mp3"
        result = _repair_ogg_container(str(p))
        assert result == str(p)
        assert _sniff_audio_container(str(p)) == "ogg"


class TestOggCodecRepair:
    """Container-level repair was not enough: the codec inside the Ogg has to be Opus.

    Regression (Telegram voice replies arriving as plain attachments, not bubbles): piper's
    auto-TTS reply is a WAV converted by a bare ``ffmpeg -i sidecar.wav out.ogg``. ffmpeg's
    ``.ogg`` muxer default codec is Vorbis; without libvorbis it silently writes Ogg/FLAC and
    exits 0. The container sniff saw ``OggS`` and passed the file through; the tool then
    reported ``voice_compatible=False``, and the gateway's media routing sends a plain ``.ogg``
    by extension — so the reply landed as an attachment instead of a voice bubble.
    """

    def test_wav_native_output_targeting_ogg_is_opus(self, tmp_path):
        if not _ffmpeg_has("libopus"):
            pytest.skip("ffmpeg without libopus cannot produce voice-bubble audio")
        wav, out = tmp_path / "sidecar.wav", tmp_path / "reply.ogg"
        if not _sine_wav(wav):
            pytest.skip("ffmpeg on PATH cannot synthesize a wav")

        assert _finalize_wav_output(str(wav), str(out)) == str(out)
        assert _sniff_audio_container(str(out)) == "ogg"
        assert _sniff_ogg_codec(str(out)) == "opus"


    def test_ogg_flac_repaired_when_the_platform_needs_opus(self, tmp_path):
        if not _ffmpeg_has("libopus"):
            pytest.skip("ffmpeg without libopus cannot produce voice-bubble audio")
        wav, out = tmp_path / "sidecar.wav", tmp_path / "reply.ogg"
        if not _sine_wav(wav) or not _ogg_flac(wav, out):
            pytest.skip("ffmpeg cannot produce ogg/flac here")

        assert _repair_ogg_container(str(out), want_opus=True) == str(out)
        assert _sniff_ogg_codec(str(out)) == "opus"


    def test_ogg_flac_untouched_when_opus_is_not_needed(self, tmp_path):
        """Non-Opus platforms accept Ogg/Vorbis and Ogg/FLAC as plain audio: never re-encode."""
        wav, out = tmp_path / "sidecar.wav", tmp_path / "reply.ogg"
        if not _sine_wav(wav) or not _ogg_flac(wav, out):
            pytest.skip("ffmpeg cannot produce ogg/flac here")
        original = out.read_bytes()

        assert _repair_ogg_container(str(out)) == str(out)
        assert out.read_bytes() == original


class TestOpusPlatformSet:
    def test_opus_platforms_cover_voice_bubble_platforms(self):
        # Behavior contract: the platforms whose adapters deliver native
        # voice bubbles only for Ogg/Opus must be recognized.
        for platform in ("telegram", "matrix", "feishu", "whatsapp", "signal"):
            assert platform in OPUS_VOICE_PLATFORMS

