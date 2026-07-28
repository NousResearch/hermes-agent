"""Tests for the class-level TTS .ogg container repair.

Root cause class (#57048, #54589, #57213, #58845, #14841, #45557, #57049):
several TTS backends silently write MP3/WAV bytes into a ``.ogg`` output
path (Edge only emits MP3; Piper writes WAV; xAI writes MP3; some
OpenAI-compatible servers ignore ``response_format="opus"``). Platforms
that require real Ogg/Opus for native voice bubbles (Telegram, Matrix,
Feishu, WhatsApp, Signal) then render a broken 0-second bubble.

Instead of per-provider fixes, ``text_to_speech_tool`` sniffs the magic
bytes once after synthesis and repairs the container centrally.
"""

import struct
from unittest.mock import patch

import pytest

from tools.tts_tool import (
    OPUS_VOICE_PLATFORMS,
    _repair_ogg_container,
    _sniff_audio_container,
)

MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64
MP3_FRAME = b"\xff\xfb\x90\x00" + b"\x00" * 64
OGG = b"OggS\x00\x02" + b"\x00" * 64
FLAC = b"fLaC" + b"\x00" * 64


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

    def test_wav(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(_wav_bytes())
        assert _sniff_audio_container(str(p)) == "wav"

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

    def test_non_ogg_extension_untouched(self, tmp_path):
        p = tmp_path / "v.mp3"
        p.write_bytes(MP3_ID3)
        assert _repair_ogg_container(str(p)) == str(p)

    def test_mp3_in_ogg_transcoded(self, tmp_path):
        p = tmp_path / "v.ogg"
        p.write_bytes(MP3_ID3)

        def fake_transcode(input_path, ogg_path):
            # simulate in-place ffmpeg success
            with open(ogg_path, "wb") as fh:
                fh.write(OGG)
            return ogg_path

        with patch("tools.tts_tool._ffmpeg_transcode_to_opus", fake_transcode):
            result = _repair_ogg_container(str(p))

        assert result == str(p)
        assert p.read_bytes()[:4] == b"OggS"

    def test_wav_in_ogg_transcoded(self, tmp_path):
        p = tmp_path / "v.ogg"
        p.write_bytes(_wav_bytes())
        with patch("tools.tts_tool._ffmpeg_transcode_to_opus",
                   lambda i, o: (open(o, "wb").write(OGG), o)[1]):
            assert _repair_ogg_container(str(p)) == str(p)
        assert p.read_bytes()[:4] == b"OggS"

    def test_no_ffmpeg_renames_to_honest_extension(self, tmp_path):
        p = tmp_path / "v.ogg"
        p.write_bytes(MP3_FRAME)
        with patch("tools.tts_tool._ffmpeg_transcode_to_opus", lambda i, o: None):
            result = _repair_ogg_container(str(p))
        assert result == str(tmp_path / "v.mp3")
        assert not p.exists()
        assert (tmp_path / "v.mp3").exists()

    def test_ffmpeg_real_transcode_if_available(self, tmp_path):
        """Live ffmpeg round-trip when the binary exists (skipped otherwise)."""
        import shutil as _shutil
        import subprocess as _sp

        if not _shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")
        # Synthesize a real tiny mp3 with ffmpeg, misname it .ogg
        p = tmp_path / "v.ogg"
        _sp.run(
            ["ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
             "-acodec", "libmp3lame", "-f", "mp3", str(p), "-y"],
            capture_output=True, check=True,
        )
        assert _sniff_audio_container(str(p)) == "mp3"
        result = _repair_ogg_container(str(p))
        assert result == str(p)
        assert _sniff_audio_container(str(p)) == "ogg"


class TestOpusPlatformSet:
    def test_opus_platforms_cover_voice_bubble_platforms(self):
        # Behavior contract: the platforms whose adapters deliver native
        # voice bubbles only for Ogg/Opus must be recognized.
        for platform in ("telegram", "matrix", "feishu", "whatsapp", "signal"):
            assert platform in OPUS_VOICE_PLATFORMS

    def test_cli_not_included(self):
        assert "" not in OPUS_VOICE_PLATFORMS
        assert "cli" not in OPUS_VOICE_PLATFORMS
