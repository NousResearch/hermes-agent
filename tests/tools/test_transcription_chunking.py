"""Cloud STT inputs over the provider's upload cap are fitted, not rejected (real ffmpeg, real files)."""

from __future__ import annotations

import shutil
import struct
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.transcription_chunking as chunking
import tools.transcription_tools as tt
from tools.transcription_audio import _probe_audio_duration

_HAS_FFMPEG = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


def _tone_with_pauses(path: Path, seconds: int, rate: int = 16000) -> None:
    """Alternating 1 s tone / 1 s silence, so silencedetect finds a pause every 2 s."""
    frames = bytearray()
    for second in range(seconds):
        amplitude = 6000 if second % 2 == 0 else 0
        frames += struct.pack(f"<{rate}h", *([amplitude, -amplitude] * (rate // 2)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))


def _run(path: Path, provider: str, *, model: str | None = None):
    """Run the real transcribe_audio with only the provider's network call replaced."""
    uploads: list = []

    def handler(file_path, model_name, **_kw):
        uploads.append({"suffix": Path(file_path).suffix, "bytes": Path(file_path).stat().st_size,
                        "seconds": _probe_audio_duration(file_path), "model": model_name})
        return {"success": True, "transcript": f"part{len(uploads)}", "provider": provider}

    with patch.object(tt, "_load_stt_config", return_value={"provider": provider, "cloud_trim_silence": False}), \
         patch.object(tt, "_get_provider", return_value=provider), \
         patch.object(tt, f"_transcribe_{provider}", side_effect=handler):
        return tt.transcribe_audio(str(path), model=model), uploads


def test_oversized_wav_is_reencoded_into_one_upload(tmp_path, monkeypatch):
    """A WAV over the byte cap that compresses under it goes up once, as m4a, with no seams."""
    monkeypatch.setitem(chunking._PROVIDER_MAX_BYTES, "groq", 400_000)
    wav = tmp_path / "note.wav"
    _tone_with_pauses(wav, 20)  # 640 KB of PCM, ~80 KB as 32 kbps AAC
    result, uploads = _run(wav, "groq")
    assert result == {"success": True, "transcript": "part1", "provider": "groq"}
    assert len(uploads) == 1 and uploads[0]["suffix"] == ".m4a" and uploads[0]["bytes"] < 400_000


def test_over_duration_cap_splits_in_order_under_the_cap(tmp_path, monkeypatch):
    """A model duration cap forces segments: each fits the cap, transcripts join in order, all audio is kept."""
    monkeypatch.setitem(chunking._OPENAI_MODEL_MAX_SECONDS, "gpt-4o-transcribe", 10.0)
    wav = tmp_path / "meeting.wav"
    _tone_with_pauses(wav, 30)
    result, uploads = _run(wav, "openai", model="gpt-4o-transcribe")
    assert result["success"] is True
    assert result["segments"] == len(uploads) >= 3
    assert result["transcript"] == " ".join(f"part{i}" for i in range(1, len(uploads) + 1))
    assert all(u["model"] == "gpt-4o-transcribe" and u["seconds"] <= 10.0 for u in uploads)
    assert sum(u["seconds"] for u in uploads) == pytest.approx(30.0, abs=0.5)
