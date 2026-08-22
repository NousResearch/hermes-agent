"""Codec guarantees shared by the local TTS providers."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import tts_tool


class _FakePiperVoice:
    @classmethod
    def load(cls, model_path, use_cuda=False):
        return cls()

    def synthesize_wav(self, text, wav_file, syn_config=None):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 2400)


def _fake_piper_config(tmp_path: Path) -> dict:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    return {"piper": {"voice": str(model), "voices_dir": str(tmp_path)}}


def test_local_wav_ogg_conversion_explicitly_selects_opus(monkeypatch, tmp_path):
    wav_path = tmp_path / "voice.wav"
    output_path = tmp_path / "voice.ogg"
    wav_path.write_bytes(b"wav")
    seen = []

    def _run(command, **kwargs):
        seen.append((command, kwargs))
        output_path.write_bytes(b"ogg")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tts_tool.subprocess, "run", _run)

    tts_tool._convert_local_wav_output(str(wav_path), str(output_path))

    command, kwargs = seen[0]
    assert command[command.index("-acodec") + 1] == "libopus"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-f") + 1] == "ogg"
    assert kwargs["check"] is True
    assert not wav_path.exists()
    assert output_path.exists()


def test_local_wav_non_ogg_keeps_normal_ffmpeg_codec_selection(monkeypatch, tmp_path):
    wav_path = tmp_path / "voice.wav"
    output_path = tmp_path / "voice.mp3"
    wav_path.write_bytes(b"wav")
    seen = []

    def _run(command, **kwargs):
        seen.append(command)
        output_path.write_bytes(b"mp3")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tts_tool.subprocess, "run", _run)

    tts_tool._convert_local_wav_output(str(wav_path), str(output_path))

    assert "libopus" not in seen[0]
    assert output_path.exists()


def test_local_wav_conversion_without_ffmpeg_preserves_audio(monkeypatch, tmp_path):
    wav_path = tmp_path / "voice.wav"
    output_path = tmp_path / "voice.ogg"
    wav_path.write_bytes(b"RIFF-audio")
    monkeypatch.setattr(tts_tool.shutil, "which", lambda name: None)

    tts_tool._convert_local_wav_output(str(wav_path), str(output_path))

    assert not wav_path.exists()
    assert output_path.read_bytes() == b"RIFF-audio"


def test_local_wav_cleanup_failure_is_reported(monkeypatch, tmp_path, caplog):
    wav_path = tmp_path / "voice.wav"
    output_path = tmp_path / "voice.ogg"
    wav_path.write_bytes(b"wav")

    def _run(command, **kwargs):
        output_path.write_bytes(b"ogg")
        return subprocess.CompletedProcess(command, 0)

    def _remove(_path):
        raise OSError("file is busy")

    monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tts_tool.subprocess, "run", _run)
    monkeypatch.setattr(tts_tool.os, "remove", _remove)

    with caplog.at_level(logging.WARNING):
        tts_tool._convert_local_wav_output(str(wav_path), str(output_path))

    assert output_path.exists()
    assert "Failed to remove temporary local TTS WAV" in caplog.text
    assert str(wav_path) in caplog.text


def test_neutts_uses_shared_local_conversion(monkeypatch, tmp_path):
    output_path = tmp_path / "voice.ogg"
    converted = MagicMock()

    def _synthesize(command, **kwargs):
        wav_path = Path(command[command.index("--out") + 1])
        wav_path.write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tts_tool.subprocess, "run", _synthesize)
    monkeypatch.setattr(tts_tool, "_convert_local_wav_output", converted)

    assert tts_tool._generate_neutts("hello", str(output_path), {}) == str(output_path)
    converted.assert_called_once_with(str(tmp_path / "voice.wav"), str(output_path))


def test_piper_uses_shared_local_conversion(monkeypatch, tmp_path):
    output_path = tmp_path / "voice.ogg"
    converted = MagicMock()
    tts_tool._piper_voice_cache.clear()
    monkeypatch.setattr(tts_tool, "_import_piper", lambda: _FakePiperVoice)
    monkeypatch.setattr(tts_tool, "_convert_local_wav_output", converted)

    assert tts_tool._generate_piper_tts(
        "hello", str(output_path), _fake_piper_config(tmp_path)
    ) == str(output_path)
    converted.assert_called_once_with(str(tmp_path / "voice.wav"), str(output_path))


def test_kittentts_uses_shared_local_conversion(monkeypatch, tmp_path):
    output_path = tmp_path / "voice.ogg"
    converted = MagicMock()
    model = MagicMock()
    model.generate.return_value = [0.0] * 2400
    fake_soundfile = types.SimpleNamespace(
        write=lambda path, audio, samplerate: Path(path).write_bytes(b"wav")
    )
    tts_tool._kittentts_model_cache.clear()
    monkeypatch.setattr(tts_tool, "_import_kittentts", lambda: MagicMock(return_value=model))
    monkeypatch.setattr(tts_tool, "_convert_local_wav_output", converted)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    assert tts_tool._generate_kittentts("hello", str(output_path), {}) == str(output_path)
    converted.assert_called_once_with(str(tmp_path / "voice.wav"), str(output_path))


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required for codec verification",
)
def test_piper_ogg_output_is_real_opus(monkeypatch, tmp_path):
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if "libopus" not in encoders:
        pytest.skip("ffmpeg build does not include libopus")

    tts_tool._piper_voice_cache.clear()
    monkeypatch.setattr(tts_tool, "_import_piper", lambda: _FakePiperVoice)
    output_path = tmp_path / "voice.ogg"

    tts_tool._generate_piper_tts(
        "hello", str(output_path), _fake_piper_config(tmp_path)
    )

    codec = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ],
        text=True,
    ).strip()
    assert codec == "opus"
