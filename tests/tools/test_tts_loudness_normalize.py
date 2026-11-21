"""Opt-in TTS loudness normalization (``tts.normalize_loudness``).

Voices differ by tens of LU across providers and inside one voice library, so a voice note can
be inaudible on a phone while the next is loud. When the option is on, every synthesized file
gets one ffmpeg EBU R128 ``loudnorm`` pass in its own container; off (the default), or without
ffmpeg, the file is delivered byte-for-byte as the provider wrote it.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from gateway.session_context import _UNSET, _VAR_MAP
from tools import tts_tool, tts_tool_delivery
from tools.tts_tool_delivery import _normalize_loudness, _normalize_loudness_enabled


@pytest.fixture(autouse=True)
def _clean_session_platform(monkeypatch):
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


@pytest.mark.parametrize("raw,expected", [
    (None, False), (False, False), (True, True), ("true", True), ("On", True), ("yes", True),
    ("off", False), ("no", False), ("loud", False), (1, True), (0, False), ([], False),
])
def test_config_spellings(raw, expected):
    assert _normalize_loudness_enabled({"normalize_loudness": raw}) is expected
    assert _normalize_loudness_enabled(None) is False


def test_off_by_default_never_runs_ffmpeg(tmp_path, monkeypatch):
    audio = tmp_path / "v.mp3"
    audio.write_bytes(b"original")
    run = Mock()
    monkeypatch.setattr(tts_tool_delivery, "_ffmpeg_run", run)

    assert _normalize_loudness(str(audio), {"provider": "edge"}) == str(audio)
    assert audio.read_bytes() == b"original"
    run.assert_not_called()


def test_missing_ffmpeg_keeps_the_file(tmp_path, monkeypatch, caplog):
    audio = tmp_path / "v.ogg"
    audio.write_bytes(b"original")
    monkeypatch.setattr(tts_tool_delivery.shutil, "which", lambda _name: None)

    with caplog.at_level("WARNING", logger="tools.tts_tool"):
        assert _normalize_loudness(str(audio), {"normalize_loudness": True}) == str(audio)
    assert audio.read_bytes() == b"original"
    assert "ffmpeg is not installed" in caplog.text


def test_failed_pass_keeps_the_file_and_removes_the_temp(tmp_path, monkeypatch):
    audio = tmp_path / "v.mp3"
    audio.write_bytes(b"original")
    monkeypatch.setattr(tts_tool_delivery.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def fail(_ffmpeg, args, **_kw):
        Path(args[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(args, 1, b"", b"boom")

    monkeypatch.setattr(tts_tool_delivery, "_ffmpeg_run", fail)

    assert _normalize_loudness(str(audio), {"normalize_loudness": True}) == str(audio)
    assert audio.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.loudnorm*"))


def test_timeout_keeps_the_file(tmp_path, monkeypatch):
    audio = tmp_path / "v.mp3"
    audio.write_bytes(b"original")
    monkeypatch.setattr(tts_tool_delivery.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def hang(_ffmpeg, args, **_kw):
        raise subprocess.TimeoutExpired(args, 120)

    monkeypatch.setattr(tts_tool_delivery, "_ffmpeg_run", hang)
    assert _normalize_loudness(str(audio), {"normalize_loudness": True}) == str(audio)
    assert audio.read_bytes() == b"original"


@pytest.mark.parametrize("suffix,codec_args", [
    (".ogg", ["-acodec", "libopus"]), (".mp3", ["-acodec", "libmp3lame"]), (".wav", []),
])
def test_encodes_in_the_same_container(tmp_path, monkeypatch, suffix, codec_args):
    audio = tmp_path / f"v{suffix}"
    audio.write_bytes(b"original")
    calls = []
    monkeypatch.setattr(tts_tool_delivery.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def ok(_ffmpeg, args, **_kw):
        calls.append(args)
        Path(args[-1]).write_bytes(b"normalized")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(tts_tool_delivery, "_ffmpeg_run", ok)

    assert _normalize_loudness(str(audio), {"normalize_loudness": True}) == str(audio)
    assert audio.read_bytes() == b"normalized"
    (args,) = calls
    assert args[args.index("-af") + 1].startswith("loudnorm=I=-16")
    assert args[-1].endswith(f".loudnorm{suffix}")
    assert "-ar" in args  # loudnorm otherwise upsamples to 192 kHz
    joined = " ".join(args)
    assert all(arg in joined for arg in codec_args)


def test_tool_applies_it_to_every_synthesized_file(tmp_path, monkeypatch):
    async def _write_edge_output(_text, output_path, _cfg):
        Path(output_path).write_bytes(b"mp3")
        return output_path

    seen = []

    def normalize(path, cfg):
        seen.append((path, cfg.get("normalize_loudness")))
        return path

    out = tmp_path / "speech.mp3"
    monkeypatch.setattr(tts_tool, "_load_tts_config",
                        lambda: {"provider": "edge", "normalize_loudness": True})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", _write_edge_output)
    monkeypatch.setattr(tts_tool, "_normalize_loudness", normalize)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=str(out)))

    assert result["success"] is True
    assert seen == [(str(out), True)]


def _ffmpeg_or_skip():
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")


def _integrated_lufs(path: Path) -> float:
    probe = subprocess.run(
        ["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True, check=False, timeout=60)
    values = re.findall(r"I:\s+(-?[\d.]+) LUFS", probe.stderr)
    assert values, probe.stderr[-500:]
    return float(values[-1])


@pytest.mark.parametrize("suffix,encoder", [(".ogg", "libopus"), (".mp3", "libmp3lame")])
def test_real_ffmpeg_lifts_a_quiet_voice_note(tmp_path, suffix, encoder):
    """Live ffmpeg round-trip when the binary (and encoder) exist; skipped otherwise."""
    _ffmpeg_or_skip()
    audio = tmp_path / f"quiet{suffix}"
    synth = subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=24000:duration=3",
         "-af", "volume=-40dB", "-acodec", encoder, str(audio), "-y"],
        capture_output=True, check=False, timeout=60)
    if synth.returncode != 0:
        pytest.skip(f"ffmpeg on PATH cannot encode {encoder} (exit {synth.returncode})")
    before = _integrated_lufs(audio)

    _normalize_loudness(str(audio), {"normalize_loudness": True})

    after = _integrated_lufs(audio)
    assert before < -40
    assert abs(after - (-16)) < 2.5, (before, after)
    assert not list(tmp_path.glob("*.loudnorm*"))
    if suffix == ".ogg":
        from tools.tts_tool_delivery import _sniff_audio_container
        assert _sniff_audio_container(str(audio)) == "ogg"
