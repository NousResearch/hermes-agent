"""Public Photon voice eligibility must not change internal Opus encoding decisions."""

import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest

from gateway.session_context import _VAR_MAP
from tools import tts_tool
from tools.registry import registry
from tools.tts_tool_delivery import _wrap_pcm_as_wav, _write_wav_bytes_as


@contextmanager
def _platform(name):
    var = _VAR_MAP["HERMES_SESSION_PLATFORM"]
    token = var.set(name)
    try:
        yield
    finally:
        var.reset(token)


@pytest.fixture
def synthetic_audio(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg isn't available")
    wav = _wrap_pcm_as_wav(b"\x00\x00" * 2400)

    def synthesize(_text, output_path, _config):
        return _write_wav_bytes_as(wav, output_path)

    monkeypatch.setattr(tts_tool, "_generate_gemini_tts", synthesize)
    monkeypatch.setattr(tts_tool, "_default_output_dir", lambda: str(tmp_path))
    return synthesize


def _invoke(text="Hello.", **kwargs):
    return json.loads(registry.get_entry("text_to_speech").handler({"text": text, **kwargs}))


@pytest.mark.parametrize("suffix", [".mp3", ".wav"])
@pytest.mark.parametrize("multiple", [False, True])
def test_photon_voice_eligibility_after_native_packing(tmp_path, monkeypatch, synthetic_audio, suffix, multiple):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    config = {"provider": "gemini", "gemini": {"max_text_length": 7 if multiple else 32000}}
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
    convert = Mock(side_effect=AssertionError("Photon must not request Opus conversion"))
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)
    pack = Mock(wraps=tts_tool._build_audio_delivery_files)
    monkeypatch.setattr(tts_tool, "_build_audio_delivery_files", pack)
    kwargs = {"output_path": str(tmp_path / "voice.wav")} if suffix == ".wav" else {}
    with _platform("photon"):
        result = _invoke("Hello. Again." if multiple else "Hello.", **kwargs)
    assert result["success"] is True
    assert result["chunk_count"] == (2 if multiple else 1)
    assert result["voice_compatible"] is True
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{result['file_path']}"
    assert result["delivery_profile"]["platform"] == "photon"
    assert pack.call_args.kwargs["voice_compatible"] is False
    assert result["combined_chunks"] is multiple
    output = Path(result["file_path"])
    assert output.suffix == suffix
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(output), "-f", "null", "-"],
                           capture_output=True, text=True, check=True, timeout=10)
    assert ("Audio: mp3" if suffix == ".mp3" else "Audio: pcm_s16le") in probe.stderr
    assert "opus" not in probe.stderr.lower()
    convert.assert_not_called()
    assert "delivered" not in result


@pytest.mark.parametrize("platform,suffix,compatible", [
    ("telegram", ".ogg", True), ("matrix", ".ogg", True), ("", ".mp3", False),
])
def test_other_bound_platforms_keep_existing_result(monkeypatch, synthetic_audio, platform, suffix, compatible):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "photon")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
    pack = Mock(wraps=tts_tool._build_audio_delivery_files)
    monkeypatch.setattr(tts_tool, "_build_audio_delivery_files", pack)
    with _platform(platform):
        result = _invoke()
    assert result["success"] is True
    assert result["voice_compatible"] is compatible
    assert pack.call_args.kwargs["voice_compatible"] is compatible
    assert Path(result["file_path"]).suffix == suffix
    assert result["media_tag"].startswith("[[audio_as_voice]]") is compatible


@pytest.mark.parametrize("kind", ["command", "plugin"])
@pytest.mark.parametrize("opted_in", [False, True])
def test_photon_custom_provider_opt_in_is_unchanged(tmp_path, monkeypatch, kind, opted_in):
    config = {"provider": "synthetic"}
    if kind == "command":
        config["providers"] = {"synthetic": {
            "type": "command", "command": "never-executed", "voice_compatible": opted_in}}
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
    monkeypatch.setattr(tts_tool, "_default_output_dir", lambda: str(tmp_path))

    def synthesize(_text, path, *_args):
        Path(path).write_bytes(b"ID3synthetic")
        return path

    monkeypatch.setattr(tts_tool, "_generate_command_tts", synthesize)
    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", synthesize)
    monkeypatch.setattr(tts_tool, "_plugin_provider_is_voice_compatible", lambda _provider: opted_in)

    def to_opus(path):
        output = Path(path).with_suffix(".ogg")
        output.write_bytes(b"OggSsynthetic")
        return str(output)

    convert = Mock(side_effect=to_opus)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)
    with _platform("photon"):
        result = _invoke()
    assert result["success"] is True
    assert result["voice_compatible"] is opted_in
    assert result["media_tag"].startswith("[[audio_as_voice]]") is opted_in
    assert convert.call_count == int(opted_in)


def test_photon_does_not_label_non_audio_output_voice_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
    monkeypatch.setattr(tts_tool, "_generate_gemini_tts", lambda _text, path, _cfg: Path(path).write_bytes(b"text"))
    with _platform("photon"):
        result = _invoke(output_path=str(tmp_path / "not-audio.txt"))
    assert result["success"] is True
    assert result["voice_compatible"] is False
    assert result["media_tag"] == f"MEDIA:{result['file_path']}"


@pytest.mark.parametrize("failure", ["generation", "conversion", "packing"])
def test_photon_failures_do_not_claim_success(monkeypatch, synthetic_audio, failure):
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "gemini"})
    error = Mock(side_effect=RuntimeError(f"synthetic {failure} failure"))
    seam = "_build_audio_delivery_files" if failure == "packing" else "_generate_gemini_tts"
    if failure == "conversion":
        monkeypatch.setattr(tts_tool, seam, lambda _text, path, _cfg: _write_wav_bytes_as(b"not-wav", path))
    else:
        monkeypatch.setattr(tts_tool, seam, error)
    with _platform("photon"):
        result = _invoke()
    assert result["success"] is False
    assert ("ffmpeg conversion failed" if failure == "conversion" else f"synthetic {failure} failure") in result["error"]
    assert "voice_compatible" not in result
    assert "media_tag" not in result
