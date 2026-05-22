import json
import sys
import time
import types
from pathlib import Path

from tools import tts_tool


def _write_ref_files(tmp_path: Path):
    ref_audio = tmp_path / "ref.wav"
    ref_text = tmp_path / "ref.txt"
    ref_audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    ref_text.write_text("reference words", encoding="utf-8")
    return ref_audio, ref_text


def _install_fake_neutts(monkeypatch):
    calls = {"construct": 0, "encode": 0, "infer": 0}

    class FakeNeuTTS:
        def __init__(self, **kwargs):
            calls["construct"] += 1
            self.kwargs = kwargs

        def encode_reference(self, path):
            calls["encode"] += 1
            return {"path": path}

        def infer(self, text, ref_codes, ref_text):
            calls["infer"] += 1
            return [0.0, 0.1, -0.1, 0.0]

    monkeypatch.setitem(sys.modules, "neutts", types.SimpleNamespace(NeuTTS=FakeNeuTTS))
    return calls


def _reset_neutts_cache():
    if hasattr(tts_tool, "_clear_neutts_cache"):
        tts_tool._clear_neutts_cache("test")


def test_neutts_warm_cache_reuses_model_and_reference(tmp_path, monkeypatch):
    _reset_neutts_cache()
    calls = _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    cfg = {
        "neutts": {
            "warm_cache": True,
            "idle_unload_seconds": 60,
            "ref_audio": str(ref_audio),
            "ref_text": str(ref_text),
            "model": "fake/model",
            "device": "cpu",
        }
    }

    out1 = tts_tool._generate_neutts("hello", str(tmp_path / "one.wav"), cfg)
    out2 = tts_tool._generate_neutts("again", str(tmp_path / "two.wav"), cfg)

    assert Path(out1).exists()
    assert Path(out2).exists()
    assert calls == {"construct": 1, "encode": 1, "infer": 2}
    _reset_neutts_cache()


def test_neutts_warm_cache_key_changes_on_model(tmp_path, monkeypatch):
    _reset_neutts_cache()
    calls = _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    base = {
        "warm_cache": True,
        "idle_unload_seconds": 60,
        "ref_audio": str(ref_audio),
        "ref_text": str(ref_text),
        "device": "cpu",
    }

    tts_tool._generate_neutts("hello", str(tmp_path / "one.wav"), {"neutts": {**base, "model": "fake/a"}})
    tts_tool._generate_neutts("hello", str(tmp_path / "two.wav"), {"neutts": {**base, "model": "fake/b"}})

    assert calls["construct"] == 2
    assert calls["encode"] == 2
    assert calls["infer"] == 2
    _reset_neutts_cache()


def test_neutts_idle_unload_clears_cache(tmp_path, monkeypatch):
    _reset_neutts_cache()
    calls = _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    cfg = {
        "neutts": {
            "warm_cache": True,
            "idle_unload_seconds": 0.05,
            "ref_audio": str(ref_audio),
            "ref_text": str(ref_text),
            "model": "fake/model",
            "device": "cpu",
        }
    }

    tts_tool._generate_neutts("hello", str(tmp_path / "one.wav"), cfg)
    time.sleep(0.15)
    assert not getattr(tts_tool, "_neutts_cache", {})

    tts_tool._generate_neutts("again", str(tmp_path / "two.wav"), cfg)
    assert calls["construct"] == 2
    _reset_neutts_cache()


def test_neutts_warm_cache_can_be_disabled(tmp_path, monkeypatch):
    _reset_neutts_cache()
    used = {"subprocess": False}

    def fake_subprocess(text, output_path, tts_config):
        used["subprocess"] = True
        Path(output_path).write_bytes(b"not real audio")
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_neutts_subprocess", fake_subprocess, raising=False)

    out = tts_tool._generate_neutts(
        "hello",
        str(tmp_path / "out.mp3"),
        {"neutts": {"warm_cache": False}},
    )

    assert used["subprocess"] is True
    assert out.endswith(".mp3")


def test_neutts_output_format_m4a_uses_aac_container(tmp_path, monkeypatch):
    _reset_neutts_cache()
    calls = _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    converted = {"cmd": None}

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {
        "provider": "neutts",
        "neutts": {
            "warm_cache": True,
            "ref_audio": str(ref_audio),
            "ref_text": str(ref_text),
            "model": "fake/model",
            "device": "cpu",
            "output_format": "m4a",
        },
    })
    monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: True)
    monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    def fake_run(cmd, check=False, timeout=None, **kwargs):
        converted["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"m4a")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tts_tool.subprocess, "run", fake_run)

    result = json.loads(tts_tool.text_to_speech_tool("hello", output_path=None))

    assert result["success"] is True
    assert result["file_path"].endswith(".m4a")
    assert result["provider"] == "neutts"
    assert calls["infer"] == 1
    assert "aac" in converted["cmd"] or "-c:a" in converted["cmd"]
    _reset_neutts_cache()
