import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

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


def test_neutts_warm_cache_is_opt_in(tmp_path, monkeypatch):
    used = {"warm": False, "subprocess": False}

    def fake_warm(text, output_path, tts_config):
        used["warm"] = True
        return output_path

    def fake_subprocess(text, output_path, tts_config):
        used["subprocess"] = True
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_neutts_warm", fake_warm)
    monkeypatch.setattr(tts_tool, "_generate_neutts_subprocess", fake_subprocess)

    tts_tool._generate_neutts("hello", str(tmp_path / "out.wav"), {"neutts": {}})

    assert used == {"warm": False, "subprocess": True}


def test_neutts_null_config_uses_subprocess_default(tmp_path, monkeypatch):
    used = {"warm": False, "subprocess": False}

    def fake_warm(text, output_path, tts_config):
        used["warm"] = True
        return output_path

    def fake_subprocess(text, output_path, tts_config):
        used["subprocess"] = True
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_neutts_warm", fake_warm)
    monkeypatch.setattr(tts_tool, "_generate_neutts_subprocess", fake_subprocess)

    tts_tool._generate_neutts("hello", str(tmp_path / "out.wav"), {"neutts": None})

    assert used == {"warm": False, "subprocess": True}


def test_neutts_warm_cache_config_defaults_to_off():
    from hermes_cli.config import DEFAULT_CONFIG

    neutts_config = DEFAULT_CONFIG["tts"]["neutts"]
    assert neutts_config["codec_repo"] == "neuphonic/neucodec"
    assert neutts_config["output_format"] == "mp3"
    assert neutts_config["warm_cache"] is False
    assert neutts_config["idle_unload_seconds"] == 1800


def test_neutts_conversion_uses_windows_safe_subprocess_kwargs(tmp_path, monkeypatch):
    wav_path = tmp_path / "source.wav"
    output_path = tmp_path / "output.mp3"
    wav_path.write_bytes(b"wav")
    captured = {}

    monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tts_tool, "windows_hide_flags", lambda: 0x08000000)

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        output_path.write_bytes(b"mp3")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tts_tool.subprocess, "run", fake_run)

    result = tts_tool._convert_neutts_wav(str(wav_path), str(output_path))

    assert result == str(output_path)
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["creationflags"] == 0x08000000


def test_neutts_subprocess_passes_custom_codec_repo(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(tts_tool.subprocess, "run", fake_run)
    monkeypatch.setattr(
        tts_tool,
        "_convert_neutts_wav",
        lambda wav_path, output_path: output_path,
    )

    output_path = str(tmp_path / "output.wav")
    result = tts_tool._generate_neutts_subprocess(
        "hello",
        output_path,
        {"neutts": {"codec_repo": "custom/codec"}},
    )

    assert result == output_path
    codec_index = captured["cmd"].index("--codec-repo")
    assert captured["cmd"][codec_index + 1] == "custom/codec"


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


def test_failed_first_warm_synthesis_clears_cache(tmp_path, monkeypatch):
    _reset_neutts_cache()
    ref_audio, ref_text = _write_ref_files(tmp_path)

    class FailingNeuTTS:
        def __init__(self, **kwargs):
            pass

        def encode_reference(self, path):
            return {"path": path}

        def infer(self, text, ref_codes, ref_text_value):
            raise RuntimeError("synthetic failure")

    monkeypatch.setitem(
        sys.modules,
        "neutts",
        types.SimpleNamespace(NeuTTS=FailingNeuTTS),
    )
    cfg = {
        "neutts": {
            "warm_cache": True,
            "idle_unload_seconds": 60,
            "ref_audio": str(ref_audio),
            "ref_text": str(ref_text),
        }
    }

    with pytest.raises(RuntimeError, match="synthetic failure"):
        tts_tool._generate_neutts("hello", str(tmp_path / "out.wav"), cfg)

    assert not tts_tool._neutts_cache
    assert tts_tool._neutts_idle_timer is None


def test_neutts_warm_cache_can_be_disabled(tmp_path, monkeypatch):
    _reset_neutts_cache()
    used = {"subprocess": False}
    _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    tts_tool._generate_neutts(
        "warm",
        str(tmp_path / "warm.wav"),
        {
            "neutts": {
                "warm_cache": True,
                "idle_unload_seconds": 60,
                "ref_audio": str(ref_audio),
                "ref_text": str(ref_text),
            }
        },
    )
    assert tts_tool._neutts_cache

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
    assert not tts_tool._neutts_cache
    assert tts_tool._neutts_idle_timer is None


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
