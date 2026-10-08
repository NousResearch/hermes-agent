import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tools import tts_tool as tts_tool_api
from tools import tts_tool_lifecycle
from tools import tts_tool_local as tts_tool
local_tts = tts_tool


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
    if hasattr(local_tts, "_clear_neutts_cache"):
        local_tts._clear_neutts_cache("test")


def test_neutts_warm_cache_is_opt_in(tmp_path, monkeypatch):
    used = {"warm": False, "subprocess": False}

    def fake_warm(text, output_path, tts_config):
        used["warm"] = True
        return output_path

    def fake_subprocess(text, output_path, tts_config):
        used["subprocess"] = True
        return output_path

    monkeypatch.setattr(local_tts, "_generate_neutts_warm", fake_warm)
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

    monkeypatch.setattr(local_tts, "_generate_neutts_warm", fake_warm)
    monkeypatch.setattr(tts_tool, "_generate_neutts_subprocess", fake_subprocess)

    tts_tool._generate_neutts("hello", str(tmp_path / "out.wav"), {"neutts": None})

    assert used == {"warm": False, "subprocess": True}


def test_neutts_null_config_public_path_uses_default_output_format(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        tts_tool_api,
        "_load_tts_config",
        lambda: {"provider": "neutts", "neutts": None},
    )
    monkeypatch.setattr(tts_tool_api, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tts_tool_api, "_check_neutts_available", lambda: True)

    def fake_generate(text, output_path, tts_config):
        Path(output_path).write_bytes(b"not real audio")
        return output_path

    monkeypatch.setattr(tts_tool_api, "_generate_neutts", fake_generate)

    result = json.loads(tts_tool_api.text_to_speech_tool("hello", output_path=None))

    assert result["success"] is True
    assert result["provider"] == "neutts"
    assert result["file_path"].endswith(".mp3")


def test_neutts_warm_cache_config_defaults_to_off():
    from hermes_cli.config import DEFAULT_CONFIG

    neutts_config = DEFAULT_CONFIG["tts"]["neutts"]
    assert neutts_config["codec_repo"] == "neuphonic/neucodec"
    assert neutts_config["output_format"] == "mp3"
    assert neutts_config["warm_cache"] is False
    assert neutts_config["idle_unload_seconds"] == 1800


def test_neutts_tts_lease_prewarms_only_when_opted_in_and_release_clears_cache(tmp_path, monkeypatch):
    _reset_neutts_cache()
    calls = _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    cfg = {
        "provider": "neutts",
        "neutts": {
            "warm_cache": False,
            "idle_unload_seconds": 60,
            "ref_audio": str(ref_audio),
            "ref_text": str(ref_text),
            "model": "fake/model",
            "device": "cpu",
        },
    }
    try:
        assert tts_tool_lifecycle.warm_tts_provider(cfg)["action"] == "noop"
        assert calls == {"construct": 0, "encode": 0, "infer": 0}
        assert not tts_tool._neutts_cache

        cfg["neutts"]["warm_cache"] = True
        warmed = tts_tool_lifecycle.warm_tts_provider(cfg)
        assert warmed["warmed"] is True
        assert warmed["action"] == "loaded"
        assert calls == {"construct": 1, "encode": 1, "infer": 0}
        assert tts_tool._neutts_cache
        assert tts_tool._neutts_idle_timer is not None

        cached = tts_tool_lifecycle.warm_tts_provider(cfg)
        assert cached["action"] == "cached"
        assert calls["construct"] == 1

        cfg["neutts"]["model"] = "fake/replacement"
        replaced = tts_tool_lifecycle.warm_tts_provider(cfg)
        assert replaced["warmed"] is True
        assert replaced["action"] == "loaded"
        assert calls["construct"] == 2
        assert len(tts_tool._neutts_cache) == 1

        assert tts_tool_lifecycle.release_tts_provider("neutts") == {"released": 1}
        assert not tts_tool._neutts_cache
        assert tts_tool._neutts_idle_timer is None
    finally:
        _reset_neutts_cache()


def test_neutts_warm_action_synchronizes_cache_replacement(monkeypatch):
    import threading

    _reset_neutts_cache()
    cache_lock = tts_tool._neutts_cache_lock
    cache = tts_tool._neutts_cache
    barrier = threading.Barrier(2)
    checked = threading.Event()
    result = {}

    def race_cache(_cfg):
        def replace_cache():
            barrier.wait(timeout=5)
            acquired = cache_lock.acquire(blocking=False)
            result["acquired"] = acquired
            if acquired:
                try:
                    cache["concurrent-replacement"] = object()
                finally:
                    cache_lock.release()
            checked.set()

        worker = threading.Thread(target=replace_cache, daemon=True)
        worker.start()
        barrier.wait(timeout=5)
        assert checked.wait(timeout=5)
        worker.join(timeout=5)
        assert not worker.is_alive()

    monkeypatch.setattr(
        tts_tool_lifecycle,
        "_local_tts_warmers",
        lambda: {"neutts": race_cache},
    )
    try:
        warmed = tts_tool_lifecycle.warm_tts_provider(
            {"provider": "neutts", "neutts": {"warm_cache": True}}
        )
        assert warmed["warmed"] is True
        assert result["acquired"] is False
    finally:
        _reset_neutts_cache()


def test_neutts_conversion_uses_windows_safe_subprocess_kwargs(tmp_path, monkeypatch):
    wav_path = tmp_path / "source.wav"
    output_path = tmp_path / "output.mp3"
    wav_path.write_bytes(b"wav")
    captured = {}

    from tools import tts_tool_delivery
    monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tts_tool_delivery, "windows_hide_flags", lambda: 0x08000000)

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        Path(cmd[-1]).write_bytes(b"mp3")
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
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(tts_tool.subprocess, "run", fake_run)
    monkeypatch.setattr(tts_tool, "windows_hide_flags", lambda: 0x08000000)
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
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["creationflags"] == 0x08000000


def test_failed_subprocess_preserves_existing_output_and_wav_sidecar(tmp_path, monkeypatch):
    output_path = tmp_path / "speech.mp3"
    existing_sidecar = tmp_path / "speech.wav"
    output_path.write_bytes(b"existing output")
    existing_sidecar.write_bytes(b"existing user sidecar")

    def fail_helper(command, timeout):
        return types.SimpleNamespace(returncode=1, stderr="synthetic failure")

    monkeypatch.setattr(tts_tool, "_run_helper", fail_helper)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        tts_tool._generate_neutts_subprocess(
            "hello", str(output_path), {"neutts": None}
        )

    assert output_path.read_bytes() == b"existing output"
    assert existing_sidecar.read_bytes() == b"existing user sidecar"
    assert list(tmp_path.glob("hermes-neutts-*.wav")) == []


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


def test_neutts_warm_cache_strips_utf8_bom_from_reference_text(tmp_path, monkeypatch):
    _reset_neutts_cache()
    ref_audio, ref_text = _write_ref_files(tmp_path)
    ref_text.write_bytes(bytes([0xEF, 0xBB, 0xBF]) + b"reference words")
    captured = {}

    class FakeNeuTTS:
        def __init__(self, **kwargs):
            pass

        def encode_reference(self, path):
            return {"path": path}

        def infer(self, text, ref_codes, ref_text_value):
            captured["ref_text"] = ref_text_value
            return [0.0, 0.1, -0.1, 0.0]

    monkeypatch.setitem(sys.modules, "neutts", types.SimpleNamespace(NeuTTS=FakeNeuTTS))
    cfg = {
        "neutts": {
            "warm_cache": True,
            "idle_unload_seconds": 60,
            "ref_audio": str(ref_audio),
            "ref_text": str(ref_text),
        }
    }
    try:
        tts_tool._generate_neutts("hello", str(tmp_path / "bom.wav"), cfg)
        assert captured["ref_text"] == "reference words"
    finally:
        _reset_neutts_cache()


def test_neutts_subprocess_strips_utf8_bom_from_reference_text(tmp_path, monkeypatch):
    from tools import neutts_synth

    ref_audio, ref_text = _write_ref_files(tmp_path)
    ref_text.write_bytes(bytes([0xEF, 0xBB, 0xBF]) + b"reference words")
    captured = {}

    class FakeNeuTTS:
        def __init__(self, **kwargs):
            pass

        def encode_reference(self, path):
            return {"path": path}

        def infer(self, text, ref_codes, ref_text_value):
            captured["ref_text"] = ref_text_value
            return [0.0]

    monkeypatch.setitem(sys.modules, "neutts", types.SimpleNamespace(NeuTTS=FakeNeuTTS))
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(write=lambda *args: None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neutts_synth",
            "--text", "hello",
            "--out", str(tmp_path / "out.wav"),
            "--ref-audio", str(ref_audio),
            "--ref-text", str(ref_text),
        ],
    )

    neutts_synth.main()

    assert captured["ref_text"] == "reference words"


def test_neutts_warm_cache_translates_cuda_for_backbone(tmp_path, monkeypatch):
    _reset_neutts_cache()
    captured = {}
    ref_audio, ref_text = _write_ref_files(tmp_path)

    class FakeNeuTTS:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def encode_reference(self, path):
            return {"path": path}

        def infer(self, text, ref_codes, ref_text_value):
            return [0.0, 0.1, -0.1, 0.0]

    monkeypatch.setitem(sys.modules, "neutts", types.SimpleNamespace(NeuTTS=FakeNeuTTS))
    cfg = {
        "neutts": {
            "warm_cache": True,
            "idle_unload_seconds": 60,
            "ref_audio": str(ref_audio),
            "ref_text": str(ref_text),
            "device": "cuda",
        }
    }

    tts_tool._generate_neutts("hello", str(tmp_path / "out.wav"), cfg)

    assert captured["backbone_device"] == "gpu"
    assert captured["codec_device"] == "cuda"
    _reset_neutts_cache()


def test_neutts_warm_cache_clears_after_conversion_failure(tmp_path, monkeypatch):
    _reset_neutts_cache()
    _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    cfg = {
        "neutts": {
            "warm_cache": True,
            "idle_unload_seconds": 60,
            "ref_audio": str(ref_audio),
            "ref_text": str(ref_text),
        }
    }

    def fail_conversion(wav_path, output_path):
        raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr(tts_tool, "_convert_neutts_wav", fail_conversion)

    out_path = tmp_path / "speech.mp3"
    existing_sidecar = tmp_path / "speech.wav"
    out_path.write_bytes(b"existing output")
    existing_sidecar.write_bytes(b"existing user sidecar")
    with pytest.raises(RuntimeError, match="synthetic conversion failure"):
        tts_tool._generate_neutts("hello", str(out_path), cfg)

    assert not tts_tool._neutts_cache
    assert tts_tool._neutts_idle_timer is None
    assert out_path.read_bytes() == b"existing output"
    assert existing_sidecar.read_bytes() == b"existing user sidecar"
    assert list(tmp_path.glob("hermes-neutts-*.wav")) == []


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


def test_failed_cache_key_transition_releases_stale_model(tmp_path, monkeypatch):
    _reset_neutts_cache()
    _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    base = {
        "warm_cache": True,
        "idle_unload_seconds": 60,
        "ref_audio": str(ref_audio),
        "ref_text": str(ref_text),
        "device": "cpu",
    }

    tts_tool._generate_neutts(
        "hello",
        str(tmp_path / "first.wav"),
        {"neutts": {**base, "model": "fake/first"}},
    )
    assert tts_tool._neutts_cache
    assert tts_tool._neutts_idle_timer is not None

    class FailingReplacementNeuTTS:
        def __init__(self, **kwargs):
            pass

        def encode_reference(self, path):
            raise RuntimeError("synthetic replacement failure")

    monkeypatch.setitem(
        sys.modules,
        "neutts",
        types.SimpleNamespace(NeuTTS=FailingReplacementNeuTTS),
    )

    with pytest.raises(RuntimeError, match="synthetic replacement failure"):
        tts_tool._generate_neutts(
            "again",
            str(tmp_path / "second.wav"),
            {"neutts": {**base, "model": "fake/replacement"}},
        )

    assert not tts_tool._neutts_cache
    assert tts_tool._neutts_idle_timer is None
    assert not (tmp_path / "second.wav").exists()


def test_neutts_idle_unload_clears_cache(tmp_path, monkeypatch):
    _reset_neutts_cache()
    calls = _install_fake_neutts(monkeypatch)
    ref_audio, ref_text = _write_ref_files(tmp_path)
    timers = []

    class FakeTimer:
        def __init__(self, interval, callback, *, args=()):
            self.interval = interval
            self.callback = callback
            self.args = args
            self.daemon = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

        def fire(self):
            self.callback(*self.args)

    monkeypatch.setattr(tts_tool, "_make_neutts_idle_timer", FakeTimer)
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
    entry = next(iter(tts_tool._neutts_cache.values()))
    entry["last_used_at"] -= 1.0
    assert tts_tool._neutts_idle_timer is timers[-1]
    timers[-1].fire()
    assert not tts_tool._neutts_cache
    assert tts_tool._neutts_idle_timer is None

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

    out_path = tmp_path / "speech.mp3"
    existing_sidecar = tmp_path / "speech.wav"
    out_path.write_bytes(b"existing output")
    existing_sidecar.write_bytes(b"existing user sidecar")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        tts_tool._generate_neutts("hello", str(out_path), cfg)

    assert not tts_tool._neutts_cache
    assert tts_tool._neutts_idle_timer is None
    assert out_path.read_bytes() == b"existing output"
    assert existing_sidecar.read_bytes() == b"existing user sidecar"
    assert list(tmp_path.glob("hermes-neutts-*.wav")) == []


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

    monkeypatch.setattr(tts_tool_api, "_load_tts_config", lambda: {
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
    monkeypatch.setattr(tts_tool_api, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tts_tool_api, "_check_neutts_available", lambda: True)
    monkeypatch.setattr(tts_tool.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    def fake_run(cmd, check=False, timeout=None, **kwargs):
        converted["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"m4a")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tts_tool.subprocess, "run", fake_run)

    result = json.loads(tts_tool_api.text_to_speech_tool("hello", output_path=None))

    assert result["success"] is True
    assert result["file_path"].endswith(".m4a")
    assert result["provider"] == "neutts"
    assert calls["infer"] == 1
    assert "aac" in converted["cmd"] or "-c:a" in converted["cmd"]
    _reset_neutts_cache()
