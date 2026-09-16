"""LuxTTS provider contracts: one resident runtime and 48 kHz sentence streaming."""

from types import SimpleNamespace

import numpy as np

from tools import tts_streaming, tts_tool, tts_tool_lifecycle, tts_tool_local


def _config(ref_audio, **overrides):
    section = {
        "model": "test/LuxTTS",
        "ref_audio": str(ref_audio),
        "consent_confirmed": True,
        "device": "cuda",
        "threads": 2,
        "ref_duration": 5,
        **overrides,
    }
    return {"provider": "luxtts", "luxtts": section}


def test_warm_and_repeated_sentences_share_model_and_encoded_prompt(monkeypatch, tmp_path):
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"RIFF-test")
    calls = {"loads": 0, "encodes": 0, "sentences": []}

    class FakeLuxTTS:
        def __init__(self, model, **kwargs):
            calls["loads"] += 1
            calls["init"] = (model, kwargs)

        def encode_prompt(self, path, **kwargs):
            calls["encodes"] += 1
            return (path, kwargs)

        def generate_speech(self, text, prompt, **kwargs):
            calls["sentences"].append((text, prompt, kwargs))
            return np.array([[0.0, 0.25, -0.25]], dtype=np.float32)

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        mps=SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setattr(tts_tool, "_import_luxtts", lambda: FakeLuxTTS)
    monkeypatch.setattr(tts_tool, "_import_torch", lambda: fake_torch)
    tts_tool_local._luxtts_runtime_cache.clear()
    cfg = _config(ref_audio)

    warmed = tts_tool_lifecycle.warm_tts_provider(cfg)
    first = tts_tool_local._generate_luxtts_waveform("First sentence.", cfg)
    second = tts_tool_local._generate_luxtts_waveform("Second sentence.", cfg)

    assert warmed["action"] == "loaded"
    assert calls["loads"] == calls["encodes"] == 1
    assert calls["init"] == ("test/LuxTTS", {"device": "cpu", "threads": 2})
    assert [entry[0] for entry in calls["sentences"]] == ["First sentence.", "Second sentence."]
    assert first.shape == second.shape == (3,)
    assert tts_tool_lifecycle.release_tts_provider("luxtts") == {"released": 1}
    assert tts_tool_local._luxtts_runtime_cache == {}


def test_streamer_requires_consent_and_emits_48khz_int16_pcm(monkeypatch, tmp_path):
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"RIFF-test")
    monkeypatch.setattr(tts_tool, "_check_luxtts_available", lambda: True)
    monkeypatch.setattr(
        tts_tool_local, "_generate_luxtts_waveform",
        lambda text, cfg: np.array([-1.0, 0.0, 1.0], dtype=np.float32),
    )

    without_consent = _config(ref_audio)
    without_consent["luxtts"]["consent_confirmed"] = False
    assert tts_streaming.resolve_streaming_provider(without_consent) is None

    streamer = tts_streaming.resolve_streaming_provider(_config(ref_audio))
    assert isinstance(streamer, tts_streaming.LuxTTSStreamer)
    assert streamer.sample_rate == 48000
    pcm = b"".join(streamer.stream("A completed sentence."))
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [-32767, 0, 32767]
