"""TTS engine lifecycle driven by speech-output toggles (issue #100881).

Local engines load lazily on first synthesis, so the first spoken reply after
"read replies aloud" / voice conversation turns on pays the model load as dead
air. The toggles now hold *leases*: acquiring warms the configured provider
into the SAME cache slot synthesis reads; releasing the last lease unloads
resident local models.
"""

from __future__ import annotations

import pytest

from tools import tts_tool


@pytest.fixture(autouse=True)
def _clean_lifecycle(monkeypatch):
    tts_tool._reset_tts_leases_for_tests()
    for cache in tts_tool._LOCAL_TTS_MODEL_CACHES.values():
        cache.clear()
    yield
    tts_tool._reset_tts_leases_for_tests()
    for cache in tts_tool._LOCAL_TTS_MODEL_CACHES.values():
        cache.clear()


class _FakePiperVoice:
    loads = 0
    synthesized: list = []

    @classmethod
    def load(cls, model_path, use_cuda=False):
        cls.loads += 1
        inst = cls()
        inst.model_path = model_path
        return inst

    def synthesize_wav(self, text, wav_file, syn_config=None):
        type(self).synthesized.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)


@pytest.fixture
def fake_piper(monkeypatch, tmp_path):
    _FakePiperVoice.loads = 0
    _FakePiperVoice.synthesized = []
    monkeypatch.setattr(tts_tool, "_import_piper", lambda: _FakePiperVoice)
    # Pretend the voice is already on disk so no download subprocess runs.
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "en_US-test-medium.onnx").write_bytes(b"onnx")
    (voices_dir / "en_US-test-medium.onnx.json").write_text("{}")
    cfg = {"provider": "piper", "piper": {"voice": "en_US-test-medium", "voices_dir": str(voices_dir)}}
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: cfg)
    return cfg


# --------------------------------------------------------------------------
# warm_tts_provider: warm-up populates the exact slot synthesis reads
# --------------------------------------------------------------------------


def test_warm_loads_piper_into_synthesis_cache(fake_piper, tmp_path):
    result = tts_tool.warm_tts_provider(fake_piper)

    assert result["warmed"] is True
    assert result["action"] == "loaded"
    assert result["provider"] == "piper"
    assert _FakePiperVoice.loads == 1
    assert len(tts_tool._piper_voice_cache) == 1

    # The load that would have happened on the first reply is already done:
    # synthesis reuses the warmed instance without loading again.
    out = tts_tool._generate_piper_tts("hello", str(tmp_path / "out.wav"), fake_piper)
    assert out.endswith(".wav")
    assert _FakePiperVoice.loads == 1
    assert _FakePiperVoice.synthesized == ["hello"]


def test_warm_twice_is_a_cache_hit(fake_piper):
    tts_tool.warm_tts_provider(fake_piper)
    second = tts_tool.warm_tts_provider(fake_piper)

    assert second["action"] == "cached"
    assert _FakePiperVoice.loads == 1


def test_warm_reads_configured_provider_when_none_given(fake_piper):
    result = tts_tool.warm_tts_provider()
    assert result["provider"] == "piper"
    assert result["action"] == "loaded"


def test_warm_never_raises_on_engine_failure(monkeypatch):
    def _boom():
        raise ImportError("No module named 'piper'")

    monkeypatch.setattr(tts_tool, "_import_piper", _boom)
    result = tts_tool.warm_tts_provider({"provider": "piper"})

    assert result["warmed"] is False
    assert result["action"] == "error"
    assert "piper" in result["error"]
    assert tts_tool._piper_voice_cache == {}


def test_warm_is_noop_for_cloud_provider_without_lazy_sdk(monkeypatch):
    result = tts_tool.warm_tts_provider({"provider": "openai"})
    assert result == {"provider": "openai", "warmed": False, "action": "noop"}


def test_warm_lazy_sdk_provider_reports_cached_when_installed(monkeypatch):
    import types

    fake = types.SimpleNamespace(
        is_available=lambda feature: feature == "tts.edge",
        ensure=lambda *a, **k: pytest.fail("ensure must not run when the SDK is present"),
    )
    monkeypatch.setitem(__import__("sys").modules, "tools.lazy_deps", fake)
    result = tts_tool.warm_tts_provider({"provider": "edge"})
    assert result["warmed"] is True
    assert result["action"] == "cached"


def test_warm_lazy_sdk_provider_installs_when_missing(monkeypatch):
    import types

    calls = []
    fake = types.SimpleNamespace(
        is_available=lambda feature: False,
        ensure=lambda feature, prompt: calls.append((feature, prompt)),
    )
    monkeypatch.setitem(__import__("sys").modules, "tools.lazy_deps", fake)
    result = tts_tool.warm_tts_provider({"provider": "edge"})
    assert result["action"] == "installed"
    assert calls == [("tts.edge", False)]


# --------------------------------------------------------------------------
# release_tts_provider
# --------------------------------------------------------------------------


def test_release_drops_every_local_cache(fake_piper):
    tts_tool.warm_tts_provider(fake_piper)
    tts_tool._kittentts_model_cache["m"] = object()

    assert tts_tool.release_tts_provider() == {"released": 2}
    assert tts_tool._piper_voice_cache == {}
    assert tts_tool._kittentts_model_cache == {}


def test_release_scoped_to_one_provider(fake_piper):
    tts_tool.warm_tts_provider(fake_piper)
    tts_tool._kittentts_model_cache["m"] = object()

    assert tts_tool.release_tts_provider("kittentts") == {"released": 1}
    assert len(tts_tool._piper_voice_cache) == 1


def test_release_with_nothing_resident_is_zero():
    assert tts_tool.release_tts_provider() == {"released": 0}


# --------------------------------------------------------------------------
# Leases: warm on acquire, unload only when the LAST holder releases
# --------------------------------------------------------------------------


def test_acquire_warms_and_counts(fake_piper):
    result = tts_tool.acquire_tts_lease("desktop:read-aloud")
    assert result["leases"] == 1
    assert result["action"] == "loaded"
    assert tts_tool.tts_lease_holders() == ["desktop:read-aloud"]


def test_last_release_unloads_but_earlier_release_does_not(fake_piper):
    tts_tool.acquire_tts_lease("desktop:read-aloud")
    tts_tool.acquire_tts_lease("tui:voice-tts")
    assert len(tts_tool._piper_voice_cache) == 1

    # One surface turning speech off must not pull the model from under the
    # other surface that still speaks through this process.
    first = tts_tool.release_tts_lease("desktop:read-aloud")
    assert first == {"leases": 1, "released": 0}
    assert len(tts_tool._piper_voice_cache) == 1

    last = tts_tool.release_tts_lease("tui:voice-tts")
    assert last == {"leases": 0, "released": 1}
    assert tts_tool._piper_voice_cache == {}


def test_reacquire_is_idempotent_and_reheals_cache(fake_piper):
    tts_tool.acquire_tts_lease("cli:voice-tts")
    tts_tool.release_tts_provider()  # something else dropped the model
    result = tts_tool.acquire_tts_lease("cli:voice-tts")

    assert result["leases"] == 1
    assert result["action"] == "loaded"
    assert _FakePiperVoice.loads == 2


def test_release_unknown_lease_is_noop(fake_piper):
    tts_tool.acquire_tts_lease("a")
    assert tts_tool.release_tts_lease("never-acquired") == {"leases": 1, "released": 0}
    assert len(tts_tool._piper_voice_cache) == 1


def test_acquire_failure_still_registers_lease(monkeypatch):
    def _boom():
        raise RuntimeError("engine missing")

    monkeypatch.setattr(tts_tool, "_import_piper", _boom)
    result = tts_tool.acquire_tts_lease("desktop:conversation", {"provider": "piper"})
    assert result["action"] == "error"
    assert result["leases"] == 1
    assert tts_tool.tts_lease_holders() == ["desktop:conversation"]


# --------------------------------------------------------------------------
# Registry invariant: every local engine cache is release-able
# --------------------------------------------------------------------------


def test_every_local_warmer_has_a_registered_cache():
    warmers = tts_tool._local_tts_warmers()
    assert set(warmers) == set(tts_tool._LOCAL_TTS_MODEL_CACHES)
    assert tts_tool._LOCAL_TTS_MODEL_CACHES["piper"] is tts_tool._piper_voice_cache
    assert tts_tool._LOCAL_TTS_MODEL_CACHES["kittentts"] is tts_tool._kittentts_model_cache
