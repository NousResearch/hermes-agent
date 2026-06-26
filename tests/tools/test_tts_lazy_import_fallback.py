"""Regression tests for TTS lazy import fallback behavior."""

import sys
import types


def _make_ensure_failure(*_args, **_kwargs):
    raise RuntimeError("lazy install unavailable")


def test_edge_tts_raw_import_still_runs_when_lazy_deps_ensure_fails(monkeypatch):
    """edge-tts may be importable from PYTHONPATH even when lazy install fails."""
    import tools.lazy_deps as lazy_deps
    from tools.tts_tool import _import_edge_tts

    fake_edge_tts = types.ModuleType("edge_tts")
    monkeypatch.setitem(sys.modules, "edge_tts", fake_edge_tts)
    monkeypatch.setattr(lazy_deps, "ensure", _make_ensure_failure)

    assert _import_edge_tts() is fake_edge_tts


def test_elevenlabs_raw_import_still_runs_when_lazy_deps_ensure_fails(monkeypatch):
    """ElevenLabs may be importable from PYTHONPATH even when lazy install fails."""
    import tools.lazy_deps as lazy_deps
    from tools.tts_tool import _import_elevenlabs

    class FakeElevenLabs:
        pass

    fake_elevenlabs = types.ModuleType("elevenlabs")
    fake_client = types.ModuleType("elevenlabs.client")
    setattr(fake_client, "ElevenLabs", FakeElevenLabs)
    setattr(fake_elevenlabs, "client", fake_client)
    monkeypatch.setitem(sys.modules, "elevenlabs", fake_elevenlabs)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", fake_client)
    monkeypatch.setattr(lazy_deps, "ensure", _make_ensure_failure)

    assert _import_elevenlabs() is FakeElevenLabs


def test_mistral_raw_import_still_runs_when_lazy_deps_ensure_fails(monkeypatch):
    """Mistral uses the same lazy import pattern and should preserve fallback."""
    import tools.lazy_deps as lazy_deps
    from tools.tts_tool import _import_mistral_client

    class FakeMistral:
        pass

    fake_mistralai = types.ModuleType("mistralai")
    fake_client = types.ModuleType("mistralai.client")
    setattr(fake_client, "Mistral", FakeMistral)
    setattr(fake_mistralai, "client", fake_client)
    monkeypatch.setitem(sys.modules, "mistralai", fake_mistralai)
    monkeypatch.setitem(sys.modules, "mistralai.client", fake_client)
    monkeypatch.setattr(lazy_deps, "ensure", _make_ensure_failure)

    assert _import_mistral_client() is FakeMistral
