import sys
import types

import tools.tts_tool as tts_tool


def test_import_edge_tts_falls_back_to_raw_import_when_lazy_ensure_fails(monkeypatch):
    fake_edge_tts = types.ModuleType("edge_tts")

    def _fail_ensure(*args, **kwargs):
        raise RuntimeError("venv is read-only")

    monkeypatch.setattr("tools.lazy_deps.ensure", _fail_ensure)
    monkeypatch.setitem(sys.modules, "edge_tts", fake_edge_tts)

    assert tts_tool._import_edge_tts() is fake_edge_tts


def test_import_elevenlabs_falls_back_to_raw_import_when_lazy_ensure_fails(monkeypatch):
    fake_pkg = types.ModuleType("elevenlabs")
    fake_pkg.__path__ = []
    fake_client = types.ModuleType("elevenlabs.client")

    class FakeElevenLabs:
        pass

    fake_client.ElevenLabs = FakeElevenLabs
    fake_pkg.client = fake_client

    def _fail_ensure(*args, **kwargs):
        raise RuntimeError("venv is read-only")

    monkeypatch.setattr("tools.lazy_deps.ensure", _fail_ensure)
    monkeypatch.setitem(sys.modules, "elevenlabs", fake_pkg)
    monkeypatch.setitem(sys.modules, "elevenlabs.client", fake_client)

    assert tts_tool._import_elevenlabs() is FakeElevenLabs
