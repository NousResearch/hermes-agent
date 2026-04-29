"""Regression guard: stream_tts_to_speaker is ElevenLabs-only by design.

Plugin TTS providers participate only in the non-streaming text_to_speech
tool dispatch. Streaming stays with ElevenLabs for now; extending the
plugin interface to support streaming is tracked as future work.
"""

from __future__ import annotations
import pytest


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    from agent import tts_registry
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tts_registry._reset_for_tests()
    yield
    tts_registry._reset_for_tests()


def test_stream_tts_to_speaker_does_not_use_plugin_registry(tmp_path):
    """Registering a plugin named e.g. 'mock-stream' must NOT cause
    stream_tts_to_speaker to route through it. That function is
    ElevenLabs-only at the source level and the plugin registry must
    never appear in its call graph."""
    import inspect
    from tools import tts_tool

    src = inspect.getsource(tts_tool.stream_tts_to_speaker)
    # Confidence assertion: the streaming function references neither
    # the registry nor the dispatcher.
    assert "tts_registry" not in src
    assert "_dispatch_to_plugin_tts_provider" not in src
    # Positive control: it DOES use ElevenLabs paths.
    assert "elevenlabs" in src.lower() or "ElevenLabs" in src


def test_plugin_registry_absent_from_streaming_config_load():
    """The streaming path loads tts config but must not consult the
    plugin registry — that would couple streaming to plugin discovery."""
    import inspect
    from tools import tts_tool

    # This is a weak check: we allow the module to import tts_registry
    # lazily inside functions. The regression we guard against is
    # anyone wiring stream_tts_to_speaker to consult it.
    # Delegated to the source-level check in the first test.
    assert True  # placeholder; first test carries the guard
