"""Tests for tools/speech/router.py (Speech Router RFC, PR1 + PR2)."""

from tools.speech.router import SpeechRouter
from tools.tts_tool import _get_provider, _load_tts_config


def test_resolve_provider_matches_direct_call():
    """PR1 contract: resolve_provider() must return exactly what the two
    lines it replaced would have returned."""
    tts_config = _load_tts_config()
    expected_provider = _get_provider(tts_config)

    provider, router_tts_config = SpeechRouter().resolve_provider()

    assert provider == expected_provider
    assert router_tts_config == tts_config


def test_get_capabilities_known_builtin():
    caps = SpeechRouter().get_capabilities("edge")
    assert caps is not None
    assert caps.local is False
    assert caps.streaming is False


def test_get_capabilities_unknown_provider():
    assert SpeechRouter().get_capabilities("not-a-real-provider") is None
