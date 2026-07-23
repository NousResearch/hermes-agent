"""Tests for tools/speech/capabilities.py (Speech Router RFC, PR2)."""

from tools.speech.capabilities import BUILTIN_CAPABILITIES, get_builtin_capabilities
from tools.tts_tool import BUILTIN_TTS_PROVIDERS


def test_every_builtin_provider_has_capabilities():
    """Catches drift: a new builtin added to tts_tool.py without a matching
    capabilities entry here would otherwise silently return None forever."""
    missing = BUILTIN_TTS_PROVIDERS - set(BUILTIN_CAPABILITIES)
    assert not missing, f"builtin providers missing capability entries: {missing}"


def test_no_unknown_extra_entries():
    """Catches the opposite drift: a stale entry for a provider that was
    removed from BUILTIN_TTS_PROVIDERS."""
    extra = set(BUILTIN_CAPABILITIES) - BUILTIN_TTS_PROVIDERS
    assert not extra, f"capability entries for non-builtin providers: {extra}"


def test_unknown_provider_returns_none():
    assert get_builtin_capabilities("not-a-real-provider") is None


def test_elevenlabs_streaming_is_true():
    """Grounded in tools.tts_tool.stream_tts_to_speaker (CLI streaming path)."""
    caps = get_builtin_capabilities("elevenlabs")
    assert caps is not None
    assert caps.streaming is True


def test_local_providers_flagged_local():
    for name in ("neutts", "kittentts", "piper"):
        caps = get_builtin_capabilities(name)
        assert caps is not None
        assert caps.local is True, f"{name} should be local=True"


def test_cloud_providers_flagged_not_local():
    for name in ("edge", "elevenlabs", "openai", "minimax", "xai", "mistral", "gemini"):
        caps = get_builtin_capabilities(name)
        assert caps is not None
        assert caps.local is False, f"{name} should be local=False"


def test_no_provider_claims_ssml():
    """This codebase's own prompt text says "Do not use SSML" in two places;
    no builtin integration here implements it."""
    for name, caps in BUILTIN_CAPABILITIES.items():
        assert caps.ssml is False, f"{name} unexpectedly claims ssml=True"
