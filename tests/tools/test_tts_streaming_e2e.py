"""End-to-end tests for streaming TTS providers. Gated on real API keys.

Salvaged from PR #47588 (@Cdddo) and adapted to the current provider ABC
(``StreamingTTSProvider(tts_config, section)``). These tests are SKIPPED by
default — they only run when the relevant credential is present. They're
useful for catching provider-API drift and verifying the integration
end-to-end, but they shouldn't run in CI without secrets configured.
"""
from __future__ import annotations

import os

import pytest


def _has_xai_creds() -> bool:
    # Plain os.environ, matching the other gates here: the repo's test
    # conftest scrubs dotenv/pooled secrets, so gating on the richer
    # resolver would collect a test whose runtime credentials are gone.
    return bool(os.environ.get("XAI_API_KEY"))


# --- ElevenLabs ---


@pytest.mark.skipif(
    not os.environ.get("ELEVENLABS_API_KEY"),
    reason="ELEVENLABS_API_KEY not set",
)
def test_elevenlabs_streaming_real():
    """Generate audio from the real ElevenLabs API and verify non-empty chunks."""
    from tools.tts_streaming import ElevenLabsStreamer

    provider = ElevenLabsStreamer({}, {})
    chunks = list(provider.stream("Hello world, this is a test."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    # 1s of PCM at 24kHz mono int16 = 48000 bytes; expect at least 1k for real audio
    assert total_bytes > 1000


# --- Gemini ---


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY/GOOGLE_API_KEY not set",
)
def test_gemini_streaming_real():
    """Generate audio from the real Gemini SSE API and verify non-empty chunks."""
    from tools.tts_streaming import GeminiStreamer

    provider = GeminiStreamer({}, {})
    chunks = list(provider.stream("Hola, esto es una prueba."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000


# --- OpenAI ---


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_openai_streaming_real():
    """Generate audio from the real OpenAI API and verify non-empty chunks."""
    from tools.tts_streaming import OpenAIStreamer

    provider = OpenAIStreamer({}, {})
    chunks = list(provider.stream("Hello, this is a test."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000


# --- xAI ---


@pytest.mark.skipif(not _has_xai_creds(), reason="no xAI credentials")
def test_xai_streaming_real():
    """Generate audio from the real xAI WebSocket API and verify non-empty frames."""
    from tools.tts_streaming import XAIStreamer

    provider = XAIStreamer({}, {})
    chunks = list(provider.stream("Hello, this is a test."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000


# --- Resolver integration (no network; requires at least one key) ---


@pytest.mark.skipif(
    not (
        os.environ.get("ELEVENLABS_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    ),
    reason="no streaming-capable TTS key set",
)
def test_auto_resolver_finds_a_real_provider():
    from tools.tts_streaming import StreamingTTSProvider, resolve_streaming_provider

    provider = resolve_streaming_provider({"streaming": {"provider": "auto"}})
    assert isinstance(provider, StreamingTTSProvider)
