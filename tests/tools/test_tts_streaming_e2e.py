"""End-to-end tests for streaming TTS providers. Gated on real API keys.

These tests are SKIPPED by default — they only run when the relevant
environment variable is set. They're useful for catching provider-API drift
and verifying the integration end-to-end, but they shouldn't run in CI
without secrets configured.
"""
from __future__ import annotations

import os

import pytest


# --- ElevenLabs ---


@pytest.mark.skipif(
    not os.environ.get("ELEVENLABS_API_KEY"),
    reason="ELEVENLABS_API_KEY not set",
)
def test_elevenlabs_streaming_real():
    """Generate audio from the real ElevenLabs API and verify non-empty chunks."""
    from tools.tts_streaming import ElevenLabsStreamingProvider

    provider = ElevenLabsStreamingProvider({})
    chunks = list(provider.stream("Hello world, this is a test."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    # 1 second of PCM at 24kHz mono int16 = 48000 bytes; expect at least 1k for any real audio
    assert total_bytes > 1000


@pytest.mark.skipif(
    not os.environ.get("ELEVENLABS_API_KEY"),
    reason="ELEVENLABS_API_KEY not set",
)
def test_elevenlabs_streaming_respects_stop_event():
    """Stop event set before call returns no chunks."""
    import threading

    from tools.tts_streaming import ElevenLabsStreamingProvider

    stop = threading.Event()
    stop.set()
    provider = ElevenLabsStreamingProvider({}, stop_event=stop)
    chunks = list(provider.stream("Hello world."))
    assert chunks == []


# --- Gemini ---


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY/GOOGLE_API_KEY not set",
)
def test_gemini_streaming_real():
    """Generate audio from the real Gemini API and verify non-empty chunks."""
    from tools.tts_streaming import GeminiStreamingProvider

    provider = GeminiStreamingProvider({})
    chunks = list(provider.stream("Hola, esto es una prueba."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY/GOOGLE_API_KEY not set",
)
def test_gemini_streaming_respects_stop_event():
    """Stop event set before call returns no chunks."""
    import threading

    from tools.tts_streaming import GeminiStreamingProvider

    stop = threading.Event()
    stop.set()
    provider = GeminiStreamingProvider({}, stop_event=stop)
    chunks = list(provider.stream("Hola."))
    assert chunks == []


# --- OpenAI ---


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_openai_streaming_real():
    """Generate audio from the real OpenAI API and verify non-empty chunks."""
    from tools.tts_streaming import OpenAIStreamingProvider

    provider = OpenAIStreamingProvider({})
    chunks = list(provider.stream("Hello, this is a test."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000


# --- xAI ---


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY"),
    reason="XAI_API_KEY not set",
)
def test_xai_streaming_real():
    """Generate audio from the real xAI WebSocket API and verify non-empty chunks."""
    from tools.tts_streaming import XAIStreamingProvider

    provider = XAIStreamingProvider({})
    chunks = list(provider.stream("Hello, this is a test."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000


# --- Edge (no key required) ---


def test_edge_streaming_real_or_skip():
    """Generate audio from real edge-tts. No key required but may be blocked by network."""
    import socket

    try:
        # Quick reachability check for the Edge TTS endpoint
        socket.create_connection(("api.dict.cn", 443), timeout=2).close()
    except OSError:
        pytest.skip("Edge TTS endpoint unreachable")

    from tools.tts_streaming import EdgeStreamingProvider

    provider = EdgeStreamingProvider({})
    chunks = list(provider.stream("Hello, this is a test."))
    assert len(chunks) > 0
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 100


# --- Dispatcher integration ---


@pytest.mark.skipif(
    not (
        os.environ.get("ELEVENLABS_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("XAI_API_KEY")
    ),
    reason="No streaming TTS API key set",
)
def test_resolve_streaming_provider_picks_available():
    """The resolver picks a real provider when at least one key is set."""
    from tools.tts_streaming import resolve_streaming_provider

    # Provide a config without an explicit streaming.provider so the
    # resolver falls back to the priority list.
    result = resolve_streaming_provider({}, preferred=None)
    assert result in {"elevenlabs", "gemini", "openai", "xai", "edge"}
