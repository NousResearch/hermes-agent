#!/usr/bin/env python3
"""
Generic streaming TTS dispatcher (ABC + registry only — implementations land in
follow-up tasks).

This module is the generic dispatcher for chunked TTS output. Each provider
implements the ``StreamingTTSProvider`` ABC to yield raw PCM bytes; the
dispatcher (added in a later task) opens the ``sounddevice`` ``OutputStream``
once, picks a registered provider by name, and writes each chunk as it arrives.

A single ABC + registry keeps the sync path in ``tools.tts_tool.py`` untouched:
providers that already support streaming just register a class here, and the
existing ``stream_tts_to_speaker`` entry point becomes a thin wrapper over the
dispatcher.
"""

# Why this exists
# ---------------
# ``tools.tts_tool.py`` was originally written with ElevenLabs' chunked HTTP
# API inlined into ``stream_tts_to_speaker``. Every other provider that exposes
# chunked output (Gemini SSE, OpenAI stream, xAI WebSocket, edge-tts) needs the
# same shape: open a stream, yield PCM bytes, let the caller decide when to
# stop. Hard-coding each one would duplicate the ``sounddevice`` plumbing and
# the sentence-buffer loop. Putting the contract behind an ABC means any new
# streaming-capable provider plugs in with a single ``@register("name")`` and
# gets the dispatcher — including ``OutputStream`` lifecycle, stop-event
# handling, and provider priority fallback — for free.

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Type

logger = logging.getLogger(__name__)


class StreamingTTSProvider(ABC):
    """Abstract base class for TTS providers that can stream PCM chunks.

    Concrete subclasses declare the audio format (sample_rate, channels,
    sample_width) and implement ``stream()`` to yield raw PCM bytes.

    The dispatcher (added in a later task) opens a ``sounddevice.OutputStream``
    using the subclass's format attributes and writes each yielded chunk to it,
    checking the stop event between chunks. Subclasses should therefore yield
    small chunks often — chunking is the whole point of this interface.
    """

    sample_rate: int
    channels: int
    sample_width: int  # bytes per sample (2 for int16)

    @abstractmethod
    def stream(self, text: str) -> Iterator[bytes]:
        """Yield raw PCM chunks for the given text.

        Caller is responsible for opening the audio output device and
        writing each chunk. Implementations should NOT block on long
        operations; chunking is the whole point of this interface.
        """
        raise NotImplementedError


# Module-level registry mapping a lower-cased provider name to the
# ``StreamingTTSProvider`` subclass that handles it. Populated via the
# ``@register("name")`` class decorator below; no provider implementations
# register themselves in this task — that comes with Tasks 3-7.
_PROVIDERS: Dict[str, Type[StreamingTTSProvider]] = {}


def register(name: str) -> Callable[[Type[StreamingTTSProvider]], Type[StreamingTTSProvider]]:
    """Class decorator to register a ``StreamingTTSProvider`` under ``name``.

    Names are normalized to lower-case + stripped before storage so look-ups
    are case-insensitive. Re-registering an existing name logs a debug line
    and overwrites the previous class — useful for tests, surprising in
    production, so providers should only be registered at import time.
    """
    key = name.lower().strip()

    def _decorator(cls: Type[StreamingTTSProvider]) -> Type[StreamingTTSProvider]:
        if not isinstance(cls, type) or not issubclass(cls, StreamingTTSProvider):
            raise TypeError(f"{cls!r} must subclass StreamingTTSProvider")
        if key in _PROVIDERS:
            logger.debug("Overriding existing streaming TTS provider: %s", key)
        _PROVIDERS[key] = cls
        return cls

    return _decorator


def get(name: str) -> Type[StreamingTTSProvider]:
    """Return the registered provider class for ``name``.

    Raises ``KeyError`` with the list of available providers if ``name`` is
    not registered, so the caller can surface a helpful error to the user
    or fall back to a default.
    """
    key = name.lower().strip()
    if key not in _PROVIDERS:
        raise KeyError(
            f"Unknown streaming TTS provider: {name!r}. "
            f"Available: {sorted(_PROVIDERS.keys())}"
        )
    return _PROVIDERS[key]


def available() -> List[str]:
    """Return sorted list of registered provider names."""
    return sorted(_PROVIDERS.keys())


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------
#
# Each provider is a thin subclass of :class:`StreamingTTSProvider` that
# yields raw PCM bytes from the underlying SDK call. New providers should
# mirror the shape of the first one below (:class:`ElevenLabsStreamingProvider`)
# — that's why we call it the *reference implementation*. The same
# three concerns show up everywhere:
#
#   1. Lazy import of the SDK (so users who only enable one provider
#      don't pay the import cost of the others). This is what the
#      ``_import_*`` helpers wrap.
#   2. Validate configuration in ``__init__`` and fail loud with
#      ``RuntimeError`` / ``ImportError`` before the dispatcher starts
#      pulling chunks — the alternative is a confusing mid-stream error.
#   3. Honor ``self._stop_event`` between yields so the dispatcher can
#      abort the iteration cheaply (skip the per-chunk try/except dance).


def _import_elevenlabs_client():
    """Lazy import the ElevenLabs SDK client class.

    Returns the ``ElevenLabs`` class (not an instance) so callers can pass
    an ``api_key`` at construction time. Raises ``ImportError`` with the
    install command when the package isn't available — same shape as the
    ``_import_elevenlabs`` helper in ``tools.tts_tool.py`` so error
    messages stay consistent across the module boundary.

    This helper is the seam the test suite monkeypatches; keeping the
    import isolated in a single function means tests never need to
    install the real ``elevenlabs`` package.
    """
    try:
        from elevenlabs.client import ElevenLabs  # type: ignore
    except ImportError as e:
        raise ImportError(
            "elevenlabs package not installed. Run: pip install elevenlabs"
        ) from e
    return ElevenLabs


@register("elevenlabs")
class ElevenLabsStreamingProvider(StreamingTTSProvider):
    """Streaming TTS provider for ElevenLabs' ``text_to_speech.convert`` API.

    This is the **reference implementation** that the other providers
    (Gemini, OpenAI, xAI, edge-tts) will mirror. New providers should
    follow the same three-step shape: lazy SDK import, ``__init__``
    config validation, ``stream()`` that iterates the upstream iterator
    and checks ``self._stop_event`` between yields.

    The output format is fixed at ``pcm_24000`` (24 kHz, 16-bit signed
    little-endian, mono) because that's the only format ElevenLabs
    exposes for chunked streaming and the format ``sounddevice`` opens
    the ``OutputStream`` with.
    """

    # ElevenLabs' pcm_24000 format: 24 kHz, mono, int16 (2 bytes/sample).
    sample_rate = 24000
    channels = 1
    sample_width = 2

    def __init__(self, config: dict, *, stop_event: Optional[threading.Event] = None):
        # Config keys mirror the ``tts.elevenlabs`` block from config.yaml.
        # Defaults match the constants in tools.tts_tool.py so behaviour
        # stays consistent with the existing inlined streaming path.
        self._voice_id = config.get("voice_id", "pNInz6obpgDQGcFmaJgB")
        self._model_id = config.get("streaming_model_id", "eleven_flash_v2_5")
        # The stop event is optional so this class is usable outside the
        # dispatcher (e.g. one-off scripts). When None, the stream() loop
        # simply skips the stop check.
        self._stop_event = stop_event

        # API key resolution: prefer the live config helper (lets the
        # test suite + dotenv fallback take effect), then fall back to
        # a direct os.environ read. Empty/None both mean "not set".
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "ELEVENLABS_API_KEY not set; ElevenLabs streaming TTS disabled"
            )
            raise RuntimeError(
                "ELEVENLABS_API_KEY not set; cannot use ElevenLabs streaming TTS"
            )

        # Lazy SDK import — wrapped in a helper so the test suite can
        # monkeypatch it without installing the real package. We surface
        # the import error loudly because silently disabling the
        # provider would mask config bugs the user can fix.
        try:
            elevenlabs_cls = _import_elevenlabs_client()
        except ImportError as e:
            logger.warning("elevenlabs package not installed: %s", e)
            raise

        # Instantiate the SDK client. ElevenLabs() reads the api_key
        # from the constructor and from ELEVENLABS_API_KEY in the
        # environment; we pass explicitly so behaviour is deterministic.
        self._client = elevenlabs_cls(api_key=api_key)

    def stream(self, text: str) -> Iterator[bytes]:
        """Yield PCM chunks for *text* as they arrive from ElevenLabs.

        Mirrors the call shape at tools.tts_tool.py:2567 (the existing
        inlined streaming path) so behaviour matches the legacy code
        until Task 11's refactor deletes it. The ``try/except`` here is
        deliberately broad — a streaming response that fails halfway
        through should log a warning and stop yielding rather than
        crash the dispatcher's audio callback.
        """
        try:
            audio_iter = self._client.text_to_speech.convert(
                text=text,
                voice_id=self._voice_id,
                model_id=self._model_id,
                output_format="pcm_24000",
            )
            for chunk in audio_iter:
                if self._stop_event is not None and self._stop_event.is_set():
                    break
                yield chunk
        except Exception as exc:
            logger.warning("ElevenLabs streaming TTS failed: %s", exc)
            return


__all__ = [
    "StreamingTTSProvider",
    "register",
    "get",
    "available",
    "ElevenLabsStreamingProvider",
]
