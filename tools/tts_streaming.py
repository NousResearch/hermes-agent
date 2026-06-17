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
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Type

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


__all__ = ["StreamingTTSProvider", "register", "get", "available"]
