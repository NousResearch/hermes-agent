"""Provider-agnostic streaming TTS: sentence text → int16 PCM chunk iterator.

The keystone of Hermes' conversational voice UX. `stream_tts_to_speaker`
(``tools.tts_tool``) owns the sentence buffer, sounddevice output, and
stop/queue protocol; this module owns the *provider* half — turning text
into audio the moment it's ready, so playback starts on sentence one
instead of after the whole reply.

Two provider shapes, one contract (int16 mono PCM at ``sample_rate``):

* **True streamers** (`StreamingTTSProvider.stream`) — chunked APIs
  (ElevenLabs pcm_24000, OpenAI pcm, …) that yield audio as it synthesizes.
  Lowest time-to-first-audio. The speaker pipeline fires a per-sentence
  prefetch: each sentence gets its own ``stream()`` call the moment it's
  complete, and a background thread fires the HTTP request immediately,
  buffering PCM while the previous sentence plays — no inter-sentence gap,
  no batching.
* **Everyone else** — providers with no chunked API still get per-*sentence*
  playback via the proven sync `text_to_speech_tool` path (handled by the
  dispatcher, not here), so edge (the default) is conversational too.

Adding a streamer is `@register("name")` on a `StreamingTTSProvider` subclass;
the dispatcher, config gate (`tts.<name>.streaming`), and resolver come free.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, Dict, Iterator, List, Optional

from tools.tts_tool import _get_provider, get_env_value

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interruption latch — lets the model know it was cut off mid-speech
# ---------------------------------------------------------------------------
# When the user barges in on a spoken reply (talks over it, types, hits the
# record key), the surface marks the latch; the next turn's submit path takes
# it and prepends SPEECH_INTERRUPTED_NOTE to the model-bound message (API-call
# local — never persisted, same as the CLI's model-switch notes). The TTL
# keeps a stale barge from annotating an unrelated message minutes later.

SPEECH_INTERRUPTED_NOTE = (
    "[Note: the user interrupted your previous spoken reply before it finished.]"
)
_INTERRUPT_TTL_S = 120.0
_interrupted_at: Optional[float] = None


def mark_speech_interrupted() -> None:
    global _interrupted_at
    _interrupted_at = time.monotonic()


def take_speech_interrupted() -> bool:
    """Pop the latch; True when a barge happened within the TTL."""
    global _interrupted_at
    at, _interrupted_at = _interrupted_at, None
    return at is not None and time.monotonic() - at < _INTERRUPT_TTL_S

# Sentence boundary: after .!? followed by whitespace, or a blank line.
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:\s|\n)|(?:\n\n)")
_THINK_BLOCK_RE = re.compile(r"<think[\s>].*?</think>", flags=re.DOTALL)


class SentenceChunker:
    """Incremental sentence cutter for LLM token deltas.

    Shared by the speaker pipeline (`stream_tts_to_speaker`) and the
    speak-stream WebSocket so every surface cuts speech identically. Strips
    ``<think>`` blocks (even split across deltas) and merges fragments shorter
    than *min_len* into the following sentence, so "Ha!" rides along with the
    sentence after it instead of stalling as a tiny clip.
    """

    def __init__(self, min_len: int = 20):
        self.min_len = min_len
        self.buf = ""

    def feed(self, delta: str) -> List[str]:
        """Absorb *delta*; return every complete sentence now ready to speak."""
        self.buf = _THINK_BLOCK_RE.sub("", self.buf + delta)
        if "<think" in self.buf and "</think>" not in self.buf:
            return []  # open think tag — the closing tag may arrive next delta
        out: List[str] = []
        start = 0  # skip boundaries that would leave the head too short
        while m := SENTENCE_BOUNDARY_RE.search(self.buf, start):
            head = self.buf[: m.end()]
            if len(head.strip()) < self.min_len:
                start = m.end()
                continue
            out.append(head)
            self.buf = self.buf[m.end():]
            start = 0
        return out

    def flush(self) -> List[str]:
        """Drain the tail (end-of-text or long-idle flush)."""
        tail = _THINK_BLOCK_RE.sub("", self.buf).strip()
        self.buf = ""
        return [tail] if tail else []


# ---------------------------------------------------------------------------
# ABC + registry
# ---------------------------------------------------------------------------

class StreamingTTSProvider(ABC):
    """Yields raw int16, little-endian, mono PCM chunks at ``sample_rate``."""

    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2  # bytes/sample (int16)

    def __init__(self, tts_config: Dict, section: Dict):
        self.tts_config = tts_config
        self.section = section

    @staticmethod
    @abstractmethod
    def available() -> bool:
        """True when this provider's credentials/SDK are usable right now."""

    @abstractmethod
    def stream(self, text: str) -> Iterator[bytes]:
        """Yield PCM chunks for ``text``. Raise on failure (caller logs)."""


_REGISTRY: Dict[str, type[StreamingTTSProvider]] = {}


def register(name: str) -> Callable[[type[StreamingTTSProvider]], type[StreamingTTSProvider]]:
    def _wrap(cls: type[StreamingTTSProvider]) -> type[StreamingTTSProvider]:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def resolve_streaming_provider(
    tts_config: Dict,
    preferred: Optional[str] = None,
) -> Optional[StreamingTTSProvider]:
    """Return a ready streamer for the *configured* provider, else ``None``.

    ``None`` means "no chunked API for this provider" — the dispatcher then
    speaks per-sentence via the sync path, preserving the user's chosen voice.
    We never silently swap to a different provider just to get streaming.
    """
    name = (preferred or _get_provider(tts_config)).lower().strip()
    cls = _REGISTRY.get(name)
    if cls is None or not cls.available():
        return None
    try:
        return cls(tts_config, tts_config.get(name) or {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("streaming provider %s init failed: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@register("elevenlabs")
class ElevenLabsStreamer(StreamingTTSProvider):
    """ElevenLabs chunked HTTP → pcm_24000 (the original reference path)."""

    sample_rate = 24000

    @staticmethod
    def available() -> bool:
        return bool(get_env_value("ELEVENLABS_API_KEY"))

    def stream(self, text: str) -> Iterator[bytes]:
        from tools.tts_tool import (
            DEFAULT_ELEVENLABS_STREAMING_MODEL_ID,
            DEFAULT_ELEVENLABS_VOICE_ID,
            _import_elevenlabs,
        )

        client = _import_elevenlabs()(api_key=get_env_value("ELEVENLABS_API_KEY"))
        voice_id = self.section.get("voice_id", DEFAULT_ELEVENLABS_VOICE_ID)
        model_id = self.section.get(
            "streaming_model_id",
            self.section.get("model_id", DEFAULT_ELEVENLABS_STREAMING_MODEL_ID),
        )
        yield from client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format="pcm_24000",
        )


@register("openai")
class OpenAIStreamer(StreamingTTSProvider):
    """OpenAI speech with ``response_format=pcm`` (24 kHz mono int16).

    Supports both direct OpenAI credentials and the Nous managed audio
    gateway. When ``tts.use_gateway`` is set (or no direct key is present
    but the managed gateway is available), the streamer resolves the
    gateway's token and base_url via the existing auth chain in
    ``tts_tool._resolve_openai_audio_client_config`` — the same path the
    sync synthesizer uses — so the prefetch pipeline fires through the
    gateway too, not just direct API.
    """

    sample_rate = 24000

    @staticmethod
    def available() -> bool:
        # Direct key path.
        if get_env_value("OPENAI_API_KEY") or get_env_value("VOICE_TOOLS_OPENAI_KEY"):
            return True
        # Managed gateway path — note: _has_openai_audio_backend calls
        # resolve_managed_tool_gateway() which can trigger a synchronous
        # token refresh. Acceptable for availability checks (called once
        # at pipeline setup), not in a hot loop.
        from tools.tts_tool import _has_openai_audio_backend
        return _has_openai_audio_backend()

    @cached_property
    def _client_config(self):
        """Lazily resolve ``(client, is_managed)`` and cache the OpenAI
        client — one HTTP/2 connection pool for every ``stream()`` call on
        this instance, so per-sentence requests share a single TCP+TLS
        connection (no handshake overhead).

        Returns ``(client, is_managed)`` so ``stream()`` can coerce the
        model for the managed gateway (same logic as the sync path).
        """
        from openai import OpenAI
        from tools.tts_tool import _resolve_openai_audio_client_config

        api_key, base_url, is_managed = _resolve_openai_audio_client_config()
        # Honor OPENAI_BASE_URL env var for direct-key usage with custom
        # endpoints (e.g. OpenAI-compatible proxies). The managed gateway
        # path already provides the correct base_url.
        if not is_managed and not base_url:
            base_url = get_env_value("OPENAI_BASE_URL") or None
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, is_managed

    @property
    def _client(self):
        """Backwards-compat alias — delegates to the cached config."""
        return self._client_config[0]

    def stream(self, text: str) -> Iterator[bytes]:
        from tools.tts_tool import MANAGED_OPENAI_TTS_MODELS, DEFAULT_OPENAI_MODEL

        model = self.section.get("model", "gpt-4o-mini-tts")
        voice = self.section.get("voice", "alloy")
        _client, is_managed = self._client_config
        # The managed OpenAI audio gateway only proxies
        # MANAGED_OPENAI_TTS_MODELS — coerce like the sync path does.
        # Skip coercion when the user explicitly set a custom base_url
        # (non-gateway endpoint), matching the sync path's behavior.
        _explicit_base = bool(get_env_value("OPENAI_BASE_URL") or self.section.get("base_url"))
        if is_managed and not _explicit_base and model not in MANAGED_OPENAI_TTS_MODELS:
            model = DEFAULT_OPENAI_MODEL
        with _client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            response_format="pcm",
        ) as response:
            yield from response.iter_bytes()
