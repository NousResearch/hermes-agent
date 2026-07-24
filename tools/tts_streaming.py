"""Provider-agnostic streaming TTS: sentence text → int16 PCM chunk iterator.

The keystone of Hermes' conversational voice UX. `stream_tts_to_speaker`
(``tools.tts_tool``) owns the sentence buffer, sounddevice output, and
stop/queue protocol; this module owns the *provider* half — turning one
sentence into audio the moment it's ready, so playback starts on sentence one
instead of after the whole reply.

Two provider shapes, one contract (int16 mono PCM at ``sample_rate``):

* **True streamers** (`StreamingTTSProvider.stream`) — chunked APIs
  (ElevenLabs pcm_24000, OpenAI pcm, …) that yield audio as it synthesizes.
  Lowest time-to-first-audio.
* **Everyone else** — direct dispatcher callers retain per-*sentence* sync
  playback. The CLI instead routes these providers through whole-response batch
  TTS so separate synthesis requests cannot introduce inter-sentence pauses.

Adding a streamer is `@register("name")` on a `StreamingTTSProvider` subclass;
the dispatcher, config gate (`tts.<name>.streaming`), and resolver come free.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
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

    ``None`` means "no chunked API for this provider". Direct dispatcher callers
    retain the sync path, while the CLI uses whole-response batch TTS. We never
    silently swap to a different provider just to get streaming.
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
        if not get_env_value("ELEVENLABS_API_KEY"):
            return False
        try:
            from tools.tts_tool import _import_elevenlabs

            _import_elevenlabs()
            return True
        except ImportError:
            return False

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
    """OpenAI speech with ``response_format=pcm`` (24 kHz mono int16)."""

    sample_rate = 24000

    @staticmethod
    def available() -> bool:
        try:
            from tools.tts_tool import (
                _import_openai_client,
                _resolve_openai_audio_client_config,
            )

            _import_openai_client()
            _resolve_openai_audio_client_config()
            return True
        except (ImportError, ValueError):
            return False

    def stream(self, text: str) -> Iterator[bytes]:
        from tools.tts_tool import (
            DEFAULT_OPENAI_MODEL,
            DEFAULT_OPENAI_VOICE,
            MANAGED_OPENAI_TTS_MODELS,
            _import_openai_client,
            _resolve_openai_audio_client_config,
        )

        api_key, resolved_base_url, is_managed = _resolve_openai_audio_client_config()
        configured_base_url = self.section.get("base_url")
        base_url = resolved_base_url if is_managed else configured_base_url or resolved_base_url
        model = self.section.get("model", DEFAULT_OPENAI_MODEL)
        if is_managed and model not in MANAGED_OPENAI_TTS_MODELS:
            model = DEFAULT_OPENAI_MODEL
        voice = self.section.get("voice", DEFAULT_OPENAI_VOICE)
        client = _import_openai_client()(api_key=api_key, base_url=base_url)
        try:
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                response_format="pcm",
                extra_headers={"x-idempotency-key": str(uuid.uuid4())},
            ) as response:
                yield from response.iter_bytes()
        finally:
            client.close()
