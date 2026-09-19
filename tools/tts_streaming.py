"""Provider-agnostic streaming TTS: sentence text → int16 mono PCM chunk iterator.

``stream_tts_to_speaker`` (``tools.tts_tool``) owns the sentence buffer, sounddevice
output and stop/queue protocol; this module owns the *provider* half so playback
starts on sentence one. True streamers (``StreamingTTSProvider.stream``) wrap chunked
APIs; providers with no chunked API (edge, the default) get per-sentence playback via
the sync ``text_to_speech_tool`` path. Adding a streamer is ``@register("name")`` on
a subclass; the dispatcher, config gate (``tts.<name>.streaming``) and resolver come free.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

from tools.tool_backend_helpers import resolve_openai_audio_api_key
from tools.tts_tool import _get_provider, _load_tts_config

logger = logging.getLogger(__name__)

# Per-sentence PCM byte cap, mirroring the sync providers' 16 MiB bounded-body invariant.
_STREAM_SENTENCE_BYTE_CAP = 16 * 1024 * 1024

# --- Kokoro (local, self-hosted) -------------------------------------------
# A self-hosted OpenAI-compatible Kokoro service. Loopback by default: local
# speech synthesis must not become a network service by accident.
DEFAULT_KOKORO_BASE_URL = "http://127.0.0.1:11640/v1"
DEFAULT_KOKORO_MODEL = "kokoro"
DEFAULT_KOKORO_VOICE = "af_heart"
DEFAULT_KOKORO_TIMEOUT_S = 60.0
# Availability runs on the `auto` resolution path, so a down service must cost
# milliseconds. Long enough for a loopback round trip, short enough not to stall.
KOKORO_PROBE_TIMEOUT_S = 1.5
MAX_CONFIGURED_FALLBACK_DEPTH = 32
# 20 ms of 24 kHz int16 mono. Small chunks keep first-audio latency low; larger
# ones would buffer speech that the user is waiting to hear.
KOKORO_CHUNK_BYTES = 960


def _resolve_key(env_var: str, provider_id: str) -> str:
    """Provider secret lookup (config > env/.env > credential pool); seam over ``tts_tool._resolve_provider_key``.
    ALL streaming-provider key lookups go through here — never bare ``get_env_value``."""
    try:
        from tools.tts_tool import _resolve_provider_key
        return _resolve_provider_key(env_var, provider_id) or ""
    except Exception:
        from hermes_cli.config import get_env_value
        return get_env_value(env_var) or ""


def _gemini_key() -> str:
    return _resolve_key("GEMINI_API_KEY", "gemini") or _resolve_key("GOOGLE_API_KEY", "gemini")


# Interruption latch: a barge-in on a spoken reply marks it; the next turn's submit path takes it
# and prepends SPEECH_INTERRUPTED_NOTE to the model-bound message (API-call local, never
# persisted). The TTL keeps a stale barge from annotating an unrelated message minutes later.
SPEECH_INTERRUPTED_NOTE = "[Note: the user interrupted your previous spoken reply before it finished.]"
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
    """Incremental sentence cutter for LLM token deltas, shared by the speaker pipeline and the
    speak-stream WebSocket so every surface cuts speech identically. Strips ``<think>`` blocks (even
    split across deltas) and merges fragments shorter than *min_len* into the following sentence."""

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
        tail, self.buf = _THINK_BLOCK_RE.sub("", self.buf).strip(), ""
        return [tail] if tail else []


class StreamingTTSProvider(ABC):
    """Yields raw int16, little-endian, mono PCM chunks at ``sample_rate`` (built-ins: 24 kHz)."""

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

# Preserves the decorated subclass's own type. Without it ``@register`` erases
# every streamer to the base class, so a type checker rejects any access to a
# subclass-specific member (``KokoroStreamer._base_url``).
_ProviderT = TypeVar("_ProviderT", bound=type[StreamingTTSProvider])


def register(name: str) -> Callable[[_ProviderT], _ProviderT]:
    def _wrap(cls: _ProviderT) -> _ProviderT:
        _REGISTRY[name] = cls
        return cls
    return _wrap


def _try_instantiate(name: str, tts_config: Dict) -> Optional[StreamingTTSProvider]:
    """Construct the registered streamer *name* if it's usable, else None."""
    cls = _REGISTRY.get(name)
    if cls is None:
        return None
    section = tts_config.get(name) or {}
    available = cls.available(section) if cls is KokoroStreamer else cls.available()
    if not available:
        return None
    try:
        return cls(tts_config, section)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("streaming provider %s init failed: %s", name, exc)
        return None


# Fallback priority for ``tts.streaming.provider: auto`` — best chunked latency/quality
# first. Deliberately hard-coded (a UX decision); edge is absent (no chunked-PCM API).
# ``kokoro`` leads: it is the only LOCAL streamer, so it costs nothing per minute and
# keeps speech on-premises. It is skipped in a millisecond when no service is running.
_PROVIDER_PRIORITY: List[str] = ["kokoro", "elevenlabs", "gemini", "openai", "xai"]


def resolve_streaming_provider(
    tts_config: Dict, preferred: Optional[str] = None) -> Optional[StreamingTTSProvider]:
    """Return a ready streamer for the *configured* provider, else ``None``.
    ``tts.streaming.provider`` when set: a name pins that exact streamer (``None`` if unusable);
    ``auto`` returns the first usable in ``_PROVIDER_PRIORITY``. Otherwise the configured TTS
    provider (or ``preferred``): ``None`` means "no chunked API" — the dispatcher speaks
    per-sentence via the sync path, preserving the user's chosen voice. We never silently swap
    providers just to get streaming.

    The ONE swap allowed is an EXPLICITLY configured one: ``tts.<name>.fallback_provider``.
    A local provider can be down (its service is not running) in a way a cloud provider with
    a valid key cannot, so "Kokoro, but Piper when the box is down" has to be expressible.
    It is opt-in per provider and never inferred.
    """
    pinned = str((tts_config.get("streaming") or {}).get("provider") or "").lower().strip()
    if pinned == "auto":
        return next((inst for name in _PROVIDER_PRIORITY
                     if (inst := _try_instantiate(name, tts_config))), None)
    name = pinned or (preferred or _get_provider(tts_config)).lower().strip()
    instance = _try_instantiate(name, tts_config)
    if instance is not None:
        return instance
    return _try_configured_fallback(name, tts_config)


def _try_configured_fallback(
    name: str, tts_config: Dict, _seen: Optional[set] = None,
    _depth: int = 0) -> Optional[StreamingTTSProvider]:
    """Follow ``tts.<name>.fallback_provider`` when *name* is unusable.

    Returns ``None`` when no fallback is configured, the fallback is itself
    unusable, or it has no streamer (e.g. Piper — a sync-only local provider,
    which the caller then speaks per sentence through the sync path).
    ``_seen`` breaks a config cycle (a → b → a) instead of recursing forever.
    """
    seen = _seen if _seen is not None else set()
    if _depth >= MAX_CONFIGURED_FALLBACK_DEPTH:
        logger.warning("TTS fallback chain exceeded %d hops; stopping", MAX_CONFIGURED_FALLBACK_DEPTH)
        return None
    if name in seen:
        logger.warning("TTS fallback cycle at %r; stopping", name)
        return None
    seen.add(name)
    section = tts_config.get(name)
    fallback = str((section or {}).get("fallback_provider") or "").lower().strip()
    if not fallback:
        return None
    logger.info("streaming TTS %r unavailable; falling back to %r", name, fallback)
    return _try_instantiate(fallback, tts_config) or _try_configured_fallback(
        fallback, tts_config, seen, _depth + 1)


def _capped(chunks: Iterator[bytes], label: str) -> Iterator[bytes]:
    """Pass chunks through, aborting past the per-sentence byte cap (runaway/hostile upstream)."""
    total = 0
    for chunk in chunks:
        total += len(chunk)
        if total > _STREAM_SENTENCE_BYTE_CAP:
            logger.warning("%s exceeded %d bytes for one sentence; truncating", label, _STREAM_SENTENCE_BYTE_CAP)
            return
        yield chunk


@register("elevenlabs")
class ElevenLabsStreamer(StreamingTTSProvider):
    """ElevenLabs chunked HTTP → pcm_24000 (the original reference path)."""

    @staticmethod
    def available() -> bool:
        return bool(_resolve_key("ELEVENLABS_API_KEY", "elevenlabs"))

    def stream(self, text: str) -> Iterator[bytes]:
        from tools.tts_tool import _import_elevenlabs
        from tools.tts_tool_providers import (
            DEFAULT_ELEVENLABS_STREAMING_MODEL_ID, DEFAULT_ELEVENLABS_VOICE_ID, _elevenlabs_environment_kwargs,
        )
        client = _import_elevenlabs()(
            api_key=_resolve_key("ELEVENLABS_API_KEY", "elevenlabs"), **_elevenlabs_environment_kwargs(self.section),
        )
        yield from client.text_to_speech.convert(
            text=text, voice_id=self.section.get("voice_id", DEFAULT_ELEVENLABS_VOICE_ID),
            model_id=self.section.get("streaming_model_id",
                                      self.section.get("model_id", DEFAULT_ELEVENLABS_STREAMING_MODEL_ID)),
            output_format="pcm_24000")


@register("kokoro")
class KokoroStreamer(StreamingTTSProvider):
    """Local Kokoro speech service → chunked int16 PCM. No API key, no per-minute cost.

    Talks to a self-hosted OpenAI-compatible Kokoro server
    (``POST <base_url>/audio/speech`` with ``response_format=pcm``), NOT to an
    in-process model. That split is deliberate:

    * Kokoro needs torch+CUDA. Importing it into the Hermes venv would drag a
      multi-gigabyte GPU stack into every Hermes process, including CLI ones
      that never speak.
    * A persistent service keeps the model resident, so first-audio latency is
      synthesis time and not model load time, and the GPU is shared across
      sessions rather than re-warmed per process.
    * ``available()`` is then a cheap socket probe, which is what makes the
      automatic fall-through to a cloud provider (or to the per-sentence sync
      Piper path) fast instead of a multi-second import stall.

    Config (``tts.kokoro``): ``base_url`` (default ``http://127.0.0.1:11640/v1``),
    ``voice``, ``model``, ``speed``, ``sample_rate``, ``timeout``.

    Cancellation: the caller stops consuming the iterator and the streamed
    response is closed, which drops the HTTP connection and the server's
    synthesis with it. There is no request-id to revoke because the socket
    itself is the handle.
    """

    # Kokoro's native output rate. Overridable because a self-hosted build may
    # be configured otherwise; the WS handshake sends whatever we report here,
    # and a mismatch plays back at the wrong pitch.
    sample_rate: int = 24000

    def __init__(self, tts_config: Dict, section: Dict):
        super().__init__(tts_config, section)
        rate = section.get("sample_rate")
        if rate:
            self.sample_rate = int(rate)

    @staticmethod
    def _base_url(section: Optional[Dict] = None) -> str:
        if section is None:
            try:
                section = _load_tts_config().get("kokoro") or {}
            except Exception:
                section = {}
        return str((section or {}).get("base_url") or DEFAULT_KOKORO_BASE_URL).strip().rstrip("/")

    @staticmethod
    def available(section: Optional[Dict] = None) -> bool:
        """True when a Kokoro service answers within one probe budget."""
        import requests
        url = KokoroStreamer._base_url(section)
        deadline = time.monotonic() + KOKORO_PROBE_TIMEOUT_S
        for index, path in enumerate(("/audio/voices", "/models")):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            timeout = KOKORO_PROBE_TIMEOUT_S if index == 0 else remaining
            try:
                if requests.get(f"{url}{path}", timeout=timeout).status_code < 500:
                    return True
            except requests.RequestException:
                continue
        return False

    def stream(self, text: str) -> Iterator[bytes]:
        import requests
        section = self.section
        payload: Dict[str, Any] = {
            "model": str(section.get("model") or DEFAULT_KOKORO_MODEL),
            "voice": str(section.get("voice") or DEFAULT_KOKORO_VOICE),
            "input": text,
            "response_format": "pcm",
        }
        speed = section.get("speed", self.tts_config.get("speed"))
        if speed:
            payload["speed"] = float(speed)
        timeout = float(section.get("timeout") or DEFAULT_KOKORO_TIMEOUT_S)

        def _chunks() -> Iterator[bytes]:
            # stream=True + closing(): abandoning the iterator on barge-in tears
            # the socket down instead of leaking a synthesis worker per interrupt.
            with contextlib.closing(requests.post(
                f"{self._base_url(section)}/audio/speech", json=payload,
                timeout=timeout, stream=True,
            )) as response:
                if response.status_code != 200:
                    chunk = next(response.iter_content(chunk_size=300), b"")
                    detail = chunk[:300].decode("utf-8", "replace")
                    raise RuntimeError(f"Kokoro TTS failed ({response.status_code}): {detail}")
                for chunk in response.iter_content(chunk_size=KOKORO_CHUNK_BYTES):
                    if chunk:
                        yield chunk

        yield from _capped(_chunks(), "Kokoro streaming TTS")


def _openai_config_api_key() -> str:
    """Return ``tts.openai.api_key`` from config.yaml, or empty string."""
    try:
        return (_load_tts_config().get("openai") or {}).get("api_key") or ""
    except Exception:
        return ""


@register("openai")
class OpenAIStreamer(StreamingTTSProvider):
    """OpenAI speech with ``response_format=pcm`` (24 kHz mono int16)."""

    @staticmethod
    def available() -> bool:
        return bool(_openai_config_api_key() or resolve_openai_audio_api_key())

    def stream(self, text: str) -> Iterator[bytes]:
        from openai import OpenAI
        from hermes_cli.config import get_env_value
        client = OpenAI(
            api_key=(self.section.get("api_key") or resolve_openai_audio_api_key()),
            base_url=(self.section.get("base_url") or get_env_value("OPENAI_BASE_URL") or None))
        with client.audio.speech.with_streaming_response.create(
            model=self.section.get("model", "gpt-4o-mini-tts"), voice=self.section.get("voice", "alloy"),
            input=text, response_format="pcm",
        ) as response:
            yield from _capped(response.iter_bytes(), "OpenAI streaming TTS")


@register("gemini")
class GeminiStreamer(StreamingTTSProvider):
    """Gemini ``streamGenerateContent?alt=sse`` → SSE feed of base64 PCM chunks (24 kHz), bounded streamed body.

    Salvaged from PR #47588 (@Cdddo) and rebased onto the post-campaign infrastructure: credentials via the
    provider-secret resolver, requests (not httpx) with a bounded streamed body, and main's provider ABC.
    """

    @staticmethod
    def available() -> bool:
        return bool(_gemini_key())

    def stream(self, text: str) -> Iterator[bytes]:
        import base64
        import json as _json
        import requests
        from tools.tts_tool_providers import (
            DEFAULT_GEMINI_TTS_BASE_URL, DEFAULT_GEMINI_TTS_MODEL, DEFAULT_GEMINI_TTS_VOICE)
        from hermes_cli.config import get_env_value
        api_key = _gemini_key()
        model = str(self.section.get("model", DEFAULT_GEMINI_TTS_MODEL)).strip() or DEFAULT_GEMINI_TTS_MODEL
        voice = str(self.section.get("voice", DEFAULT_GEMINI_TTS_VOICE)).strip() or DEFAULT_GEMINI_TTS_VOICE
        from agent.gemini_native_adapter import normalize_gemini_base_url
        base_url = normalize_gemini_base_url(
            self.section.get("base_url") or get_env_value("GEMINI_BASE_URL") or DEFAULT_GEMINI_TTS_BASE_URL, api_key,
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
        url = f"{base_url}/models/{model}:streamGenerateContent"

        def _sse_chunks() -> Iterator[bytes]:
            with requests.post(
                url, params={"alt": "sse", "key": api_key}, json=payload, timeout=60, stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        parts = _json.loads(line[len("data: "):])["candidates"][0]["content"]["parts"]
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    for part in parts:
                        b64 = (part.get("inlineData") or part.get("inline_data") or {}).get("data", "")
                        if not b64:
                            continue
                        try:
                            yield base64.b64decode(b64)
                        except (ValueError, TypeError) as exc:
                            logger.warning("Gemini SSE: bad base64 audio: %s", exc)

        yield from _capped(_sse_chunks(), "Gemini streaming TTS")


@register("xai")
class XAIStreamer(StreamingTTSProvider):
    """xAI WebSocket TTS (``wss://api.x.ai/v1/tts``) → binary PCM frames (24 kHz mono int16).
    Credentials route through ``resolve_xai_http_credentials`` (OAuth or XAI_API_KEY), same as the
    sync path. ``_collect_async`` bridges the async WS loop to the sync iterator contract (test
    seam).

    Salvaged from PR #47588 (@Cdddo): xAI's chunked TTS API is WebSocket-only (``wss://api.x.ai/v1/tts``).
    """

    @staticmethod
    def available() -> bool:
        try:
            from tools.xai_http import resolve_xai_http_credentials
            # Same ordering as the sync path: the subscription OAuth bearer
            # authorizes but 403s on metered TTS, so an explicit key wins (#87045).
            return bool(str(resolve_xai_http_credentials(prefer_api_key=True).get("api_key") or "").strip())
        except Exception:
            return False

    def stream(self, text: str) -> Iterator[bytes]:
        yield from _capped(iter(self._collect_async(text)), "xAI streaming TTS")

    def _collect_async(self, text: str) -> List[bytes]:
        import asyncio

        async def _drain() -> List[bytes]:
            return [frame async for frame in self._async_frames(text)]
        return asyncio.run(_drain())

    async def _async_frames(self, text: str):
        import json as _json
        import websockets
        from tools.tts_tool_providers import DEFAULT_XAI_VOICE_ID
        from tools.xai_http import resolve_xai_http_credentials
        api_key = str(resolve_xai_http_credentials(prefer_api_key=True).get("api_key") or "").strip()
        if not api_key:
            raise RuntimeError("No xAI credentials for streaming TTS")
        voice = str(self.section.get("voice_id", DEFAULT_XAI_VOICE_ID)).strip() or DEFAULT_XAI_VOICE_ID
        ws_url = str(self.section.get("streaming_url") or "wss://api.x.ai/v1/tts").strip()
        async with websockets.connect(ws_url, additional_headers={"Authorization": f"Bearer {api_key}"}) as ws:
            await ws.send(_json.dumps({"text": text, "voice_id": voice, "response_format": "pcm"}))
            try:
                while True:
                    message = await ws.recv()
                    if isinstance(message, (bytes, bytearray, memoryview)):
                        yield bytes(message)
                        continue
                    try:
                        envelope = _json.loads(message)
                    except (ValueError, TypeError):
                        if message == "done":
                            return
                        continue
                    etype = envelope.get("type")
                    if etype == "error":
                        logger.warning(
                            "xAI WS error envelope: %s", envelope.get("error") or envelope.get("message") or envelope,
                        )
                    if etype in ("done", "error"):
                        return
            except Exception as exc:
                if exc.__class__.__name__ != "ConnectionClosed":
                    logger.warning("xAI WS receive failed: %s", exc)
                return
