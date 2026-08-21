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
* **Everyone else** — providers with no chunked API still get per-*sentence*
  playback via the proven sync `text_to_speech_tool` path (handled by the
  dispatcher, not here), so edge (the default) is conversational too.

Adding a streamer is `@register("name")` on a `StreamingTTSProvider` subclass;
the dispatcher, config gate (`tts.<name>.streaming`), and resolver come free.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional

from tools.tool_backend_helpers import resolve_openai_audio_api_key
from tools.tts_tool import _get_provider, _load_tts_config, get_env_value

logger = logging.getLogger(__name__)

# Upper bound on the PCM bytes accepted from one provider stream for one
# sentence. Mirrors the 16 MiB bounded-upstream-body invariant of the sync
# providers (``_read_tts_response_bytes`` in tools.tts_tool): a buggy or
# hostile endpoint must not be able to feed us unbounded audio.
_STREAM_SENTENCE_BYTE_CAP = 16 * 1024 * 1024
_GEMINI_STREAM_DEADLINE_S = 60.0
_GEMINI_RAW_SSE_BYTE_CAP = 24 * 1024 * 1024
_GEMINI_SSE_EVENT_BYTE_CAP = 4 * 1024 * 1024


class StreamingTTSLimitError(RuntimeError):
    """A bounded streaming transport exceeded an integrity/resource limit."""


def _resolve_key(env_var: str, provider_id: str) -> str:
    """Provider secret lookup: config > env/.env > credential pool.

    Thin, monkeypatchable seam over ``tools.tts_tool._resolve_provider_key``
    (which delegates to ``resolve_provider_secret``). ALL streaming-provider
    key lookups go through here — never bare ``get_env_value``.
    """
    try:
        from tools.tts_tool import _resolve_provider_key

        return _resolve_provider_key(env_var, provider_id) or ""
    except Exception:
        return get_env_value(env_var) or ""


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
    # Gateway streaming requires native async cancellation that can stop DNS,
    # connect, and header waits. Legacy synchronous stream() implementations
    # may keep best-effort cancel() without qualifying for gateway use.
    async_transport_cancellable: bool = False

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

    def cancel(self) -> None:
        """Best-effort cancellation of an active provider request."""


_REGISTRY: Dict[str, type[StreamingTTSProvider]] = {}


def register(name: str) -> Callable[[type[StreamingTTSProvider]], type[StreamingTTSProvider]]:
    def _wrap(cls: type[StreamingTTSProvider]) -> type[StreamingTTSProvider]:
        _REGISTRY[name] = cls
        return cls

    return _wrap


def _try_instantiate(name: str, tts_config: Dict) -> Optional[StreamingTTSProvider]:
    """Construct the registered streamer *name* if it's usable, else None."""
    cls = _REGISTRY.get(name)
    if cls is None or not cls.available():
        return None
    try:
        return cls(tts_config, tts_config.get(name) or {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("streaming provider %s init failed: %s", name, exc)
        return None


# Fallback priority for ``tts.streaming.provider: auto`` — best chunked
# latency/quality first. Deliberately hard-coded (a UX decision, not a
# config knob); edge is absent because it has no chunked-PCM API — the
# dispatcher's per-sentence sync path keeps it conversational instead.
_PROVIDER_PRIORITY: List[str] = ["elevenlabs", "gemini", "openai", "xai"]


def resolve_streaming_provider(
    tts_config: Dict,
    preferred: Optional[str] = None,
    *,
    require_transport_cancellation: bool = False,
) -> Optional[StreamingTTSProvider]:
    """Return a ready streamer for the *configured* provider, else ``None``.

    Resolution order:

    1. ``tts.streaming.provider`` (config knob) when set:
       * a provider name pins that exact streamer (or ``None`` if unusable);
       * ``auto`` walks the priority list (``elevenlabs → gemini → openai
         → xai``) and returns the first usable streamer — an explicit
         opt-in to "give me the best chunked voice available".
    2. Otherwise the *configured* TTS provider (or ``preferred`` override).
       ``None`` means "no chunked API for this provider" — the dispatcher
       then speaks per-sentence via the sync path, preserving the user's
       chosen voice. We never silently swap to a different provider just
       to get streaming.
    """
    def _acceptable(
        instance: Optional[StreamingTTSProvider],
    ) -> Optional[StreamingTTSProvider]:
        if instance is None:
            return None
        if require_transport_cancellation and (
            not getattr(instance, "async_transport_cancellable", False)
            or not callable(getattr(instance, "astream", None))
        ):
            return None
        return instance

    streaming_cfg = tts_config.get("streaming") or {}
    pinned = str(streaming_cfg.get("provider") or "").lower().strip()
    if pinned == "auto":
        for name in _PROVIDER_PRIORITY:
            inst = _acceptable(_try_instantiate(name, tts_config))
            if inst is not None:
                return inst
        return None
    if pinned:
        return _acceptable(_try_instantiate(pinned, tts_config))

    name = (preferred or _get_provider(tts_config)).lower().strip()
    return _acceptable(_try_instantiate(name, tts_config))


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@register("elevenlabs")
class ElevenLabsStreamer(StreamingTTSProvider):
    """ElevenLabs chunked HTTP → pcm_24000 (the original reference path)."""

    sample_rate = 24000

    @staticmethod
    def available() -> bool:
        return bool(_resolve_key("ELEVENLABS_API_KEY", "elevenlabs"))

    def stream(self, text: str) -> Iterator[bytes]:
        from tools.tts_tool import (
            DEFAULT_ELEVENLABS_STREAMING_MODEL_ID,
            DEFAULT_ELEVENLABS_VOICE_ID,
            _elevenlabs_environment_kwargs,
            _import_elevenlabs,
        )

        client = _import_elevenlabs()(
            api_key=_resolve_key("ELEVENLABS_API_KEY", "elevenlabs"),
            **_elevenlabs_environment_kwargs(self.section),
        )
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


def _openai_config_api_key() -> str:
    """Return ``tts.openai.api_key`` from config.yaml, or empty string."""
    try:
        openai_cfg = (_load_tts_config().get("openai") or {})
    except Exception:
        return ""
    return openai_cfg.get("api_key") or ""


@register("openai")
class OpenAIStreamer(StreamingTTSProvider):
    """OpenAI speech with ``response_format=pcm`` (24 kHz mono int16)."""

    sample_rate = 24000

    @staticmethod
    def available() -> bool:
        return bool(_openai_config_api_key() or resolve_openai_audio_api_key())

    def stream(self, text: str) -> Iterator[bytes]:
        from openai import OpenAI

        client = OpenAI(
            api_key=(self.section.get("api_key") or resolve_openai_audio_api_key()),
            base_url=(
                self.section.get("base_url")
                or get_env_value("OPENAI_BASE_URL")
                or None
            ),
        )
        model = self.section.get("model", "gpt-4o-mini-tts")
        voice = self.section.get("voice", "alloy")
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            response_format="pcm",
        ) as response:
            yield from _capped(response.iter_bytes(), "OpenAI streaming TTS")


def _capped(chunks: Iterator[bytes], label: str) -> Iterator[bytes]:
    """Pass chunks through, aborting past the 16 MiB per-sentence cap.

    The streaming mirror of ``_read_tts_response_bytes``'s bounded-body
    invariant: one sentence of PCM should never approach the cap, so
    exceeding it means a runaway/hostile upstream — stop pulling.
    """
    total = 0
    for chunk in chunks:
        total += len(chunk)
        if total > _STREAM_SENTENCE_BYTE_CAP:
            raise StreamingTTSLimitError(
                f"{label} exceeded its decoded audio cap of "
                f"{_STREAM_SENTENCE_BYTE_CAP} bytes"
            )
        yield chunk


def _safe_close_streaming_response(response) -> None:
    try:
        response.close()
    except Exception:
        pass


def _iter_bounded_sse_data(
    response,
    *,
    label: str,
    deadline: float,
    raw_byte_cap: int = _GEMINI_RAW_SSE_BYTE_CAP,
) -> Iterator[bytes]:
    """Yield bounded SSE data and close blocked reads at the absolute deadline."""
    content_type = str(response.headers.get("content-type", ""))
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "text/event-stream":
        raise RuntimeError(f"{label} returned unexpected content type {content_type!r}")

    deadline_timer = threading.Timer(
        max(0.0, deadline - time.monotonic()),
        _safe_close_streaming_response,
        args=(response,),
    )
    deadline_timer.daemon = True
    deadline_timer.start()
    event_cap = _GEMINI_SSE_EVENT_BYTE_CAP
    raw_total = 0
    pending = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=8192):
            if time.monotonic() > deadline:
                raise TimeoutError(f"{label} exceeded its absolute deadline")
            if not chunk:
                continue
            raw_total += len(chunk)
            if raw_total > raw_byte_cap:
                raise RuntimeError(f"{label} exceeded {raw_byte_cap} raw SSE bytes")
            pending.extend(chunk)
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    if len(pending) > event_cap:
                        raise RuntimeError(f"{label} SSE event exceeds {event_cap} bytes")
                    break
                line = bytes(pending[:newline]).rstrip(b"\r")
                del pending[:newline + 1]
                if len(line) > event_cap:
                    raise RuntimeError(f"{label} SSE event exceeds {event_cap} bytes")
                if line.startswith(b"data:"):
                    yield line[len(b"data:"):].lstrip()
        if time.monotonic() > deadline:
            raise TimeoutError(f"{label} exceeded its absolute deadline")
        if len(pending) > event_cap:
            raise RuntimeError(f"{label} SSE event exceeds {event_cap} bytes")
        line = bytes(pending).rstrip(b"\r")
        if line.startswith(b"data:"):
            yield line[len(b"data:"):].lstrip()
    finally:
        deadline_timer.cancel()


def _decode_gemini_sse_audio(raw: bytes) -> List[bytes]:
    """Decode supported mono 24 kHz L16 parts from one Gemini SSE event."""
    import base64
    import json

    try:
        event = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(event, dict):
        return []
    try:
        parts = event["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return []

    chunks: List[bytes] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData") or part.get("inline_data") or {}
        if not isinstance(inline, dict):
            continue
        mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
        mime_parts = [item.strip().lower() for item in mime.split(";")]
        media_type = mime_parts[0] if mime_parts else ""
        params: Dict[str, str] = {}
        malformed = False
        for item in mime_parts[1:]:
            if "=" not in item:
                malformed = True
                break
            key, value = item.split("=", 1)
            if not key or key in params:
                malformed = True
                break
            params[key] = value
        if (
            malformed
            or media_type != "audio/l16"
            or params.get("rate") != "24000"
            or params.get("channels", "1") != "1"
        ):
            logger.warning(
                "Gemini SSE: ignoring unsupported audio format %r",
                mime,
            )
            continue
        b64 = inline.get("data", "")
        if not b64:
            continue
        try:
            chunks.append(base64.b64decode(b64, validate=True))
        except (ValueError, TypeError) as exc:
            logger.warning("Gemini SSE: bad base64 audio: %s", exc)
    return chunks


@register("gemini")
class GeminiStreamer(StreamingTTSProvider):
    """Gemini ``streamGenerateContent?alt=sse`` → PCM chunks (24 kHz mono).

    ``gemini-3.1-flash-tts-preview`` is currently the only Gemini TTS model
    that emits audio incrementally. Credentials are resolved through the
    provider-secret chain and sent in ``x-goog-api-key``, never the URL.
    """

    sample_rate = 24000
    async_transport_cancellable = True

    def __init__(self, tts_config: Dict, section: Dict):
        super().__init__(tts_config, section)
        self._response_lock = threading.Lock()
        self._active_response = None
        self._active_async_loop = None
        self._active_async_task = None
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._response_lock:
            response = self._active_response
            loop = self._active_async_loop
            task = self._active_async_task
        if response is not None:
            _safe_close_streaming_response(response)
        if loop is not None and task is not None and not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)

    @staticmethod
    def available() -> bool:
        return bool(
            _resolve_key("GEMINI_API_KEY", "gemini")
            or _resolve_key("GOOGLE_API_KEY", "gemini")
        )

    async def astream(self, text: str):
        """Async Gemini transport used by gateway voice for prompt cancellation."""
        import asyncio

        import httpx

        from tools.tts_tool import (
            DEFAULT_GEMINI_TTS_BASE_URL,
            DEFAULT_GEMINI_TTS_MODEL,
            DEFAULT_GEMINI_TTS_VOICE,
        )

        if self._cancelled.is_set():
            return
        api_key = (
            _resolve_key("GEMINI_API_KEY", "gemini")
            or _resolve_key("GOOGLE_API_KEY", "gemini")
        )
        model = str(
            self.section.get("model", DEFAULT_GEMINI_TTS_MODEL)
        ).strip() or DEFAULT_GEMINI_TTS_MODEL
        voice = str(
            self.section.get("voice", DEFAULT_GEMINI_TTS_VOICE)
        ).strip() or DEFAULT_GEMINI_TTS_VOICE
        base_url = str(
            self.section.get("base_url")
            or get_env_value("GEMINI_BASE_URL")
            or DEFAULT_GEMINI_TTS_BASE_URL
        ).strip().rstrip("/")
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice},
                    },
                },
            },
        }
        url = f"{base_url}/models/{model}:streamGenerateContent"
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        with self._response_lock:
            self._active_async_loop = loop
            self._active_async_task = task

        raw_byte_cap = _GEMINI_RAW_SSE_BYTE_CAP
        event_cap = _GEMINI_SSE_EVENT_BYTE_CAP
        raw_total = 0
        decoded_total = 0
        pending = bytearray()
        try:
            timeout = httpx.Timeout(
                _GEMINI_STREAM_DEADLINE_S,
                connect=min(10.0, _GEMINI_STREAM_DEADLINE_S),
            )
            async with asyncio.timeout(_GEMINI_STREAM_DEADLINE_S):
                async with httpx.AsyncClient(
                    follow_redirects=False,
                    timeout=timeout,
                ) as client:
                    async with client.stream(
                        "POST",
                        url,
                        params={"alt": "sse"},
                        headers={"x-goog-api-key": api_key},
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        content_type = str(response.headers.get("content-type", ""))
                        media_type = content_type.split(";", 1)[0].strip().lower()
                        if media_type != "text/event-stream":
                            raise RuntimeError(
                                "Gemini streaming TTS returned unexpected "
                                f"content type {content_type!r}"
                            )
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            if not chunk:
                                continue
                            raw_total += len(chunk)
                            if raw_total > raw_byte_cap:
                                raise RuntimeError(
                                    "Gemini streaming TTS exceeded "
                                    f"{raw_byte_cap} raw SSE bytes"
                                )
                            pending.extend(chunk)
                            while True:
                                newline = pending.find(b"\n")
                                if newline < 0:
                                    if len(pending) > event_cap:
                                        raise RuntimeError(
                                            "Gemini streaming TTS SSE event "
                                            f"exceeds {event_cap} bytes"
                                        )
                                    break
                                line = bytes(pending[:newline]).rstrip(b"\r")
                                del pending[:newline + 1]
                                if len(line) > event_cap:
                                    raise RuntimeError(
                                        "Gemini streaming TTS SSE event "
                                        f"exceeds {event_cap} bytes"
                                    )
                                if not line.startswith(b"data:"):
                                    continue
                                for pcm in _decode_gemini_sse_audio(
                                    line[len(b"data:"):].lstrip()
                                ):
                                    decoded_total += len(pcm)
                                    if decoded_total > _STREAM_SENTENCE_BYTE_CAP:
                                        raise StreamingTTSLimitError(
                                            "Gemini streaming TTS exceeded its "
                                            "decoded audio cap of "
                                            f"{_STREAM_SENTENCE_BYTE_CAP} bytes"
                                        )
                                    yield pcm
                        if len(pending) > event_cap:
                            raise RuntimeError(
                                "Gemini streaming TTS SSE event "
                                f"exceeds {event_cap} bytes"
                            )
                        line = bytes(pending).rstrip(b"\r")
                        if line.startswith(b"data:"):
                            for pcm in _decode_gemini_sse_audio(
                                line[len(b"data:"):].lstrip()
                            ):
                                decoded_total += len(pcm)
                                if decoded_total > _STREAM_SENTENCE_BYTE_CAP:
                                    raise StreamingTTSLimitError(
                                        "Gemini streaming TTS exceeded its "
                                        "decoded audio cap of "
                                        f"{_STREAM_SENTENCE_BYTE_CAP} bytes"
                                    )
                                yield pcm
        finally:
            with self._response_lock:
                if self._active_async_task is task:
                    self._active_async_loop = None
                    self._active_async_task = None

    def stream(self, text: str) -> Iterator[bytes]:
        """Legacy synchronous compatibility path.

        Gateway playback never selects this path: only ``astream()`` advertises
        cancellation across DNS/connect/header/body. Here cancel() is best-effort
        once ``requests`` has returned a response, with phase timeouts providing
        the pre-header bound.
        """
        import requests

        from tools.tts_tool import (
            DEFAULT_GEMINI_TTS_BASE_URL,
            DEFAULT_GEMINI_TTS_MODEL,
            DEFAULT_GEMINI_TTS_VOICE,
        )

        api_key = (
            _resolve_key("GEMINI_API_KEY", "gemini")
            or _resolve_key("GOOGLE_API_KEY", "gemini")
        )
        model = str(self.section.get("model", DEFAULT_GEMINI_TTS_MODEL)).strip() or DEFAULT_GEMINI_TTS_MODEL
        voice = str(self.section.get("voice", DEFAULT_GEMINI_TTS_VOICE)).strip() or DEFAULT_GEMINI_TTS_VOICE
        base_url = str(
            self.section.get("base_url")
            or get_env_value("GEMINI_BASE_URL")
            or DEFAULT_GEMINI_TTS_BASE_URL
        ).strip().rstrip("/")

        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice},
                    },
                },
            },
        }
        url = f"{base_url}/models/{model}:streamGenerateContent"

        def _sse_chunks() -> Iterator[bytes]:
            deadline = time.monotonic() + _GEMINI_STREAM_DEADLINE_S
            if self._cancelled.is_set():
                return
            remaining = max(0.001, deadline - time.monotonic())
            with requests.post(
                url,
                params={"alt": "sse"},
                headers={"x-goog-api-key": api_key},
                json=payload,
                timeout=(min(10.0, remaining), min(15.0, remaining)),
                stream=True,
                # requests may preserve custom headers across redirects. Do
                # not risk forwarding the API key to a different origin.
                allow_redirects=False,
            ) as response:
                with self._response_lock:
                    self._active_response = response
                try:
                    if self._cancelled.is_set():
                        _safe_close_streaming_response(response)
                        return
                    response.raise_for_status()
                    for raw in _iter_bounded_sse_data(
                        response,
                        label="Gemini streaming TTS",
                        deadline=deadline,
                    ):
                        yield from _decode_gemini_sse_audio(raw)
                    if not self._cancelled.is_set() and time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Gemini streaming TTS exceeded its absolute deadline"
                        )
                finally:
                    with self._response_lock:
                        if self._active_response is response:
                            self._active_response = None

        yield from _capped(_sse_chunks(), "Gemini streaming TTS")


@register("xai")
class XAIStreamer(StreamingTTSProvider):
    """xAI WebSocket TTS → binary PCM frames (24 kHz mono int16).

    Salvaged from PR #47588 (@Cdddo): xAI's chunked TTS API is
    WebSocket-only (``wss://api.x.ai/v1/tts``). Credentials route through
    ``resolve_xai_http_credentials`` (OAuth or XAI_API_KEY), same as the
    sync ``_generate_xai_tts`` path. The async WS loop is bridged to the
    sync iterator contract via ``_collect_async`` — the seam unit tests
    monkeypatch.
    """

    sample_rate = 24000

    @staticmethod
    def available() -> bool:
        try:
            from tools.xai_http import resolve_xai_http_credentials

            creds = resolve_xai_http_credentials()
            return bool(str(creds.get("api_key") or "").strip())
        except Exception:
            return False

    def stream(self, text: str) -> Iterator[bytes]:
        yield from _capped(iter(self._collect_async(text)), "xAI streaming TTS")

    # -- async→sync bridge (test seam) ------------------------------------

    def _collect_async(self, text: str) -> List[bytes]:
        import asyncio

        return asyncio.run(self._drain_async(text))

    async def _drain_async(self, text: str) -> List[bytes]:
        frames: List[bytes] = []
        async for frame in self._async_frames(text):
            frames.append(frame)
        return frames

    async def _async_frames(self, text: str):
        import json as _json

        import websockets

        from tools.tts_tool import DEFAULT_XAI_VOICE_ID
        from tools.xai_http import resolve_xai_http_credentials

        creds = resolve_xai_http_credentials()
        api_key = str(creds.get("api_key") or "").strip()
        if not api_key:
            raise RuntimeError("No xAI credentials for streaming TTS")
        voice = str(self.section.get("voice_id", DEFAULT_XAI_VOICE_ID)).strip() or DEFAULT_XAI_VOICE_ID
        ws_url = str(
            self.section.get("streaming_url") or "wss://api.x.ai/v1/tts"
        ).strip()

        async with websockets.connect(
            ws_url, extra_headers={"Authorization": f"Bearer {api_key}"}
        ) as ws:
            await ws.send(_json.dumps({
                "text": text,
                "voice_id": voice,
                "response_format": "pcm",
            }))
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
                    if etype == "done":
                        return
                    if etype == "error":
                        logger.warning("xAI WS error envelope: %s",
                                       envelope.get("error") or envelope.get("message") or envelope)
                        return
            except Exception as exc:
                if exc.__class__.__name__ == "ConnectionClosed":
                    return
                logger.warning("xAI WS receive failed: %s", exc)
                return
