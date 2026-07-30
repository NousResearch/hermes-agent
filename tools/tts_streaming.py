"""Provider-agnostic streaming TTS: sentence text → int16 mono PCM chunk iterator.

``stream_tts_to_speaker`` (``tools.tts_tool``) owns the sentence buffer, sounddevice
output and stop/queue protocol; this module owns the *provider* half so playback
starts on sentence one. True streamers (``StreamingTTSProvider.stream``) wrap chunked
APIs; providers with no chunked API (edge, the default) get per-sentence playback via
the sync ``text_to_speech_tool`` path. Adding a streamer is ``@register("name")`` on
a subclass; the dispatcher, config gate (``tts.<name>.streaming``) and resolver come free.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional

from tools.tool_backend_helpers import resolve_openai_audio_api_key
from tools.tts_tool import _get_provider, _load_tts_config

logger = logging.getLogger(__name__)

# Per-sentence PCM byte cap, mirroring the sync providers' 16 MiB bounded-body invariant.
_STREAM_SENTENCE_BYTE_CAP = 16 * 1024 * 1024


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

    @classmethod
    def from_config(cls, tts_config: Dict) -> "SentenceChunker":
        """Chunker honouring ``tts.streaming.min_len``. 20 suits English; a CJK opener of 5–7
        characters is a whole clause, so voice setups lower it to speak the first sentence
        alone instead of buffering it behind the second. Floor 1: 0 would emit every boundary."""
        try:
            return cls(min_len=max(1, int((tts_config.get("streaming") or {}).get("min_len", 20))))
        except (AttributeError, TypeError, ValueError):  # non-mapping / non-numeric → default
            return cls()

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
    """Yields raw int16, little-endian, mono PCM chunks at ``sample_rate`` (built-ins: 24 kHz).

    ``sample_rate`` is provisional until ``stream()`` has yielded its first chunk: a provider may
    update the instance attribute once the endpoint's real format is known (OpenAI-compatible
    servers advertise it in the response headers), so consumers open their output device or WAV
    header after pulling the first chunk, never at construction.
    """

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


# Fallback priority for ``tts.streaming.provider: auto`` — best chunked latency/quality
# first. Deliberately hard-coded (a UX decision); edge is absent (no chunked-PCM API).
_PROVIDER_PRIORITY: List[str] = ["elevenlabs", "gemini", "openai", "xai"]


def resolve_streaming_provider(
    tts_config: Dict, preferred: Optional[str] = None) -> Optional[StreamingTTSProvider]:
    """Return a ready streamer for the *configured* provider, else ``None``.
    ``tts.streaming.provider`` when set: a name pins that exact streamer (``None`` if unusable);
    ``auto`` returns the first usable in ``_PROVIDER_PRIORITY``. Otherwise the configured TTS
    provider (or ``preferred``): ``None`` means "no chunked API" — the dispatcher speaks
    per-sentence via the sync path, preserving the user's chosen voice. We never silently swap
    providers just to get streaming."""
    pinned = str((tts_config.get("streaming") or {}).get("provider") or "").lower().strip()
    if pinned == "auto":
        return next((inst for name in _PROVIDER_PRIORITY
                     if (inst := _try_instantiate(name, tts_config))), None)
    return _try_instantiate(pinned or (preferred or _get_provider(tts_config)).lower().strip(), tts_config)


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


def _openai_config_api_key() -> str:
    """Return ``tts.openai.api_key`` from config.yaml, or empty string."""
    try:
        return (_load_tts_config().get("openai") or {}).get("api_key") or ""
    except Exception:
        return ""


def _sample_rate_from_headers(headers) -> Optional[int]:
    """Rate an OpenAI-compatible TTS endpoint advertises: ``X-Audio-Sample-Rate`` (the convention
    local servers use) or ``rate=`` in ``Content-Type`` (``audio/pcm; rate=44100``); None if absent."""
    if not headers:
        return None
    raw = headers.get("x-audio-sample-rate")
    if raw is None:
        m = re.search(r"(?:^|[;\s])rate\s*=\s*(\d+)", str(headers.get("content-type") or ""), re.IGNORECASE)
        raw = m.group(1) if m else None
    try:
        rate = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return rate if rate > 0 else None


@register("openai")
class OpenAIStreamer(StreamingTTSProvider):
    """OpenAI speech with ``response_format=pcm`` (OpenAI itself: 24 kHz mono int16).

    Compatible servers may emit another rate: ``tts.openai.pcm_sample_rate`` sets the expected
    rate up front and a rate reported by the response (``X-Audio-Sample-Rate`` / Content-Type
    ``rate=``) overrides it before the first chunk is yielded (#76466).
    """

    def __init__(self, tts_config: Dict, section: Dict):
        super().__init__(tts_config, section)
        configured = section.get("pcm_sample_rate", self.sample_rate)
        if isinstance(configured, bool) or not isinstance(configured, (int, float, str)) \
                or not str(configured).strip().isdigit() or int(str(configured).strip()) <= 0:
            logger.warning("Invalid tts.openai.pcm_sample_rate %r; using %d Hz", configured, self.sample_rate)
        else:
            self.sample_rate = int(str(configured).strip())

    @staticmethod
    def available() -> bool:
        return bool(_openai_config_api_key() or resolve_openai_audio_api_key())

    def stream(self, text: str) -> Iterator[bytes]:
        from openai import OpenAI
        from hermes_cli.config import get_env_value
        client = OpenAI(
            api_key=(self.section.get("api_key") or resolve_openai_audio_api_key()),
            base_url=(self.section.get("base_url") or get_env_value("OPENAI_BASE_URL") or None))
        from tools.tts_tool_openai import _openai_extra_body
        extra = {"extra_body": body} if (body := _openai_extra_body(self.section)) else {}
        with client.audio.speech.with_streaming_response.create(
            model=self.section.get("model", "gpt-4o-mini-tts"), voice=self.section.get("voice", "alloy"),
            input=text, response_format="pcm", **extra,
        ) as response:
            # Runs on the first next(), before any audio is yielded, so consumers reading
            # ``sample_rate`` after the first chunk open their device at the endpoint's rate.
            rate = _sample_rate_from_headers(getattr(response, "headers", None))
            if rate is not None and rate != self.sample_rate:
                logger.info("TTS endpoint reports %d Hz PCM (expected %d Hz); honoring it", rate, self.sample_rate)
                self.sample_rate = rate
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
            self.section.get("base_url") or get_env_value("GEMINI_BASE_URL") or DEFAULT_GEMINI_TTS_BASE_URL,
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
        url = f"{base_url}/models/{model}:streamGenerateContent"

        def _sse_chunks() -> Iterator[bytes]:
            with requests.post(
                url, params={"alt": "sse"}, headers={"x-goog-api-key": api_key},
                json=payload, timeout=60, stream=True,
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
    """xAI WebSocket TTS → base64 PCM ``audio.delta`` frames (24 kHz mono int16).

    Salvaged from PR #47588 (@Cdddo) and rewritten against the real wire
    protocol: voice/language/codec/sample_rate ride in the URL query string
    (the server 400s the bare path at handshake), the client sends
    ``text.delta`` + ``text.done``, and the server streams JSON
    ``audio.delta`` envelopes (base64 PCM in ``delta``) until ``audio.done``.
    Credentials route through ``resolve_xai_http_credentials`` (XAI_API_KEY
    first, OAuth as fallback), same as the sync ``_generate_xai_tts`` path. The async WS
    loop runs on a background thread feeding a queue, so ``stream()`` yields
    chunks as they arrive and works from sync CLI code and from gateway
    adapters that already run an event loop.
    """

    sample_rate = 24000

    _RECV_TIMEOUT_S = 60  # a sentence of TTS should never gap this long

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
        yield from _capped(self._queued_frames(text), "xAI streaming TTS")

    # -- async→sync bridge -------------------------------------------------

    def _queued_frames(self, text: str) -> Iterator[bytes]:
        """Yield PCM chunks as the WS delivers them, on a pump thread.

        The thread owns a fresh event loop, so ``asyncio.run`` never fights
        a loop the caller is already running (gateway adapters are async).
        Exceptions from the pump are re-raised on the consumer side so the
        caller's "raise on failure" contract holds.

        ``_capped()`` runs on the consumer side of the queue, after
        ``q.get()`` — so the byte budget is ALSO enforced in the pump before
        each enqueue (see ``_pump``), and a consumer that stops early flips
        ``stop`` so the pump closes the socket instead of piling decoded PCM
        into a queue nobody drains.
        """
        import asyncio
        import queue
        import threading
        from contextvars import copy_context

        q: "queue.Queue[object]" = queue.Queue()
        done = object()
        stop = threading.Event()

        def _pump_thread() -> None:
            try:
                asyncio.run(self._pump(text, q, stop))
            except BaseException as exc:  # hand failures to the consumer
                q.put(exc)
            finally:
                q.put(done)

        threading.Thread(
            target=copy_context().run, args=(_pump_thread,), name="xai-tts-pump", daemon=True
        ).start()
        try:
            while True:
                item = q.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Generator closed early (cap, playback failure, caller dropped
            # it): stop the pump at its next frame.
            stop.set()

    async def _pump(self, text: str, q, stop) -> None:
        import asyncio
        import base64
        import json as _json
        from urllib.parse import urlencode

        import websockets

        from tools.tts_tool_providers import DEFAULT_XAI_LANGUAGE, DEFAULT_XAI_VOICE_ID
        from tools.xai_http import resolve_xai_http_credentials
        api_key = str(resolve_xai_http_credentials(prefer_api_key=True).get("api_key") or "").strip()
        if not api_key:
            raise RuntimeError("No xAI credentials for streaming TTS")
        voice = str(self.section.get("voice_id", DEFAULT_XAI_VOICE_ID)).strip() or DEFAULT_XAI_VOICE_ID
        language = str(self.section.get("language", DEFAULT_XAI_LANGUAGE)).strip() or DEFAULT_XAI_LANGUAGE
        base = str(
            self.section.get("streaming_url") or "wss://api.x.ai/v1/tts"
        ).strip()
        params = urlencode({
            "voice": voice,
            "language": language,
            "codec": "pcm",
            "sample_rate": self.sample_rate,
        })
        sep = "&" if "?" in base else "?"
        ws_url = f"{base}{sep}{params}"

        async with websockets.connect(
            ws_url, additional_headers={"Authorization": f"Bearer {api_key}"}
        ) as ws:
            await ws.send(_json.dumps({"type": "text.delta", "delta": text}))
            await ws.send(_json.dumps({"type": "text.done"}))
            enqueued = 0
            while True:
                try:
                    message = await asyncio.wait_for(
                        ws.recv(), timeout=self._RECV_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError("xAI streaming TTS: no audio for 60s")
                except websockets.exceptions.ConnectionClosedOK:
                    return  # clean close, with or without audio.done
                except websockets.exceptions.ConnectionClosedError as exc:
                    raise RuntimeError(f"xAI WS closed mid-stream: {exc}") from exc
                if isinstance(message, (bytes, bytearray, memoryview)):
                    continue  # the server speaks JSON; binary is not expected
                try:
                    envelope = _json.loads(message)
                except (ValueError, TypeError):
                    continue
                msg_type = envelope.get("type")
                if msg_type == "audio.delta":
                    b64 = envelope.get("delta") or ""
                    if b64:
                        pcm = base64.b64decode(b64)
                        enqueued += len(pcm)
                        if enqueued > _STREAM_SENTENCE_BYTE_CAP:
                            # Enforce the per-sentence byte budget BEFORE
                            # enqueueing: the consumer-side _capped() only
                            # runs after q.get(), so without this a runaway
                            # upstream piles decoded PCM into the queue.
                            # Returning exits the context manager, which
                            # closes the socket — we stop reading upstream.
                            logger.warning(
                                "xAI streaming TTS exceeded %d bytes for one "
                                "sentence; closing upstream",
                                _STREAM_SENTENCE_BYTE_CAP,
                            )
                            return
                        if stop.is_set():
                            # Consumer went away (cap, playback failure):
                            # close and stop reading rather than filling a
                            # queue nobody drains.
                            return
                        q.put(pcm)
                elif msg_type == "audio.done":
                    return
                elif msg_type == "error":
                    detail = (
                        envelope.get("message") or envelope.get("error") or envelope
                    )
                    raise RuntimeError(f"xAI streaming TTS error: {detail}")
