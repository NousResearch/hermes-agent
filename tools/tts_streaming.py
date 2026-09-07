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
from tools.tts_tool import _get_provider, _load_tts_config, get_env_value

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


@register("openai")
class OpenAIStreamer(StreamingTTSProvider):
    """OpenAI speech with ``response_format=pcm`` (24 kHz mono int16)."""

    @staticmethod
    def available() -> bool:
        return bool(_openai_config_api_key() or resolve_openai_audio_api_key())

    def stream(self, text: str) -> Iterator[bytes]:
        from openai import OpenAI
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
        api_key = _gemini_key()
        model = str(self.section.get("model", DEFAULT_GEMINI_TTS_MODEL)).strip() or DEFAULT_GEMINI_TTS_MODEL
        voice = str(self.section.get("voice", DEFAULT_GEMINI_TTS_VOICE)).strip() or DEFAULT_GEMINI_TTS_VOICE
        base_url = str(
            self.section.get("base_url") or get_env_value("GEMINI_BASE_URL") or DEFAULT_GEMINI_TTS_BASE_URL
        ).strip().rstrip("/")
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
            return bool(str(resolve_xai_http_credentials().get("api_key") or "").strip())
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
        api_key = str(resolve_xai_http_credentials().get("api_key") or "").strip()
        if not api_key:
            raise RuntimeError("No xAI credentials for streaming TTS")
        voice = str(self.section.get("voice_id", DEFAULT_XAI_VOICE_ID)).strip() or DEFAULT_XAI_VOICE_ID
        ws_url = str(self.section.get("streaming_url") or "wss://api.x.ai/v1/tts").strip()
        async with websockets.connect(ws_url, extra_headers={"Authorization": f"Bearer {api_key}"}) as ws:
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


# ---------------------------------------------------------------------------
# Breeze — local warm-GPU streaming server (self-hosted, no cloud key)
# ---------------------------------------------------------------------------

DEFAULT_BREEZE_BASE_URL = "http://127.0.0.1:7860"

# Cache the /health probe briefly so ``available()`` (called during every
# provider resolution) doesn't hit the loopback socket on a hot path.
_BREEZE_HEALTH_TTL = 5.0
_breeze_health_cache: dict[str, tuple[float, bool, Optional[int]]] = {}
_breeze_health_lock = None  # lazily set to a threading.Lock in _breeze_health


def _breeze_base_url() -> str:
    """Resolve the Breeze server base URL: config > env > localhost default."""
    try:
        section = (_load_tts_config().get("breeze") or {})
        configured = str(section.get("base_url") or "").strip()
        if configured:
            return configured.rstrip("/")
    except Exception:
        pass
    return (get_env_value("BREEZE_TTS_URL") or DEFAULT_BREEZE_BASE_URL).rstrip("/")


def _breeze_health(base_url: str) -> tuple[bool, Optional[int]]:
    """Return ``(reachable, sample_rate)`` for the Breeze server at *base_url*.

    Cached for ``_BREEZE_HEALTH_TTL`` seconds per URL. Never raises — an
    unreachable or malformed server reports ``(False, None)`` so the resolver
    quietly falls back to the whole-file path.
    """
    global _breeze_health_lock
    import threading as _threading
    import time as _time

    if _breeze_health_lock is None:
        _breeze_health_lock = _threading.Lock()

    now = _time.monotonic()
    with _breeze_health_lock:
        cached = _breeze_health_cache.get(base_url)
        if cached is not None and (now - cached[0]) < _BREEZE_HEALTH_TTL:
            return cached[1], cached[2]

    ok, sr = False, None
    try:
        import json as _json
        import urllib.request as _urlreq

        with _urlreq.urlopen(f"{base_url}/health", timeout=1.5) as resp:
            payload = _json.loads(resp.read().decode("utf-8") or "{}")
        ok = str(payload.get("status") or "").lower() == "ok"
        try:
            sr = int(payload.get("sample_rate")) if payload.get("sample_rate") else None
        except (TypeError, ValueError):
            sr = None
    except Exception as exc:
        logger.debug("Breeze health check failed for %s: %s", base_url, exc)

    with _breeze_health_lock:
        _breeze_health_cache[base_url] = (now, ok, sr)
    return ok, sr


@register("breeze")
class BreezeStreamer(StreamingTTSProvider):
    """Local Breeze TTS 2 warm streaming server → s16le mono PCM.

    Breeze serves chunked ``audio/pcm`` from ``POST /v1/audio/speech`` (see the
    ``breeze-tts`` project: https://github.com/breezeblue-ai/breeze-tts). It runs
    *below* real-time (~0.7x on a GB10), so we
    buffer each clause to completion before yielding it: the consumer then hands
    the adapter a gap-free clause instead of risking a within-word underrun.
    The latency win comes from sentence-level pipelining upstream
    (:class:`SentenceChunker`) — synthesis of the next clause overlaps playback
    of the current one — not from sub-clause streaming, which this server is too
    slow to sustain against a real-time sink.

    Config (``tts.breeze`` in config.yaml)::

        tts:
          breeze:
            base_url: http://127.0.0.1:7860   # or $BREEZE_TTS_URL
            ref_audio: /path/to/reference.mp3  # optional voice-clone reference
            ref_text: "Transcript of the reference clip."
            instruction: "Speak clearly and naturally."
            cfg_scale: 1.0
            seed: 42
          streaming:
            provider: breeze                  # pin (base provider may differ)

    ``ref_audio`` + ``ref_text`` must be supplied together (Breeze rejects one
    without the other); omit both for the server's default speaker.
    """

    sample_rate = 24000  # refined from the server's X-Sample-Rate at init

    _BOUNDARY = "----HermesBreezeStreamerBoundary"

    def __init__(self, tts_config: Dict, section: Dict):
        super().__init__(tts_config, section)
        self.base_url = str(section.get("base_url") or _breeze_base_url()).rstrip("/")
        self.instruction = str(section.get("instruction") or "Speak clearly and naturally.")
        self.cfg_scale = section.get("cfg_scale", 1.0)
        self.seed = section.get("seed", 42)
        self.ref_audio = str(section.get("ref_audio") or "").strip()
        self.ref_text = str(section.get("ref_text") or "").strip()
        # Breeze serves one synthesis at a time (HTTP 409 while busy). Verbal
        # acks, the whole-file fallback, and overlapping turns can hold the lock,
        # so a clause request may need to wait. Poll until it frees rather than
        # dropping the clause — up to ``busy_timeout`` seconds.
        try:
            self.busy_timeout = float(section.get("busy_timeout", 20.0))
        except (TypeError, ValueError):
            self.busy_timeout = 20.0
        # Prefer an explicitly configured rate; otherwise adopt the server's.
        configured_sr = section.get("sample_rate")
        if configured_sr:
            try:
                self.sample_rate = int(configured_sr)
            except (TypeError, ValueError):
                pass
        else:
            _ok, sr = _breeze_health(self.base_url)
            if sr:
                self.sample_rate = sr

    @staticmethod
    def available() -> bool:
        ok, _sr = _breeze_health(_breeze_base_url())
        return ok

    def stream(self, text: str) -> Iterator[bytes]:
        yield from _capped(self._request(text), "Breeze streaming TTS")

    # -- HTTP ------------------------------------------------------------

    def _request_form(self) -> tuple[str, Dict[str, str]]:
        """Return ``(mode, fields)`` for the speech request.

        ``mode`` is ``"multipart"`` when a readable ``ref_audio`` is configured
        (voice clone), else ``"urlencoded"`` (text-only). ``fields["text"]`` is
        left ``None`` here and filled per-request in :meth:`_request`.
        """
        import os as _os

        fields = {
            "text": None,  # filled per-request in _request
            "instruction": self.instruction,
            "cfg_scale": str(self.cfg_scale),
            "seed": str(self.seed),
        }
        if self.ref_audio and self.ref_text and _os.path.isfile(self.ref_audio):
            fields["ref_text"] = self.ref_text
            return "multipart", fields
        return "urlencoded", fields

    def _request(self, text: str) -> Iterator[bytes]:
        import time as _time
        import urllib.error as _urlerr
        import urllib.parse as _urlparse
        import urllib.request as _urlreq

        mode, fields = self._request_form()
        fields = dict(fields)
        fields["text"] = text
        url = f"{self.base_url}/v1/audio/speech"

        if mode == "multipart":
            data, content_type = self._multipart(fields)
        else:
            data = _urlparse.urlencode(fields).encode("utf-8")
            content_type = "application/x-www-form-urlencoded"

        # The server synthesises one request at a time (HTTP 409 while busy).
        # Poll until it frees rather than dropping the clause. The 409 is raised
        # by urlopen() before any body arrives (the lock is taken server-side at
        # request start), so retrying before the first chunk never re-emits
        # already-yielded audio.
        deadline = _time.monotonic() + self.busy_timeout
        while True:
            req = _urlreq.Request(
                url, data=data, method="POST",
                headers={"Content-Type": content_type},
            )
            try:
                with _urlreq.urlopen(req, timeout=180) as resp:
                    while True:
                        chunk = resp.read(16384)
                        if not chunk:
                            return
                        yield chunk
                return
            except _urlerr.HTTPError as exc:
                if exc.code == 409 and _time.monotonic() < deadline:
                    _time.sleep(0.4)
                    continue
                raise
            except _urlerr.URLError as exc:
                raise RuntimeError(
                    f"Breeze server unreachable at {self.base_url}: {exc}"
                ) from exc

    def _multipart(self, fields: Dict[str, str]) -> tuple[bytes, str]:
        import os as _os

        boundary = self._BOUNDARY
        chunks: List[bytes] = []
        for key, value in fields.items():
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            )
            chunks.append(str(value).encode("utf-8") + b"\r\n")

        with open(self.ref_audio, "rb") as fh:
            ref_bytes = fh.read()
        filename = _os.path.basename(self.ref_audio)
        ctype = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="ref_audio"; '
            f'filename="{filename}"\r\n'.encode()
        )
        chunks.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        chunks.append(ref_bytes + b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
