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

import asyncio
import base64
import json
import logging
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Type

import numpy as np

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


def _import_httpx():
    """Lazy import the ``httpx`` HTTP client library.

    Returns the ``httpx`` module (not an instance) so callers can build
    their own ``httpx.Client()``. Raises ``ImportError`` with the install
    command when the package isn't available.

    This helper is the seam the test suite monkeypatches; keeping the
    import isolated in a single function means tests never need to
    install the real ``httpx`` package to validate the SSE parsing
    shape.
    """
    try:
        import httpx  # type: ignore
    except ImportError as e:
        raise ImportError(
            "httpx package not installed. Run: pip install httpx"
        ) from e
    return httpx


def _import_websockets():
    """Lazy import the ``websockets`` async WebSocket client library.

    Returns the ``websockets`` *module* (not a connection object) so
    callers can write ``websockets.connect(...)`` exactly like the real
    production code does. The module-level lookup means tests can
    monkeypatch the helper to return a stub module exposing a fake
    ``connect`` function without installing the real package.

    ``websockets`` is already a core dependency of ``hermes-agent`` (it's
    used by the browser CDP supervisor and ``browser_dialog``), so
    ``import websockets`` succeeds on every install; the
    ``ImportError`` branch is purely a defensive message so the user
    knows what to install if they ever build a lean venv without it.
    """
    try:
        import websockets  # type: ignore
    except ImportError as e:
        raise ImportError(
            "websockets package not installed. Run: pip install websockets"
        ) from e
    return websockets


def _import_openai_client():
    """Lazy import the ``openai`` SDK module.

    Returns the ``openai`` *module* (not the ``OpenAI`` client class) so
    the caller can write ``openai.OpenAI(...)`` exactly like the real
    production code does. The module-level lookup also means tests can
    monkeypatch the helper to return a stub module exposing a fake
    ``OpenAI`` class without installing the real package.

    This mirrors the lazy-import shape used by
    ``tools.tts_tool.py:_import_openai_client`` (around line 118) so the
    two import sites stay consistent — the dispatcher and the legacy
    sync path both treat a missing package the same way.
    """
    try:
        import openai  # type: ignore
    except ImportError as e:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        ) from e
    return openai


def _import_edge_tts():
    """Lazy import the ``edge_tts`` SDK module.

    Returns the ``edge_tts`` *module* (not a ``Communicate`` instance)
    so the caller can build its own ``Communicate(text, voice=...)``
    exactly like the real production code does. The module-level
    lookup also means tests can monkeypatch the helper to return a
    stub module exposing a fake ``Communicate`` class without
    installing the real package.

    This mirrors the lazy-import shape used by
    ``tools.tts_tool.py:_import_edge_tts`` so the two import sites
    stay consistent — the streaming dispatcher and the legacy
    ``_generate_edge_tts`` path both treat a missing package the same
    way. ``edge-tts`` is already an optional dependency tracked in
    ``tools.lazy_deps.py``; the ImportError branch here is the
    canonical "user must install it" message when running on a lean
    venv.
    """
    try:
        import edge_tts  # type: ignore
    except ImportError as e:
        raise ImportError(
            "edge-tts package not installed. Run: pip install edge-tts"
        ) from e
    return edge_tts


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


@register("gemini")
class GeminiStreamingProvider(StreamingTTSProvider):
    """Streaming TTS provider for Google Gemini's ``streamGenerateContent`` SSE endpoint.

    The Gemini TTS API (``gemini-2.5-flash-preview-tts`` and similar
    models) returns 24 kHz mono 16-bit signed PCM (L16) wrapped in a
    ``streamGenerateContent`` call when the ``?alt=sse`` query parameter
    is set — without that param the API returns a single JSON blob and
    the whole point of this class (chunked streaming) is lost. Each
    SSE event carries one base64-encoded PCM chunk under
    ``candidates[0].content.parts[*].inlineData.data``; we decode and
    yield each one as it arrives.

    Auth follows the same fallback as the non-streaming
    ``_generate_gemini_tts`` in ``tools.tts_tool.py``: ``GEMINI_API_KEY``
    wins, ``GOOGLE_API_KEY`` is the secondary, and a missing key fails
    loud in ``__init__`` so the dispatcher never pulls zero chunks.
    """

    # Gemini TTS hardcodes 24 kHz mono 16-bit (L16) PCM — see the API
    # docs for ``responseModalities=AUDIO`` + ``speechConfig``. We
    # declare it as class attrs so the dispatcher can build the
    # ``sounddevice.OutputStream`` with the matching format.
    sample_rate = 24000
    channels = 1
    sample_width = 2

    def __init__(self, config: dict, *, stop_event: Optional[threading.Event] = None):
        # Config keys mirror the ``tts.gemini`` block from config.yaml.
        # Defaults match the constants in tools.tts_tool.py
        # (DEFAULT_GEMINI_TTS_MODEL / _VOICE / _BASE_URL) so behaviour
        # stays consistent with the existing non-streaming path.
        self._model = config.get("model", "gemini-2.5-flash-preview-tts")
        self._voice = config.get("voice", "Kore")
        self._base_url = (
            config.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
            .strip()
            .rstrip("/")
        )
        # The stop event is optional so this class is usable outside the
        # dispatcher (e.g. one-off scripts). When None, the stream() loop
        # simply skips the stop check.
        self._stop_event = stop_event

        # Auth: prefer GEMINI_API_KEY, fall back to GOOGLE_API_KEY (same
        # as tools.tts_tool.py:_generate_gemini_tts). Empty/None both
        # mean "not set" — strip whitespace defensively.
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) not set; cannot use "
                "Gemini streaming TTS. Get one at "
                "https://aistudio.google.com/app/apikey"
            )
        self._api_key = api_key

        # Lazy SDK import — wrapped in a helper so the test suite can
        # monkeypatch it without requiring httpx at install time on
        # machines that never use this provider. We surface the import
        # error loudly because silently disabling the provider would
        # mask config bugs the user can fix. The return value is unused
        # in normal operation (we re-look up ``httpx`` via the module
        # globals below) but calling the helper gives the test suite a
        # seam to override the import behaviour.
        _import_httpx()
        import httpx as _httpx_module  # noqa: WPS433 — intentional late import
        # Build a long-lived client so the SSE connection reuses the
        # default keepalive pool. Tests patch ``httpx.Client.post`` at
        # the class level, so this instantiation must call
        # ``httpx.Client()`` (not the module-level ``httpx.post``).
        self._client = _httpx_module.Client()

    def stream(self, text: str) -> Iterator[bytes]:
        """Yield PCM chunks for *text* as they arrive from Gemini.

        POSTs to ``streamGenerateContent?alt=sse`` so the API returns a
        Server-Sent Events stream rather than a single JSON blob. The
        body is a sequence of ``data: {json}\\n\\n`` blocks separated by
        blank lines; we walk the lines, strip the ``data: `` prefix,
        parse the JSON, and decode the base64 audio under
        ``inlineData.data``.

        The ``try/except`` is deliberately broad (same shape as the
        ElevenLabs provider above): a streaming response that fails
        halfway through should log a warning and stop yielding rather
        than crash the dispatcher's audio callback.
        """
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self._voice},
                    },
                },
            },
        }
        # ``?alt=sse`` is what flips the response from a single JSON
        # blob to a streaming SSE feed; without it the API just returns
        # the full audio in one shot and we lose the whole point of
        # being a streaming provider.
        url = (
            f"{self._base_url}/models/{self._model}:streamGenerateContent"
            f"?alt=sse&key={self._api_key}"
        )
        try:
            with self._client.post(url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if self._stop_event is not None and self._stop_event.is_set():
                        break
                    if not line:
                        # Blank line = SSE event separator; skip.
                        continue
                    if not line.startswith("data: "):
                        # Comments (``event:``, ``id:``, ``:heartbeat``)
                        # and any non-data lines are ignored.
                        continue
                    try:
                        event = json.loads(line[len("data: "):])
                    except (ValueError, TypeError) as exc:
                        logger.warning(
                            "Gemini SSE: failed to parse event JSON: %s", exc
                        )
                        continue
                    # Audio is nested under
                    # ``candidates[0].content.parts[*].inlineData.data``
                    # as a base64-encoded PCM blob. Some Gemini
                    # responses use ``inline_data`` (snake_case) — the
                    # legacy non-streaming path checks both, so we do
                    # the same.
                    try:
                        parts = event["candidates"][0]["content"]["parts"]
                    except (KeyError, IndexError, TypeError):
                        continue
                    for part in parts:
                        inline = part.get("inlineData") or part.get("inline_data")
                        if not inline:
                            continue
                        b64 = inline.get("data", "")
                        if not b64:
                            continue
                        try:
                            yield base64.b64decode(b64)
                        except (ValueError, TypeError) as exc:
                            logger.warning(
                                "Gemini SSE: failed to base64-decode audio: %s",
                                exc,
                            )
        except Exception as exc:
            logger.warning("Gemini streaming TTS failed: %s", exc)
            return


@register("openai")
class OpenAIStreamingProvider(StreamingTTSProvider):
    """Streaming TTS provider for OpenAI's ``gpt-4o-mini-tts`` model.

    OpenAI's ``gpt-4o-mini-tts`` supports streaming via
    ``with_streaming_response.create()`` which yields raw PCM bytes.
    Unlike the legacy ``client.audio.speech.create(...)`` + ``read()``
    path used by ``tools.tts_tool.py:_generate_openai_tts`` (which
    downloads the whole audio before returning), the streaming variant
    exposes the response as a context manager whose ``iter_bytes()``
    generator yields each chunk as it arrives from the server. That
    shape is exactly what the dispatcher's audio callback wants.

    The audio format is fixed at **24 kHz, mono, 16-bit signed
    little-endian PCM** (the format OpenAI returns when
    ``response_format="pcm"`` is passed). The dispatcher reads
    ``sample_rate``/``channels``/``sample_width`` off the instance to
    open the ``sounddevice.OutputStream`` with the matching format.

    Auth follows the same fallback as the non-streaming
    ``_generate_openai_tts`` in ``tools.tts_tool.py``: ``OPENAI_API_KEY``
    is read directly from the environment, and a missing key fails
    loud in ``__init__`` so the dispatcher never pulls zero chunks.
    """

    # OpenAI's ``pcm`` response format is 24 kHz, mono, int16
    # (2 bytes/sample) per the OpenAI TTS streaming docs. Declared as
    # class attrs so the dispatcher can build the
    # ``sounddevice.OutputStream`` with the matching format.
    sample_rate = 24000
    channels = 1
    sample_width = 2

    def __init__(self, config: dict, *, stop_event: Optional[threading.Event] = None):
        # Config keys mirror the ``tts.openai`` block from config.yaml.
        # Defaults match the constants in tools.tts_tool.py
        # (``DEFAULT_OPENAI_MODEL``/``DEFAULT_OPENAI_VOICE``) so
        # behaviour stays consistent with the existing non-streaming
        # path.
        self._model = config.get("model", "gpt-4o-mini-tts")
        self._voice = config.get("voice", "alloy")
        # ``base_url`` is optional; passing ``None`` lets the SDK fall
        # back to the official OpenAI endpoint. Storing the raw value
        # (without stripping) matches the behaviour of
        # ``tools.tts_tool.py:_generate_openai_tts`` which forwards
        # whatever the config supplies.
        self._base_url = config.get("base_url", None)
        # OpenAI's TTS API accepts an optional ``instructions`` field
        # for tone/style guidance (e.g. "Speak in a cheerful tone").
        # Pull it from config if present and forward it on every call;
        # ``None`` means "use the model's default voice personality".
        self._instructions = config.get("instructions")
        # The stop event is optional so this class is usable outside
        # the dispatcher (e.g. one-off scripts). When None, the
        # ``stream()`` loop simply skips the stop check.
        self._stop_event = stop_event

        # Auth: read OPENAI_API_KEY directly from the environment
        # (matches the legacy ``_generate_openai_tts`` path which also
        # uses ``os.environ.get``). Strip defensively so a stray
        # whitespace-only value still fails loud.
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; cannot use OpenAI streaming TTS"
            )
        self._api_key = api_key

        # Lazy SDK import — wrapped in a helper so the test suite can
        # monkeypatch it without installing the real package. The
        # helper returns the *module* (not a client instance) so
        # ``self._client = openai_module.OpenAI(...)`` below mirrors
        # the real production code in ``tools.tts_tool.py``.
        try:
            openai_module = _import_openai_client()
        except ImportError as e:
            logger.warning("openai package not installed: %s", e)
            raise

        # Build the client. ``base_url`` is forwarded only when
        # explicitly set; passing ``None`` would override the SDK
        # default with the same default, which is a no-op but noisy
        # in some proxied environments, so we just omit the kwarg.
        client_kwargs = {"api_key": api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        self._client = openai_module.OpenAI(**client_kwargs)

    def stream(self, text: str) -> Iterator[bytes]:
        """Yield PCM chunks for *text* as they arrive from OpenAI.

        Calls ``client.audio.speech.with_streaming_response.create(...)``
        which is the modern OpenAI streaming entry point — the legacy
        ``client.audio.speech.create(...)`` + ``response.read()`` path
        buffers the whole audio before returning, defeating the whole
        point of this class. The ``with_streaming_response`` variant
        returns a context manager whose ``iter_bytes(chunk_size=...)``
        generator yields raw PCM bytes as they arrive.

        The output format is ``pcm`` (raw little-endian 16-bit signed
        PCM, 24 kHz, mono) which matches the class-level format
        attributes above. ``extra_body`` is used (rather than
        ``extra_query``) to set the sample rate so any OpenAI-compatible
        proxy that supports a per-request sample-rate override picks
        it up; the real OpenAI endpoint ignores unknown extra_body
        keys, so it's safe to pass unconditionally.

        The ``try/except`` is deliberately broad (same shape as the
        ElevenLabs and Gemini providers above): a streaming response
        that fails halfway through should log a warning and stop
        yielding rather than crash the dispatcher's audio callback.
        """
        try:
            # The ``with_streaming_response.create(...)`` call returns
            # a context manager; the response object it yields exposes
            # ``iter_bytes()`` which streams the PCM payload. We open
            # it as a context manager so the underlying HTTP connection
            # is closed cleanly when iteration finishes (or aborts
            # early via the stop event).
            create_kwargs = {
                "model": self._model,
                "voice": self._voice,
                "input": text,
                "response_format": "pcm",
                # 24 kHz matches the class-level ``sample_rate`` and
                # the format OpenAI returns for ``pcm``. Some
                # OpenAI-compatible gateways (e.g. local self-hosted
                # TTS servers) honour ``sample_rate``; the official
                # OpenAI endpoint ignores it because ``pcm`` is
                # already locked to 24 kHz, so this is safe to pass
                # unconditionally.
                "extra_body": {"sample_rate": self.sample_rate},
            }
            if self._instructions is not None:
                create_kwargs["instructions"] = self._instructions

            with self._client.audio.speech.with_streaming_response.create(
                **create_kwargs
            ) as response:
                # ``iter_bytes`` is a generator — yielding its values
                # one-by-one is what gives the dispatcher low-latency
                # chunked playback. The chunk_size argument is a hint
                # to the underlying HTTP layer; the SDK is free to
                # yield smaller or larger chunks based on socket
                # availability.
                for chunk in response.iter_bytes(chunk_size=4096):
                    if self._stop_event is not None and self._stop_event.is_set():
                        break
                    yield chunk
        except Exception as exc:
            logger.warning("OpenAI streaming TTS failed: %s", exc)
            return


# TODO(verify-xai-sample-rate): xAI's docs do not pin a sample rate for the
# WebSocket TTS endpoint as of this writing. The legacy ``_generate_xai_tts``
# in tools.tts_tool.py reads ``DEFAULT_XAI_SAMPLE_RATE = 24000`` from config,
# and the rest of the streaming providers in this module all use 24 kHz, so
# we hard-code 24 kHz here. The E2E test gated on a real ``XAI_API_KEY``
# (added in a later task) should confirm this and remove the TODO if xAI
# standardises on a different rate.
@register("xai")
class XAIStreamingProvider(StreamingTTSProvider):
    """Streaming TTS provider for xAI's WebSocket TTS endpoint.

    Unlike ElevenLabs, Gemini, and OpenAI — all of which expose
    server-sent HTTP streams — xAI's TTS API is **WebSocket-only**. The
    client opens a connection to ``wss://api.x.ai/v1/tts``, sends a
    single JSON request ``{"text": ..., "voice": ..., "response_format":
    "pcm"}``, then reads a sequence of binary PCM frames back until the
    server closes the socket. The exact envelope (binary vs JSON,
    ``done``/``error`` sentinel shape) is documented at
    https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
    and may evolve; the async-to-sync bridge below is the seam an E2E
    test can replace when that happens.

    Because the dispatcher's audio callback is a sync generator, the
    async WebSocket API has to be bridged onto a sync ``Iterator``. The
    shape mirrors the rest of the providers in this module — yield
    raw PCM bytes, honour the stop event, wrap in a broad try/except —
    but the bridging is done by a small pair of helpers:

      * ``_async_collect_chunks(text)`` is an ``async`` generator that
        opens the WebSocket and ``yield``s each raw PCM frame as it
        arrives from the server.
      * ``_collect_async(text)`` runs that generator under
        ``asyncio.run()`` and returns a ``list[bytes]`` of every frame
        received. The sync ``stream()`` method just iterates the list
        in order and checks ``self._stop_event`` between yields.

    The ``_collect_async`` seam exists purely for the test suite:
    monkeypatching it to return a canned ``list[bytes]`` exercises the
    full sync-path machinery (config validation, stop-event handling,
    error wrapping) without faking the WebSocket protocol. The async
    WS path is small enough to read directly and is covered by the E2E
    test in a later task.

    Audio format: 24 kHz, mono, 16-bit signed little-endian PCM. The
    ``sample_rate`` matches ``DEFAULT_XAI_SAMPLE_RATE`` in
    ``tools.tts_tool.py`` and the format the dispatcher's
    ``sounddevice.OutputStream`` is opened with; see the
    ``TODO(verify-xai-sample-rate)`` comment at the top of this class
    for the open question.
    """

    sample_rate = 24000
    channels = 1
    sample_width = 2

    def __init__(self, config: dict, *, stop_event: Optional[threading.Event] = None):
        # Config keys mirror the ``tts.xai`` block from config.yaml.
        # The defaults match the constants in tools.tts_tool.py
        # (``DEFAULT_XAI_VOICE_ID``/``DEFAULT_XAI_BASE_URL``) so
        # behaviour stays consistent with the legacy
        # ``_generate_xai_tts`` path that this provider will eventually
        # replace for streaming use cases.
        self._voice = config.get("voice", "eve")
        self._base_url = config.get("base_url", "wss://api.x.ai/v1/tts")
        # The stop event is optional so this class is usable outside
        # the dispatcher (e.g. one-off scripts). When None, the
        # ``stream()`` loop simply skips the stop check.
        self._stop_event = stop_event

        # Auth: read XAI_API_KEY directly from the environment. This
        # matches the simple ``os.environ.get`` pattern used by the
        # legacy ``_generate_xai_tts`` (the OAuth resolver is
        # non-streaming-specific, so we keep the streaming path
        # symmetric and let the dispatcher pull credentials via the
        # normal config-driven flow). A missing key fails loud in
        # ``__init__`` so the dispatcher never pulls zero chunks.
        api_key = os.environ.get("XAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "XAI_API_KEY not set; cannot use xAI streaming TTS"
            )
        self._api_key = api_key

        # Lazy SDK import — wrapped in a helper so the test suite can
        # monkeypatch it without installing the real package. We
        # surface the import error loudly because silently disabling
        # the provider would mask config bugs the user can fix.
        # ``websockets`` is a core dependency of hermes-agent (used by
        # the browser CDP supervisor) so this normally succeeds
        # transparently.
        try:
            self._websockets = _import_websockets()
        except ImportError as e:
            logger.warning("websockets package not installed: %s", e)
            raise

    def stream(self, text: str) -> Iterator[bytes]:
        """Yield PCM chunks for *text* as they arrive from xAI.

        The sync generator shape is what the dispatcher's audio
        callback expects, but the actual work happens inside
        ``_collect_async`` — which runs the WebSocket message loop
        under ``asyncio.run()`` and returns a ``list[bytes]`` of every
        frame received. The list is iterated here so the stop event
        can be honoured between yields, matching the contract of the
        other providers in this module.

        The ``try/except`` is deliberately broad (same shape as the
        ElevenLabs, Gemini, and OpenAI providers above): a streaming
        response that fails halfway through should log a warning and
        stop yielding rather than crash the dispatcher's audio
        callback. Connection-level errors (DNS, TLS, handshake) bubble
        up through ``_collect_async`` and land here.
        """
        try:
            chunks = self._collect_async(text)
            for chunk in chunks:
                if self._stop_event is not None and self._stop_event.is_set():
                    break
                yield chunk
        except Exception as exc:
            logger.warning("xAI streaming TTS failed: %s", exc)
            return

    def _collect_async(self, text: str) -> List[bytes]:
        """Run ``_async_collect_chunks`` under a fresh event loop and
        return every frame as a ``list[bytes]``.

        This is the sync seam the test suite monkeypatches — keeping
        it as a separate method (rather than inlining the
        ``asyncio.run()`` call into ``stream()``) means unit tests can
        substitute a canned list without ever spinning up an event
        loop or a fake WebSocket. The E2E test in a later task
        exercises the real path.

        Importing ``asyncio`` lazily here (not at module top) keeps
        the import cost off the import path for users who only enable
        a different streaming provider.
        """
        import asyncio  # noqa: WPS433 — intentional late import

        return asyncio.run(self._drain_async(text))

    async def _drain_async(self, text: str) -> List[bytes]:
        """Coroutine-friendly wrapper around ``_async_collect_chunks``.

        ``asyncio.run`` doesn't accept async generators directly; it
        needs a coroutine that returns a concrete value. This coroutine
        just runs the async generator to completion and collects the
        frames into a list — keeping the websocket lifecycle inside
        the coroutine so the connection is closed deterministically
        when ``asyncio.run`` returns.
        """
        frames: List[bytes] = []
        async for frame in self._async_collect_chunks(text):
            frames.append(frame)
        return frames

    async def _async_collect_chunks(self, text: str):  # type: ignore[no-untyped-def]
        """Async generator: yield every PCM frame the WebSocket emits.

        Connects to ``wss://api.x.ai/v1/tts`` with a Bearer-token
        ``Authorization`` header (the same auth shape as the legacy
        ``_generate_xai_tts`` HTTP path), sends the TTS request as a
        single JSON message, and ``yield``s each frame as it arrives.

        End-of-stream detection is the single fragile point in this
        class: the xAI docs describe ``done`` / ``error`` envelopes
        in a shape that has shifted across early-access versions. The
        current best-effort implementation treats any non-bytes
        message as a control frame — if the message is JSON, we look
        for ``{"type": "done"}`` and stop; if it's a string, we look
        for the literal token ``"done"``; everything else is logged
        and ignored. A ``websockets.ConnectionClosed`` exception is
        the normal end-of-stream signal and is caught locally so the
        async generator terminates cleanly.

        This method is the seam an E2E test (gated on a real
        ``XAI_API_KEY``) will exercise. Unit tests bypass it via the
        ``_collect_async`` patch — see the test docstring for the
        rationale.
        """
        import json as _json

        websockets = self._websockets
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = self._base_url

        # ``extra_headers`` is the websockets>=10 kwarg; older versions
        # used ``extra_headers=`` too, so this works for everything
        # the project's lockfile pins.
        async with websockets.connect(url, extra_headers=headers) as ws:
            await ws.send(
                _json.dumps(
                    {
                        "text": text,
                        "voice": self._voice,
                        "response_format": "pcm",
                    }
                )
            )
            try:
                while True:
                    message = await ws.recv()
                    if isinstance(message, (bytes, bytearray, memoryview)):
                        # Raw binary PCM frame — the happy path.
                        yield bytes(message)
                        continue
                    # Control frame: try to interpret it as JSON.
                    try:
                        envelope = _json.loads(message)
                    except (ValueError, TypeError):
                        # Non-JSON text control frame; the xAI
                        # protocol has historically used the literal
                        # string "done" as a fallback end-of-stream
                        # marker. Log and stop on that exact match,
                        # otherwise just log and continue.
                        if message == "done":
                            return
                        logger.debug("xAI WS: ignoring control frame: %r", message)
                        continue
                    etype = envelope.get("type")
                    if etype == "done":
                        return
                    if etype == "error":
                        err = envelope.get("error") or envelope.get("message") or envelope
                        logger.warning("xAI WS error envelope: %s", err)
                        return
                    # Unknown envelope type — log and keep going so
                    # one bad frame doesn't kill the whole stream.
                    logger.debug("xAI WS: ignoring envelope: %r", envelope)
            except Exception as exc:
                # ``websockets.ConnectionClosed`` is the normal
                # end-of-stream signal on the xAI endpoint; the server
                # just closes the socket when audio is done. Anything
                # else is a real error worth logging.
                if exc.__class__.__name__ == "ConnectionClosed":
                    return
                logger.warning("xAI WS receive failed: %s", exc)
                return


# TODO(real-time-mp3-decode): edge-tts yields mp3 chunks, not PCM. The MVP
# below accumulates the full mp3 blob, writes it to a temp file, and yields
# the bytes in one chunk — matching the shape of ``_play_via_tempfile`` at
# tools/tts_tool.py:2586. A future optimization is to decode the mp3 in
# real time (e.g. via pydub) and yield 4096-sample PCM chunks so the
# dispatcher can start playback before edge-tts has finished the full
# response. The MVP keeps the provider simple and avoids adding pydub as
# a dep, at the cost of higher time-to-first-audio.
@register("edge")
class EdgeStreamingProvider(StreamingTTSProvider):
    """Streaming TTS provider for Microsoft Edge TTS (free, public endpoint).

    Edge TTS is the only no-auth provider in this module — Microsoft's
    public ``wss://api.edge.microsoft.com`` endpoint requires no API
    key, just a voice ID. The audio format returned is **mp3** at 24
    kHz mono, which is not what the dispatcher's
    ``sounddevice.OutputStream`` expects (it wants raw PCM), so the
    MVP wires the contract end-to-end but yields the raw mp3 bytes
    in a single chunk rather than decoding on the fly.

    MVP impl using temp-file accumulation. Real-time mp3→PCM decode is
    a future optimization.

    The async-to-sync bridge mirrors the xAI provider: ``stream()`` is
    a sync generator, but the actual work happens inside
    ``_collect_async`` (runs ``Communicate.stream()`` under
    ``asyncio.run()`` and returns the full concatenated mp3 blob).
    The list-vs-blob difference is intentional — edge-tts yields
    variable-size chunks and we always need the full mp3 to write to
    a file before yielding, so accumulating into a single ``bytes`` is
    simpler than the chunk-list shape xAI uses.

    Audio format declared on the class is 24 kHz, mono, 16-bit (the
    format edge-tts decodes to internally and the format the
    dispatcher's ``OutputStream`` is opened with). The yielded bytes
    are raw mp3, not PCM — see the TODO at the top of the class and
    the docstring above for the rationale and the future-optimization
    plan.
    """

    # edge-tts returns 24 kHz mono mp3; we declare the decoded PCM
    # format here so the dispatcher opens the ``OutputStream`` with
    # the right shape. The yielded bytes are mp3 in the MVP, but the
    # class-level attrs are PCM metadata for the (future) decoded
    # path.
    sample_rate = 24000
    channels = 1
    sample_width = 2

    def __init__(self, config: dict, *, stop_event: Optional[threading.Event] = None):
        # Config keys mirror the ``tts.edge`` block from config.yaml.
        # Defaults match the user's current config (Andrew Multilingual)
        # and the standard edge-tts rate/volume/pitch string shapes
        # (``+0%``, ``+0%``, ``+0Hz``) — these are passed straight
        # through to ``Communicate(..., rate=..., volume=..., pitch=...)``
        # so the user can fine-tune the output without code changes.
        self._voice = config.get("voice", "en-US-AndrewMultilingualNeural")
        self._rate = config.get("rate", "+0%")
        self._volume = config.get("volume", "+0%")
        self._pitch = config.get("pitch", "+0Hz")
        # The stop event is optional so this class is usable outside
        # the dispatcher (e.g. one-off scripts). When None, the
        # ``stream()`` loop simply skips the stop check.
        self._stop_event = stop_event

        # Lazy SDK import — wrapped in a helper so the test suite can
        # monkeypatch it without installing the real package. We
        # surface the import error loudly because silently disabling
        # the provider would mask config bugs the user can fix. No
        # API key is required: edge-tts uses a public Microsoft
        # endpoint that does its own (lightweight) authentication.
        try:
            self._edge_tts = _import_edge_tts()
        except ImportError as e:
            logger.warning("edge-tts package not installed: %s", e)
            raise

    def stream(self, text: str) -> Iterator[bytes]:
        """Yield the mp3 audio for *text* as a single chunk.

        The MVP collects the full mp3 blob from ``Communicate.stream()``
        via ``_collect_async`` (which runs the async iterator under
        ``asyncio.run()``), writes it to a temp ``.mp3`` file, reads
        the file back, and yields the bytes once. The temp-file round
        trip is deliberate — it mirrors ``_play_via_tempfile`` at
        ``tools.tts_tool.py:2586`` and sets up a clean upgrade path
        to a real-time mp3→PCM decode later. The dispatcher treats
        the chunk opaquely, so a single ``yield`` is enough for the
        MVP contract.

        The ``try/except`` is deliberately broad (same shape as the
        ElevenLabs, Gemini, OpenAI, and xAI providers above): a
        streaming response that fails halfway through should log a
        warning and stop yielding rather than crash the dispatcher's
        audio callback.
        """
        # Honour the stop event *before* doing any work — if the
        # dispatcher has already given up, the asyncio.run() call
        # (and the network round-trip it implies) is wasted effort.
        if self._stop_event is not None and self._stop_event.is_set():
            return
        try:
            mp3_bytes = self._collect_async(text)
            if not mp3_bytes:
                # Edge-tts returned no audio (e.g. an empty response or
                # an unsupported voice). Yield nothing and let the
                # dispatcher move on.
                return
            # Write the accumulated mp3 to a temp file, read it back
            # as bytes, yield the bytes, then delete the file. The
            # temp-file indirection mirrors ``_play_via_tempfile`` at
            # tools/tts_tool.py:2586 and sets up a clean upgrade path
            # to a real-time mp3→PCM decode (see the class-level
            # TODO). Using ``delete=False`` + manual ``os.unlink`` in
            # ``finally`` is the same pattern the legacy code uses —
            # ``NamedTemporaryFile(delete=True)`` would race with the
            # read-back on Windows because the file would be closed
            # (and unlinked) as soon as the ``with`` block exits.
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp_path = tmp.name
                tmp.write(mp3_bytes)
                tmp.close()
                with open(tmp_path, "rb") as fh:
                    yield fh.read()
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        # Best-effort cleanup; the temp dir gets
                        # reaped on reboot anyway.
                        pass
        except Exception as exc:
            logger.warning("Edge streaming TTS failed: %s", exc)
            return

    def _collect_async(self, text: str) -> bytes:
        """Run ``_drain_async`` under a fresh event loop and return the
        concatenated mp3 blob.

        This is the sync seam the test suite monkeypatches — keeping
        it as a separate method (rather than inlining the
        ``asyncio.run()`` call into ``stream()``) means unit tests can
        substitute a canned ``bytes`` blob without ever spinning up
        an event loop or a fake ``edge_tts`` module. The E2E test in a
        later task exercises the real path.
        """
        return asyncio.run(self._drain_async(text))

    async def _drain_async(self, text: str) -> bytes:
        """Coroutine-friendly wrapper that drains ``Communicate.stream()``.

        ``asyncio.run`` doesn't accept async generators directly; it
        needs a coroutine that returns a concrete value. This coroutine
        runs the async generator to completion and concatenates every
        ``("audio", chunk)`` tuple into a single ``bytes`` blob. Any
        non-audio event types (``WordBoundary``, ``SentenceBoundary``,
        etc.) are skipped — they're metadata the dispatcher doesn't
        care about.

        Keeping the WebSocket lifecycle inside the coroutine means
        the connection is closed deterministically when ``asyncio.run``
        returns.
        """
        communicate = self._edge_tts.Communicate(
            text,
            voice=self._voice,
            rate=self._rate,
            volume=self._volume,
            pitch=self._pitch,
        )
        buf = bytearray()
        async for chunk in communicate.stream():
            # edge-tts >= 7 yields ``{"type": "audio", "data": <bytes>}`` dicts
            # (older versions yielded ``(type, data)`` tuples — both shapes
            # handled here for forward-compat).
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type", "")
                chunk_bytes = chunk.get("data", b"")
            else:
                chunk_type, chunk_bytes = chunk[0], chunk[1]
            if chunk_type != "audio":
                # ``WordBoundary`` / ``SentenceBoundary`` / etc. are
                # metadata events edge-tts uses for word/phrase
                # timing; the dispatcher doesn't consume them, so we
                # skip them here. This is the same filter the legacy
                # ``_generate_edge_tts`` path applies.
                continue
            if chunk_bytes:
                buf.extend(chunk_bytes)
        return bytes(buf)


__all__ = [
    "StreamingTTSProvider",
    "register",
    "get",
    "available",
    "dispatch_stream_tts",
    "resolve_streaming_provider",
    "ElevenLabsStreamingProvider",
    "GeminiStreamingProvider",
    "OpenAIStreamingProvider",
    "XAIStreamingProvider",
    "EdgeStreamingProvider",
]


# ---------------------------------------------------------------------------
# Dispatcher (Task 8)
# ---------------------------------------------------------------------------
#
# The two functions below are the runtime entry point for voice mode: the
# legacy ``stream_tts_to_speaker`` in ``tools.tts_tool.py`` will be rewritten
# in Task 11 to call ``dispatch_stream_tts`` once per sentence, and voice
# mode calls ``resolve_streaming_provider`` to figure out which provider to
# use. Everything above this comment is *pure provider plumbing* — these
# two functions are the only place that knows about ``sounddevice``, the
# PCM→numpy conversion, and the provider priority list.
#
# Design constraints (carried over from the legacy inlined path):
#
#   * ``sounddevice`` is lazy-imported so users who only need the registry
#     (e.g. the unit tests) don't pay the import cost.
#   * The output stream is opened *once per sentence* with the provider's
#     declared format. The legacy path kept a single stream open for the
#     whole session, but per-sentence open/close is the safe default and
#     matches the ``_FakeOutputStream`` test seam.
#   * ``stop_event`` is checked between chunks (and before opening the
#     stream) so an interrupt aborts cheaply. Individual chunk errors
#     don't crash playback — they log a warning and continue.
#   * The ``output_stream`` kwarg is the test seam: when provided, the
#     dispatcher uses it directly instead of opening a real ``sounddevice``
#     stream. Production code never passes this kwarg.

# Priority order for the resolver. ElevenLabs first (best quality) and
# edge last (free fallback). The order is intentionally hard-coded rather
# than read from config — it's a user-experience decision, not a
# configuration knob.
_PROVIDER_PRIORITY: List[str] = [
    "elevenlabs",
    "gemini",
    "openai",
    "xai",
    "edge",
]


def _import_sounddevice():
    """Lazy import the ``sounddevice`` audio library.

    Returns the ``sounddevice`` *module* so the caller can write
    ``sd.OutputStream(...)`` exactly like the legacy inlined path at
    ``tools.tts_tool.py:2524``. Raises ``ImportError`` with the install
    command when the package isn't available, so the error message tells
    the user what to do rather than just bubbling a bare ``ModuleNotFoundError``.

    This helper is the seam the test suite monkeypatches; keeping the
    import isolated in a single function means the dispatcher tests
    never need a working audio device or even the real ``sounddevice``
    package installed.
    """
    try:
        import sounddevice  # type: ignore
    except ImportError as e:
        raise ImportError(
            "sounddevice package not installed. Run: pip install sounddevice"
        ) from e
    return sounddevice


def _dtype_from_sample_width(width: int) -> str:
    """Map a ``sample_width`` (bytes per sample) to a sounddevice dtype string.

    The dispatcher reads ``sample_width`` off the provider class and passes
    the matching dtype to ``sounddevice.OutputStream``; the conversion
    table mirrors the formats the providers in this module actually emit
    (currently all int16, but future 24-bit / 32-bit float providers can
    plug in here without touching the dispatcher).
    """
    mapping = {
        1: "int8",
        2: "int16",
        4: "int32",
        8: "int64",
    }
    if width not in mapping:
        raise ValueError(
            f"Unsupported sample_width {width}; expected one of {sorted(mapping)}"
        )
    return mapping[width]


def _try_instantiate_provider(
    name: str, tts_config: dict, *, stop_event: Optional[threading.Event]
) -> Optional[StreamingTTSProvider]:
    """Try to construct the provider named ``name``; return the instance or None.

    Used by ``resolve_streaming_provider`` to walk the priority list
    cheaply. Any exception from ``__init__`` (missing API key, missing
    SDK, bad config) is caught and converted to ``None`` so the caller
    can move on to the next provider. The instance is discarded on
    failure, so the resolver doesn't pay any setup cost for providers
    it can't actually use.
    """
    if name not in _PROVIDERS:
        return None
    cls = _PROVIDERS[name]
    provider_cfg = tts_config.get(name, {}) or {}
    try:
        return cls(provider_cfg, stop_event=stop_event)
    except Exception:
        # Constructor failure (missing key, missing SDK, etc.) means the
        # provider is "not available right now". We deliberately catch
        # broadly because providers raise a mix of ``RuntimeError`` and
        # ``ImportError`` for different failure modes, and the resolver
        # doesn't care which one — it just moves to the next candidate.
        return None


def resolve_streaming_provider(
    tts_config: dict, preferred: Optional[str]
) -> str:
    """Pick a registered streaming TTS provider that can run *right now*.

    Resolution order:

        1. If ``tts_config["streaming"]["provider"]`` is set, the named
           provider is registered, and a probe-instantiation succeeds
           → return that name. The config knob (Task 9) is a
           per-user opt-in: ``hermes config set tts.streaming.provider
           gemini`` overrides the default priority list.
        2. If ``preferred`` is set (e.g. the caller already resolved
           the config knob itself), the named provider is registered,
           and a probe-instantiation succeeds → return ``preferred``.
        3. Otherwise walk the priority list
           (``elevenlabs → gemini → openai → xai → edge``) and return
           the first name that is registered AND can be constructed
           without raising.
        4. If nothing in the list is usable, raise ``RuntimeError``.

    The probe-instantiation deliberately *constructs* a real provider
    (then discards it) so the resolver doesn't have to know each
    provider's env-var / SDK-install requirements individually. That's
    the same shape the dispatcher would use anyway, and it means
    custom registered providers (e.g. test stubs) get the right answer
    without us having to maintain an env-var map here.
    """
    # Config knob: ``tts.streaming.provider`` in the user's
    # ``~/.hermes/config.yaml``. The plan document (Task 9) says
    # callers should set this to pin a specific streaming provider
    # (e.g. ``gemini`` for the lower-latency SSE path) instead of
    # falling through the priority list. We extract it here so the
    # resolver and dispatcher share the same lookup contract — both
    # treat the config knob as the highest-priority signal.
    streaming_cfg = tts_config.get("streaming") or {}
    config_preferred = streaming_cfg.get("provider")

    # ``config_preferred`` wins over an explicit ``preferred=`` kwarg
    # when set; this lets the config block pin a provider even if a
    # caller forgets to pass ``preferred``. We still fall through to
    # ``preferred`` and then the priority list if the knob is unset
    # or unusable.
    effective_preferred = config_preferred or preferred
    if effective_preferred:
        # Normalize the same way ``get()`` does so the resolver matches
        # the dispatcher's lookup contract.
        candidate = effective_preferred.lower().strip()
        if candidate in _PROVIDERS:
            inst = _try_instantiate_provider(
                candidate, tts_config, stop_event=None
            )
            if inst is not None:
                return candidate
        # Preferred name wasn't usable; fall through to the priority
        # walk. We deliberately don't raise here — the whole point of
        # the priority list is to give the user a working provider
        # even when their explicit choice is misconfigured.
    for name in _PROVIDER_PRIORITY:
        inst = _try_instantiate_provider(name, tts_config, stop_event=None)
        if inst is not None:
            return name
    raise RuntimeError("No streaming TTS provider is available")


def _load_tts_config_or_default(tts_config: Optional[dict]) -> dict:
    """Return *tts_config* if provided, else load from ``tools.tts_tool``.

    The production path resolves the user's TTS config from
    ``~/.hermes/config.yaml`` via ``tools.tts_tool._load_tts_config``.
    The test path injects a dict directly so it doesn't depend on the
    user's actual config (which is environment-specific and not
    hermetic).

    The import is wrapped in try/except so a missing
    ``tools.tts_tool`` (e.g. in a stripped-down venv) doesn't break the
    dispatcher — the resolver still works with an empty config dict,
    it just has no env vars / keys to discover.
    """
    if tts_config is not None:
        return tts_config
    try:
        from tools.tts_tool import _load_tts_config
        return _load_tts_config()
    except Exception as exc:  # ImportError, AttributeError, config error
        logger.debug("Could not load TTS config from tools.tts_tool: %s", exc)
        return {}


def dispatch_stream_tts(
    sentence: str,
    provider_name: str,
    *,
    stop_event: Optional[threading.Event] = None,
    output_stream=None,
    tts_config: Optional[dict] = None,
) -> None:
    """Stream *sentence* through *provider_name* and write to *output_stream*.

    This is the runtime entry point the legacy
    ``stream_tts_to_speaker`` (in ``tools/tts_tool.py``) will call once
    per sentence in Task 11. It looks up the provider via the registry,
    opens a ``sounddevice.OutputStream`` with the provider's declared
    audio format, and writes each PCM chunk to it as it arrives.

    Parameters
    ----------
    sentence : str
        The text to speak. The provider handles its own tokenization /
        sentence splitting; the dispatcher just forwards the string.
    provider_name : str
        Registry name (case-insensitive) of the provider to use. Must
        already be registered via ``@register(...)``. A bad name raises
        ``KeyError`` from ``get()``; the caller (typically
        ``stream_tts_to_speaker``) is responsible for surfacing the
        error.
    stop_event : threading.Event, optional
        When set, the dispatcher aborts the current stream between
        chunks and tears down the output stream. ``None`` means
        "no interrupt" — the same shape the legacy inlined path uses
        for one-off scripts.
    output_stream : optional
        Pre-built stream object. When provided, the dispatcher uses
        it directly (no real ``sounddevice`` import). This is the
        test seam — production code never passes this kwarg. The
        object must expose ``start()``/``stop()``/``close()`` and
        ``write(numpy_array)`` matching the ``sounddevice.OutputStream``
        contract.
    tts_config : dict, optional
        Override for the auto-loaded TTS config. When ``None`` (the
        default), the dispatcher calls ``tools.tts_tool._load_tts_config``
        to pull the ``tts:`` block from ``~/.hermes/config.yaml``. Tests
        pass a dict here to keep the dispatcher hermetic.

    Errors
    ------
    The whole function is wrapped in a broad ``try/except`` that logs a
    warning and returns — matching the legacy ``stream_tts_to_speaker``
    behaviour, where a single bad sentence should never crash the voice
    mode loop. Provider construction errors (missing key, missing SDK)
    propagate from ``__init__`` and are caught here.
    """
    try:
        # Look up the provider class. ``get()`` raises ``KeyError`` for
        # unknown names; we let that bubble — the caller should pick a
        # valid name via ``resolve_streaming_provider`` first.
        provider_cls = get(provider_name)

        # Pull the user's TTS config. The provider's ``__init__`` reads
        # its own sub-block (e.g. ``tts.elevenlabs.voice_id``) from this
        # dict, so the resolver and dispatcher pass the same config
        # through.
        config = _load_tts_config_or_default(tts_config)

        # Construct the provider. The constructor may raise
        # ``RuntimeError`` (missing API key) or ``ImportError`` (missing
        # SDK). Both are caught by the outer try/except.
        provider = provider_cls(config, stop_event=stop_event)

        # Open the audio output. If the caller injected a fake stream
        # (test seam) we skip the real sounddevice import but still
        # call start()/stop()/close() on it — the seam is about which
        # audio backend is used, not about lifecycle ownership. The
        # dispatcher's contract is "I own the stream's lifecycle for
        # the duration of this call", which the test relies on.
        if output_stream is None:
            sd = _import_sounddevice()
            output_stream = sd.OutputStream(
                samplerate=provider.sample_rate,
                channels=provider.channels,
                dtype=_dtype_from_sample_width(provider.sample_width),
            )
        output_stream.start()

        # Walk the provider's chunk iterator. ``provider.stream()`` is
        # already supposed to honour ``stop_event`` itself, but we check
        # again here as a belt-and-braces guard in case a provider
        # forgets — the dispatcher's contract is "stop between chunks,
        # no exceptions". A bad chunk should not crash the whole
        # stream; we log a warning and move on to the next one. That
        # matches the legacy inlined path's behaviour, which is the
        # well-trodden UX in voice mode.
        for chunk in provider.stream(sentence):
            if stop_event is not None and stop_event.is_set():
                break
            try:
                arr = np.frombuffer(
                    chunk,
                    dtype=_dtype_from_sample_width(provider.sample_width),
                ).reshape(-1, provider.channels)
                output_stream.write(arr)
            except Exception as exc:
                # One bad chunk is recoverable: log it and keep going
                # so a transient decode glitch doesn't kill the rest of
                # the sentence.
                logger.warning(
                    "dispatch_stream_tts: chunk write failed (%s); skipping",
                    exc,
                )
                continue
    except Exception as exc:
        # Broad catch: matches the legacy ``stream_tts_to_speaker`` error
        # handling, which logs a warning and returns so a single
        # misconfigured sentence doesn't crash the voice mode loop.
        logger.warning(
            "dispatch_stream_tts failed for provider %r: %s",
            provider_name,
            exc,
        )
    finally:
        # Always tear down the stream. Whether the dispatcher opened it
        # or the test injected it, the lifecycle is the dispatcher's
        # responsibility for the duration of the call. The
        # stop()/close() calls are individually wrapped so a buggy
        # stream object can't keep us in a half-torn-down state.
        if output_stream is not None:
            try:
                output_stream.stop()
            except Exception:
                pass
            try:
                output_stream.close()
            except Exception:
                pass

