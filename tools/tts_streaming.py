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

import base64
import json
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


__all__ = [
    "StreamingTTSProvider",
    "register",
    "get",
    "available",
    "ElevenLabsStreamingProvider",
    "GeminiStreamingProvider",
    "OpenAIStreamingProvider",
]
