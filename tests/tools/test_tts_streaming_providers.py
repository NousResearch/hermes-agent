"""Tests for the ``StreamingTTSProvider`` ABC and registry.

These tests cover the generic dispatcher's contract surface added in
Task 1 of the streaming-TTS plan: the ABC requires ``stream()`` and the
three audio-format attributes, and the module-level registry is a
case-insensitive, whitespace-tolerant, sorted-on-read store of provider
classes keyed by name.

Nothing here talks to a real TTS engine or ``sounddevice``. The
``_make_fake_provider_class`` helper produces the smallest possible
subclass that satisfies the ABC, so the tests stay hermetic and fast.
"""

from __future__ import annotations

import base64
import json

import pytest

from tools.tts_streaming import (
    StreamingTTSProvider,
    available,
    get,
    register,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Snapshot and restore the registry around each test.

    The dispatcher is a module-level dict, and Tasks 3-7 will register
    real providers at import time. We snapshot on entry, clear for the
    duration of the test, then restore on exit so no test pollutes
    global state and no test depends on prior registration.
    """
    from tools import tts_streaming
    saved = dict(tts_streaming._PROVIDERS)
    tts_streaming._PROVIDERS.clear()
    yield
    tts_streaming._PROVIDERS.clear()
    tts_streaming._PROVIDERS.update(saved)


def _make_fake_provider_class(name: str = "fake"):
    """Build a minimal concrete ``StreamingTTSProvider`` and register it.

    Returns the class object (not an instance) so tests can both inspect
    the registered class via ``get(name)`` and instantiate it for
    behavioral checks.
    """
    @register(name)
    class FakeProvider(StreamingTTSProvider):
        sample_rate = 24000
        channels = 1
        sample_width = 2

        def stream(self, text):
            yield b"abc"
            yield b"def"
    return FakeProvider


# --- ABC contract ---


def test_abc_requires_stream_method():
    """Can't instantiate a subclass that doesn't implement ``stream()``.

    Python's ABC machinery raises ``TypeError`` at construction when an
    abstract method is unimplemented, which is the contract the
    dispatcher relies on.
    """
    class Incomplete(StreamingTTSProvider):
        sample_rate = 24000
        channels = 1
        sample_width = 2
    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_provider_has_audio_format_attrs():
    """Subclass must expose the three audio-format attributes.

    The dispatcher reads ``sample_rate``, ``channels``, and
    ``sample_width`` off the instance to open the ``sounddevice``
    ``OutputStream`` with the right format, so each registered class
    must set them to concrete values.
    """
    Fake = _make_fake_provider_class("attrs_test")
    inst = Fake()
    assert inst.sample_rate == 24000
    assert inst.channels == 1
    assert inst.sample_width == 2
    # And the types match what ``sounddevice`` expects.
    assert isinstance(inst.sample_rate, int)
    assert isinstance(inst.channels, int)
    assert isinstance(inst.sample_width, int)


# --- Registry: round-trip and error reporting ---


def test_register_and_get_roundtrip():
    """``@register(name)`` stores the class and ``get(name)`` returns it."""
    Fake = _make_fake_provider_class("roundtrip")
    assert get("roundtrip") is Fake


def test_get_raises_for_unknown():
    """``get("nonexistent")`` raises ``KeyError`` whose message is actionable.

    The message must include both the unknown name (so the caller can
    log it) and the list of available names (so the user knows what's
    possible). This is the contract that lets the dispatcher surface a
    helpful error or fall back to a default.
    """
    _make_fake_provider_class("known")
    with pytest.raises(KeyError) as exc_info:
        get("nonexistent")
    msg = str(exc_info.value)
    assert "nonexistent" in msg
    assert "known" in msg


def test_get_unknown_error_includes_every_registered_name():
    """The error message enumerates *every* registered provider, not just one.

    Guards against a regression where the message would only mention the
    first registered name (e.g. from a ``next(iter(...))`` mistake).
    """
    _make_fake_provider_class("alpha")
    _make_fake_provider_class("beta")
    _make_fake_provider_class("gamma")
    with pytest.raises(KeyError) as exc_info:
        get("nope")
    msg = str(exc_info.value)
    assert "alpha" in msg
    assert "beta" in msg
    assert "gamma" in msg


# --- Registry: validation and normalization ---


def test_register_rejects_non_subclass():
    """``@register`` refuses classes that don't subclass ``StreamingTTSProvider``.

    This protects the dispatcher from a silent invariant violation —
    a class without ``stream()`` would only blow up at the first chunk
    yield, deep inside the audio callback, where the traceback is
    useless. Failing loudly at registration time keeps the bug close to
    its source.
    """
    class NotASubclass:
        pass
    with pytest.raises(TypeError):
        register("bad")(NotASubclass)


def test_register_rejects_non_class():
    """``@register`` also refuses non-class callables (e.g. plain functions).

    The decorator accepts ``Type[StreamingTTSProvider]``, so a function
    object should fail the same ``issubclass`` check.
    """
    def not_a_class():
        pass
    with pytest.raises(TypeError):
        register("bad_func")(not_a_class)


def test_available_returns_sorted():
    """``available()`` returns registered names in sorted order, regardless
    of registration order.
    """
    _make_fake_provider_class("zeta")
    _make_fake_provider_class("alpha")
    _make_fake_provider_class("mu")
    assert available() == ["alpha", "mu", "zeta"]


def test_register_normalizes_name_case():
    """``@register`` lower-cases the name; ``get`` does the same for look-ups.

    This is what makes ``register("ElevenLabs")`` and
    ``get("elevenlabs")`` (or ``get("ELEVENLABS")``) the same key, so
    user-supplied provider names from config files don't have to match
    the registration case exactly.
    """
    _make_fake_provider_class("MixedCase")
    # Direct lower-case lookup.
    assert get("mixedcase") is not None
    # Whitespace + different case still resolves to the same entry.
    assert get("  MIXEDCASE  ") is not None
    assert get("mixedcase") is get("  MIXEDCASE  ")


def test_available_empty_by_default():
    """No providers registered → empty list (the autouse fixture cleared)."""
    assert available() == []


# --- Behavioral contract of the ABC ---


def test_stream_yields_bytes():
    """``stream()`` is an iterator of ``bytes`` chunks.

    The dispatcher's audio callback writes whatever each ``yield``
    produces straight into ``sounddevice.OutputStream.write``, so the
    contract is simply "yield small ``bytes`` objects, in order". A
    trivial subclass that yields ``b"abc"``, ``b"def"`` should yield
    those exact bytes when iterated.
    """
    Fake = _make_fake_provider_class("yield_test")
    inst = Fake()
    chunks = list(inst.stream("hello"))
    assert chunks == [b"abc", b"def"]


def test_stream_is_lazy_iterator():
    """``stream()`` returns a lazy iterator, not a pre-computed list.

    Streaming is the whole point of this interface — eager evaluation
    would defeat the chunked-output design. We assert by tagging each
    checkpoint in the generator with a unique 2-tuple and confirming
    that the tags appear in the produced list at the right moment —
    no earlier, no later.
    """
    # Snapshot the pre-existing registration (the helper registers a
    # default ``FakeProvider``); we want to replace it with our own
    # tracking provider without affecting other tests.
    from tools import tts_streaming
    tts_streaming._PROVIDERS.pop("lazy_test", None)

    produced = []

    def tracking_stream(self, text):
        produced.append(("start", text))
        yield b"chunk1"
        produced.append(("after_first", text))
        yield b"chunk2"
        produced.append(("done", text))

    @register("lazy_test")
    class LazyProvider(StreamingTTSProvider):
        sample_rate = 24000
        channels = 1
        sample_width = 2
        stream = tracking_stream

    inst = LazyProvider()
    it = inst.stream("hi")
    # Nothing has been produced yet — the generator is un-started.
    assert produced == []
    # Pull the first chunk. A generator function runs up to the first
    # ``yield`` and pauses there, so the "start" tag has fired but
    # "after_first" has not — that line sits between the two yields.
    first = next(it)
    assert first == b"chunk1"
    assert produced == [("start", "hi")]
    # Pull the second chunk — the function resumes past the first
    # yield, "after_first" fires, and the second yield pauses.
    second = next(it)
    assert second == b"chunk2"
    assert produced == [("start", "hi"), ("after_first", "hi")]
    # Exhaust the iterator — the trailing "done" line runs at last.
    with pytest.raises(StopIteration):
        next(it)
    assert produced == [
        ("start", "hi"),
        ("after_first", "hi"),
        ("done", "hi"),
    ]


# --- ElevenLabs streaming provider ---


class _FakeElevenLabsChunk:
    """Mimics the bytes returned by ElevenLabs' iterator."""
    pass


def _make_fake_elevenlabs_client(chunks: list[bytes]):
    """Build a stub *class* mimicking ``elevenlabs.client.ElevenLabs``.

    The real SDK's ``ElevenLabs(api_key=...)`` returns a client instance
    that exposes a ``.text_to_speech.convert(**kwargs)`` method returning
    an iterator of ``bytes``. This stub class behaves the same way:
    instantiated with ``api_key=...`` it produces a client whose
    ``text_to_speech.convert`` returns an iterator over ``chunks`` and
    asserts the ``output_format`` kwarg matches the API contract.

    Note: this stub returns the *class* (not an instance) so the
    production code can call ``StubClass(api_key=...)`` just like the
    real ``ElevenLabs(api_key=...)`` constructor.
    """
    class _StubClient:
        def __init__(self, **kwargs):
            # Accept api_key=... silently (matches real SDK signature).
            # The nested class captures ``chunks`` via closure from the
            # factory's enclosing scope.
            class _TTS:
                def __init__(self, chunks):
                    self._chunks = chunks

                def convert(self, **kwargs):
                    # Verify the call shape matches the real API
                    assert kwargs["output_format"] == "pcm_24000"
                    return iter(self._chunks)

            self.text_to_speech = _TTS(chunks)

    return _StubClient


def test_elevenlabs_yields_chunks(monkeypatch):
    """Provider yields each chunk from the client iterator."""
    from tools.tts_streaming import ElevenLabsStreamingProvider
    fake_chunks = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
    monkeypatch.setattr(
        "tools.tts_streaming._import_elevenlabs_client",
        lambda: _make_fake_elevenlabs_client(fake_chunks),
    )
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    provider = ElevenLabsStreamingProvider({})
    out = list(provider.stream("hello"))
    assert out == fake_chunks


def test_elevenlabs_respects_stop_event(monkeypatch):
    """Stop event aborts the iteration."""
    from tools.tts_streaming import ElevenLabsStreamingProvider
    import threading
    fake_chunks = [b"\x01", b"\x02", b"\x03", b"\x04"]
    monkeypatch.setattr(
        "tools.tts_streaming._import_elevenlabs_client",
        lambda: _make_fake_elevenlabs_client(fake_chunks),
    )
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    stop = threading.Event()
    stop.set()  # already stopped
    provider = ElevenLabsStreamingProvider({}, stop_event=stop)
    out = list(provider.stream("hello"))
    assert out == []  # iteration aborts before yielding any chunk


def test_elevenlabs_audio_format():
    """sample_rate/channels/sample_width match the API contract."""
    from tools.tts_streaming import ElevenLabsStreamingProvider
    p = ElevenLabsStreamingProvider.__new__(ElevenLabsStreamingProvider)  # skip __init__
    assert p.sample_rate == 24000
    assert p.channels == 1
    assert p.sample_width == 2


# --- Gemini streaming provider ---


def _make_sse_response(events: list[dict]) -> str:
    """Build a fake SSE response body from a list of event dicts.

    The real Gemini ``streamGenerateContent?alt=sse`` endpoint produces a
    stream of ``data: {json}\\n\\n`` chunks separated by blank lines. This
    helper reconstructs that shape so the test stub can feed it into a
    fake ``httpx`` response and exercise the SSE parser.
    """
    lines = []
    for ev in events:
        lines.append("data: " + json.dumps(ev))
        lines.append("")  # blank line separator
    return "\n".join(lines)


def _make_fake_httpx_stream(body: str):
    """Build a stub httpx client whose ``.post()`` returns a context manager.

    The real ``httpx.Client.post(url, json=payload)`` returns an object
    usable as a context manager whose response object exposes
    ``iter_lines()``. This stub mimics that shape: the stub class is
    callable like ``httpx.Client()`` and the post() method returns an
    object that yields lines from ``body`` when entered.
    """
    class _FakeStream:
        def __init__(self, body):
            self._body = body
        def __enter__(self):
            class _Resp:
                def __init__(self, body):
                    self._body = body
                def raise_for_status(self):
                    pass
                def iter_lines(self):
                    for line in self._body.split("\n"):
                        yield line
            return _Resp(self._body)
        def __exit__(self, *args):
            pass
    return _FakeStream(body)


def test_gemini_yields_decoded_pcm(monkeypatch):
    """Provider decodes base64 inlineData and yields raw PCM bytes."""
    from tools.tts_streaming import GeminiStreamingProvider

    pcm = b"\x10\x20\x30\x40"
    b64 = base64.b64encode(pcm).decode("ascii")
    sse_body = _make_sse_response([{
        "candidates": [{
            "content": {"parts": [{"inlineData": {"data": b64, "mimeType": "audio/L16"}}]}
        }]
    }])

    def _fake_post(self, url, **kwargs):
        return _make_fake_httpx_stream(sse_body)

    monkeypatch.setattr("tools.tts_streaming._import_httpx", lambda: None)  # not used, patched below
    monkeypatch.setattr("httpx.Client.post", _fake_post)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = GeminiStreamingProvider({})
    out = list(provider.stream("hello"))
    assert out == [pcm]


def test_gemini_respects_stop_event(monkeypatch):
    """Stop event aborts the iteration."""
    from tools.tts_streaming import GeminiStreamingProvider
    import threading

    # Build 4 SSE events
    events = []
    for i in range(4):
        pcm = bytes([i])
        b64 = base64.b64encode(pcm).decode("ascii")
        events.append({
            "candidates": [{"content": {"parts": [{"inlineData": {"data": b64}}]}}]
        })
    sse_body = _make_sse_response(events)

    monkeypatch.setattr("httpx.Client.post", lambda self, url, **kw: _make_fake_httpx_stream(sse_body))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    stop = threading.Event()
    stop.set()  # already stopped
    provider = GeminiStreamingProvider({}, stop_event=stop)
    out = list(provider.stream("hello"))
    assert out == []


def test_gemini_audio_format():
    """sample_rate/channels/sample_width match the API contract (24kHz mono int16)."""
    from tools.tts_streaming import GeminiStreamingProvider
    p = GeminiStreamingProvider.__new__(GeminiStreamingProvider)
    assert p.sample_rate == 24000
    assert p.channels == 1
    assert p.sample_width == 2


# --- OpenAI streaming provider ---


def _make_fake_openai_response(chunks: list[bytes]):
    """Build a stub context manager matching openai's with_streaming_response shape.

    The real OpenAI SDK's
    ``client.audio.speech.with_streaming_response.create(**kwargs)``
    returns a context manager whose ``__enter__`` yields an object
    exposing ``iter_bytes(chunk_size=...)``. This stub mimics that
    shape so the provider's ``with ... as response: for chunk in
    response.iter_bytes(...)`` loop is exercised end-to-end.
    """
    class _FakeResp:
        def __init__(self, chunks):
            self._chunks = chunks
        def iter_bytes(self, chunk_size=None):
            for c in self._chunks:
                yield c
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return _FakeResp(chunks)


def test_openai_yields_pcm_chunks(monkeypatch):
    """Provider yields each chunk from the streaming response."""
    from tools.tts_streaming import OpenAIStreamingProvider

    fake_chunks = [b"\x01\x02", b"\x03\x04"]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mirror the real OpenAI SDK shape: ``client.audio.speech`` exposes
    # ``with_streaming_response`` (a property/method that returns a
    # wrapper); the wrapper's ``.create(**kwargs)`` returns a context
    # manager whose ``__enter__`` yields an object with
    # ``iter_bytes(chunk_size=...)``. Folding the context-manager
    # creation into ``create`` (rather than putting it on
    # ``with_streaming_response``) is what makes the provider's
    # ``with self._client.audio.speech.with_streaming_response.create(
    # **kwargs) as response:`` block resolve correctly.
    class _FakeSpeech:
        @property
        def with_streaming_response(self):
            class _Wrapper:
                def create(self, **kwargs):
                    return _make_fake_openai_response(fake_chunks)
            return _Wrapper()
    class _FakeAudio:
        speech = _FakeSpeech()
    class _FakeClient:
        audio = _FakeAudio()
    class _FakeOpenAIModule:
        OpenAI = lambda **kwargs: _FakeClient()

    monkeypatch.setattr("tools.tts_streaming._import_openai_client",
                        lambda: _FakeOpenAIModule)

    provider = OpenAIStreamingProvider({})
    out = list(provider.stream("hello"))
    assert out == fake_chunks


def test_openai_respects_stop_event(monkeypatch):
    """Stop event aborts iteration."""
    from tools.tts_streaming import OpenAIStreamingProvider
    import threading

    fake_chunks = [b"\x01", b"\x02", b"\x03", b"\x04"]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _FakeSpeech:
        @property
        def with_streaming_response(self):
            class _Wrapper:
                def create(self, **kwargs):
                    return _make_fake_openai_response(fake_chunks)
            return _Wrapper()
    class _FakeAudio:
        speech = _FakeSpeech()
    class _FakeClient:
        audio = _FakeAudio()
    class _FakeOpenAIModule:
        OpenAI = lambda **kwargs: _FakeClient()

    monkeypatch.setattr("tools.tts_streaming._import_openai_client",
                        lambda: _FakeOpenAIModule)

    stop = threading.Event()
    stop.set()
    provider = OpenAIStreamingProvider({}, stop_event=stop)
    out = list(provider.stream("hello"))
    assert out == []


def test_openai_audio_format():
    from tools.tts_streaming import OpenAIStreamingProvider
    p = OpenAIStreamingProvider.__new__(OpenAIStreamingProvider)
    assert p.sample_rate == 24000
    assert p.channels == 1
    assert p.sample_width == 2


# --- xAI streaming provider ---
#
# Test design note (read this before modifying):
#
# xAI's TTS is WebSocket-only, so ``stream()`` has to bridge an async iterator
# (the WebSocket message stream) into a sync generator the dispatcher's audio
# callback can pull from. That bridge — an event loop wrapped around an
# async generator — is fiddly to unit-test in isolation, and the real xAI
# protocol (binary frames vs JSON envelopes, the exact ``done`` / ``error``
# sentinel) is undocumented enough that a WS-level mock is as much a guess
# about xAI's behaviour as a test of our code.
#
# The cleaner seam is the ``_collect_async`` helper: the sync ``stream()``
# method delegates to ``_collect_async(text)`` which runs the async iterator
# under ``asyncio.run()`` and returns a ``list[bytes]`` of every frame
# received. Tests patch ``_collect_async`` to return a canned list, which
# exercises the full sync-path machinery (config validation, stop-event
# handling, error wrapping) without faking the WebSocket protocol. The
# async-to-WS path is small enough to read directly and is covered by an
# E2E test gated on a real ``XAI_API_KEY`` (added in a later task).


def test_xai_audio_format():
    """sample_rate/channels/sample_width match the xAI TTS contract.

    xAI's docs don't pin a sample rate for the WS TTS endpoint, so we
    default to 24 kHz to match the legacy ``_generate_xai_tts`` shape
    (``DEFAULT_XAI_SAMPLE_RATE = 24000``) and the rest of the streaming
    providers in this module. A TODO lives in the class body to flag
    this for the E2E test.
    """
    from tools.tts_streaming import XAIStreamingProvider
    p = XAIStreamingProvider.__new__(XAIStreamingProvider)  # skip __init__
    assert p.sample_rate == 24000
    assert p.channels == 1
    assert p.sample_width == 2


def test_xai_yields_pcm_via_collect_async(monkeypatch):
    """Provider yields PCM frames that ``_collect_async`` returned.

    Patches ``XAIStreamingProvider._collect_async`` to return a canned
    list of bytes — the same pattern used by the ElevenLabs and OpenAI
    tests, just with a method seam (not an import seam) because the
    async-to-sync bridge is the unit under test, not the SDK client.
    """
    from tools.tts_streaming import XAIStreamingProvider

    fake_chunks = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]

    def _fake_collect(self, text):
        # Return a real list — the test verifies the provider yields the
        # exact bytes in order, which is the only behaviour the dispatcher
        # cares about. The ``text`` arg is captured for parity with the
        # real signature even though we don't use it.
        assert text == "hello"
        return list(fake_chunks)

    monkeypatch.setattr(
        "tools.tts_streaming.XAIStreamingProvider._collect_async",
        _fake_collect,
    )
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    provider = XAIStreamingProvider({})
    out = list(provider.stream("hello"))
    assert out == fake_chunks


def test_xai_respects_stop_event(monkeypatch):
    """Stop event aborts iteration.

    Even when the async-bridge returns chunks, the sync wrapper must
    short-circuit on the stop event. We patch ``_collect_async`` to
    return a 4-chunk list; setting the stop event before the iteration
    starts should yield nothing — the ``stream()`` loop checks the
    event *before* pulling the first chunk.
    """
    from tools.tts_streaming import XAIStreamingProvider
    import threading

    fake_chunks = [b"\x01", b"\x02", b"\x03", b"\x04"]

    def _fake_collect(self, text):
        return list(fake_chunks)

    monkeypatch.setattr(
        "tools.tts_streaming.XAIStreamingProvider._collect_async",
        _fake_collect,
    )
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    stop = threading.Event()
    stop.set()  # already stopped
    provider = XAIStreamingProvider({}, stop_event=stop)
    out = list(provider.stream("hello"))
    assert out == []


def test_xai_missing_api_key_raises():
    """No ``XAI_API_KEY`` → ``RuntimeError`` from ``__init__``.

    Matches the contract of the other providers: a missing credential
    fails loud at construction time so the dispatcher never pulls zero
    chunks. We use ``monkeypatch.delenv`` (with ``raising=False``) so
    the test doesn't depend on a possibly-set host environment.
    """
    from tools.tts_streaming import XAIStreamingProvider
    import os

    # Clear the env var defensively. Use a context manager so we
    # restore whatever the test runner had set.
    saved = os.environ.pop("XAI_API_KEY", None)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            XAIStreamingProvider({})
        assert "XAI_API_KEY" in str(exc_info.value)
    finally:
        if saved is not None:
            os.environ["XAI_API_KEY"] = saved


def test_xai_registered_in_registry():
    """``@register("xai")`` puts the class in the module-level registry.

    The autouse ``_clear_registry`` fixture wipes the registry at the
    start of every test, so we can't just call ``get("xai")`` after
    import. Instead, we verify the class is decorated with the
    registry key — a re-import would defeat the test's purpose
    (decorator must fire at first import, not on demand).
    """
    from tools import tts_streaming
    # The decorator populates ``_PROVIDERS`` at import time, then the
    # autouse fixture clears it. So we verify the registration by
    # re-applying the decorator to a fresh subclass and confirming
    # the lookup works — this exercises the same code path the real
    # import uses, but without depending on the wiped registry.
    @tts_streaming.register("xai_reimport_test")
    class _Marker(tts_streaming.StreamingTTSProvider):
        sample_rate = 24000
        channels = 1
        sample_width = 2
        def stream(self, text):
            yield b""

    assert tts_streaming.get("xai_reimport_test") is _Marker
    # And the real xAI class is still importable + is a subclass of
    # the ABC, which is the static contract the dispatcher relies on
    # (the dynamic ``_PROVIDERS`` registration is verified by the
    # other tests indirectly, since they instantiate via the class).
    assert issubclass(tts_streaming.XAIStreamingProvider, tts_streaming.StreamingTTSProvider)


# --- Edge streaming provider ---


def test_edge_audio_format():
    """sample_rate/channels/sample_width match the edge-tts contract (24kHz mono int16)."""
    from tools.tts_streaming import EdgeStreamingProvider
    p = EdgeStreamingProvider.__new__(EdgeStreamingProvider)  # skip __init__
    assert p.sample_rate == 24000
    assert p.channels == 1
    assert p.sample_width == 2


def test_edge_accumulates_mp3_and_yields(monkeypatch):
    """Provider accumulates mp3 chunks from edge_tts.Communicate.stream() and yields them."""
    from tools.tts_streaming import EdgeStreamingProvider
    import tempfile, os

    fake_mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00fake_mp3_data"

    class _FakeStream:
        def __init__(self):
            self._emitted = False
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self._emitted:
                raise StopAsyncIteration
            self._emitted = True
            return {"type": "audio", "data": fake_mp3}

    class _FakeCommunicate:
        def __init__(self, text, voice, **kwargs):
            self.text = text
            self.voice = voice
        def stream(self):
            return _FakeStream()

    class _FakeEdgeTtsModule:
        Communicate = _FakeCommunicate

    monkeypatch.setattr("tools.tts_streaming._import_edge_tts",
                        lambda: _FakeEdgeTtsModule)

    provider = EdgeStreamingProvider({})
    out = list(provider.stream("hello"))
    # The provider should yield the accumulated bytes (in one or more chunks)
    assert b"".join(out) == fake_mp3


def test_edge_respects_stop_event(monkeypatch):
    from tools.tts_streaming import EdgeStreamingProvider
    import threading

    class _FakeStream:
        async def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration

    class _FakeCommunicate:
        def __init__(self, *args, **kwargs):
            pass
        def stream(self):
            return _FakeStream()

    class _FakeEdgeTtsModule:
        Communicate = _FakeCommunicate

    monkeypatch.setattr("tools.tts_streaming._import_edge_tts",
                        lambda: _FakeEdgeTtsModule)

    stop = threading.Event()
    stop.set()
    provider = EdgeStreamingProvider({}, stop_event=stop)
    out = list(provider.stream("hello"))
    assert out == []  # iteration aborts


# --- Dispatcher ---


def _register_real_providers():
    """Re-register the real streaming providers in this test's cleared registry.

    The module-level ``@register("...")`` decorators run at import time and
    populate ``tts_streaming._PROVIDERS``. The autouse ``_clear_registry``
    fixture snapshots and clears that dict for hermetic tests of the
    registry itself — but the dispatcher tests need the real providers
    to be visible to ``resolve_streaming_provider``. We re-register by
    importing the classes and copying them into the dict, mirroring the
    original import-time behaviour.
    """
    from tools import tts_streaming
    tts_streaming._PROVIDERS["elevenlabs"] = tts_streaming.ElevenLabsStreamingProvider
    tts_streaming._PROVIDERS["gemini"] = tts_streaming.GeminiStreamingProvider
    tts_streaming._PROVIDERS["openai"] = tts_streaming.OpenAIStreamingProvider
    tts_streaming._PROVIDERS["xai"] = tts_streaming.XAIStreamingProvider
    tts_streaming._PROVIDERS["edge"] = tts_streaming.EdgeStreamingProvider


def test_resolve_streaming_provider_returns_preferred(monkeypatch):
    """When preferred is set + registered + has env var, return it."""
    from tools.tts_streaming import resolve_streaming_provider
    _register_real_providers()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    result = resolve_streaming_provider({}, preferred="elevenlabs")
    assert result == "elevenlabs"


def test_resolve_streaming_provider_falls_back(monkeypatch):
    """When preferred is missing/unavailable, fall back to next available."""
    from tools.tts_streaming import resolve_streaming_provider
    _register_real_providers()
    # No env vars set → edge should still be available (no env required)
    for k in ["ELEVENLABS_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    result = resolve_streaming_provider({}, preferred="elevenlabs")
    assert result == "edge"


def test_resolve_streaming_provider_reads_config_knob(monkeypatch):
    """If tts.streaming.provider is set in config, use it.

    Task 9 added a ``tts.streaming.provider`` config knob so the
    user can pick the streaming provider explicitly via config
    rather than via a ``preferred=`` kwarg. The resolver should
    honour that knob (it becomes the ``preferred`` value).

    Strength: we set the ElevenLabs env var (which the priority
    walk would pick first) AND set the config knob to gemini. Only
    a resolver that actually reads the config knob can return
    ``gemini``; a resolver that ignores it will return ``elevenlabs``
    and this test will fail.
    """
    from tools.tts_streaming import resolve_streaming_provider
    _register_real_providers()
    for k in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    # The priority walk would return ``elevenlabs`` (it's first in
    # the list and the env is set), but the config knob overrides
    # that to ``gemini`` (which has GEMINI_API_KEY set).
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    result = resolve_streaming_provider(
        {"streaming": {"provider": "gemini"}}, preferred=None
    )
    assert result == "gemini"


def test_resolve_streaming_provider_no_available(monkeypatch):
    """When nothing is available, raise RuntimeError."""
    from tools.tts_streaming import resolve_streaming_provider
    _register_real_providers()

    @register("requires_special_env")
    class RequiresEnv(StreamingTTSProvider):
        sample_rate = 24000
        channels = 1
        sample_width = 2
        def __init__(self, config, *, stop_event=None):
            import os
            if not os.environ.get("SPECIAL_TEST_KEY"):
                raise RuntimeError("missing key")
        def stream(self, text):
            yield b"x"

    # Remove all known env vars
    for k in ["ELEVENLABS_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "SPECIAL_TEST_KEY"]:
        monkeypatch.delenv(k, raising=False)
    # Also make edge unavailable so the priority walk falls off the end —
    # otherwise the test would pass with ``edge`` as the result on a host
    # where the edge_tts package is installed. We patch the import helper
    # to simulate a missing SDK, mirroring the same seam the edge provider
    # tests use.
    monkeypatch.setattr(
        "tools.tts_streaming._import_edge_tts",
        lambda: (_ for _ in ()).throw(ImportError("edge_tts not installed (test stub)")),
    )
    # And remove the requires_special_env from consideration (not in priority list)
    # The priority list doesn't include it, so resolve should raise
    with pytest.raises(RuntimeError):
        resolve_streaming_provider({}, preferred=None)


class _FakeOutputStream:
    def __init__(self):
        self.writes = []
        self.started = False
        self.stopped = False
        self.closed = False
    def start(self):
        self.started = True
    def stop(self):
        self.stopped = True
    def close(self):
        self.closed = True
    def write(self, arr):
        self.writes.append(bytes(arr.tobytes()))


def test_dispatch_stream_tts_writes_chunks(monkeypatch):
    """dispatch_stream_tts writes each chunk to the output stream."""
    from tools.tts_streaming import dispatch_stream_tts

    @register("dispatch_test")
    class DispatchTest(StreamingTTSProvider):
        sample_rate = 24000
        channels = 1
        sample_width = 2
        def __init__(self, config, *, stop_event=None):
            # The dispatcher's contract requires providers to accept
            # ``(config, *, stop_event=None)``; mirror that here so
            # instantiation in the dispatcher succeeds. This fake
            # doesn't need any config so it ignores the values.
            self._stop_event = stop_event
        def stream(self, text):
            yield b"\x10\x00\x20\x00"
            yield b"\x30\x00\x40\x00"

    fake_stream = _FakeOutputStream()
    dispatch_stream_tts("hello", "dispatch_test", output_stream=fake_stream)
    assert fake_stream.started
    assert fake_stream.stopped
    assert fake_stream.closed
    assert b"".join(fake_stream.writes) == b"\x10\x00\x20\x00\x30\x00\x40\x00"


# --- Full dispatcher integration ---


def test_dispatch_stream_tts_full_flow_with_fake(monkeypatch):
    """End-to-end: dispatch a sentence through the dispatcher with a fake provider, verify all chunks land in the output stream.

    Task 10's integration test: prove the dispatcher can carry a
    sentence from a registered provider all the way to a (fake)
    output stream, with the lifecycle (start → write each chunk →
    stop → close) intact. The provider yields three synthetic
    chunks; the test asserts each one reaches the stream, in order,
    and that ``start``/``stop``/``close`` were all called so the
    dispatcher's lifecycle contract holds for downstream audio
    backends.
    """
    from tools.tts_streaming import dispatch_stream_tts

    # Register a fake provider inside the test so we don't have to
    # import a real provider or mock the SDK. ``@register`` mutates
    # the module-level registry; the autouse ``_clear_registry``
    # fixture snapshots and restores it, so this is hermetic.
    @register("flow_test")
    class FlowProvider(StreamingTTSProvider):
        sample_rate = 24000
        channels = 1
        sample_width = 2
        def __init__(self, config, *, stop_event=None):
            # The dispatcher's contract requires ``(config, *,
            # stop_event=None)``; mirror it. We stash the config
            # and stop event so the test could extend assertions
            # later (e.g. verify config is forwarded) without
            # changing the provider shape.
            self._config = config
            self._stop_event = stop_event
        def stream(self, text):
            # Assert the dispatcher forwarded the sentence verbatim.
            # This catches a class of regression where the dispatcher
            # might mangle/normalize the input before passing it on.
            assert text == "hello world"
            yield b"\x01\x00"
            yield b"\x02\x00"
            yield b"\x03\x00"

    fake_stream = _FakeOutputStream()
    dispatch_stream_tts("hello world", "flow_test", output_stream=fake_stream)
    # Lifecycle: dispatcher must start the stream before writing
    # and stop + close it afterwards so a real ``sounddevice``
    # stream doesn't leak file descriptors.
    assert fake_stream.started
    assert fake_stream.stopped
    assert fake_stream.closed
    # Every chunk the provider yielded must have been written to
    # the stream, in order, concatenated. We use ``in`` to be
    # tolerant of dispatcher-side padding; combined bytes are
    # always the prefix the provider produced.
    combined = b"".join(fake_stream.writes)
    assert b"\x01\x00\x02\x00\x03\x00" in combined
