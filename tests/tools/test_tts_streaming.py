import pytest
import tools.tts_streaming as ts


def test_openai_available_via_managed_gateway(monkeypatch):
    """The streamer must also be available when only the managed gateway is
    ready — no direct OPENAI_API_KEY or VOICE_TOOLS_OPENAI_KEY needed.
    """
    # No direct keys.
    monkeypatch.setattr(ts, "get_env_value", lambda k, *a: None)
    # But the managed gateway is ready.
    monkeypatch.setattr(
        "tools.tts_tool._has_openai_audio_backend",
        lambda: True,
    )
    assert ts.OpenAIStreamer.available() is True
    # And unavailable when neither path is open.
    monkeypatch.setattr(
        "tools.tts_tool._has_openai_audio_backend",
        lambda: False,
    )
    assert ts.OpenAIStreamer.available() is False



def test_openai_streamer_coerces_model_for_gateway(monkeypatch):
    """When going through the managed gateway, a non-supported model must be
    coerced to the default managed model (gpt-4o-mini-tts).
    """
    seen_models: list[str] = []

    class _FakeResp:
        @staticmethod
        def iter_bytes():
            yield b"\x00\x00" * 10

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeCreate:
        def __call__(self, **kw):
            seen_models.append(kw.get("model", ""))
            return _FakeResp()

    class _FakeWithStreamingResponse:
        create = _FakeCreate()

    class _FakeSpeech:
        with_streaming_response = _FakeWithStreamingResponse()

    class _FakeAudio:
        speech = _FakeSpeech()

    class _FakeOpenAI:
        def __init__(self, **kw):
            pass

        audio = _FakeAudio()

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI, raising=False)
    monkeypatch.setattr(
        "tools.tts_tool._resolve_openai_audio_client_config",
        lambda: ("gw-token", "https://gw.example.com/v1", True),
    )

    streamer = ts.OpenAIStreamer({}, {"model": "tts-1-hd", "voice": "nova"})
    chunks = list(streamer.stream("hello world"))
    assert chunks  # got audio back
    assert seen_models == ["gpt-4o-mini-tts"], (
        f"expected model coerced to gpt-4o-mini-tts, got {seen_models}"
    )





def test_openai_client_created_once_and_reused(monkeypatch):
    """The OpenAI client must be created once via cached_property and reused
    across multiple stream() calls — no fresh TLS handshake per sentence.
    """
    client_instances: list[int] = []

    class _FakeResp:
        @staticmethod
        def iter_bytes():
            yield b"\x00\x00" * 10

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeCreate:
        def __call__(self, **kw):
            return _FakeResp()

    class _FakeWithStreamingResponse:
        create = _FakeCreate()

    class _FakeSpeech:
        with_streaming_response = _FakeWithStreamingResponse()

    class _FakeAudio:
        speech = _FakeSpeech()

    class _FakeOpenAI:
        def __init__(self, **kw):
            client_instances.append(id(self))

        audio = _FakeAudio()

    import tools.tts_streaming as _ts
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI, raising=False)
    # Stub the auth chain so _client_config resolves without real credentials.
    monkeypatch.setattr(
        "tools.tts_tool._resolve_openai_audio_client_config",
        lambda: ("fake-key", None, False),
    )

    streamer = _ts.OpenAIStreamer({}, {})
    list(streamer.stream("first sentence."))
    list(streamer.stream("second sentence."))
    list(streamer.stream("third sentence."))

    assert len(client_instances) == 1, (
        f"expected 1 OpenAI client (reused), got {len(client_instances)}: "
        f"{client_instances}"
    )


# ── Dispatch: chunked streamer path ──────────────────────────────────────


