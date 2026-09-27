"""OpenAI streaming TTS instruction routing."""

import asyncio
import json
import queue
import threading
from unittest.mock import MagicMock

import pytest

import tools.tts_streaming as ts


LONG_SENTENCE = (
    "The narrator reads every word of this carefully paced sentence so listeners hear the "
    "entire response and never lose the ending after a long clause."
)


@pytest.mark.parametrize(
    ("instructions_config", "expected_instructions"),
    [
        pytest.param(
            {"instructions": "global", "openai": {"instructions": "provider"}},
            "provider",
            id="provider-override",
        ),
        pytest.param(
            {"instructions": "global", "openai": {}},
            "global",
            id="global-default",
        ),
        pytest.param(
            {"instructions": "global", "openai": {"instructions": ""}},
            None,
            id="provider-empty-suppresses-global",
        ),
        pytest.param(
            {"openai": {}},
            None,
            id="unset",
        ),
    ],
)
def test_openai_streamer_uses_resolved_config(
    monkeypatch,
    instructions_config,
    expected_instructions,
):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self):
            yield b"\x01\x00"

    class _StreamingCreate:
        @staticmethod
        def create(**kwargs):
            captured["request"] = kwargs
            return _Response()

    class _OpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.audio = MagicMock()
            self.audio.speech.with_streaming_response = _StreamingCreate()

    monkeypatch.setattr(ts, "resolve_openai_audio_api_key", lambda: "env-key")
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda key, *args: None)
    monkeypatch.setattr("openai.OpenAI", _OpenAI)

    openai_config = instructions_config["openai"]
    config = dict(instructions_config)
    config["provider"] = "openai"
    config["openai"] = {
        **openai_config,
        "api_key": "cfg-key",
        "base_url": "http://local-tts.example/v1",
    }
    streamer = ts.resolve_streaming_provider(config)

    assert streamer is not None
    assert list(streamer.stream("Streaming test.")) == [b"\x01\x00"]

    expected_request = {
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": "Streaming test.",
        "response_format": "pcm",
    }
    if expected_instructions is not None:
        expected_request["instructions"] = expected_instructions

    assert captured == {
        "client": {
            "api_key": "cfg-key",
            "base_url": "http://local-tts.example/v1",
        },
        "request": expected_request,
    }


@pytest.mark.parametrize(
    ("sync_model", "streaming_model", "expected_text", "expected_limit"),
    [
        ("eleven_v3", "eleven_flash_v2_5", "Hello there.", 60),
        ("eleven_multilingual_v2", "eleven_v3", "[excited] Hello there.", 48),
    ],
)
def test_elevenlabs_stream_uses_effective_model_for_audio_tags(
    monkeypatch, sync_model, streaming_model, expected_text, expected_limit,
):
    client = MagicMock()
    client.text_to_speech.convert.return_value = [b"\x01\x00"]
    monkeypatch.setattr(ts, "_resolve_key", lambda *_args: "test-key")
    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: lambda **_kwargs: client)
    config = {
        "provider": "elevenlabs",
        "instructions": "[excited]",
        "elevenlabs": {"model_id": sync_model, "streaming_model_id": streaming_model},
    }

    streamer = ts.resolve_streaming_provider(config)

    assert list(streamer.stream("Hello there.")) == [b"\x01\x00"]
    assert {
        "text": client.text_to_speech.convert.call_args.kwargs["text"],
        "model_id": client.text_to_speech.convert.call_args.kwargs["model_id"],
        "text_limit": ts.streaming_text_limit(streamer, "elevenlabs", config, 60),
    } == {"text": expected_text, "model_id": streaming_model, "text_limit": expected_limit}


def test_gemini_stream_sends_style_and_persona_as_direction(monkeypatch, tmp_path):
    import requests

    persona = tmp_path / "persona.md"
    persona.write_text("AUDIO PROFILE: radio host\n{transcript}", encoding="utf-8")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self, decode_unicode=True):
            return iter(())

    def fake_post(_url, **kwargs):
        captured["request"] = kwargs["json"]
        return Response()

    monkeypatch.setattr(ts, "_resolve_key", lambda *_args: "test-key")
    monkeypatch.setattr(requests, "post", fake_post)
    config = {
        "provider": "gemini",
        "instructions": "calm and measured",
        "gemini": {"persona_prompt_file": str(persona)},
    }

    streamer = ts.resolve_streaming_provider(config)

    assert list(streamer.stream("Hello there.")) == []
    prompt = captured["request"]["contents"][0]["parts"][0]["text"]
    assert "#### STYLE DIRECTION\ncalm and measured" in prompt
    assert "AUDIO PROFILE: radio host\nHello there." in prompt
    assert prompt.count("Hello there.") == 1
    limit = ts.streaming_text_limit(streamer, "gemini", config, 400)
    from tools.tts_tool_providers import _gemini_prompt_with_instructions

    assert len(_gemini_prompt_with_instructions("x" * limit, streamer.section, streamer.tts_config,
                                                "gemini-2.5-flash-preview-tts")) <= 400
    assert len(_gemini_prompt_with_instructions("x" * (limit + 1), streamer.section, streamer.tts_config,
                                                "gemini-2.5-flash-preview-tts")) > 400


def test_gemini_stream_rejects_prompt_that_exceeds_request_limit(monkeypatch):
    import requests

    monkeypatch.setattr(ts, "_resolve_key", lambda *_args: "test-key")
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: pytest.fail("unexpected request"))
    config = {
        "provider": "gemini",
        "instructions": "calm and measured",
        "gemini": {"max_text_length": 20},
    }

    streamer = ts.resolve_streaming_provider(config)

    with pytest.raises(ValueError, match="Gemini TTS composed prompt exceeds"):
        list(streamer.stream("Hello there."))


def test_cli_stream_splits_gemini_clause_without_losing_words(monkeypatch):
    from tools import tts_tool, tts_tool_speaker
    from tools.tts_tool_providers import _gemini_prompt_with_instructions

    config = {
        "provider": "gemini",
        "instructions": "calm and measured",
        "gemini": {"max_text_length": 280},
    }
    streamer = ts.GeminiStreamer(config, config["gemini"])
    requests = []

    class Playback:
        def __init__(self, *_args):
            pass

        def speak(self, text):
            requests.append(text)

        def finish(self):
            pass

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
    monkeypatch.setattr(ts, "resolve_streaming_provider", lambda *_args, **_kwargs: streamer)
    monkeypatch.setattr(tts_tool_speaker, "_StreamerPlayback", Playback)
    text_queue = queue.Queue()
    text_queue.put(LONG_SENTENCE)
    text_queue.put(None)
    done = threading.Event()

    tts_tool_speaker.stream_tts_to_speaker(text_queue, threading.Event(), done)

    assert done.is_set()
    assert len(requests) > 1
    assert " ".join(requests) == LONG_SENTENCE
    assert all(len(_gemini_prompt_with_instructions(piece, config["gemini"], config,
                                                    "gemini-2.5-flash-preview-tts")) <= 280
               for piece in requests)


def test_gemini_streaming_budget_requires_room_for_transcript():
    from tools.tts_tool_providers import _gemini_prompt_with_instructions

    config = {
        "provider": "gemini",
        "instructions": "calm and measured",
        "gemini": {},
    }
    streamer = ts.GeminiStreamer(config, config["gemini"])
    minimum = len(_gemini_prompt_with_instructions(
        "x", config["gemini"], config, "gemini-2.5-flash-preview-tts",
    ))

    config["gemini"]["max_text_length"] = minimum
    assert ts.streaming_text_limit(streamer, "gemini", config, minimum) == 1

    config["gemini"]["max_text_length"] = minimum - 1
    with pytest.raises(ValueError, match="Gemini TTS composed prompt exceeds"):
        ts.streaming_text_limit(streamer, "gemini", config, minimum - 1)


@pytest.mark.parametrize(
    ("instructions", "auto_tags", "expected_text"),
    [
        ("whisper", False, "<whisper>Hello there.</whisper>"),
        ("gravelly narrator", False, "Hello there."),
        ("gravelly narrator", True, "<soft>Hello there.</soft>"),
    ],
)
def test_xai_stream_sends_rendered_speech_text(
    monkeypatch, instructions, auto_tags, expected_text,
):
    import websockets

    sent = []

    class Socket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def send(self, message):
            sent.append(json.loads(message))

        async def recv(self):
            return json.dumps({"type": "audio.done"})

    monkeypatch.setattr(websockets, "connect", lambda *_args, **_kwargs: Socket())
    monkeypatch.setattr(
        "tools.xai_http.resolve_xai_http_credentials",
        lambda **_kwargs: {"api_key": "test-key"},
    )
    if auto_tags:
        def rewrite(_text, direction=""):
            assert direction == instructions
            return "<soft>Hello there.</soft>"

        monkeypatch.setattr("tools.tts_tool_providers._apply_xai_auto_speech_tags", rewrite)
    config = {
        "provider": "xai",
        "instructions": instructions,
        "xai": {"auto_speech_tags": auto_tags},
    }

    streamer = ts.resolve_streaming_provider(config)
    asyncio.run(streamer._pump("Hello there.", queue.Queue(), threading.Event()))

    assert sent == [
        {"type": "text.delta", "delta": expected_text},
        {"type": "text.done"},
    ]
