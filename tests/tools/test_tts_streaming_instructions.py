"""OpenAI streaming TTS instruction routing."""

from unittest.mock import MagicMock

import pytest

import tools.tts_streaming as ts


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
