"""Exercise ElevenLabs HTTP synthesis with the real SDK and an in-memory transport."""

import json
from functools import partial

import httpx
import pytest
from elevenlabs.client import ElevenLabs

from tools import tts_streaming, tts_tool


@pytest.mark.parametrize("base_url", [None, "http://127.0.0.1:9876/proxy/"])
@pytest.mark.parametrize("streaming", [False, True])
def test_elevenlabs_synthesis_uses_configured_endpoint(tmp_path, monkeypatch, base_url, streaming):
    """Both synthesis paths preserve custom URL schemes, ports and path prefixes."""
    urls = []
    audio = b"ID3\x03\x00\x00\x00\x00\x00\x00"
    config = {"provider": "elevenlabs", "elevenlabs": {"voice_id": "test-voice", "base_url": base_url}}

    def respond(request):
        urls.append(str(request.url.copy_with(query=None)))
        return httpx.Response(200, content=audio)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: partial(ElevenLabs, httpx_client=transport))
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *_args: "test-key")
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
        if streaming:
            provider = tts_streaming.resolve_streaming_provider(config)
            assert provider is not None
            result = b"".join(provider.stream("Hello."))
        else:
            from tools.registry import registry

            output = tmp_path / "speech.mp3"
            response = json.loads(registry.get_entry("text_to_speech").handler(
                {"text": "Hello.", "output_path": str(output)}))
            assert response["success"], response
            result = output.read_bytes()

    assert result == audio
    expected_base = (base_url or "https://api.elevenlabs.io").rstrip("/")
    assert urls == [f"{expected_base}/v1/text-to-speech/test-voice"]
