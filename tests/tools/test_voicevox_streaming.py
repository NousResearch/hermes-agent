"""Real loopback HTTP tests for the VOICEVOX streaming provider contract."""
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from tools.tts_streaming import VoicevoxStreamer, resolve_streaming_provider


@pytest.fixture
def engine():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            url = urlsplit(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.server.calls.append((url.path, parse_qs(url.query), body))
            data = (b'{"outputSamplingRate":48000,"outputStereo":true}'
                    if url.path.endswith("/audio_query") else self.server.audio)
            self.send_response(self.server.status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.calls = []
        server.status = 200
        server.audio = _wav()
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield server
        finally:
            server.shutdown()
            worker.join(timeout=5)


def _wav(rate=24000, channels=1, width=2):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(b"\x00" * 100 * channels * width)
    return output.getvalue()


def _provider(engine):
    config = {
        "provider": "edge", "streaming": {"provider": "voicevox"},
        "voicevox": {"base_url": f"http://127.0.0.1:{engine.server_port}/engine/", "speaker": 3},
    }
    provider = resolve_streaming_provider(config)
    assert isinstance(provider, VoicevoxStreamer)
    return provider


def test_configured_endpoint_speaker_and_advertised_pcm_format(engine):
    provider = _provider(engine)
    assert engine.calls == []  # no probe to an unrelated localhost engine
    assert b"".join(provider.stream("日本語 & test？")) == b"\x00" * 200
    query, synthesis = engine.calls
    assert query[:2] == ("/engine/audio_query", {"text": ["日本語 & test？"], "speaker": ["3"]})
    assert synthesis[0:2] == ("/engine/synthesis", {"speaker": ["3"]})
    assert json.loads(synthesis[2]) == {"outputSamplingRate": 24000, "outputStereo": False}
    assert provider.sample_rate == 24000


def test_long_sentence_is_not_truncated(engine):
    text = "長" * 500 + "最後の文字。"
    list(_provider(engine).stream(text))
    assert engine.calls[0][1]["text"] == [text]


@pytest.mark.parametrize("format", [(48000, 1, 2), (24000, 2, 2), (24000, 1, 1)])
def test_mismatched_format_never_reaches_playback(engine, format):
    engine.audio = _wav(*format)
    provider = _provider(engine)
    with pytest.raises(RuntimeError, match="24000 Hz mono int16"):
        next(provider.stream("うん。"))
    assert provider.sample_rate == 24000


def test_http_error_reaches_caller_without_retries(engine):
    engine.status = 503
    with pytest.raises(HTTPError) as caught:
        list(_provider(engine).stream("うん。"))
    caught.value.close()
    assert len(engine.calls) == 1


def test_oversized_response_is_rejected(engine, monkeypatch):
    monkeypatch.setattr("tools.tts_streaming._STREAM_SENTENCE_BYTE_CAP", 64)
    with pytest.raises(RuntimeError, match="byte limit"):
        list(_provider(engine).stream("うん。"))
