"""OpenAI streaming timeout and single-attempt behavior over real loopback HTTP."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import openai
import pytest

from tools import tts_streaming as ts


@pytest.mark.parametrize("value", [0.05, 1, 15.5, 120, 120.0])
def test_timeout_accepts_finite_numbers_in_range(value, caplog):
    assert ts._openai_stream_timeout_seconds({"stream_timeout_seconds": value}) == value
    assert not caplog.records


@pytest.mark.parametrize("value", [
    True, False, None, 0, -1, 120.01, 10 ** 1000,
    float("nan"), float("inf"), float("-inf"),
    "15", "", "https://user:private-token@example.test", [], {},
])
def test_invalid_timeout_falls_back_without_logging_private_values(value, caplog):
    assert ts._openai_stream_timeout_seconds({"stream_timeout_seconds": value}) == 15.0
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "tts.openai.stream_timeout_seconds" in caplog.text
    assert "private-token" not in caplog.text


def test_missing_timeout_uses_default_without_warning(caplog):
    assert ts._openai_stream_timeout_seconds({}) == 15.0
    assert not caplog.records


@pytest.fixture
def speech_server():
    state = SimpleNamespace(
        mode="success", requests=[], received=threading.Event(),
        release=threading.Event(), pcm=b"\x01\x00\x02\x00",
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state.requests.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": json.loads(body),
            })
            if state.mode == "headers":
                state.received.set()
                state.release.wait(10)
            status = 500 if state.mode == "error" else 200
            payload = b'{"error":{"message":"unavailable"}}' if status == 500 else state.pcm
            self.send_response(status)
            self.send_header("Content-Type", "application/json" if status == 500 else "audio/pcm")
            self.send_header("X-Audio-Sample-Rate", "44100")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.flush()
            if state.mode == "body":
                state.received.set()
                state.release.wait(10)
            else:
                state.received.set()
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Expected when the client times out before the test releases the response.
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _consume(streamer):
    result = SimpleNamespace(chunks=[], error=None, done=threading.Event())

    def run():
        try:
            result.chunks.extend(streamer.stream("A spoken sentence."))
        except Exception as exc:
            result.error = exc
        finally:
            result.done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return result, thread


@pytest.mark.parametrize("stall", ["headers", "body"])
def test_active_profile_timeout_stops_stalled_http_without_retry(
    tmp_path, monkeypatch, speech_server, stall,
):
    """The real TTS loader must read the served profile, not the launch profile."""
    from hermes_cli.config import atomic_config_write
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "served-profile"
    launch_home.mkdir()
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    section = {
        "api_key": "profile-key", "base_url": speech_server.base_url,
        "model": "profile-model", "voice": "profile-voice", "pcm_sample_rate": 22050,
    }
    atomic_config_write(launch_home / "config.yaml", {
        "tts": {"provider": "openai", "openai": {**section, "stream_timeout_seconds": 120}},
    })
    atomic_config_write(profile_home / "config.yaml", {
        "tts": {"provider": "openai", "openai": {**section, "stream_timeout_seconds": 0.1}},
    })
    token = set_hermes_home_override(profile_home)
    try:
        streamer = ts.resolve_streaming_provider(ts._load_tts_config())
    finally:
        reset_hermes_home_override(token)
    assert streamer is not None
    speech_server.mode = stall
    result, worker = _consume(streamer)
    try:
        assert speech_server.received.wait(5), "speech endpoint was never reached"
        assert result.done.wait(5), "configured timeout did not stop the stalled stream"
    finally:
        speech_server.release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    # Header acquisition is wrapped by the SDK; body iteration exposes httpx's read timeout.
    assert isinstance(result.error, (openai.APITimeoutError, httpx.ReadTimeout))
    assert result.chunks == []
    assert len(speech_server.requests) == 1
    if stall == "body":
        assert streamer.sample_rate == 44100


def test_http_failure_does_not_retry_sentence(speech_server):
    speech_server.mode = "error"
    section = {"api_key": "local-key", "base_url": speech_server.base_url}
    streamer = ts.OpenAIStreamer({"openai": section}, section)
    with pytest.raises(openai.InternalServerError):
        list(streamer.stream("Do not replay this sentence."))
    assert len(speech_server.requests) == 1


def test_success_keeps_pcm_and_endpoint_sample_rate(speech_server):
    section = {
        "api_key": "local-key", "base_url": speech_server.base_url,
        "stream_timeout_seconds": 120, "pcm_sample_rate": 22050,
    }
    streamer = ts.OpenAIStreamer({"openai": section}, section)
    assert b"".join(streamer.stream("A spoken sentence.")) == speech_server.pcm
    assert streamer.sample_rate == 44100
    assert len(speech_server.requests) == 1
