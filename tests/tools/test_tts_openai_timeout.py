"""``tts.openai.timeout`` bounds a stalled OpenAI speech request.

A speech endpoint that sends part of a reply and then goes quiet used to hold the voice reply
until a proxy cut the body (~300 s). Each SDK attempt now gives up after ``tts.openai.timeout``
seconds without bytes, and the SDK's transport retry sends a fresh request.
"""

import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.config as hermes_config
from gateway.session_context import _UNSET, _VAR_MAP
from hermes_constants import get_hermes_home
from tools import tts_tool


def _ogg_page(header_type: int, granule: int, payload: bytes, sequence: int) -> bytes:
    lacing = [255] * (len(payload) // 255) + [len(payload) % 255]
    header = struct.pack("<4sBBqIIIB", b"OggS", 0, header_type, granule, 1, sequence, 0, len(lacing))
    return header + bytes(lacing) + payload


HEAD_PAGE = _ogg_page(0x02, 0, b"OpusHead" + bytes([1, 1]) + struct.pack("<HIhB", 312, 48000, 0, 0), 0)
COMPLETE_OGG = HEAD_PAGE + _ogg_page(0x04, 312 + 3 * 48000, b"\x55" * 600, 1)


@pytest.fixture(autouse=True)
def _no_session_platform(monkeypatch):
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


def _write_tts_config(openai_section: dict) -> None:
    config = {"tts": {"provider": "openai", "openai": openai_section}}  # JSON is valid YAML
    (get_hermes_home() / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    hermes_config._LOAD_CONFIG_CACHE.clear()


@pytest.fixture
def stalling_speech_server():
    """A speech endpoint whose first response sends the Ogg head page and then stalls."""
    release = threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "audio/ogg")
            self.send_header("Content-Length", str(len(COMPLETE_OGG)))
            self.end_headers()
            if len(requests) == 1:
                self.wfile.write(HEAD_PAGE)
                self.wfile.flush()
                release.wait(timeout=30)
                return
            self.wfile.write(COMPLETE_OGG)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_stalled_speech_stream_times_out_and_the_retry_delivers(tmp_path, stalling_speech_server):
    base_url, requests = stalling_speech_server
    _write_tts_config({"base_url": base_url, "timeout": 0.5})
    output = tmp_path / "reply.ogg"

    started = time.monotonic()
    with patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
               return_value=("test-key", None, False)):
        result = json.loads(tts_tool.text_to_speech_tool("Sure, the meeting moved to Thursday.",
                                                         output_path=str(output)))
    elapsed = time.monotonic() - started

    assert result["success"] is True, result
    assert requests == ["/v1/audio/speech", "/v1/audio/speech"]
    assert output.read_bytes() == COMPLETE_OGG
    assert elapsed < 15, f"a stalled stream must fail within tts.openai.timeout, took {elapsed:.1f}s"


@pytest.mark.parametrize("openai_section, expected", [
    pytest.param({}, 120.0, id="default"),
    pytest.param({"timeout": 45}, 45.0, id="configured"),
    pytest.param({"timeout": "soon"}, 120.0, id="invalid-falls-back"),
])
def test_openai_tts_client_timeout_comes_from_config(tmp_path, openai_section, expected):
    _write_tts_config(openai_section)
    factory = MagicMock()
    factory.return_value.audio.speech.create.return_value.stream_to_file.side_effect = (
        lambda path: open(path, "wb").write(COMPLETE_OGG))

    with patch("tools.tts_tool._import_openai_client", return_value=factory), \
         patch("tools.tts_tool_openai._resolve_openai_audio_client_config",
               return_value=("test-key", None, False)):
        result = json.loads(tts_tool.text_to_speech_tool("Hello there.", output_path=str(tmp_path / "a.ogg")))

    assert result["success"] is True, result
    assert factory.call_args.kwargs["timeout"] == expected
