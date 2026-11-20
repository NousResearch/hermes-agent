"""Caller-chosen audio extensions retain their spelling and format semantics."""

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import tts_tool
from tools.registry import registry
from tools.tts_tool_delivery import _repair_ogg_container, _wav_sidecar_path
from tools.tts_tool_providers import _tts_response_format_from_path


@pytest.mark.parametrize("suffix", ["wav", "WAV", "WaV"])
def test_registered_tts_requests_wav_for_case_variants(tmp_path, monkeypatch, suffix):
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * 80)
    requests = []

    class Speech(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            valid = self.path == "/v1/audio/speech" and payload["response_format"] == "wav"
            body = audio.getvalue() if valid else b'{"error":{"message":"wav response_format required"}}'
            self.send_response(200 if valid else 400)
            self.send_header("Content-Type", "audio/wav" if valid else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Speech)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(json.dumps({"tts": {
        "provider": "openai", "openai": {"api_key": "local-fixture-key",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1"},
    }}), encoding="utf-8")
    output = tmp_path / f"speech.{suffix}"
    try:
        result = json.loads(registry.dispatch("text_to_speech", {"text": "hello", "output_path": str(output)}))
        assert result["success"], result
        assert requests[0]["response_format"] == "wav"
        assert result["file_path"] == str(output)
        with wave.open(str(output), "rb") as generated:
            assert generated.getnframes() == 80
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


@pytest.mark.parametrize("suffix,expected", [
    (".ogg", "opus"), (".OGG", "opus"), (".wav", "wav"), (".WAV", "wav"),
    (".flac", "flac"), (".FLAC", "flac"), (".mp3", "mp3"), (".MP3", "mp3"),
])
def test_extension_selection_preserves_output_spelling(tmp_path, suffix, expected):
    path = str(tmp_path / f"speech{suffix}")
    assert _tts_response_format_from_path(path) == expected
    assert _wav_sidecar_path(path) == (path if expected == "wav" else str(tmp_path / "speech.wav"))
    assert tts_tool._finalize_voice_delivery(path, "openai", None, True) == (path, expected == "opus")


@pytest.mark.parametrize("suffix", ["ogg", "OGG", "OgG"])
def test_case_variants_do_not_skip_container_repair(tmp_path, monkeypatch, suffix):
    from gateway.platforms import base

    path = tmp_path / f"speech.{suffix}"
    path.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20)
    monkeypatch.setattr(base, "transcode_to_ogg_opus", lambda *_args, **_kwargs: None)
    repaired = _repair_ogg_container(str(path))
    assert repaired == str(tmp_path / "speech.wav")
    assert not path.exists()
