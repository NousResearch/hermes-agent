"""Tests for the Ollama speech backend (``tools/tts_tool_ollama.py``).

Everything runs against a real socket: a throwaway ``HTTPServer`` stands in for ``ollama serve``,
so URL joining, JSON bodies, error mapping and the WAV hand-off are exercised for real instead of
being mocked away. No Ollama install and no network access required.
"""

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def _wav_bytes(seconds: float = 0.25, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class _FakeOllama(BaseHTTPRequestHandler):
    """Records requests; behaviour is driven by the instance's ``routes`` dict on the server."""

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send(self, code: int, body: bytes, content_type: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        server = self.server
        server.requests.append((self.path, body))
        speech = server.routes.get("speech")
        if self.path.endswith("/v1/audio/speech"):
            if isinstance(speech, tuple):  # (status, payload)
                status, payload = speech
                self._send(status, payload if isinstance(payload, bytes) else json.dumps(payload).encode())
            else:
                self._send(200, _wav_bytes(), "audio/wav")
            return
        if self.path.endswith("/api/show"):
            self._send(200, json.dumps(server.routes.get("show", {})).encode())
            return
        self._send(404, b'{"error":"not found"}')

    def do_GET(self):
        self.server.requests.append((self.path, None))
        if self.path.endswith("/api/tags"):
            self._send(200, json.dumps(self.server.routes.get("tags", {"models": []})).encode())
            return
        self._send(404, b'{"error":"not found"}')


@pytest.fixture
def ollama_server():
    """A live fake Ollama; ``server.routes`` is mutated by individual tests."""
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    server.routes = {}
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class TestBaseUrl:
    @pytest.mark.parametrize("raw,expected", [
        (None, "http://localhost:11434/v1"),
        ("", "http://localhost:11434/v1"),
        ("localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
        ("https://box.lan:11434", "https://box.lan:11434/v1"),
    ])
    def test_normalisation(self, raw, expected):
        from tools.tts_tool_ollama import _ollama_base_url
        assert _ollama_base_url(raw) == expected

    def test_root_is_the_native_api(self):
        from tools.tts_tool_ollama import _ollama_root
        assert _ollama_root("http://h:11434/v1") == "http://h:11434"


class TestGenerate:
    def test_request_body_and_output(self, tmp_path, ollama_server):
        from tools.tts_tool_ollama import _generate_ollama_tts

        output = str(tmp_path / "out.wav")
        result = _generate_ollama_tts(
            "Hello there", output,
            {"ollama": {"base_url": ollama_server.base_url, "model": "phonon:31m-en",
                        "voice": "Easton"}})
        assert result == output
        assert (tmp_path / "out.wav").read_bytes().startswith(b"RIFF")
        path, body = ollama_server.requests[0]
        assert path == "/v1/audio/speech"
        assert body == {"model": "phonon:31m-en", "input": "Hello there",
                        "voice": "Easton", "response_format": "wav"}

    def test_defaults_reach_the_request(self, tmp_path, ollama_server):
        from tools.tts_tool_ollama import _generate_ollama_tts

        _generate_ollama_tts("Hi", str(tmp_path / "o.wav"),
                             {"ollama": {"base_url": ollama_server.base_url}})
        _, body = ollama_server.requests[0]
        assert body["model"] == "phonon"
        assert body["voice"] == "default"

    def test_unknown_voice_error_is_surfaced_with_hint(self, tmp_path, ollama_server):
        """Ollama answers an unknown voice with 400 + the available voices: keep that text."""
        from tools.tts_tool_ollama import _generate_ollama_tts

        ollama_server.routes["speech"] = (
            400, {"error": 'voice "Zzz" not found; available voices: default'})
        with pytest.raises(RuntimeError) as excinfo:
            _generate_ollama_tts("Hi", str(tmp_path / "o.wav"),
                                 {"ollama": {"base_url": ollama_server.base_url, "voice": "Zzz"}})
        message = str(excinfo.value)
        assert "available voices: default" in message
        assert "tts.ollama.voice" in message

    def test_missing_model_points_at_pull_and_version(self, tmp_path, ollama_server):
        from tools.tts_tool_ollama import _generate_ollama_tts

        ollama_server.routes["speech"] = (404, {"error": "model not found"})
        with pytest.raises(RuntimeError) as excinfo:
            _generate_ollama_tts("Hi", str(tmp_path / "o.wav"),
                                 {"ollama": {"base_url": ollama_server.base_url}})
        message = str(excinfo.value)
        assert "ollama pull phonon" in message
        assert "0.40.9" in message

    def test_unreachable_daemon(self, tmp_path):
        from tools.tts_tool_ollama import _generate_ollama_tts

        with pytest.raises(RuntimeError) as excinfo:
            _generate_ollama_tts("Hi", str(tmp_path / "o.wav"),
                                 {"ollama": {"base_url": "http://127.0.0.1:1", "timeout": 2}})
        assert "Cannot reach Ollama" in str(excinfo.value)


class TestDiscovery:
    def test_voices_come_from_api_show(self, ollama_server):
        from tools.tts_tool_ollama import ollama_voices
        ollama_server.routes["show"] = {"capabilities": ["speech"], "voices": ["default"]}
        assert ollama_voices("phonon", ollama_server.base_url) == ["default"]

    def test_voices_empty_when_daemon_is_down(self):
        from tools.tts_tool_ollama import ollama_voices
        assert ollama_voices("phonon", "http://127.0.0.1:1", timeout=2) == []

    def test_speech_models_filter_by_capability(self, ollama_server):
        """Only models that report the ``speech`` capability are offered as TTS models."""
        from tools.tts_tool_ollama import ollama_speech_models

        ollama_server.routes["tags"] = {"models": [{"name": "phonon:31m-en"}, {"name": "llama3:8b"}]}
        ollama_server.routes["show"] = {"capabilities": ["completion"]}
        assert ollama_speech_models(ollama_server.base_url) == []

    def test_discovery_never_raises_offline(self):
        from tools.tts_tool_ollama import ollama_speech_models
        assert ollama_speech_models("http://127.0.0.1:1", timeout=2) == []


class TestRegistration:
    def test_cap_matches_the_documented_limit(self):
        from tools.tts_tool_delivery import _resolve_max_text_length
        from tools.tts_tool_ollama import OLLAMA_MAX_TEXT_LENGTH
        assert OLLAMA_MAX_TEXT_LENGTH == 4096
        assert _resolve_max_text_length("ollama") == OLLAMA_MAX_TEXT_LENGTH

    def test_no_sdk_no_credential(self):
        """The provider must stay usable without any package or key installed."""
        from tools.tts_tool import _BUILTIN_DISPATCH, _BUILTIN_REQUIREMENTS
        available, _label, generator, missing = _BUILTIN_DISPATCH["ollama"]
        assert available is None and missing is None
        assert generator == "_generate_ollama_tts"
        assert _BUILTIN_REQUIREMENTS["ollama"]() is True
