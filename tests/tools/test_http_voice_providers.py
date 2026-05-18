"""Tests for OpenClaw-compatible HTTP STT/TTS voice providers."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest


class _FakeVoiceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def _handle(self):
        length = int(self.headers.get("content-length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        route = self.server.routes.get((self.command, self.path))
        if route is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return

        if callable(route):
            status, headers, response_body = route(self, body)
        else:
            status, headers, response_body = route

        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, _format, *args):
        return


@contextmanager
def _fake_voice_server(routes):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVoiceHandler)
    server.routes = routes
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _json_response(payload, status=200):
    return (
        status,
        {"Content-Type": "application/json"},
        json.dumps(payload).encode("utf-8"),
    )


def _text_response(text, status=200):
    return (
        status,
        {"Content-Type": "text/plain; charset=utf-8"},
        text.encode("utf-8"),
    )


def _audio_response(data=b"RIFF\x10\x00\x00\x00WAVEfmt fake"):
    return 200, {"Content-Type": "audio/wav"}, data


def test_get_provider_honors_explicit_whisper_http():
    from tools.transcription_tools import _get_provider

    assert _get_provider({"provider": "whisper_http"}) == "whisper_http"


def test_transcribe_audio_dispatches_to_whisper_http(tmp_path):
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    with patch(
        "tools.transcription_tools._load_stt_config",
        return_value={"provider": "whisper_http", "whisper_http": {"model": "whisper-1"}},
    ), patch(
        "tools.transcription_tools._get_provider",
        return_value="whisper_http",
    ), patch(
        "tools.transcription_tools._transcribe_whisper_http",
        return_value={"success": True, "transcript": "hi", "provider": "whisper_http"},
    ) as mock_whisper_http:
        from tools.transcription_tools import transcribe_audio

        result = transcribe_audio(str(audio_file))

    assert result["success"] is True
    mock_whisper_http.assert_called_once_with(str(audio_file), "whisper-1")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (" plain text ", "plain text"),
        ({"text": "json text"}, "json text"),
        ({"transcript": "json transcript"}, "json transcript"),
        ({"segments": [{"text": "first"}, {"text": "second"}]}, "first second"),
    ],
)
def test_whisper_http_extracts_common_response_shapes(payload, expected):
    from tools.transcription_tools import _extract_whisper_http_transcript

    assert _extract_whisper_http_transcript(payload) == expected


def test_whisper_http_openai_compatible_plain_text(tmp_path):
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    routes = {
        ("POST", "/v1/audio/transcriptions"): _text_response("привет из http"),
    }
    with _fake_voice_server(routes) as base_url, patch(
        "tools.transcription_tools._load_stt_config",
        return_value={
            "provider": "whisper_http",
            "whisper_http": {
                "base_url": base_url,
                "path": "/v1/audio/transcriptions",
                "model": "whisper-podlodka-turbo",
                "timeout": 5,
            },
        },
    ):
        from tools.transcription_tools import transcribe_audio

        result = transcribe_audio(str(audio_file))

    assert result == {
        "success": True,
        "transcript": "привет из http",
        "provider": "whisper_http",
    }


def test_whisper_http_openclaw_inference_segments(tmp_path):
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    routes = {
        ("POST", "/inference"): _json_response(
            {"segments": [{"text": "проверка"}, {"text": "связи"}]}
        ),
    }
    with _fake_voice_server(routes) as base_url, patch(
        "tools.transcription_tools._load_stt_config",
        return_value={
            "provider": "whisper_http",
            "whisper_http": {
                "base_url": base_url,
                "path": "/inference",
                "model": "whisper-podlodka-turbo",
                "language": "ru",
                "timeout": 5,
            },
        },
    ):
        from tools.transcription_tools import transcribe_audio

        result = transcribe_audio(str(audio_file))

    assert result["success"] is True
    assert result["transcript"] == "проверка связи"
    assert result["provider"] == "whisper_http"


@pytest.mark.parametrize(
    ("route", "error_part"),
    [
        (_json_response({"text": ""}), "empty transcript"),
        (_text_response("upstream failed", status=503), "HTTP 503"),
    ],
)
def test_whisper_http_surfaces_empty_and_non_200_errors(tmp_path, route, error_part):
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    with _fake_voice_server({("POST", "/v1/audio/transcriptions"): route}) as base_url, patch(
        "tools.transcription_tools._load_stt_config",
        return_value={
            "provider": "whisper_http",
            "whisper_http": {
                "base_url": base_url,
                "path": "/v1/audio/transcriptions",
                "timeout": 5,
            },
        },
    ):
        from tools.transcription_tools import transcribe_audio

        result = transcribe_audio(str(audio_file))

    assert result["success"] is False
    assert error_part in result["error"]


def test_whisper_http_connection_error_is_helpful(tmp_path):
    import requests

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    with patch(
        "tools.transcription_tools._load_stt_config",
        return_value={
            "provider": "whisper_http",
            "whisper_http": {"base_url": "http://127.0.0.1:1", "timeout": 5},
        },
    ), patch(
        "requests.post",
        side_effect=requests.exceptions.ConnectionError("connection refused"),
    ):
        from tools.transcription_tools import transcribe_audio

        result = transcribe_audio(str(audio_file))

    assert result["success"] is False
    assert "connection error" in result["error"].lower()


@pytest.mark.parametrize("provider", ["silero_http", "piper_http"])
def test_http_tts_providers_create_audio_file(tmp_path, provider, monkeypatch):
    output_path = tmp_path / f"{provider}.wav"
    config = {
        "provider": provider,
        provider: {
            "base_url": "",
            "path": "/tts",
            "timeout": 5,
        },
    }
    if provider == "silero_http":
        config[provider]["speaker"] = "eugene"
    else:
        config[provider]["voice_id"] = "ru_RU-test"

    with _fake_voice_server({("POST", "/tts"): _audio_response()}) as base_url:
        config[provider]["base_url"] = base_url
        monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)
        monkeypatch.setattr("tools.tts_tool._convert_to_opus", lambda _path: None)

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("hello", output_path=str(output_path)))

    assert result["success"] is True
    assert result["provider"] == provider
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_piper_http_passes_edge_compatible_voice_and_speed(tmp_path, monkeypatch):
    output_path = tmp_path / "piper.wav"
    captured = {}

    def handle_tts(_handler, body):
        captured["payload"] = json.loads(body.decode("utf-8"))
        return _audio_response()

    with _fake_voice_server({("POST", "/tts"): handle_tts}) as base_url:
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {
                "provider": "piper_http",
                "piper_http": {
                    "base_url": base_url,
                    "path": "/tts",
                    "voice_id": "en-US-AndrewMultilingualNeural",
                    "speed": 1.7,
                    "output_format": "wav",
                    "timeout": 5,
                },
            },
        )
        monkeypatch.setattr("tools.tts_tool._convert_to_opus", lambda _path: None)

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("hello", output_path=str(output_path)))

    assert result["success"] is True
    assert captured["payload"]["voice_id"] == "en-US-AndrewMultilingualNeural"
    assert captured["payload"]["speed"] == 1.7
    assert captured["payload"]["output_format"] == "wav"


@pytest.mark.parametrize("output_format", ["mp3", "ogg"])
def test_piper_http_default_output_path_matches_output_format(
    tmp_path, monkeypatch, output_format
):
    captured = {}

    def handle_tts(_handler, body):
        captured["payload"] = json.loads(body.decode("utf-8"))
        return _audio_response()

    with _fake_voice_server({("POST", "/tts"): handle_tts}) as base_url:
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {
                "provider": "piper_http",
                "piper_http": {
                    "base_url": base_url,
                    "path": "/tts",
                    "output_format": output_format,
                    "timeout": 5,
                },
            },
        )
        monkeypatch.setattr("tools.tts_tool.DEFAULT_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setattr("tools.tts_tool._convert_to_opus", lambda _path: None)

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("hello"))

    assert result["success"] is True
    assert result["file_path"].endswith(f".{output_format}")
    assert result["media_tag"].endswith(f".{output_format}")
    assert captured["payload"]["output_format"] == output_format


def test_piper_http_explicit_output_path_suffix_matches_output_format(
    tmp_path, monkeypatch
):
    requested_output = tmp_path / "piper.wav"
    expected_output = tmp_path / "piper.ogg"

    with _fake_voice_server({("POST", "/tts"): _audio_response()}) as base_url:
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {
                "provider": "piper_http",
                "piper_http": {
                    "base_url": base_url,
                    "path": "/tts",
                    "output_format": "ogg",
                    "timeout": 5,
                },
            },
        )
        monkeypatch.setattr("tools.tts_tool._convert_to_opus", lambda _path: None)

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(
            text_to_speech_tool("hello", output_path=str(requested_output))
        )

    assert result["success"] is True
    assert result["file_path"] == str(expected_output)
    assert expected_output.exists()
    assert not requested_output.exists()


def test_piper_http_rejects_unknown_output_format(tmp_path, monkeypatch):
    with _fake_voice_server({("POST", "/tts"): _audio_response()}) as base_url:
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {
                "provider": "piper_http",
                "piper_http": {
                    "base_url": base_url,
                    "path": "/tts",
                    "output_format": "flac",
                    "timeout": 5,
                },
            },
        )

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(
            text_to_speech_tool("hello", output_path=str(tmp_path / "piper.wav"))
        )

    assert result["success"] is False
    assert "output_format" in result["error"]


def test_http_tts_keeps_wav_for_non_telegram_playback(tmp_path, monkeypatch):
    output_path = tmp_path / "silero.wav"
    with _fake_voice_server({("POST", "/tts"): _audio_response()}) as base_url:
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {
                "provider": "silero_http",
                "silero_http": {"base_url": base_url, "path": "/tts", "timeout": 5},
            },
        )

        def fail_if_called(_path):
            raise AssertionError("CLI/local playback should not be converted to Opus")

        monkeypatch.setattr("tools.tts_tool._convert_to_opus", fail_if_called)

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("hello", output_path=str(output_path)))

    assert result["success"] is True
    assert result["file_path"] == str(output_path)
    assert result["voice_compatible"] is False
    assert result["media_tag"] == f"MEDIA:{output_path}"


def test_http_tts_empty_audio_is_error(tmp_path, monkeypatch):
    with _fake_voice_server({("POST", "/tts"): _audio_response(b"")}) as base_url:
        monkeypatch.setattr(
            "tools.tts_tool._load_tts_config",
            lambda: {
                "provider": "silero_http",
                "silero_http": {"base_url": base_url, "path": "/tts", "timeout": 5},
            },
        )

        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool("hello", output_path=str(tmp_path / "out.wav")))

    assert result["success"] is False
    assert "empty audio" in result["error"]


def test_check_tts_requirements_accepts_http_provider_with_base_url(monkeypatch):
    for key in ("MINIMAX_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), patch(
        "tools.tts_tool._import_elevenlabs", side_effect=ImportError
    ), patch("tools.tts_tool._import_openai_client", side_effect=ImportError), patch(
        "tools.tts_tool._import_mistral_client", side_effect=ImportError
    ), patch(
        "tools.tts_tool._check_neutts_available", return_value=False
    ), patch(
        "tools.tts_tool._check_kittentts_available", return_value=False
    ), patch(
        "tools.tts_tool._load_tts_config",
        return_value={
            "provider": "silero_http",
            "silero_http": {"base_url": "http://127.0.0.1:9000"},
        },
    ):
        from tools.tts_tool import check_tts_requirements

        assert check_tts_requirements() is True
