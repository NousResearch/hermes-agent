"""Regression tests for bounded upstream TTS response reads."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools import tts_tool


class StreamingResponse:
    def __init__(self, chunks, *, status_code=200, headers=None):
        self._chunks = list(chunks)
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size=65536):
        del chunk_size
        yield from self._chunks

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def small_tts_body_cap(monkeypatch):
    monkeypatch.setattr(tts_tool, "TTS_RESPONSE_BODY_LIMIT_BYTES", 8)


def test_xai_tts_rejects_oversized_audio_response(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    response = StreamingResponse([b"12345", b"6789"], headers={"Content-Type": "audio/mpeg"})
    output_path = tmp_path / "out.mp3"

    with patch("requests.post", return_value=response) as post:
        with pytest.raises(RuntimeError, match="xAI TTS response exceeds 8 bytes"):
            tts_tool._generate_xai_tts("hello", str(output_path), {})

    assert post.call_args.kwargs["stream"] is True
    assert response.closed is True
    assert not output_path.exists()


def test_minimax_t2a_rejects_oversized_json_response(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    response = StreamingResponse([b'{"data":', b'"too large"}'], headers={"Content-Type": "application/json"})

    with patch("requests.post", return_value=response) as post:
        with pytest.raises(RuntimeError, match="MiniMax TTS response exceeds 8 bytes"):
            tts_tool._generate_minimax_tts("hello", str(tmp_path / "out.mp3"), {})

    assert post.call_args.kwargs["stream"] is True
    assert response.closed is True


def test_minimax_legacy_rejects_oversized_audio_response(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    response = StreamingResponse([b"12345", b"6789"], headers={"Content-Type": "audio/mpeg"})
    config = {"minimax": {"base_url": "https://api.minimax.chat/v1/text_to_speech"}}
    output_path = tmp_path / "out.mp3"

    with patch("requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="MiniMax TTS response exceeds 8 bytes"):
            tts_tool._generate_minimax_tts("hello", str(output_path), config)

    assert response.closed is True
    assert not output_path.exists()


def test_gemini_tts_rejects_oversized_json_response(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    response = StreamingResponse([b'{"candidates":', b"[{}]}"], headers={"Content-Type": "application/json"})

    with patch("requests.post", return_value=response) as post:
        with pytest.raises(RuntimeError, match="Gemini TTS response exceeds 8 bytes"):
            tts_tool._generate_gemini_tts("hello", str(tmp_path / "out.wav"), {})

    assert post.call_args.kwargs["stream"] is True
    assert response.closed is True
