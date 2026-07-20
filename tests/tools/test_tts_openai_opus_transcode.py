"""Regression tests for issue #54589.

An OpenAI-*compatible* TTS backend (e.g. self-hosted Speaches/Kokoro) may not
encode opus and only support mp3/flac/wav/pcm. Requesting response_format="opus"
against such a backend fails and no voice bubble is delivered. When the user
pins ``tts.openai.response_format`` to a supported format, Hermes must
synthesize in that format and transcode to OGG/Opus locally.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from tools import tts_tool


class _FakeResponse:
    def __init__(self, fmt: str):
        self._fmt = fmt

    def stream_to_file(self, path: str) -> None:
        Path(path).write_bytes(b"audio-" + self._fmt.encode())


class _FakeClient:
    """Captures the create() kwargs and records the synth path written."""

    def __init__(self, *args, **kwargs):
        self.created_kwargs = None
        self.audio = SimpleNamespace(speech=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.created_kwargs = kwargs
        return _FakeResponse(kwargs["response_format"])

    def close(self):
        pass


def _patch_client(monkeypatch, holder):
    def _factory(*args, **kwargs):
        client = _FakeClient()
        holder.append(client)
        return client

    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: _factory)
    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        lambda: ("key", "http://x/v1", False),
    )


def test_ogg_target_defaults_to_opus(tmp_path, monkeypatch):
    """Real OpenAI API path: .ogg target still asks the backend for opus."""
    clients: list = []
    _patch_client(monkeypatch, clients)
    convert = MagicMock()
    monkeypatch.setattr(tts_tool, "_convert_to_opus", convert)

    out = tmp_path / "speech.ogg"
    cfg = {"openai": {"model": "m", "voice": "v"}}

    result = tts_tool._generate_openai_tts("hi", str(out), cfg)

    assert result == str(out)
    assert clients[0].created_kwargs["response_format"] == "opus"
    convert.assert_not_called()


def test_ogg_target_transcodes_when_backend_lacks_opus(tmp_path, monkeypatch):
    """Speaches/Kokoro path: synthesize in mp3, transcode to .ogg locally."""
    clients: list = []
    _patch_client(monkeypatch, clients)

    def fake_convert(path: str) -> str:
        # backend-format temp file should exist when we transcode it
        assert path.endswith(".mp3")
        assert os.path.exists(path)
        ogg = path.rsplit(".", 1)[0] + ".ogg"
        Path(ogg).write_bytes(b"ogg")
        return ogg

    monkeypatch.setattr(tts_tool, "_convert_to_opus", fake_convert)

    out = tmp_path / "speech.ogg"
    cfg = {"openai": {"model": "m", "voice": "v", "response_format": "mp3"}}

    result = tts_tool._generate_openai_tts("hi", str(out), cfg)

    assert result == str(out)
    # Backend was asked for mp3, never opus.
    assert clients[0].created_kwargs["response_format"] == "mp3"
    assert out.exists()
    # Intermediate mp3 is cleaned up.
    assert not (tmp_path / "speech.mp3").exists()
