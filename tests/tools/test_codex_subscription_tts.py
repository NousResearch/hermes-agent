"""Codex OAuth subscription TTS and live-voice contracts."""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import codex_web_audio, tts_streaming, tts_tool
from hermes_cli.tools_config import TOOL_CATEGORIES


def test_codex_voice_resolution_is_explicit_and_fail_closed():
    assert codex_web_audio.resolve_voice("juniper") == "juniper"
    assert codex_web_audio.resolve_voice("alloy") == "juniper"
    with pytest.raises(ValueError, match="Unsupported Codex subscription voice"):
        codex_web_audio.resolve_voice("invented")


def test_codex_web_transport_identifies_hermes_without_android_identity():
    headers = codex_web_audio._headers("token", "device")

    assert headers["originator"] == "hermes"
    assert "Android" not in headers["user-agent"]
    assert "oai-package-name" not in headers
    assert "x-sentinel-payload" not in headers
    assert (
        codex_web_audio._conversation_body("speak")["history_and_training_disabled"]
        is True
    )


def test_codex_tts_picker_requires_codex_oauth_without_platform_key():
    row = next(
        provider
        for provider in TOOL_CATEGORIES["tts"]["providers"]
        if provider["name"] == "OpenAI Codex OAuth"
    )

    assert row["tts_provider"] == "openai-codex"
    assert row["auth_provider"] == "openai-codex"
    assert row["env_vars"] == []


def test_codex_sse_parser_applies_assistant_text_patches():
    conversation_id = "conversation-id"
    message_id = "assistant-id"
    lines = [
        "data: "
        + json.dumps({
            "type": "resume_conversation_token",
            "conversation_id": conversation_id,
        }),
        "data: "
        + json.dumps({
            "v": {
                "conversation_id": conversation_id,
                "message": {
                    "id": message_id,
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": [""]},
                },
            }
        }),
        "data: "
        + json.dumps({
            "o": "patch",
            "v": [
                {"p": "/message/content/parts/0", "o": "append", "v": "Exact speech."}
            ],
        }),
    ]

    assert codex_web_audio._parse_conversation_sse(lines) == (
        conversation_id,
        message_id,
        "Exact speech.",
    )


def test_codex_synthesis_refuses_model_changed_text(monkeypatch):
    class FakeSession:
        def get(self, *_args, **_kwargs):
            return SimpleNamespace(status_code=403)

    mismatch = "\n".join([
        "data: " + json.dumps({"conversation_id": "conversation-id"}),
        "data: "
        + json.dumps({
            "v": {
                "conversation_id": "conversation-id",
                "message": {
                    "id": "assistant-id",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Different text."]},
                },
            }
        }),
    ]).encode()
    monkeypatch.setattr(codex_web_audio.requests, "Session", FakeSession)
    monkeypatch.setattr(codex_web_audio, "_conduit", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        codex_web_audio, "_sentinel", lambda *_args, **_kwargs: ("requirement", "proof")
    )
    monkeypatch.setattr(
        codex_web_audio,
        "_request",
        lambda *_args, **_kwargs: (
            SimpleNamespace(headers={"content-type": "text/event-stream"}),
            mismatch,
        ),
    )

    with pytest.raises(RuntimeError, match="changed the requested TTS text"):
        codex_web_audio.synthesize_codex_speech("Exact speech.", "token")


def test_public_tts_dispatches_codex_and_preserves_requested_mp3(monkeypatch, tmp_path):
    output = tmp_path / "speech.mp3"
    monkeypatch.setattr(
        tts_tool, "_load_tts_config", lambda: {"provider": "openai-codex"}
    )
    _, label, generator, error = tts_tool._BUILTIN_DISPATCH["openai-codex"]
    monkeypatch.setitem(
        tts_tool._BUILTIN_DISPATCH,
        "openai-codex",
        (lambda: True, label, generator, error),
    )

    def generate(text, path, config):
        assert text == "Subscription speech."
        assert config["provider"] == "openai-codex"
        Path(path).write_bytes(b"ID3codex")

    monkeypatch.setattr(tts_tool, "_generate_openai_codex_tts", generate)
    result = json.loads(
        tts_tool.text_to_speech_tool("Subscription speech.", str(output))
    )

    assert result["success"] is True, result
    assert result["provider"] == "openai-codex"
    assert result["file_path"] == str(output)
    assert output.read_bytes() == b"ID3codex"


def test_codex_streamer_captures_profile_scoped_pool(monkeypatch):
    pool = Mock()
    credentials = {"api_key": "token", "credential_id": "credential"}
    monkeypatch.setattr(
        "tools.tts_tool_codex._codex_tts_credentials", lambda: (pool, credentials)
    )
    monkeypatch.setattr("tools.tts_tool_codex._has_codex_tts_backend", lambda: True)

    streamer = tts_streaming.OpenAICodexStreamer({"provider": "openai-codex"}, {})

    assert streamer.pool is pool
    assert streamer.credentials == credentials
    assert (
        tts_streaming.resolve_streaming_provider(
            {"provider": "openai-codex"}, preferred="openai-codex"
        ).credentials
        == credentials
    )


def test_codex_streamer_enforces_pcm_cap(monkeypatch):
    pool = Mock()
    credentials = {"api_key": "token", "credential_id": "credential"}
    proc = Mock()
    proc.stdout = io.BytesIO(b"x" * 4096)
    proc.stderr = io.BytesIO()
    proc.poll.return_value = 0
    proc.wait.return_value = 0
    monkeypatch.setattr(
        "tools.tts_tool_codex._codex_tts_credentials", lambda: (pool, credentials)
    )
    monkeypatch.setattr("tools.tts_tool_codex._has_codex_tts_backend", lambda: True)
    monkeypatch.setattr(
        "tools.codex_web_audio.synthesize_codex_speech",
        lambda *_args, **_kwargs: SimpleNamespace(audio=b"ID3audio"),
    )
    monkeypatch.setattr(tts_streaming.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(
        tts_streaming.subprocess, "Popen", lambda *_args, **_kwargs: proc
    )
    monkeypatch.setattr(tts_streaming, "_STREAM_SENTENCE_BYTE_CAP", 1024)
    streamer = tts_streaming.OpenAICodexStreamer({"provider": "openai-codex"}, {})

    with pytest.raises(ValueError, match="exceeded the per-sentence byte cap"):
        list(streamer.stream("Bound this."))

    proc.terminate.assert_called_once()
