"""Tests for the OpenAI TTS `instructions` field passthrough.

Covers #14196: forwarding the OpenAI-spec `instructions` parameter through the
`text_to_speech` tool so the agent can control tone/emotion/pacing on
gpt-4o-mini-tts and OpenAI-compatible voice-design servers.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Backend-level passthrough (_generate_openai_tts)
# ---------------------------------------------------------------------------

class TestOpenaiBackendInstructions:
    def _run(self, tmp_path, monkeypatch, *, tts_config=None, instructions=None):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = MagicMock()
        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None)):
            from tools.tts_tool import _generate_openai_tts
            kwargs = {}
            if instructions is not None:
                kwargs["instructions"] = instructions
            _generate_openai_tts(
                "Hello", str(tmp_path / "out.mp3"), tts_config or {}, **kwargs
            )
        return mock_client.audio.speech.create

    def test_instructions_forwarded_when_provided(self, tmp_path, monkeypatch):
        """Tool arg `instructions` is passed to audio.speech.create as-is."""
        create = self._run(tmp_path, monkeypatch, instructions="Speak cheerfully.")
        assert create.call_args[1]["instructions"] == "Speak cheerfully."

    def test_instructions_absent_by_default(self, tmp_path, monkeypatch):
        """No instructions arg -> key not present in create kwargs.

        Preserves behavior on `tts-1`/`tts-1-hd` and strict servers that
        reject unknown kwargs.
        """
        create = self._run(tmp_path, monkeypatch)
        assert "instructions" not in create.call_args[1]

    def test_empty_string_instructions_omitted(self, tmp_path, monkeypatch):
        """Empty string is treated as absent (not forwarded)."""
        create = self._run(tmp_path, monkeypatch, instructions="")
        assert "instructions" not in create.call_args[1]


# ---------------------------------------------------------------------------
# Tool-level plumbing (text_to_speech_tool -> _generate_openai_tts)
# ---------------------------------------------------------------------------

class TestToolLevelInstructions:
    def _invoke_tool(self, tmp_path, monkeypatch, *, instructions=None):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = MagicMock()

        def fake_stream(path):
            # Mimic OpenAI SDK's stream_to_file by writing a tiny payload.
            with open(path, "wb") as f:
                f.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")

        response = MagicMock()
        response.stream_to_file.side_effect = fake_stream
        mock_client.audio.speech.create.return_value = response

        mock_cls = MagicMock(return_value=mock_client)

        with patch("tools.tts_tool._import_openai_client", return_value=mock_cls), \
             patch("tools.tts_tool._resolve_openai_audio_client_config",
                   return_value=("test-key", None)), \
             patch("tools.tts_tool._load_tts_config",
                   return_value={"provider": "openai"}):
            from tools.tts_tool import text_to_speech_tool
            kwargs = {"output_path": str(tmp_path / "out.mp3")}
            if instructions is not None:
                kwargs["instructions"] = instructions
            result = text_to_speech_tool("Hello world", **kwargs)
        return mock_client.audio.speech.create, json.loads(result)

    def test_tool_threads_instructions_to_openai_create(
        self, tmp_path, monkeypatch
    ):
        create, result = self._invoke_tool(
            tmp_path, monkeypatch, instructions="Whisper conspiratorially."
        )
        assert result.get("success") is True
        assert create.call_args[1]["instructions"] == "Whisper conspiratorially."

    def test_tool_omits_instructions_when_not_supplied(
        self, tmp_path, monkeypatch
    ):
        create, result = self._invoke_tool(tmp_path, monkeypatch)
        assert result.get("success") is True
        assert "instructions" not in create.call_args[1]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_exposes_instructions_parameter(self):
        from tools.tts_tool import TTS_SCHEMA
        props = TTS_SCHEMA["parameters"]["properties"]
        assert "instructions" in props
        assert props["instructions"]["type"] == "string"
        # Must stay optional — current behavior must be preserved.
        assert "instructions" not in TTS_SCHEMA["parameters"].get("required", [])
