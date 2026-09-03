"""Tests for config-authoritative TTS provider selection (#90109).

``tts.provider`` in config.yaml is the only backend selector: the
model-facing ``text_to_speech`` schema must not advertise a per-call
``provider`` override, and a per-call value that disagrees with the
configured provider is ignored (with a warning) instead of rerouting
speech to another vendor.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _fake_key(tag: str) -> str:
    return "-".join(("tts", tag, "test", "key"))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)
    # The ignored-override warning is once-per-process per distinct value;
    # reset it so one test's emission never masks the next test's assertion.
    from tools import tts_tool as _tts_mod

    _tts_mod._once_warnings.clear()


class _OpenaiBackendStub:
    """Patch set that routes the configured ``openai`` provider through a
    mocked OpenAI client writing a tiny MP3 payload (same shape as
    tests/tools/test_tts_instructions.py)."""

    def __init__(self):
        self.mock_client = MagicMock()

        def fake_stream(path):
            Path(path).write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00")

        response = MagicMock()
        response.stream_to_file.side_effect = fake_stream
        self.mock_client.audio.speech.create.return_value = response

        mock_cls = MagicMock(return_value=self.mock_client)
        self.xai_patch = patch("tools.tts_tool._generate_xai_tts")
        self.xai_generator = self.xai_patch.start()
        self._patches = [
            patch("tools.tts_tool._import_openai_client", return_value=mock_cls),
            patch(
                "tools.tts_tool._resolve_openai_audio_client_config",
                return_value=(_fake_key("openai"), None, False),
            ),
            patch(
                "tools.tts_tool._load_tts_config",
                return_value={"provider": "openai"},
            ),
        ]
        for p in self._patches:
            p.start()

    @property
    def create(self):
        return self.mock_client.audio.speech.create

    def stop(self):
        for p in self._patches:
            p.stop()
        self.xai_patch.stop()


@pytest.fixture
def openai_backend():
    stub = _OpenaiBackendStub()
    yield stub
    stub.stop()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_does_not_advertise_provider(self):
        from tools.tts_tool import TTS_SCHEMA
        props = TTS_SCHEMA["parameters"]["properties"]
        assert "provider" not in props, (
            "The model-facing text_to_speech schema must not advertise a "
            "provider override: tts.provider in config.yaml is the only "
            "backend selector (#90109)."
        )


# ---------------------------------------------------------------------------
# Tool-level enforcement (text_to_speech_tool)
# ---------------------------------------------------------------------------

class TestToolLevelProviderAuthority:
    def test_disagreeing_override_is_ignored(
        self, tmp_path, monkeypatch, openai_backend
    ):
        """A per-call provider that disagrees with config is not honored:
        the configured openai backend runs and xAI never fires."""
        from tools.tts_tool import text_to_speech_tool

        result = json.loads(
            text_to_speech_tool(
                "Hello world",
                output_path=str(tmp_path / "out.mp3"),
                provider="xai",
            )
        )
        assert result.get("success") is True
        assert result.get("provider") == "openai"
        openai_backend.create.assert_called_once()
        openai_backend.xai_generator.assert_not_called()

    def test_disagreeing_override_logs_warning(
        self, tmp_path, monkeypatch, openai_backend, caplog
    ):
        from tools.tts_tool import text_to_speech_tool

        with caplog.at_level(logging.WARNING, logger="tools.tts_tool"):
            text_to_speech_tool(
                "Hello world",
                output_path=str(tmp_path / "out.mp3"),
                provider="xai",
            )
        assert any(
            "Ignoring per-call TTS provider override" in rec.message
            for rec in caplog.records
        ), "An ignored provider override must be visible in the logs."

    def test_matching_override_is_accepted(
        self, tmp_path, monkeypatch, openai_backend, caplog
    ):
        """A per-call provider that agrees with config stays a no-op."""
        from tools.tts_tool import text_to_speech_tool

        with caplog.at_level(logging.WARNING, logger="tools.tts_tool"):
            result = json.loads(
                text_to_speech_tool(
                    "Hello world",
                    output_path=str(tmp_path / "out.mp3"),
                    provider="openai",
                )
            )
        assert result.get("success") is True
        assert result.get("provider") == "openai"
        assert not any(
            "Ignoring per-call TTS provider override" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Registered handler (the model-args path)
# ---------------------------------------------------------------------------

class TestRegisteredHandler:
    def test_handler_does_not_forward_provider_arg(
        self, tmp_path, monkeypatch, openai_backend
    ):
        """Even if a model (or a leaked platform hint) sends provider= in
        its tool args, the registered handler must not forward it."""
        from tools.registry import registry

        entry = registry.get_entry("text_to_speech")
        assert entry is not None
        result = json.loads(
            entry.handler(
                {
                    "text": "Hello world",
                    "output_path": str(tmp_path / "out.mp3"),
                    "provider": "xai",
                }
            )
        )
        assert result.get("success") is True
        assert result.get("provider") == "openai"
        openai_backend.xai_generator.assert_not_called()


# ---------------------------------------------------------------------------
# Inner single-chunk helper
# ---------------------------------------------------------------------------

class TestSingleChunkHelperAuthority:
    def test_single_helper_ignores_disagreeing_provider(
        self, tmp_path, monkeypatch, openai_backend
    ):
        from tools.tts_tool import _text_to_speech_single

        result = json.loads(
            _text_to_speech_single(
                "Hello world",
                str(tmp_path / "out.mp3"),
                provider="xai",
                tts_config_override={"provider": "openai"},
            )
        )
        assert result.get("success") is True
        assert result.get("provider") == "openai"
        openai_backend.xai_generator.assert_not_called()
