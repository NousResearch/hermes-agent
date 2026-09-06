"""
Tests for the STT command-provider registry (``stt.providers.<name>``).

Mirrors ``tests/tools/test_tts_command_providers.py`` — same shape, same
invariants, adapted for the input=audio → output=transcript flow.

Covers:
- Resolution: built-in precedence, missing/unknown name, type/command gating
- Placeholder rendering: shell-quote-aware, doubled-brace preservation
- Helpers: timeout fallback, output_format validation, iter/has-any
- End-to-end via transcribe_audio(): command-provider wins when configured,
  built-ins still win when name collides, plugin coexistence

Nothing here talks to a real STT engine. The shell command writes a static
transcript to ``{output_path}`` using ``python -c`` so the tests run
identically on Linux, macOS, and Windows (with minor quoting differences).
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path
from unittest.mock import patch


from tools.transcription_common import BUILTIN_STT_PROVIDERS
from tools.transcription_command import (
    COMMAND_STT_OUTPUT_FORMATS,
    DEFAULT_COMMAND_STT_LANGUAGE,
    DEFAULT_COMMAND_STT_OUTPUT_FORMAT,
    DEFAULT_COMMAND_STT_TIMEOUT_SECONDS,
    _get_command_stt_output_format,
    _get_command_stt_timeout,
    _get_named_stt_provider_config,
    _render_command_stt_template,
    _resolve_command_stt_provider_config,
    _transcribe_command_stt,
)
from tools.transcription_tools import (
    transcribe_audio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_silent_wav(path: Path, seconds: float = 0.1) -> Path:
    """Write a minimal silent .wav file so _validate_audio_file accepts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        frames = b"\x00\x00" * int(8000 * seconds)
        w.writeframes(frames)
    return path


def _python_emit_command(transcript_text: str, output_placeholder: str = "{output_path}") -> str:
    """Return a portable shell command that writes ``transcript_text`` to {output_path}."""
    interpreter = sys.executable
    # Use repr() to embed the literal string safely; outer single quotes
    # avoid shell expansion of $ / ` / etc.
    payload = (
        "import sys; "
        f"open(sys.argv[1], 'w').write({transcript_text!r})"
    )
    return f'"{interpreter}" -c "{payload}" {output_placeholder}'


def _python_emit_stdout_command(transcript_text: str) -> str:
    """Return a portable shell command that writes transcript to stdout only."""
    interpreter = sys.executable
    payload = f"import sys; sys.stdout.write({transcript_text!r})"
    return f'"{interpreter}" -c "{payload}"'


# ---------------------------------------------------------------------------
# _resolve_command_stt_provider_config / built-in precedence
# ---------------------------------------------------------------------------


class TestResolveCommandSTTProviderConfig:
    def test_builtin_names_are_never_command_providers(self):
        cfg = {
            "providers": {
                "openai": {"type": "command", "command": "echo hi"},
                "groq": {"type": "command", "command": "echo hi"},
                "local": {"type": "command", "command": "echo hi"},
                "local_command": {"type": "command", "command": "echo hi"},
                "mistral": {"type": "command", "command": "echo hi"},
                "xai": {"type": "command", "command": "echo hi"},
            },
        }
        for name in BUILTIN_STT_PROVIDERS:
            assert _resolve_command_stt_provider_config(name, cfg) is None

    def test_missing_provider_returns_none(self):
        cfg = {"providers": {}}
        assert _resolve_command_stt_provider_config("nope", cfg) is None


    def test_resolution_is_case_insensitive(self):
        cfg = {"providers": {"my-cli": {"type": "command", "command": "echo hi"}}}
        assert _resolve_command_stt_provider_config("MY-CLI", cfg) is not None
        assert _resolve_command_stt_provider_config(" my-cli ", cfg) is not None


# ---------------------------------------------------------------------------
# _get_named_stt_provider_config: legacy stt.<name> fallback
# ---------------------------------------------------------------------------


class TestGetNamedSTTProviderConfig:
    def test_canonical_stt_providers_lookup(self):
        cfg = {"providers": {"my-cli": {"command": "whisper {input_path}"}}}
        result = _get_named_stt_provider_config(cfg, "my-cli")
        assert result == {"command": "whisper {input_path}"}


    def test_canonical_wins_over_legacy(self):
        cfg = {
            "providers": {"my-cli": {"command": "canonical"}},
            "my-cli": {"command": "legacy"},
        }
        assert _get_named_stt_provider_config(cfg, "my-cli")["command"] == "canonical"


# ---------------------------------------------------------------------------
# Helpers: timeout / format / iter / has-any
# ---------------------------------------------------------------------------


class TestSTTCommandHelpers:
    def test_timeout_uses_default_when_missing(self):
        assert _get_command_stt_timeout({}) == DEFAULT_COMMAND_STT_TIMEOUT_SECONDS


    def test_output_format_defaults_to_txt(self):
        assert _get_command_stt_output_format({}) == DEFAULT_COMMAND_STT_OUTPUT_FORMAT
        assert DEFAULT_COMMAND_STT_OUTPUT_FORMAT == "txt"


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestRenderCommandSTTTemplate:
    def test_renders_all_placeholders(self):
        rendered = _render_command_stt_template(
            "whisper {input_path} -o {output_path} --lang {language} --model {model}",
            {
                "input_path": "/tmp/audio.wav",
                "output_path": "/tmp/out.txt",
                "output_dir": "/tmp",
                "format": "txt",
                "language": "en",
                "model": "base",
            },
        )
        assert "/tmp/audio.wav" in rendered
        assert "/tmp/out.txt" in rendered
        assert "en" in rendered
        assert "base" in rendered

    def test_preserves_doubled_braces(self):
        rendered = _render_command_stt_template(
            'echo {{"foo": {input_path}}}',
            {"input_path": "audio.wav"},
        )
        # Doubled braces collapse to single braces — JSON snippets survive.
        assert rendered.startswith('echo {"foo":')
        assert rendered.endswith('}')
        assert "audio.wav" in rendered


    def test_placeholder_not_in_dict_passes_through(self):
        # Unknown placeholder isn't replaced — preserves literal text.
        rendered = _render_command_stt_template(
            "echo {unknown_name}",
            {"input_path": "x"},
        )
        assert rendered == "echo {unknown_name}"


# ---------------------------------------------------------------------------
# _transcribe_command_stt: end-to-end via the runner
# ---------------------------------------------------------------------------


class TestTranscribeCommandSTT:
    def test_writes_transcript_to_output_path(self, tmp_path):
        audio = _make_silent_wav(tmp_path / "input.wav")
        cfg = {
            "type": "command",
            "command": _python_emit_command("hello world"),
        }
        result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        assert result["success"] is True
        assert result["transcript"] == "hello world"
        assert result["provider"] == "fake-cli"

    def test_reads_transcript_from_stdout_when_no_file(self, tmp_path):
        audio = _make_silent_wav(tmp_path / "input.wav")
        cfg = {
            "type": "command",
            "command": _python_emit_stdout_command("stdout transcript"),
        }
        result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        assert result["success"] is True
        assert result["transcript"] == "stdout transcript"


    def test_language_defaults_to_en(self, tmp_path):
        audio = _make_silent_wav(tmp_path / "input.wav")
        interpreter = sys.executable
        payload = "import sys; open(sys.argv[2], 'w', encoding='utf-8').write(sys.argv[1])"
        cfg = {
            "command": f'"{interpreter}" -c "{payload}" {{language}} {{output_path}}',
        }
        result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        assert result["transcript"] == DEFAULT_COMMAND_STT_LANGUAGE


# ---------------------------------------------------------------------------
# End-to-end via transcribe_audio(): dispatcher integration
# ---------------------------------------------------------------------------


class TestTranscribeAudioDispatchToCommandProvider:
    """Verify ``transcribe_audio()`` picks command providers correctly.

    These tests bypass the lazy-load STT detection (faster-whisper /
    HERMES_LOCAL_STT_COMMAND) by patching ``_load_stt_config`` directly.
    """

    def _config_with_command_provider(self, name: str, command: str) -> dict:
        return {
            "provider": name,
            "providers": {
                name: {"type": "command", "command": command},
            },
        }

    def test_command_provider_dispatches_via_transcribe_audio(self, tmp_path):
        audio = _make_silent_wav(tmp_path / "audio.wav")
        cfg = self._config_with_command_provider(
            "fake-cli", _python_emit_command("dispatched via command")
        )
        with patch("tools.transcription_tools._load_stt_config", return_value=cfg):
            result = transcribe_audio(str(audio))
        assert result["success"] is True
        assert result["transcript"] == "dispatched via command"
        assert result["provider"] == "fake-cli"


    def test_unknown_provider_no_command_falls_through_to_error(self, tmp_path):
        audio = _make_silent_wav(tmp_path / "audio.wav")
        cfg = {"provider": "unknown-cli"}
        with patch("tools.transcription_tools._load_stt_config", return_value=cfg):
            result = transcribe_audio(str(audio))
        assert result["success"] is False
        # Explicitly-configured unknown providers now get a named
        # registration error instead of the generic legacy message.
        assert result["error_type"] == "provider_not_registered"
        assert "unknown-cli" in result["error"]


# ---------------------------------------------------------------------------
# Command vs plugin precedence
# ---------------------------------------------------------------------------


class TestCommandWinsOverPlugin:
    """When a name has BOTH a command provider AND a registered plugin, the
    command provider must win — same precedence rule as TTS PR #17843
    (config is more local than plugin install).
    """

    def test_command_wins_when_both_configured(self, tmp_path):
        audio = _make_silent_wav(tmp_path / "audio.wav")
        cfg = {
            "provider": "fake-cli",
            "providers": {
                "fake-cli": {
                    "type": "command",
                    "command": _python_emit_command("FROM_COMMAND"),
                },
            },
        }

        # Register a plugin under the SAME name. It must NOT fire.
        from agent.transcription_provider import TranscriptionProvider
        from agent.transcription_registry import (
            _reset_for_tests,
            register_provider,
        )

        class FakePlugin(TranscriptionProvider):
            @property
            def name(self) -> str:
                return "fake-cli"

            def transcribe(self, file_path, *, model=None, language=None, **extra):
                return {
                    "success": True,
                    "transcript": "FROM_PLUGIN",
                    "provider": self.name,
                }

        _reset_for_tests()
        try:
            register_provider(FakePlugin())
            with patch("tools.transcription_tools._load_stt_config", return_value=cfg):
                result = transcribe_audio(str(audio))
        finally:
            _reset_for_tests()

        assert result["success"] is True
        assert result["transcript"] == "FROM_COMMAND"

    def test_plugin_fires_when_no_command_provider(self, tmp_path):
        audio = _make_silent_wav(tmp_path / "audio.wav")
        cfg = {"provider": "fake-plugin"}

        from agent.transcription_provider import TranscriptionProvider
        from agent.transcription_registry import (
            _reset_for_tests,
            register_provider,
        )

        class FakePlugin(TranscriptionProvider):
            @property
            def name(self) -> str:
                return "fake-plugin"

            def transcribe(self, file_path, *, model=None, language=None, **extra):
                return {
                    "success": True,
                    "transcript": "FROM_PLUGIN",
                    "provider": self.name,
                }

        _reset_for_tests()
        try:
            register_provider(FakePlugin())
            with patch("tools.transcription_tools._load_stt_config", return_value=cfg):
                result = transcribe_audio(str(audio))
        finally:
            _reset_for_tests()

        assert result["success"] is True
        assert result["transcript"] == "FROM_PLUGIN"
