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

import shutil
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import patch

import pytest


from tools.transcription_common import BUILTIN_STT_PROVIDERS
from tools.transcription_command import (
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


def _python_echo_input_command() -> str:
    """Return a portable shell command that writes the received {input_path} to {output_path}."""
    interpreter = sys.executable
    payload = "import sys; open(sys.argv[2], 'w', encoding='utf-8').write(sys.argv[1])"
    return f'"{interpreter}" -c "{payload}" {{input_path}} {{output_path}}'


def _python_copy_input_command(copy_dest: str) -> str:
    """Command that copies the received {input_path} to *copy_dest* and echoes the path
    to {output_path} — lets a test inspect the exact bytes the provider received even
    after the runner's temp dir is cleaned up."""
    interpreter = sys.executable
    payload = ("import sys, shutil; shutil.copyfile(sys.argv[1], sys.argv[3]); "
               "open(sys.argv[2], 'w', encoding='utf-8').write(sys.argv[1])")
    return f'"{interpreter}" -c "{payload}" {{input_path}} {{output_path}} {copy_dest}'


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

    def test_silent_capture_is_success_with_empty_transcript(self, tmp_path):
        """Exit-0 provider writing an empty transcript file = no speech, not a failure:
        the voice loop must re-listen quietly like the built-in providers (#132951)."""
        audio = _make_silent_wav(tmp_path / "input.wav")
        cfg = {
            "type": "command",
            "command": _python_emit_command(""),
        }
        result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        assert result["success"] is True
        assert result["transcript"] == ""
        assert result["provider"] == "fake-cli"

    def test_missing_output_file_and_no_stdout_still_fails(self, tmp_path):
        """A provider that exits 0 without writing the output file or stdout broke its
        contract — that stays a failure (distinct from the empty-transcript case)."""
        audio = _make_silent_wav(tmp_path / "input.wav")
        cfg = {
            "type": "command",
            "command": f'"{sys.executable}" -c "pass"',
        }
        result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        assert result["success"] is False
        assert "wrote no output file" in result["error"]

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


# ---------------------------------------------------------------------------
# normalize: opt-in ffmpeg transcode (16 kHz mono) before the command runs
# ---------------------------------------------------------------------------


class TestNormalizeCommandSTTInput:
    """``stt.providers.<name>.normalize: true`` hands the command a 16 kHz mono
    m4a instead of the raw container (desktop voice notes are WebM/Opus 48 kHz;
    providers with container/sample-rate contracts reject the raw file — #81811).

    The flag is opt-in: providers that accept any container (whisper CLIs) must
    keep receiving the original file untouched.
    """

    def test_normalize_false_passes_original_file(self, tmp_path):
        import tools.transcription_command as mod

        audio = _make_silent_wav(tmp_path / "input.wav")
        cfg = {"type": "command", "command": _python_echo_input_command()}
        with patch.object(mod, "_transcode_audio_for_stt") as transcode:
            result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        transcode.assert_not_called()
        assert result["success"] is True
        assert Path(result["transcript"]) == audio

    def test_normalize_true_passes_transcoded_file(self, tmp_path):
        import tools.transcription_command as mod

        audio = _make_silent_wav(tmp_path / "input.wav")
        converted = tmp_path / "converted" / "input-stt.m4a"
        converted.parent.mkdir()
        converted.write_bytes(b"fake m4a")
        cfg = {
            "type": "command",
            "command": _python_echo_input_command(),
            "normalize": True,
        }
        with patch.object(mod, "_transcode_audio_for_stt",
                          return_value=(str(converted), None)) as transcode:
            result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        transcode.assert_called_once()
        assert result["success"] is True
        assert Path(result["transcript"]) == converted
        # The original file is never modified.
        assert audio.read_bytes() != b"fake m4a"

    def test_normalize_transcode_failure_returns_error(self, tmp_path):
        import tools.transcription_command as mod

        audio = _make_silent_wav(tmp_path / "input.wav")
        cfg = {
            "type": "command",
            "command": _python_echo_input_command(),
            "normalize": True,
        }
        with patch.object(mod, "_transcode_audio_for_stt",
                          return_value=(None, "ffmpeg was not found")):
            result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        assert result["success"] is False
        assert "normalize" in result["error"]
        assert "ffmpeg was not found" in result["error"]

    def test_normalize_string_false_passes_original_file(self, tmp_path):
        import tools.transcription_command as mod

        audio = _make_silent_wav(tmp_path / "input.wav")
        cfg = {
            "type": "command",
            "command": _python_echo_input_command(),
            "normalize": "false",
        }
        with patch.object(mod, "_transcode_audio_for_stt") as transcode:
            result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        transcode.assert_not_called()
        assert result["success"] is True
        assert Path(result["transcript"]) == audio


_HAS_FFMPEG = bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


def _make_opus_webm(path: Path) -> Path:
    """Write a minimal WebM/Opus file (48 kHz), like a desktop voice recording."""
    silent_wav = _make_silent_wav(path.with_suffix(".wav"), seconds=0.2)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent_wav),
         "-c:a", "libopus", "-b:a", "32k", str(path)],
        check=True, timeout=60)
    return path


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
class TestNormalizeCommandSTTRealFFmpeg:
    def test_webm_normalized_to_16k_mono_m4a(self, tmp_path):
        audio = _make_opus_webm(tmp_path / "voice.webm")
        received_copy = str(tmp_path / "received.m4a")
        cfg = {
            "type": "command",
            "command": _python_copy_input_command(received_copy),
            "normalize": True,
        }
        result = _transcribe_command_stt(str(audio), "fake-cli", cfg, {})
        assert result["success"] is True, result
        received = Path(result["transcript"])
        assert received != audio
        assert received.suffix == ".m4a"
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels", "-of",
             "default=noprint_wrappers=1:nokey=1", received_copy],
            check=True, capture_output=True, text=True, timeout=30)
        sample_rate, channels = probe.stdout.split()
        assert int(sample_rate) == 16000
        assert int(channels) == 1
