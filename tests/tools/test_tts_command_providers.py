"""
Tests for custom command-type TTS providers.

These tests cover the ``tts.providers.<name>`` registry: built-in
precedence, command resolution, placeholder rendering, shell-quote
context handling, timeout / failure cleanup, voice_compatible opt-in,
and max_text_length lookup.

Nothing here talks to a real TTS engine. The shell command itself is
portable: we write bytes to ``{output_path}`` using ``python -c`` so
the tests run identically on Linux, macOS, and (with minor quoting
differences) Windows.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    COMMAND_TTS_OUTPUT_FORMATS,
    DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH,
    DEFAULT_COMMAND_TTS_OUTPUT_FORMAT,
    DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS,
    _generate_command_tts,
    _get_command_tts_output_format,
    _get_command_tts_timeout,
    _get_named_provider_config,
    _has_any_command_tts_provider,
    _is_command_provider_config,
    _is_command_tts_voice_compatible,
    _iter_command_providers,
    _render_command_tts_template,
    _resolve_command_provider_config,
    _resolve_max_text_length,
    _shell_quote_context,
    check_tts_requirements,
    text_to_speech_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python_copy_command(output_placeholder: str = "{output_path}") -> str:
    """Return a cross-platform shell command that copies {input_path} -> output."""
    interpreter = sys.executable
    return (
        f'"{interpreter}" -c "import shutil, sys; '
        f'shutil.copyfile(sys.argv[1], sys.argv[2])" '
        f'{{input_path}} {output_placeholder}'
    )


# ---------------------------------------------------------------------------
# _resolve_command_provider_config / built-in precedence
# ---------------------------------------------------------------------------

class TestResolveCommandProviderConfig:
    def test_builtin_names_are_never_command_providers(self):
        cfg = {
            "providers": {
                "openai": {"type": "command", "command": "echo hi"},
                "edge": {"type": "command", "command": "echo hi"},
            },
        }
        for name in BUILTIN_TTS_PROVIDERS:
            assert _resolve_command_provider_config(name, cfg) is None

    def test_missing_provider_returns_none(self):
        cfg = {"providers": {}}
        assert _resolve_command_provider_config("nope", cfg) is None

    def test_user_declared_command_provider_resolves(self):
        cfg = {
            "providers": {
                "piper-cli": {"type": "command", "command": "piper-cli foo"},
            },
        }
        resolved = _resolve_command_provider_config("piper-cli", cfg)
        assert resolved is not None
        assert resolved["command"] == "piper-cli foo"

    def test_type_command_is_implied_when_command_is_set(self):
        cfg = {"providers": {"piper-cli": {"command": "piper-cli foo"}}}
        resolved = _resolve_command_provider_config("piper-cli", cfg)
        assert resolved is not None

    def test_other_type_values_reject(self):
        cfg = {"providers": {"piper-cli": {"type": "python", "command": "piper-cli foo"}}}
        assert _resolve_command_provider_config("piper-cli", cfg) is None

    def test_empty_command_rejects(self):
        cfg = {"providers": {"piper-cli": {"type": "command", "command": "   "}}}
        assert _resolve_command_provider_config("piper-cli", cfg) is None

    def test_case_insensitive_lookup(self):
        cfg = {"providers": {"piper-cli": {"type": "command", "command": "x"}}}
        assert _resolve_command_provider_config("PIPER-CLI", cfg) is not None

    def test_native_piper_cannot_be_shadowed_by_command_entry(self):
        """Regression guard for PR that added native Piper as a built-in.
        A user's ``tts.providers.piper`` must not override the built-in."""
        cfg = {
            "providers": {
                "piper": {"type": "command", "command": "some-script"},
            },
        }
        assert _resolve_command_provider_config("piper", cfg) is None


class TestGetNamedProviderConfig:
    def test_providers_block_wins(self):
        cfg = {"providers": {"voxcpm": {"command": "new"}},
               "voxcpm": {"command": "legacy"}}
        assert _get_named_provider_config(cfg, "voxcpm") == {"command": "new"}

    def test_legacy_tts_name_block_still_resolves(self):
        cfg = {"voxcpm": {"type": "command", "command": "legacy"}}
        assert _get_named_provider_config(cfg, "voxcpm") == {
            "type": "command", "command": "legacy"
        }

    def test_builtin_names_do_not_leak_through_legacy_path(self):
        """``tts.openai`` must never be mistaken for a command provider."""
        cfg = {"openai": {"command": "oops", "type": "command"}}
        assert _get_named_provider_config(cfg, "openai") == {}


class TestIsCommandProviderConfig:
    def test_empty_dict_is_false(self):
        assert _is_command_provider_config({}) is False

    def test_non_dict_is_false(self):
        assert _is_command_provider_config("foo") is False
        assert _is_command_provider_config(None) is False

    def test_type_mismatch_is_false(self):
        assert _is_command_provider_config({"type": "native", "command": "x"}) is False


# ---------------------------------------------------------------------------
# _iter_command_providers / _has_any_command_tts_provider
# ---------------------------------------------------------------------------

class TestIterCommandProviders:
    def test_iterates_only_user_command_providers(self):
        cfg = {
            "providers": {
                "openai": {"type": "command", "command": "shouldnt show up"},
                "piper-cli": {"type": "command", "command": "piper-cli"},
                "voxcpm": {"type": "command", "command": "voxcpm"},
                "broken": {"type": "command", "command": ""},
            },
        }
        names = sorted(name for name, _ in _iter_command_providers(cfg))
        assert names == ["piper-cli", "voxcpm"]

    def test_has_any_command_provider_detects_declared(self):
        cfg = {"providers": {"piper-cli": {"type": "command", "command": "piper-cli"}}}
        assert _has_any_command_tts_provider(cfg) is True

    def test_has_any_command_provider_when_none(self):
        assert _has_any_command_tts_provider({"providers": {}}) is False
        assert _has_any_command_tts_provider({}) is False


# ---------------------------------------------------------------------------
# config getters
# ---------------------------------------------------------------------------

class TestConfigGetters:
    def test_timeout_defaults(self):
        assert _get_command_tts_timeout({}) == float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)

    def test_timeout_coerces_string(self):
        assert _get_command_tts_timeout({"timeout": "45"}) == 45.0

    def test_timeout_rejects_non_positive(self):
        assert _get_command_tts_timeout({"timeout": 0}) == float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)
        assert _get_command_tts_timeout({"timeout": -1}) == float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)

    def test_timeout_rejects_garbage(self):
        assert _get_command_tts_timeout({"timeout": "fast"}) == float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)

    def test_timeout_seconds_alias(self):
        assert _get_command_tts_timeout({"timeout_seconds": 90}) == 90.0

    def test_output_format_defaults(self):
        assert _get_command_tts_output_format({}) == DEFAULT_COMMAND_TTS_OUTPUT_FORMAT

    def test_output_format_path_override(self):
        assert _get_command_tts_output_format({}, "/tmp/clip.wav") == "wav"

    def test_output_format_unknown_path_falls_back_to_config(self):
        assert _get_command_tts_output_format({"format": "ogg"}, "/tmp/clip.xyz") == "ogg"

    def test_output_format_rejects_unknown(self):
        assert _get_command_tts_output_format({"format": "m4a"}) == DEFAULT_COMMAND_TTS_OUTPUT_FORMAT

    def test_output_format_supported_set(self):
        assert COMMAND_TTS_OUTPUT_FORMATS == frozenset({"mp3", "wav", "ogg", "flac"})

    def test_voice_compatible_boolean(self):
        assert _is_command_tts_voice_compatible({"voice_compatible": True}) is True
        assert _is_command_tts_voice_compatible({"voice_compatible": False}) is False

    def test_voice_compatible_string(self):
        assert _is_command_tts_voice_compatible({"voice_compatible": "yes"}) is True
        assert _is_command_tts_voice_compatible({"voice_compatible": "0"}) is False

    def test_voice_compatible_default_off(self):
        assert _is_command_tts_voice_compatible({}) is False


# ---------------------------------------------------------------------------
# _resolve_max_text_length for command providers
# ---------------------------------------------------------------------------

class TestMaxTextLengthForCommandProviders:
    def test_default_for_command_provider(self):
        cfg = {"providers": {"piper-cli": {"type": "command", "command": "x"}}}
        assert _resolve_max_text_length("piper-cli", cfg) == DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH

    def test_override_under_providers(self):
        cfg = {"providers": {"piper-cli": {"type": "command", "command": "x", "max_text_length": 2500}}}
        assert _resolve_max_text_length("piper-cli", cfg) == 2500

    def test_override_under_legacy_tts_name_block(self):
        cfg = {"piper-cli": {"type": "command", "command": "x", "max_text_length": 7777}}
        assert _resolve_max_text_length("piper-cli", cfg) == 7777

    def test_non_command_unknown_provider_still_falls_back(self):
        assert _resolve_max_text_length("unknown", {}) > 0


# ---------------------------------------------------------------------------
# _shell_quote_context / template rendering
# ---------------------------------------------------------------------------

class TestShellQuoteContext:
    def test_bare_context(self):
        tpl = 'tts {output_path}'
        pos = tpl.index("{output_path}")
        assert _shell_quote_context(tpl, pos) is None

    def test_inside_single_quotes(self):
        tpl = "tts '{output_path}'"
        pos = tpl.index("{output_path}")
        assert _shell_quote_context(tpl, pos) == "'"

    def test_inside_double_quotes(self):
        tpl = 'tts "{output_path}"'
        pos = tpl.index("{output_path}")
        assert _shell_quote_context(tpl, pos) == '"'

    def test_escaped_double_quote_inside_double(self):
        tpl = r'tts "foo \" {output_path}"'
        pos = tpl.index("{output_path}")
        assert _shell_quote_context(tpl, pos) == '"'


class TestRenderCommandTtsTemplate:
    def test_substitutes_all_placeholders(self):
        placeholders = {
            "input_path": "/tmp/in.txt",
            "text_path": "/tmp/in.txt",
            "output_path": "/tmp/out.mp3",
            "format": "mp3",
            "voice": "af_sky",
            "model": "tiny",
            "speed": "1.0",
        }
        rendered = _render_command_tts_template(
            "tts --voice {voice} --in {input_path} --out {output_path}",
            placeholders,
        )
        assert "af_sky" in rendered
        assert "/tmp/out.mp3" in rendered

    def test_quotes_paths_with_spaces(self):
        placeholders = {
            "input_path": "/tmp/Jane Doe/in.txt",
            "text_path": "/tmp/Jane Doe/in.txt",
            "output_path": "/tmp/out.mp3",
            "format": "mp3",
            "voice": "",
            "model": "",
            "speed": "1.0",
        }
        rendered = _render_command_tts_template(
            "tts --in {input_path} --out {output_path}",
            placeholders,
        )
        # shlex.quote wraps space-containing paths in single quotes on POSIX.
        if os.name != "nt":
            assert "'/tmp/Jane Doe/in.txt'" in rendered

    def test_literal_braces_survive(self):
        placeholders = {
            "input_path": "/tmp/in.txt", "text_path": "/tmp/in.txt",
            "output_path": "/tmp/out.mp3", "format": "mp3",
            "voice": "", "model": "", "speed": "1.0",
        }
        rendered = _render_command_tts_template(
            "echo '{{not a placeholder}}' && tts --in {input_path}",
            placeholders,
        )
        assert "{not a placeholder}" in rendered

    def test_injection_is_neutralized(self):
        """Embedded shell metacharacters in a placeholder value must be quoted."""
        placeholders = {
            "input_path": "/tmp/in.txt", "text_path": "/tmp/in.txt",
            "output_path": "/tmp/out; rm -rf /",
            "format": "mp3",
            "voice": "$(whoami)", "model": "", "speed": "1.0",
        }
        rendered = _render_command_tts_template(
            "tts --voice {voice} --out {output_path}",
            placeholders,
        )
        # The injection payload must not appear unquoted in the rendered
        # command. On POSIX shlex.quote wraps the value in single quotes.
        if os.name != "nt":
            assert "'$(whoami)'" in rendered or "'\\''" in rendered
            assert "; rm -rf /" not in rendered.replace(
                "'/tmp/out; rm -rf /'", "",
            )

    def test_preserves_shell_quoting_style(self):
        placeholders = {
            "input_path": "/tmp/in.txt", "text_path": "/tmp/in.txt",
            "output_path": "/tmp/out.mp3", "format": "mp3",
            "voice": "bob's voice", "model": "", "speed": "1.0",
        }
        # When the template wraps the placeholder in double quotes we must
        # escape for that context, not collapse to single-quoted form.
        rendered = _render_command_tts_template(
            'tts --voice "{voice}"',
            placeholders,
        )
        assert '"bob\'s voice"' in rendered


# ---------------------------------------------------------------------------
# End-to-end: _generate_command_tts
# ---------------------------------------------------------------------------

class TestGenerateCommandTts:
    def test_writes_output_file(self, tmp_path):
        out = tmp_path / "clip.mp3"
        config = {"command": _python_copy_command()}
        result = _generate_command_tts(
            "hello world",
            str(out),
            "py-copy",
            config,
            {},
        )
        assert result == str(out)
        assert out.exists()
        # The command copied the input text file over to output, so it
        # contains the original UTF-8 text.
        assert out.read_text(encoding="utf-8") == "hello world"

    def test_empty_command_raises(self, tmp_path):
        with pytest.raises(ValueError, match="is not configured"):
            _generate_command_tts(
                "hello",
                str(tmp_path / "x.mp3"),
                "empty",
                {"command": "  "},
                {},
            )

    def test_nonzero_exit_raises_runtime(self, tmp_path):
        config = {"command": f'"{sys.executable}" -c "import sys; sys.exit(3)"'}
        with pytest.raises(RuntimeError, match="exited with code 3"):
            _generate_command_tts(
                "hello",
                str(tmp_path / "x.mp3"),
                "failing",
                config,
                {},
            )

    def test_empty_output_raises_runtime(self, tmp_path):
        # This command completes successfully but writes nothing.
        config = {"command": f'"{sys.executable}" -c "pass"'}
        with pytest.raises(RuntimeError, match="produced no output"):
            _generate_command_tts(
                "hello",
                str(tmp_path / "x.mp3"),
                "silent",
                config,
                {},
            )

    @pytest.mark.skipif(os.name == "nt", reason="POSIX-only timeout semantics")
    def test_timeout_raises_runtime(self, tmp_path):
        config = {
            "command": f'"{sys.executable}" -c "import time; time.sleep(10)"',
            "timeout": 1,
        }
        with pytest.raises(RuntimeError, match="timed out"):
            _generate_command_tts(
                "hello",
                str(tmp_path / "x.mp3"),
                "slow",
                config,
                {},
            )


# ---------------------------------------------------------------------------
# text_to_speech_tool integration
# ---------------------------------------------------------------------------

class TestTextToSpeechToolWithCommandProvider:
    def test_command_provider_dispatches_end_to_end(self, tmp_path):
        cfg = {
            "tts": {
                "provider": "py-copy",
                "providers": {
                    "py-copy": {
                        "type": "command",
                        "command": _python_copy_command(),
                        "output_format": "mp3",
                    },
                },
            },
        }
        out = tmp_path / "clip.mp3"

        # Patch the config loader used by the tool so we don't touch disk.
        def fake_load():
            return cfg["tts"]

        with patch("tools.tts_tool._load_tts_config", fake_load):
            result = text_to_speech_tool(text="hi", output_path=str(out))
        data = json.loads(result)
        assert data["success"] is True, data
        assert data["provider"] == "py-copy"
        assert data["voice_compatible"] is False
        assert Path(data["file_path"]).exists()

    def test_voice_compatible_opt_in_toggles_flag(self, tmp_path):
        """voice_compatible=true is reflected in the response when the
        file is already .ogg (no ffmpeg needed)."""
        cfg = {
            "provider": "py-copy-ogg",
            "providers": {
                "py-copy-ogg": {
                    "type": "command",
                    "command": _python_copy_command(),
                    "output_format": "ogg",
                    "voice_compatible": True,
                },
            },
        }
        out = tmp_path / "clip.ogg"

        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result = text_to_speech_tool(text="hi", output_path=str(out))
        data = json.loads(result)
        assert data["success"] is True
        assert data["voice_compatible"] is True
        assert data["media_tag"].startswith("[[audio_as_voice]]")

    def test_missing_command_falls_through_to_builtin(self, tmp_path):
        """A provider entry with an empty command is not a command
        provider; the tool should not raise a "command not configured"
        error but fall through to the built-in resolution path."""
        cfg = {
            "provider": "broken",
            "providers": {
                "broken": {"type": "command", "command": "   "},
            },
        }
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "x.mp3"))
        data = json.loads(result)
        # The response should not carry the command-provider error text.
        err = (data.get("error") or "").lower()
        assert "tts.providers.broken.command is not configured" not in err


class TestCheckTtsRequirements:
    def test_configured_command_provider_satisfies_requirement(self):
        cfg = {"providers": {"x": {"type": "command", "command": "echo x"}}}
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            assert check_tts_requirements() is True
