"""Tests for the local command TTS provider in tools.tts_tool."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_local_tts_env(monkeypatch):
    monkeypatch.delenv("HERMES_LOCAL_TTS_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)


def _write_synth_script(tmp_path: Path, exit_code: int = 0, write_output: bool = True) -> Path:
    script = tmp_path / "fake_tts_engine.py"
    script.write_text(
        f"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--format", default="")
parser.add_argument("--voice", default="")
parser.add_argument("--model", default="")
parser.add_argument("--speed", default="")
args = parser.parse_args()

if {exit_code!r}:
    print("synth failed on stderr", file=sys.stderr)
    raise SystemExit({exit_code!r})

text = open(args.input, encoding="utf-8").read()
if {write_output!r}:
    payload = (
        f"AUDIO:{{text}}|format={{args.format}}|voice={{args.voice}}|"
        f"model={{args.model}}|speed={{args.speed}}"
    )
    open(args.output, "wb").write(payload.encode("utf-8"))
""",
        encoding="utf-8",
    )
    return script


def _disable_non_local_tts_providers(monkeypatch):
    def _missing():
        raise ImportError()

    monkeypatch.setattr("tools.tts_tool._import_edge_tts", _missing)
    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", _missing)
    monkeypatch.setattr("tools.tts_tool._import_openai_client", _missing)
    monkeypatch.setattr("tools.tts_tool._import_mistral_client", _missing)
    monkeypatch.setattr("tools.tts_tool._check_neutts_available", lambda: False)
    monkeypatch.setattr("tools.tts_tool._check_kittentts_available", lambda: False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)


def test_get_local_tts_command_template_prefers_config_over_env(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_TTS_COMMAND", "env-command {text_path}")

    from tools.tts_tool import _get_local_tts_command_template

    config = {"local_command": {"command": "config-command {text_path}"}}

    assert _get_local_tts_command_template(config) == "config-command {text_path}"


def test_get_local_tts_command_template_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HERMES_LOCAL_TTS_COMMAND", "env-command {text_path}")

    from tools.tts_tool import _get_local_tts_command_template

    assert _get_local_tts_command_template({}) == "env-command {text_path}"


def test_get_local_tts_command_template_returns_none_when_missing():
    from tools.tts_tool import _get_local_tts_command_template

    assert _get_local_tts_command_template({}) is None


def test_generate_local_command_tts_writes_utf8_text_and_passes_output_path(tmp_path):
    from tools.tts_tool import _generate_local_command_tts

    script = _write_synth_script(tmp_path)
    output_path = tmp_path / "speech.mp3"
    config = {
        "local_command": {
            "command": (
                f"{sys.executable} {script} "
                "--input {{text_path}} --output {{output_path}}"
            )
        }
    }

    result = _generate_local_command_tts("café 你好", str(output_path), config)

    assert result == str(output_path)
    assert output_path.read_bytes().startswith("AUDIO:café 你好".encode("utf-8"))


def test_generate_local_command_tts_supports_template_placeholders(tmp_path):
    from tools.tts_tool import _generate_local_command_tts

    script = _write_synth_script(tmp_path)
    output_path = tmp_path / "speech.wav"
    config = {
        "local_command": {
            "command": (
                f"{sys.executable} {script} "
                "--input {{text_path}} --output {{output_path}} "
                "--format {{format}} --voice {{voice}} --model {{model}} --speed {{speed}}"
            ),
            "format": "wav",
            "voice": "Ava",
            "model": "local-small",
            "speed": 1.25,
        }
    }

    _generate_local_command_tts("hello", str(output_path), config)

    assert output_path.read_text(encoding="utf-8") == (
        "AUDIO:hello|format=wav|voice=Ava|model=local-small|speed=1.25"
    )


def test_generate_local_command_tts_output_path_suffix_overrides_config_format(tmp_path):
    from tools.tts_tool import _generate_local_command_tts

    script = _write_synth_script(tmp_path)
    output_path = tmp_path / "speech.wav"
    config = {
        "local_command": {
            "command": (
                f"{sys.executable} {script} "
                "--input {{text_path}} --output {{output_path}} --format {{format}}"
            ),
            "format": "mp3",
        }
    }

    _generate_local_command_tts("hello", str(output_path), config)

    assert output_path.read_text(encoding="utf-8") == (
        "AUDIO:hello|format=wav|voice=|model=|speed="
    )


def test_get_local_tts_output_format_accepts_leading_dot():
    from tools.tts_tool import _get_local_tts_output_format

    config = {"local_command": {"output_format": ".wav"}}

    assert _get_local_tts_output_format(config) == "wav"


def test_generate_local_command_tts_missing_command_raises_value_error(tmp_path):
    from tools.tts_tool import _generate_local_command_tts

    with pytest.raises(ValueError, match="local_command\\.command"):
        _generate_local_command_tts("hello", str(tmp_path / "speech.mp3"), {})


def test_generate_local_command_tts_invalid_placeholder_raises_value_error(tmp_path):
    from tools.tts_tool import _generate_local_command_tts

    config = {"local_command": {"command": "fake-tts {missing_placeholder}"}}

    with pytest.raises(ValueError, match="missing placeholder"):
        _generate_local_command_tts("hello", str(tmp_path / "speech.mp3"), config)


@pytest.mark.parametrize(
    "command_template",
    [
        "fake-tts --input {text_path",
        "fake-tts --input {}",
    ],
)
def test_generate_local_command_tts_invalid_template_raises_value_error(
    tmp_path,
    command_template,
):
    from tools.tts_tool import _generate_local_command_tts

    config = {"local_command": {"command": command_template}}

    with pytest.raises(ValueError, match="invalid template"):
        _generate_local_command_tts("hello", str(tmp_path / "speech.mp3"), config)


def test_generate_local_command_tts_command_failure_includes_stderr(tmp_path):
    from tools.tts_tool import _generate_local_command_tts

    script = _write_synth_script(tmp_path, exit_code=2)
    output_path = tmp_path / "speech.mp3"
    config = {
        "local_command": {
            "command": (
                f"{sys.executable} {script} "
                "--input {{text_path}} --output {{output_path}}"
            )
        }
    }

    with pytest.raises(RuntimeError, match="synth failed on stderr"):
        _generate_local_command_tts("hello", str(output_path), config)


def test_generate_local_command_tts_timeout_mentions_timed_out(tmp_path):
    from tools.tts_tool import _generate_local_command_tts

    config = {"local_command": {"command": "fake-tts --input {text_path} --output {output_path}"}}

    with patch(
        "tools.tts_tool.subprocess.run",
        side_effect=subprocess.TimeoutExpired("tts", 1),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            _generate_local_command_tts("hello", str(tmp_path / "speech.mp3"), config)


def test_generate_local_command_tts_empty_output_raises_runtime_error(tmp_path):
    from tools.tts_tool import _generate_local_command_tts

    script = _write_synth_script(tmp_path, write_output=False)
    output_path = tmp_path / "speech.mp3"
    config = {
        "local_command": {
            "command": (
                f"{sys.executable} {script} "
                "--input {{text_path}} --output {{output_path}}"
            )
        }
    }

    with pytest.raises(RuntimeError, match="produced no output"):
        _generate_local_command_tts("hello", str(output_path), config)


def test_text_to_speech_tool_routes_local_command_and_returns_json_shape(monkeypatch, tmp_path):
    from tools.tts_tool import text_to_speech_tool

    output_path = tmp_path / "speech.mp3"
    calls = {}

    def fake_local_command(text, output_path_arg, tts_config):
        calls["text"] = text
        calls["output_path"] = output_path_arg
        calls["tts_config"] = tts_config
        Path(output_path_arg).write_bytes(b"AUDIO")
        return output_path_arg

    config = {"provider": "local_command", "local_command": {"command": "fake"}}
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)
    monkeypatch.setattr("tools.tts_tool._generate_local_command_tts", fake_local_command)
    _disable_non_local_tts_providers(monkeypatch)
    monkeypatch.setattr("tools.tts_tool._convert_to_opus", lambda _: None)

    result = json.loads(text_to_speech_tool("hello", output_path=str(output_path)))

    assert result["success"] is True
    assert result["file_path"] == str(output_path)
    assert result["media_tag"] == f"MEDIA:{output_path}"
    assert result["provider"] == "local_command"
    assert result["voice_compatible"] is False
    assert calls == {
        "text": "hello",
        "output_path": str(output_path),
        "tts_config": config,
    }


def test_text_to_speech_tool_local_command_ogg_format_without_voice_compatible_stays_attachment(
    monkeypatch,
    tmp_path,
):
    from tools.tts_tool import text_to_speech_tool

    calls = {}

    def fake_local_command(text, output_path_arg, tts_config):
        calls["output_path"] = output_path_arg
        Path(output_path_arg).write_bytes(b"AUDIO")
        return output_path_arg

    def fail_convert_to_opus(_path):
        pytest.fail("local_command should not convert to opus without voice_compatible")

    config = {
        "provider": "local_command",
        "local_command": {"command": "fake", "output_format": "ogg"},
    }
    monkeypatch.setattr("tools.tts_tool.DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)
    monkeypatch.setattr("tools.tts_tool._generate_local_command_tts", fake_local_command)
    monkeypatch.setattr("tools.tts_tool._convert_to_opus", fail_convert_to_opus)
    _disable_non_local_tts_providers(monkeypatch)

    result = json.loads(text_to_speech_tool("hello"))

    assert result["success"] is True
    assert result["file_path"].endswith(".mp3")
    assert calls["output_path"].endswith(".mp3")
    assert result["media_tag"] == f"MEDIA:{result['file_path']}"
    assert result["voice_compatible"] is False


def test_text_to_speech_tool_local_command_voice_compatible_allows_ogg_output(
    monkeypatch,
    tmp_path,
):
    from tools.tts_tool import text_to_speech_tool

    def fake_local_command(text, output_path_arg, tts_config):
        Path(output_path_arg).write_bytes(b"AUDIO")
        return output_path_arg

    config = {
        "provider": "local_command",
        "local_command": {
            "command": "fake",
            "output_format": "ogg",
            "voice_compatible": True,
        },
    }
    monkeypatch.setattr("tools.tts_tool.DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: config)
    monkeypatch.setattr("tools.tts_tool._generate_local_command_tts", fake_local_command)
    _disable_non_local_tts_providers(monkeypatch)

    result = json.loads(text_to_speech_tool("hello"))

    assert result["success"] is True
    assert result["file_path"].endswith(".ogg")
    assert result["media_tag"] == f"[[audio_as_voice]]\nMEDIA:{result['file_path']}"
    assert result["voice_compatible"] is True


def test_text_to_speech_tool_returns_helpful_json_error_when_command_missing(monkeypatch, tmp_path):
    from tools.tts_tool import text_to_speech_tool

    monkeypatch.setattr(
        "tools.tts_tool._load_tts_config",
        lambda: {"provider": "local_command", "local_command": {}},
    )
    _disable_non_local_tts_providers(monkeypatch)

    result = json.loads(text_to_speech_tool("hello", output_path=str(tmp_path / "speech.mp3")))

    assert result["success"] is False
    assert "local_command.command" in result["error"]


def test_check_tts_requirements_true_when_local_command_configured(monkeypatch):
    from tools.tts_tool import check_tts_requirements

    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {
        "local_command": {"command": "fake-tts {text_path} {output_path}"}
    })
    _disable_non_local_tts_providers(monkeypatch)

    assert check_tts_requirements() is True


def test_check_tts_requirements_false_when_no_provider_available(monkeypatch):
    from tools.tts_tool import check_tts_requirements

    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    _disable_non_local_tts_providers(monkeypatch)

    assert check_tts_requirements() is False
