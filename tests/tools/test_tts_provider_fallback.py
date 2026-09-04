"""Regression coverage for opt-in TTS provider fallback.

All synthesis functions are mocked: these tests prove routing and artifact
atomicity without credentials, network requests, or real audio generation.
"""

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from agent import tts_registry
from hermes_cli import plugins
from tools import tts_tool


def _write_audio(_text: str, output_path: str, _config: dict) -> str:
    Path(output_path).write_bytes(b"fallback-audio")
    return output_path


def _fallback_config() -> dict:
    return {"provider": "elevenlabs", "fallback_provider": "openai"}


def test_no_fallback_preserves_primary_dependency_error(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "elevenlabs"})
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: (_ for _ in ()).throw(ImportError()))

    result = json.loads(tts_tool.text_to_speech_tool("hello", str(tmp_path / "out.mp3")))

    assert result["success"] is False
    assert result["error"] == (
        "ElevenLabs provider selected but 'elevenlabs' package not installed. "
        "Run: pip install elevenlabs"
    )


def test_normal_primary_path_uses_shared_dispatch_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "openai"})
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_openai_tts", _write_audio)

    output = tmp_path / "out.mp3"
    result = json.loads(tts_tool.text_to_speech_tool("hello", str(output)))

    assert result["success"] is True
    assert result["provider"] == "openai"
    assert "fallback_from" not in result
    assert output.read_bytes() == b"fallback-audio"


def test_missing_primary_sdk_uses_openai_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_tool, "_load_tts_config", _fallback_config)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_openai_tts", _write_audio)

    result = json.loads(tts_tool.text_to_speech_tool("whole reply", str(tmp_path / "out.mp3")))

    assert result["success"] is True
    assert result["provider"] == "openai"
    assert result["fallback_from"] == "elevenlabs"
    assert "elevenlabs" in result["fallback_error"].lower()
    assert (tmp_path / "out.mp3").read_bytes() == b"fallback-audio"


def test_primary_runtime_failure_retries_complete_text_with_fallback(tmp_path, monkeypatch):
    captured: list[str] = []

    def fail_primary(text: str, output_path: str, _config: dict) -> str:
        Path(output_path).write_bytes(b"partial-primary")
        raise RuntimeError("quota exceeded")

    def fallback(text: str, output_path: str, _config: dict) -> str:
        captured.append(text)
        Path(output_path).write_bytes(b"whole-fallback")
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", _fallback_config)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_elevenlabs", fail_primary)
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fallback)

    whole_text = "first chunk. second chunk. third chunk."
    output = tmp_path / "out.mp3"
    result = json.loads(tts_tool.text_to_speech_tool(whole_text, str(output)))

    assert result["success"] is True
    assert captured == [whole_text]
    assert output.read_bytes() == b"whole-fallback"
    assert not list(tmp_path.glob(".*.primary.*"))
    assert not list(tmp_path.glob(".*.fallback.*"))


def test_fallback_uses_shared_lower_provider_input_cap(tmp_path, monkeypatch, caplog):
    primary_texts: list[str] = []
    fallback_texts: list[str] = []

    def fail_primary(text: str, _output_path: str, _config: dict) -> str:
        primary_texts.append(text)
        raise RuntimeError("quota exceeded")

    def fallback(text: str, output_path: str, _config: dict) -> str:
        fallback_texts.append(text)
        Path(output_path).write_bytes(b"fallback-audio")
        return output_path

    monkeypatch.setattr(tts_tool, "_load_tts_config", _fallback_config)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_elevenlabs", fail_primary)
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fallback)

    text = "A" * 5000  # valid for ElevenLabs' default cap, over OpenAI's 4096
    result = json.loads(tts_tool.text_to_speech_tool(text, str(tmp_path / "out.mp3")))

    expected = text[:4096]
    assert result["success"] is True
    assert primary_texts == [expected]
    assert fallback_texts == [expected]
    assert any("shared script to 4096" in record.message for record in caplog.records)


@pytest.mark.parametrize("fallback", ["ELEVENLABS", "not-a-provider"])
def test_invalid_or_self_referential_fallback_is_skipped(tmp_path, monkeypatch, fallback):
    calls: list[str] = []

    def fail_primary(*_args, **_kwargs):
        calls.append("primary")
        raise RuntimeError("quota exceeded")

    config = {"provider": "elevenlabs", "fallback_provider": fallback}
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_elevenlabs", fail_primary)

    result = json.loads(tts_tool.text_to_speech_tool("hello", str(tmp_path / "out.mp3")))

    assert result["success"] is False
    assert calls == ["primary"]
    assert "quota exceeded" in result["error"]


def test_both_provider_failures_are_reported_without_secrets(tmp_path, monkeypatch):
    secret = "do-not-leak-this-key"

    monkeypatch.setattr(tts_tool, "_load_tts_config", _fallback_config)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: object())
    monkeypatch.setattr(
        tts_tool, "_generate_elevenlabs",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError(f"quota failed Authorization: Token {secret}")
        ),
    )
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
    monkeypatch.setattr(
        tts_tool, "_generate_openai_tts",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("OpenAI billing disabled")),
    )
    monkeypatch.setattr(
        tts_tool, "get_env_value",
        lambda name, default=None: secret if name == "OPENAI_API_KEY" else default,
    )

    result = json.loads(tts_tool.text_to_speech_tool("hello", str(tmp_path / "out.mp3")))

    assert result["success"] is False
    assert "quota failed" in result["error"]
    assert "OpenAI billing disabled" in result["error"]
    assert secret not in result["error"]


def test_preflight_uses_valid_fallback_when_primary_is_unavailable(monkeypatch):
    monkeypatch.setattr(tts_tool, "_load_tts_config", _fallback_config)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
    monkeypatch.setattr(tts_tool, "_has_openai_audio_backend", lambda: True)

    assert tts_tool.check_tts_requirements() is True


def test_registered_plugin_is_a_valid_fallback_without_synthesizing(tmp_path, monkeypatch):
    class PluginProvider:
        voice_compatible = False

        def __init__(self):
            self.synthesize_calls = 0

        def is_available(self):
            return True

        def synthesize(self, text, output_path, **_kwargs):
            self.synthesize_calls += 1
            Path(output_path).write_bytes(b"plugin-audio")
            return output_path

    plugin = PluginProvider()
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tts_registry, "get_provider", lambda name: plugin if name == "mock-plugin" else None,
    )

    assert tts_tool._get_tts_fallback_provider("elevenlabs", {
        "provider": "elevenlabs", "fallback_provider": "mock-plugin",
    }) == "mock-plugin"
    assert plugin.synthesize_calls == 0

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {
        "provider": "elevenlabs", "fallback_provider": "mock-plugin",
    })
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: (_ for _ in ()).throw(ImportError()))
    assert tts_tool.check_tts_requirements() is True

    result = json.loads(tts_tool.text_to_speech_tool("whole reply", str(tmp_path / "out.mp3")))
    assert result["success"] is True
    assert result["provider"] == "mock-plugin"
    assert plugin.synthesize_calls == 1


def test_command_stderr_secret_never_reaches_fallback_metadata(tmp_path, monkeypatch):
    secret = "nonstandard-command-secret"
    config = {
        "provider": "private-command",
        "fallback_provider": "openai",
        "providers": {
            "private-command": {"type": "command", "command": "not-run"},
        },
    }

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
    monkeypatch.setattr(
        tts_tool,
        "_run_command_tts",
        lambda *_args: (_ for _ in ()).throw(
            subprocess.CalledProcessError(23, "not-run", stderr=secret)
        ),
    )
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_openai_tts", _write_audio)

    result = json.loads(tts_tool.text_to_speech_tool("hello", str(tmp_path / "out.mp3")))

    assert result["success"] is True
    assert "exited with code 23" in result["fallback_error"]
    assert secret not in result["fallback_error"]


def test_command_fallback_uses_its_configured_output_format(tmp_path, monkeypatch):
    rendered_paths: list[Path] = []
    config = {
        "provider": "elevenlabs",
        "fallback_provider": "wav-command",
        "providers": {
            "wav-command": {
                "type": "command",
                "command": "render {output_path}",
                "output_format": "wav",
            },
        },
    }

    def fail_primary(*_args, **_kwargs):
        raise RuntimeError("quota exceeded")

    def render_command(command: str, _timeout: float):
        output = Path(shlex.split(command)[1])
        rendered_paths.append(output)
        output.write_bytes(b"wav-audio")

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: config)
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: object())
    monkeypatch.setattr(tts_tool, "_generate_elevenlabs", fail_primary)
    monkeypatch.setattr(tts_tool, "_run_command_tts", render_command)

    requested = tmp_path / "requested.mp3"
    result = json.loads(tts_tool.text_to_speech_tool("hello", str(requested)))

    expected = tmp_path / "requested.wav"
    assert result["success"] is True
    assert result["provider"] == "wav-command"
    assert result["file_path"] == str(expected)
    assert rendered_paths and rendered_paths[0].suffix == ".wav"
    assert expected.read_bytes() == b"wav-audio"
    assert not requested.exists()
