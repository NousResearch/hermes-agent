import importlib.util
import shlex
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_voice_command_stt.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_voice_command_stt", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_config_uses_voice_stream_transcribe_command():
    script = _load_script_module()

    config = script.build_config(
        provider="voice",
        voice_bin="/tmp/bin/voice with spaces",
        timeout=300.0,
    )

    assert "enabled: true" in config
    assert "provider: voice" in config
    assert "type: command" in config
    assert "format: txt" in config
    assert "stream-transcribe --quiet {input_path}" in config
    assert shlex.quote("/tmp/bin/voice with spaces") in config


def test_existing_config_path_requires_config_yaml(tmp_path: Path):
    script = _load_script_module()

    with pytest.raises(SystemExit) as excinfo:
        script.existing_config_path(tmp_path)

    assert "config.yaml does not exist" in str(excinfo.value)


def test_existing_config_path_returns_deployed_config(tmp_path: Path):
    script = _load_script_module()
    config = tmp_path / "config.yaml"
    config.write_text("stt:\n  provider: voice\n", encoding="utf-8")

    assert script.existing_config_path(tmp_path) == config


def test_transcript_text_accepts_transcript_or_text_field():
    script = _load_script_module()

    assert script.transcript_text({"transcript": "hello"}) == "hello"
    assert script.transcript_text({"text": "world"}) == "world"
    assert script.transcript_text({}) == ""


def test_validate_result_requires_expected_provider_and_words():
    script = _load_script_module()
    result = {"provider": "voice", "transcript": "Hello world."}

    script.validate_result(
        result,
        expected_provider="voice",
        expected_words=["hello", "world"],
    )

    with pytest.raises(SystemExit) as excinfo:
        script.validate_result(
            result,
            expected_provider="other",
            expected_words=["hello"],
        )

    assert "expected provider other" in str(excinfo.value)

    with pytest.raises(SystemExit) as excinfo:
        script.validate_result(
            result,
            expected_provider="voice",
            expected_words=["missing"],
        )

    assert "expected transcript to contain" in str(excinfo.value)
