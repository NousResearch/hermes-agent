import importlib.util
import shlex
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_voice_command_tts.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_voice_command_tts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_config_uses_explicit_ogg_opus_voice_command():
    script = _load_script_module()

    config = script.build_config(
        provider="kokoro",
        voice_bin="/tmp/bin/voice with spaces",
        voice="af_heart",
        speed="1.0",
        timeout=180.0,
    )

    assert "provider: kokoro" in config
    assert "output_format: ogg" in config
    assert "voice_compatible: true" in config
    assert "--format ogg-opus" in config
    assert "--input-file {input_path}" in config
    assert "--output {output_path}" in config
    assert shlex.quote("/tmp/bin/voice with spaces") in config


def test_parse_ffprobe_output_extracts_stream_fields():
    script = _load_script_module()

    assert script.parse_ffprobe("codec_name=opus\nsample_rate=48000\nchannels=1\n") == {
        "codec_name": "opus",
        "sample_rate": "48000",
        "channels": "1",
    }


def test_existing_config_path_requires_config_yaml(tmp_path: Path):
    script = _load_script_module()

    with pytest.raises(SystemExit) as excinfo:
        script.existing_config_path(tmp_path)

    assert "config.yaml does not exist" in str(excinfo.value)


def test_existing_config_path_returns_deployed_config(tmp_path: Path):
    script = _load_script_module()
    config = tmp_path / "config.yaml"
    config.write_text("tts:\n  provider: kokoro\n", encoding="utf-8")

    assert script.existing_config_path(tmp_path) == config


def test_validate_result_uses_expected_provider(tmp_path: Path):
    script = _load_script_module()
    audio = tmp_path / "reply.ogg"
    audio.write_bytes(b"OggS")

    result = {
        "provider": "kokoro",
        "voice_compatible": True,
        "media_tag": f"[[audio_as_voice]]\nMEDIA:{audio}",
        "file_path": str(audio),
    }
    probe = {"codec_name": "opus", "sample_rate": "48000", "channels": "1"}

    script.validate_result(result, probe, expected_provider="kokoro")

    with pytest.raises(SystemExit) as excinfo:
        script.validate_result(result, probe, expected_provider="other")

    assert "expected provider other" in str(excinfo.value)
