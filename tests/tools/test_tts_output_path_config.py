"""text_to_speech honors the configured ``tts.output_path`` (with ``~`` expansion).

Tests ``_resolve_tts_output_dir`` directly — the single helper both synthesis
entry points use to pick the output directory — rather than inspecting source.
"""
from pathlib import Path

from tools.tts_tool import DEFAULT_OUTPUT_DIR, _resolve_tts_output_dir


def test_output_path_is_honored():
    assert _resolve_tts_output_dir({"output_path": "/custom/dir"}) == Path("/custom/dir")


def test_output_path_expands_tilde():
    assert _resolve_tts_output_dir({"output_path": "~/tts"}) == Path.home() / "tts"


def test_empty_output_path_falls_back_to_default():
    assert _resolve_tts_output_dir({"output_path": ""}) == Path(DEFAULT_OUTPUT_DIR)


def test_missing_output_path_falls_back_to_default():
    assert _resolve_tts_output_dir({}) == Path(DEFAULT_OUTPUT_DIR)


def test_none_config_falls_back_to_default():
    assert _resolve_tts_output_dir(None) == Path(DEFAULT_OUTPUT_DIR)
