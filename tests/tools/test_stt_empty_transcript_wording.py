"""Regression: command-STT no-output error must say "empty transcript".

That wording is load-bearing: the desktop transcribe endpoint and the CLI
voice loop treat an empty-transcript error as silence and re-listen instead
of surfacing a failure on every quiet gap. See #80998.
"""
import pytest
from pathlib import Path
from tools.transcription_command import _read_command_stt_output


def test_no_output_raises_empty_transcript_error(tmp_path):
    with pytest.raises(RuntimeError, match="returned an empty transcript"):
        _read_command_stt_output(tmp_path / "missing.txt", "", "text")


def test_stdout_still_captured(tmp_path):
    assert _read_command_stt_output(tmp_path / "missing.txt", "  hi  ", "text") == "hi"
