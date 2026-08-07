"""Regression test for salvaged PR #43911 — microsecond TTS output timestamps.

Default output paths used second-resolution ``%Y%m%d_%H%M%S`` timestamps, so
two ``text_to_speech_tool`` calls landing in the same wall-clock second
produced the same filename and the second synthesis overwrote the first. The
format now appends ``%f`` (microseconds).
"""

import datetime
import json
import re
from pathlib import Path

import pytest

from tools import tts_tool


class TestDefaultOutputTimestampResolution:
    @pytest.fixture()
    def edge_tts(self, tmp_path, monkeypatch):
        """Point default output at a temp dir and stub synthesis."""
        out_dir = tmp_path / "voice-memos"
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(out_dir))
        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())

        async def _write(_text, output_path, _cfg):
            Path(output_path).write_bytes(b"mp3")
            return output_path

        monkeypatch.setattr(tts_tool, "_generate_edge_tts", _write)
        return out_dir

    def test_two_calls_in_the_same_second_do_not_collide(self, edge_tts):
        """The behaviour #43911 fixed: back-to-back calls keep both files.

        With second-resolution timestamps the second synthesis silently
        overwrote the first.
        """
        first = json.loads(tts_tool.text_to_speech_tool("one"))
        second = json.loads(tts_tool.text_to_speech_tool("two"))

        assert first["success"] and second["success"]
        assert first["file_path"] != second["file_path"], (
            "two calls in the same second produced the same default filename "
            "— the later synthesis would overwrite the earlier one (#43911)"
        )
        assert len(list(edge_tts.iterdir())) == 2

    def test_default_filename_carries_a_microsecond_component(self, edge_tts):
        result = json.loads(tts_tool.text_to_speech_tool("hello"))

        name = Path(result["file_path"]).stem
        assert re.fullmatch(r"tts_\d{8}_\d{6}_\d{6}", name), (
            f"default TTS filename lost its microsecond component: {name}"
        )

    def test_timestamp_component_is_filename_safe(self):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        assert re.fullmatch(r"\d{8}_\d{6}_\d{6}", stamp)
