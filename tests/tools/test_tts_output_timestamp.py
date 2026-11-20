"""Regression test for salvaged PR #43911 — microsecond TTS output timestamps.

Default output paths used second-resolution ``%Y%m%d_%H%M%S`` timestamps, so
two text_to_speech_tool calls landing in the same wall-clock second produced
the same filename and the second synthesis overwrote the first.
"""

import datetime as real_datetime
from types import SimpleNamespace

from tools import tts_tool


def test_default_output_paths_differ_within_the_same_second(tmp_path, monkeypatch):
    stamps = iter([
        real_datetime.datetime(2026, 9, 23, 12, 0, 0, 1000),
        real_datetime.datetime(2026, 9, 23, 12, 0, 0, 2000),
    ])

    class _FrozenDateTime:
        @staticmethod
        def now():
            return next(stamps)

    monkeypatch.setattr(tts_tool, "datetime", SimpleNamespace(datetime=_FrozenDateTime))
    monkeypatch.setattr(tts_tool, "_default_output_dir", lambda: str(tmp_path))

    first, err1 = tts_tool._resolve_output_base(None, "edge", None, False)
    second, err2 = tts_tool._resolve_output_base(None, "edge", None, False)

    assert err1 is None and err2 is None
    assert first.parent == tmp_path and second.parent == tmp_path
    assert first != second, "same-second TTS calls must not share an output file (#43911)"
