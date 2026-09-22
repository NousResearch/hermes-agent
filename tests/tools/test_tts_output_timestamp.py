"""Regression test for salvaged PR #43911 — microsecond TTS output timestamps.

Default output paths used second-resolution ``%Y%m%d_%H%M%S`` timestamps, so
two text_to_speech_tool calls landing in the same wall-clock second produced
the same filename and the second synthesis overwrote the first. The format
now appends ``%f`` (microseconds).
"""

import datetime
import re


class TestDefaultOutputTimestampResolution:
    def test_default_output_path_uses_microsecond_timestamp(self):
        """The source must build default filenames with a %f component."""
        import inspect

        from tools import tts_tool

        src = inspect.getsource(tts_tool._resolve_output_base)
        assert "%Y%m%d_%H%M%S_%f" in src, (
            "default TTS output timestamp lost its microsecond component — "
            "concurrent calls in the same second would collide again (#43911)"
        )


    def test_timestamp_component_is_filename_safe(self):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        assert re.fullmatch(r"\d{8}_\d{6}_\d{6}", stamp)
