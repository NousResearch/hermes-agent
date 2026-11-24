"""Edge word times for lip-sync are collected beside the tool's result, never in it."""

import json
from pathlib import Path
from unittest.mock import patch

from tools.tts_tool import text_to_speech_tool
from tools.tts_tool_providers import edge_words


class _Edge:
    class Communicate:
        def __init__(self, text, **kwargs):
            self.kwargs = kwargs

        async def save(self, path):
            assert "boundary" not in self.kwargs
            Path(path).write_bytes(b"abcd")

        async def stream(self):
            yield {"type": "WordBoundary", "offset": 1_000_000, "duration": 2_500_000, "text": "Hello"}
            yield {"type": "audio", "data": b"abcd"}


def test_words_are_collected_only_when_asked_for(tmp_path):
    def speak(name):
        return json.loads(text_to_speech_tool(text="Hello", output_path=str(tmp_path / name)))

    with patch("tools.tts_tool._import_edge_tts", return_value=_Edge), \
            patch("tools.tts_tool._load_tts_config", return_value={"provider": "edge"}):
        plain = speak("plain.mp3")
        words = ["from the previous piece"]
        token = edge_words.set(words)
        try:
            timed = speak("timed.mp3")
        finally:
            edge_words.reset(token)

    assert words == [{"text": "Hello", "kind": "word", "startMs": 146.0, "endMs": 396.0}]
    assert set(timed) == set(plain)
    assert (tmp_path / "timed.mp3").read_bytes() == b"abcd"
