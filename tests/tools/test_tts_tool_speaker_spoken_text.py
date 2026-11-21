"""``stream_tts_to_speaker`` fires ``transform_spoken_text`` once per sentence (issue #152238).

The plugin dispatch is faked at ``hermes_cli.plugins``; synthesis goes through the sync
per-sentence path with a fake ``text_to_speech_tool`` and no audio device.
"""

from __future__ import annotations

import queue
import sys
import threading
from unittest.mock import MagicMock, patch

from tools import tts_tool
from tools.tts_tool_speaker import stream_tts_to_speaker


def _fake_spoken_hook(monkeypatch, reply):
    calls = []

    def _invoke(hook_name, **kwargs):
        if hook_name != "transform_spoken_text":
            return []
        calls.append(kwargs)
        return [reply(kwargs)]

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "transform_spoken_text")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    return calls


def _speak(monkeypatch, deltas, **kwargs):
    synthesized = []

    def fake_synth(text, output_path):
        synthesized.append(text)
        with open(output_path, "wb") as fh:
            fh.write(b"x" * 100)

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_synth)
    monkeypatch.setitem(sys.modules, "tools.voice_mode", MagicMock())
    text_queue = queue.Queue()
    for delta in [*deltas, None]:
        text_queue.put(delta)
    with patch("tools.tts_streaming.resolve_streaming_provider", return_value=None):
        stream_tts_to_speaker(text_queue, threading.Event(), threading.Event(), **kwargs)
    return synthesized


SENTENCES = ["The first sentence is here. ", "The second sentence is here. "]


def test_each_sentence_is_transformed_with_its_index_and_session(monkeypatch):
    calls = _fake_spoken_hook(monkeypatch, lambda kw: f"[{kw['index']}] {kw['text']}")
    spoken = _speak(monkeypatch, SENTENCES, session_id="sess-9", platform="cli")

    assert spoken == ["[0] The first sentence is here.", "[1] The second sentence is here."]
    assert [(c["surface"], c["platform"], c["session_id"]) for c in calls] == [("streaming", "cli", "sess-9")] * 2


def test_empty_replacement_skips_only_that_sentence(monkeypatch):
    _fake_spoken_hook(monkeypatch, lambda kw: "" if kw["index"] == 0 else None)

    assert _speak(monkeypatch, SENTENCES) == ["The second sentence is here."]


def test_display_callback_still_shows_the_original_sentence(monkeypatch):
    _fake_spoken_hook(monkeypatch, lambda kw: "")
    shown = []

    assert _speak(monkeypatch, SENTENCES, display_callback=shown.append) == []
    assert [s.strip() for s in shown] == [s.strip() for s in SENTENCES]


def test_broken_hook_dispatch_still_speaks_every_sentence(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", MagicMock(side_effect=RuntimeError("boom")))

    assert _speak(monkeypatch, SENTENCES) == [s.strip() for s in SENTENCES]
