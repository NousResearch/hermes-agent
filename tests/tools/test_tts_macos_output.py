"""macOS output policy for streaming TTS.

On macOS, stream_tts_to_speaker must NOT open a sounddevice OutputStream
(PortAudio/CoreAudio init triggers a kTCCServiceMediaLibrary prompt). It
should route audio through the tempfile/afplay fallback instead.
See PR #62601 / #13291.
"""

import queue
import threading

import pytest


def _run_stream(monkeypatch, system_name):
    """Drive stream_tts_to_speaker once with a mock client on *system_name*.

    Returns True if _import_sounddevice was called during the run.
    """
    import tools.tts_tool as tts

    monkeypatch.setattr("tools.tts_tool.platform.system", lambda: system_name)
    monkeypatch.setattr("tools.tts_tool.get_env_value",
                        lambda name, default=None: "fake-key"
                        if name == "ELEVENLABS_API_KEY" else default)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})

    class _FakeTTS:
        def __init__(self, *a, **k):
            self.text_to_speech = self

        def convert(self, *a, **k):
            return iter([])  # no audio chunks needed for setup assertion

    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: _FakeTTS)

    sd_called = {"hit": False}

    def _spy_import_sd():
        sd_called["hit"] = True
        raise AssertionError("sounddevice must not be imported for output on macOS")

    monkeypatch.setattr("tools.tts_tool._import_sounddevice", _spy_import_sd)

    text_queue: queue.Queue = queue.Queue()
    text_queue.put(None)  # end-of-text sentinel: no sentence spoken
    stop_event = threading.Event()
    done_event = threading.Event()

    tts.stream_tts_to_speaker(text_queue, stop_event, done_event)
    assert done_event.is_set()
    return sd_called["hit"]


def test_streaming_tts_skips_sounddevice_on_macos(monkeypatch):
    assert _run_stream(monkeypatch, "Darwin") is False


def test_streaming_tts_uses_sounddevice_off_macos(monkeypatch):
    # Off macOS the OutputStream setup runs; _import_sounddevice raising here
    # is caught by the function's own guard, so the call itself is what we assert.
    called = _run_stream_offmac(monkeypatch)
    assert called is True


def _run_stream_offmac(monkeypatch):
    """Like _run_stream but tolerant of the sounddevice import being attempted."""
    import tools.tts_tool as tts

    monkeypatch.setattr("tools.tts_tool.platform.system", lambda: "Linux")
    monkeypatch.setattr("tools.tts_tool.get_env_value",
                        lambda name, default=None: "fake-key"
                        if name == "ELEVENLABS_API_KEY" else default)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})

    class _FakeTTS:
        def __init__(self, *a, **k):
            self.text_to_speech = self

        def convert(self, *a, **k):
            return iter([])

    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: _FakeTTS)

    sd_called = {"hit": False}

    def _spy_import_sd():
        sd_called["hit"] = True
        raise OSError("no audio device in test")  # handled by the function's guard

    monkeypatch.setattr("tools.tts_tool._import_sounddevice", _spy_import_sd)

    text_queue: queue.Queue = queue.Queue()
    text_queue.put(None)
    stop_event = threading.Event()
    done_event = threading.Event()

    tts.stream_tts_to_speaker(text_queue, stop_event, done_event)
    assert done_event.is_set()
    return sd_called["hit"]
