"""macOS output policy for streaming TTS.

On macOS, stream_tts_to_speaker must NOT open a sounddevice OutputStream
(PortAudio/CoreAudio init triggers a kTCCServiceMediaLibrary prompt). It
should route audio through the tempfile/afplay fallback instead.
See PR #62601 / #13291.
"""

import queue
import threading

import pytest


class _FakeStreamer:
    """Minimal chunked-streamer stand-in so the OutputStream setup path runs."""

    sample_rate = 24000
    channels = 1

    def stream(self, text):
        return iter([])


def _run_stream(monkeypatch):
    """Drive stream_tts_to_speaker once with a mock client on the real host.

    Returns True if _import_sounddevice was called during the run.

    No platform parameter: the two callers below are the macOS and non-macOS
    arms of the same policy, and each now runs on a host that reaches its arm
    by itself. Faking ``platform.system()`` here selected the branch without
    reproducing anything underneath it — on Darwin the branch exists because
    PortAudio init raises a TCC prompt, which no Linux runner can produce.
    """
    import tools.tts_tool as tts

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
    monkeypatch.setattr(
        "tools.tts_streaming.resolve_streaming_provider",
        lambda cfg, preferred=None: _FakeStreamer(),
    )

    sd_called = {"hit": False}

    def _spy_import_sd():
        sd_called["hit"] = True
        # OSError, not AssertionError: the function's own guard handles it, so
        # the off-macOS arm can record the call without the raise aborting the
        # run. On macOS the call must never happen at all.
        raise OSError("no audio device in test")

    monkeypatch.setattr("tools.tts_tool._import_sounddevice", _spy_import_sd)

    text_queue: queue.Queue = queue.Queue()
    text_queue.put(None)  # end-of-text sentinel: no sentence spoken
    stop_event = threading.Event()
    done_event = threading.Event()

    tts.stream_tts_to_speaker(text_queue, stop_event, done_event)
    assert done_event.is_set()
    return sd_called["hit"]


@pytest.mark.macos_only
def test_streaming_tts_skips_sounddevice_on_macos(monkeypatch):
    assert _run_stream(monkeypatch) is False


@pytest.mark.linux_only
def test_streaming_tts_uses_sounddevice_off_macos(monkeypatch):
    # Off macOS the OutputStream setup runs; _import_sounddevice raising here
    # is caught by the function's own guard, so the call itself is what we assert.
    assert _run_stream(monkeypatch) is True
