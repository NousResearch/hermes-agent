"""Tests for stream_tts_to_speaker after multi-provider refactor.

The function consumes str deltas from a queue, buffers them into
sentences, and pipes each sentence through _iter_pcm_chunks to a
sounddevice OutputStream (or a tempfile fallback). Provider selection
is parametric — _iter_pcm_chunks handles the provider-specific SDK
calls.

The streaming playback path decodes chunks via numpy.frombuffer
(int16 PCM); miniaudio is used on the edge path for MP3 decoding.
Skip this module entirely when numpy isn't installed so the tests
don't fail on lean CI environments.
"""

import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy")


def _drain_queue(sentences):
    """Build a queue preloaded with sentence deltas + end sentinel."""
    q = queue.Queue()
    for s in sentences:
        q.put(s)
    q.put(None)
    return q


def _make_sd_mock():
    """Build a mock sounddevice module whose OutputStream() succeeds."""
    mock_sd = MagicMock()
    mock_output = MagicMock()
    mock_sd.OutputStream.return_value = mock_output
    return mock_sd, mock_output


def test_provider_param_forwarded_to_iter_pcm_chunks():
    """provider kwarg reaches _iter_pcm_chunks on every spoken sentence."""
    from tools.tts_tool import stream_tts_to_speaker

    received = []

    def fake_iter(text, provider, tts_config):
        received.append((text, provider))
        yield b"\x00\x00" * 100

    mock_sd, _ = _make_sd_mock()
    q = _drain_queue(["Hello world."])
    stop = threading.Event()
    done = threading.Event()

    with patch("tools.tts_tool._import_sounddevice", return_value=mock_sd), \
         patch("tools.tts_tool._probe_streaming_deps"), \
         patch("tools.tts_tool._iter_pcm_chunks", side_effect=fake_iter):
        stream_tts_to_speaker(q, stop, done, provider="openai")

    assert received, "expected at least one _iter_pcm_chunks call"
    assert all(p == "openai" for _, p in received)


def test_audio_skipped_when_sounddevice_missing_but_display_runs():
    """No sounddevice → display_callback still fires, _iter_pcm_chunks never called."""
    from tools.tts_tool import stream_tts_to_speaker

    displayed = []
    q = _drain_queue(["Hello world."])
    stop = threading.Event()
    done = threading.Event()

    with patch("tools.tts_tool._import_sounddevice", side_effect=ImportError), \
         patch("tools.tts_tool._probe_streaming_deps"), \
         patch("tools.tts_tool._iter_pcm_chunks") as mock_iter:
        stream_tts_to_speaker(
            q, stop, done,
            display_callback=lambda s: displayed.append(s),
            provider="edge",
        )

    assert displayed == ["Hello world."]
    mock_iter.assert_not_called()


def test_stop_event_aborts_chunk_consumption():
    """Setting stop_event mid-iteration breaks the chunk loop cleanly."""
    from tools.tts_tool import stream_tts_to_speaker

    stop = threading.Event()

    def fake_iter(text, provider, tts_config):
        # Yield enough chunks to give stop_event time to fire
        for _ in range(50):
            yield b"\x00\x01" * 1000

    mock_sd, mock_output = _make_sd_mock()
    q = _drain_queue(["Sentence."])
    done = threading.Event()

    # Trip stop_event on the first write
    def _trip_on_write(*args, **kwargs):
        stop.set()
    mock_output.write.side_effect = _trip_on_write

    with patch("tools.tts_tool._import_sounddevice", return_value=mock_sd), \
         patch("tools.tts_tool._probe_streaming_deps"), \
         patch("tools.tts_tool._iter_pcm_chunks", side_effect=fake_iter):
        stream_tts_to_speaker(q, stop, done, provider="elevenlabs")

    # The function returned without error; stop was honoured
    assert stop.is_set()


def test_probe_failure_disables_audio_without_crash():
    """If provider deps are missing, _probe_streaming_deps raises and audio is skipped."""
    from tools.tts_tool import stream_tts_to_speaker

    mock_sd, _ = _make_sd_mock()
    q = _drain_queue(["Hello."])
    stop = threading.Event()
    done = threading.Event()

    with patch("tools.tts_tool._import_sounddevice", return_value=mock_sd), \
         patch("tools.tts_tool._probe_streaming_deps",
               side_effect=ImportError("elevenlabs not installed")), \
         patch("tools.tts_tool._iter_pcm_chunks") as mock_iter:
        stream_tts_to_speaker(q, stop, done, provider="elevenlabs")

    mock_iter.assert_not_called()


# ---------------------------------------------------------------------------
# Token-streaming suppression for voice mode (avoid double-box display)
# ---------------------------------------------------------------------------


def test_suppress_token_streaming_returns_restore_callable():
    """When use_streaming_tts=True, suppress stream_delta_callback and return a restore."""
    from tools.tts_tool import suppress_token_streaming_for_voice

    agent = MagicMock()
    original_cb = MagicMock(name="original_callback")
    agent.stream_delta_callback = original_cb

    restore = suppress_token_streaming_for_voice(agent, use_streaming_tts=True)

    assert callable(restore)
    assert agent.stream_delta_callback is None

    restore()

    assert agent.stream_delta_callback is original_cb


def test_suppress_token_streaming_noop_when_streaming_off():
    """When use_streaming_tts=False, no suppression."""
    from tools.tts_tool import suppress_token_streaming_for_voice

    agent = MagicMock()
    original_cb = MagicMock(name="original_callback")
    agent.stream_delta_callback = original_cb

    restore = suppress_token_streaming_for_voice(agent, use_streaming_tts=False)

    assert restore is None
    assert agent.stream_delta_callback is original_cb


def test_suppress_token_streaming_noop_when_agent_lacks_callback():
    """When agent has no stream_delta_callback, no suppression."""
    from tools.tts_tool import suppress_token_streaming_for_voice

    agent = MagicMock()
    agent.stream_delta_callback = None

    restore = suppress_token_streaming_for_voice(agent, use_streaming_tts=True)

    assert restore is None


def test_suppress_token_streaming_restore_is_idempotent():
    """Calling restore() twice is safe (no error, no double-restore side effect)."""
    from tools.tts_tool import suppress_token_streaming_for_voice

    agent = MagicMock()
    original_cb = MagicMock(name="original_callback")
    agent.stream_delta_callback = original_cb

    restore = suppress_token_streaming_for_voice(agent, use_streaming_tts=True)
    restore()
    # Overwrite with a sentinel to detect double-restore
    sentinel = MagicMock(name="sentinel")
    agent.stream_delta_callback = sentinel
    restore()  # should be no-op

    assert agent.stream_delta_callback is sentinel
