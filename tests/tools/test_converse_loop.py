"""Unit tests for the off-device voice loop primitives.

Pure/hermetic: numpy only, no sockets, no models, no audio devices.
Covers the VAD trip/bleed behaviour and the :class:`_NetworkMicStream` framing +
endpointing shim.
"""

from __future__ import annotations

import threading
import wave

import numpy as np
import pytest

from hermes_cli.web_routers._converse_loop import _NetworkMicStream, ConverseSession
from tools import voice_mode as vm


# ── _BargeDetector (via voice_mode; ConverseSession mirrors its wiring) ──

def _make_detector():
    return vm._BargeDetector(
        np, mult=vm.DEFAULT_BARGE_MULTIPLIER,
        calib_blocks=max(1, 450 // 30), trip_blocks=max(1, 300 // 30),
        grace_blocks=max(0, 500 // 30),
    )


def test_barge_detector_trips_on_speech_after_quiet_floor():
    det = _make_detector()
    # Calibrate on a quiet room (RMS well under the silence threshold).
    for _ in range(20):
        assert det.feed(80.0, playing=False) is None
    # A sustained burst of speech-level RMS (>> floor * multiplier) must trip.
    tripped = None
    for _ in range(20):
        phase = det.feed(6000.0, playing=False)
        if phase is not None:
            tripped = phase
            break
    assert tripped == "generation"


def test_barge_detector_ignores_bleed_during_grace_window():
    det = _make_detector()
    for _ in range(20):
        det.feed(80.0, playing=False)
    # Playback starts -> grace window opens. Speaker bleed (moderate RMS, below
    # the playback trigger clamp) during the grace window must NOT trip.
    tripped = False
    for _ in range(det.grace_blocks):
        # Bleed-level: above the quiet floor but below PLAYBACK_MIN_TRIGGER.
        if det.feed(1200.0, playing=True) is not None:
            tripped = True
    assert not tripped


# ── _NetworkMicStream framing ──

def test_network_mic_stream_reads_exact_blocks_from_odd_chunks():
    stop = threading.Event()
    stream = _NetworkMicStream(np, stop=stop)
    # Feed 1000 int16 samples in awkward byte-sized chunks.
    samples = np.arange(1000, dtype=np.int16)
    raw = samples.tobytes()
    for start in range(0, len(raw), 7):  # 7-byte chunks split samples across feeds
        stream.feed(raw[start:start + 7])

    block = 480
    got = []
    for _ in range(2):
        data, overflow = stream.read(block)
        assert overflow is False
        assert data.dtype == np.int16
        assert data.shape == (block,)
        got.append(data)
    # The two 480-sample reads reproduce the first 960 samples in order.
    np.testing.assert_array_equal(np.concatenate(got), samples[:960])


def test_network_mic_stream_zero_pads_and_returns_on_stop():
    stop = threading.Event()
    stream = _NetworkMicStream(np, stop=stop, poll_seconds=0.01)
    stream.feed(np.arange(100, dtype=np.int16).tobytes())
    stop.set()
    data, overflow = stream.read(480)
    assert overflow is False
    assert data.shape == (480,)
    # First 100 samples preserved, remainder zero-padded.
    np.testing.assert_array_equal(data[:100], np.arange(100, dtype=np.int16))
    assert np.all(data[100:] == 0)


# ── endpointing over the shim (canned speech-then-silence) ──

def test_capture_until_quiet_produces_wav_from_shim(tmp_path):
    stop = threading.Event()
    stream = _NetworkMicStream(np, stop=stop, poll_seconds=0.01)
    block = int(vm.SAMPLE_RATE * 0.03)  # 480

    # A few speech-level blocks then enough silence to endpoint.
    speech = (np.ones(block, dtype=np.int16) * 8000)
    silence = np.zeros(block, dtype=np.int16)
    endpoint_blocks = max(1, 1250 // 30)
    stream.feed(speech.tobytes())
    stream.feed(speech.tobytes())
    for _ in range(endpoint_blocks + 2):
        stream.feed(silence.tobytes())

    from collections import deque

    pre_roll: deque = deque(maxlen=4)
    wav_path = vm._capture_until_quiet(
        stream, np, block, pre_roll,
        endpoint_blocks=endpoint_blocks, max_blocks=max(1, 30_000 // 30),
    )
    assert wav_path
    with wave.open(wav_path, "rb") as wf:
        assert wf.getframerate() == vm.SAMPLE_RATE
        assert wf.getnchannels() == 1
        assert wf.getnframes() > 0
    vm._unlink_quietly(wav_path)


# ── ConverseSession barge-in / playing flag ──

def test_converse_session_barge_in_sets_interrupt_and_stops_tts():
    session = ConverseSession(np)
    tts_stop = threading.Event()
    session.set_playing(True, tts_stop=tts_stop)
    assert session.playing() is True
    session._trigger_barge_in()
    assert tts_stop.is_set()
    assert session.take_interrupted() is True
    # Popping the flag clears it.
    assert session.take_interrupted() is False
    assert session.playing() is False


def test_converse_session_stop_pushes_sentinel():
    session = ConverseSession(np)
    session.stop()
    assert session.stopped is True
    assert session.transcripts.get_nowait() is None
