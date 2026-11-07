"""SileroStream reframes the loop's 30 ms blocks into the model's input contract.

The model wants 512 new 16 kHz samples per call, prefixed by the previous call's last 64 samples,
with its recurrent state threaded through. The ONNX session is replaced by a recorder so this runs
without the weights; what's asserted is our buffering against that contract.
"""

import pytest

np = pytest.importorskip("numpy")

from tools.silero_vad import WINDOW, SileroStream


class _RecordingSession:
    def __init__(self, probs=None):
        self.calls = []
        self._probs = list(probs or [])

    def run(self, _outputs, feeds):
        self.calls.append({k: v.copy() for k, v in feeds.items()})
        prob = self._probs.pop(0) if self._probs else 0.0
        return np.array([[prob]], dtype=np.float32), feeds["state"] + 1.0


def _blocks(n, size=480, rate=16000):
    t = np.arange(n * size) / rate
    pcm = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    return [pcm[i * size:(i + 1) * size] for i in range(n)]


def test_blocks_are_reframed_into_contiguous_model_windows():
    session = _RecordingSession()
    stream = SileroStream(session, np)
    blocks = _blocks(16)  # 16 * 480 = 7680 samples = 15 full windows
    for b in blocks:
        stream.feed(b)

    audio = np.concatenate(blocks).astype(np.float32) / 32768.0
    assert len(session.calls) == len(audio) // WINDOW
    for i, call in enumerate(session.calls):
        chunk = call["input"][0]
        assert np.array_equal(chunk[64:], audio[i * WINDOW:(i + 1) * WINDOW])
        previous = session.calls[i - 1]["input"][0][-64:] if i else np.zeros(64, np.float32)
        assert np.array_equal(chunk[:64], previous)
        assert np.all(call["state"] == i)  # each call receives the state the previous returned


def test_probability_is_the_latest_complete_window():
    stream = SileroStream(_RecordingSession(probs=[0.2, 0.9]), np)
    first, second, third = (stream.feed(b) for b in _blocks(3))
    assert first == 0.0          # 480 samples: no full window yet
    assert second == pytest.approx(0.2)   # 960: one window
    assert third == pytest.approx(0.9)    # 1440: two windows


def test_non_16k_input_is_resampled_before_framing():
    session = _RecordingSession()
    stream = SileroStream(session, np, input_rate=48000)
    for b in _blocks(16, size=1440, rate=48000):  # 30 ms at 48 kHz
        stream.feed(b)
    assert len(session.calls) == (16 * 480) // WINDOW
