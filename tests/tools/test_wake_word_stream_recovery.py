"""Regression for #118001: a read error must not permanently disarm wake detection."""

import threading
import time
from types import SimpleNamespace

import pytest

from tools import wake_word as ww


def test_listener_recovers_after_transient_capture_failure(monkeypatch):
    recovered = threading.Event()
    closed = []
    opened = []
    engine = SimpleNamespace(frame_length=1280, reset=lambda: None,
                             close=lambda: None, process=lambda frame: False)
    detector = ww.WakeWordDetector(engine, lambda: None, input_device="chosen microphone")

    class Capture:
        rate = ww.SAMPLE_RATE

        def read(self, stop):
            if len(opened) == 1:
                raise OSError("transient microphone disconnect")
            recovered.set()
            stop.wait(5)
            return None

        def close(self):
            closed.append(self)

    def open_capture(frame_length):
        assert detector.input_device == "chosen microphone"
        assert frame_length == engine.frame_length
        if opened:
            assert opened[-1] in closed, "close failed stream before reopening"
        capture = Capture()
        opened.append(capture)
        return capture

    monkeypatch.setattr(detector, "_open_capture", open_capture)
    detector.start()
    try:
        assert recovered.wait(3), "listener never reopened capture after a transient read failure"
    finally:
        detector.stop()
    assert all(capture in closed for capture in opened)


@pytest.mark.parametrize("failure", ["read", "open", "cancel", "external"])
def test_capture_recovery_is_bounded_and_cancellable(monkeypatch, failure):
    calls, closed, failures = [], [], []
    engine = SimpleNamespace(frame_length=1280, reset=lambda: None,
                             close=lambda: None, process=lambda frame: False)
    detector = ww.WakeWordDetector(engine, lambda: None, on_failure=failures.append,
                                   external_audio=failure == "external")

    class Capture:
        def read(self, stop):
            raise OSError("device unavailable")

        def close(self):
            closed.append(self)

    def open_capture(frame_length):
        calls.append(frame_length)
        if failure == "open" and len(calls) > 1:
            raise OSError("still unavailable")
        return Capture()

    def wait(delay):
        if failure == "cancel":
            detector._stop.set()
            return True
        return False

    monkeypatch.setattr(detector, "_open_capture", open_capture)
    monkeypatch.setattr(detector._stop, "wait", wait)
    ready, errors = threading.Event(), []
    detector._run(ready, errors)
    assert ready.is_set() and not errors
    if failure in ("cancel", "external"):
        assert len(calls) == 1
    else:
        assert len(calls) == 4, "reopening must consume a finite per-arm retry budget"
    assert len(closed) == (4 if failure == "read" else 1)
    assert failures == ([] if failure == "cancel" else [detector])


def test_halt_releases_reader_wedged_in_recovered_capture(monkeypatch):
    """Interaction of #118001 recovery with the #118552 halt hook: after a read
    failure reopens the microphone, a reader wedged in the NEW capture must still
    be released by pause() — the halt hook must follow the recovered capture."""
    recovered = threading.Event()
    opened, closed = [], []
    engine = SimpleNamespace(frame_length=1280, reset=lambda: None,
                             close=lambda: None, process=lambda frame: False)
    detector = ww.WakeWordDetector(engine, lambda: None)

    class WedgedCapture:
        def __init__(self):
            self._unblock = threading.Event()

        def read(self, stop):
            if not recovered.is_set():
                recovered.set()
                raise OSError("transient microphone disconnect")
            self._unblock.wait(30)
            return None

        def close(self):
            closed.append(self)
            self._unblock.set()

    monkeypatch.setattr(detector, "_open_capture", lambda frame_length: opened.append(WedgedCapture()) or opened[-1])
    monkeypatch.setattr(ww, "_HALT_JOIN_SECONDS", 0.2)
    detector.start()
    try:
        assert recovered.wait(3), "listener never hit the transient read failure"
        deadline = time.monotonic() + 3
        while len(opened) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(opened) == 2, "listener never recovered a fresh capture"
        t0 = time.monotonic()
        detector.pause()
        assert time.monotonic() - t0 < 2.5, "pause() must abort the wedged recovered reader, not block on it"
        assert opened[1] in closed, "halt must abort the RECOVERED capture, not the dead one"
    finally:
        detector.stop()


def _fake_sounddevice(events, opens, fail_opens):
    """Minimal sounddevice stub: device query + InputStream + Pa re-init hooks."""
    from types import SimpleNamespace

    class FakeStream:
        def start(self):
            events.append("start")

        def abort(self):
            events.append("abort")

        def close(self):
            events.append("close")

    class FakeSD:
        def query_devices(self, selector, kind=None):
            return {"name": "hot-plugged mic", "max_input_channels": 1,
                    "default_samplerate": 16000.0, "hostapi": 0}

        def query_hostapis(self, index):
            return {"name": "MME"}

        def InputStream(self, **kwargs):
            opens.append(kwargs.get("device"))
            if len(opens) <= fail_opens:
                raise OSError("PaErrorCode -9986: invalid device")
            return FakeStream()

        def _terminate(self):
            events.append("terminate")

        def _initialize(self):
            events.append("initialize")

    return FakeSD(), SimpleNamespace()


def _wake_detector():
    engine = SimpleNamespace(frame_length=1280, reset=lambda: None,
                             close=lambda: None, process=lambda frame: False)
    return ww.WakeWordDetector(engine, lambda: None), engine


def test_recovery_refreshes_portaudio_snapshot_before_retry(monkeypatch):
    """#131177: recovery retried _open_capture against a stale Pa_Initialize snapshot,
    so a hot-plugged mic kept failing on its dead device ID. Recovery must
    terminate/re-initialize PortAudio before reopening."""
    detector, engine = _wake_detector()
    events, opens = [], []
    sd, np = _fake_sounddevice(events, opens, fail_opens=1)
    monkeypatch.setattr(ww, "_import_audio", lambda: (sd, np))

    with pytest.raises(OSError, match="invalid device"):
        detector._open_capture(engine.frame_length)

    cap = detector._recover_capture(engine.frame_length, iter([0]))
    try:
        assert cap is not None, "recovery must reopen capture after snapshot refresh"
        assert events[:2] == ["terminate", "initialize"], f"re-init must precede reopen: {events}"
        assert events.index("start") > events.index("initialize")
        assert len(opens) == 2
    finally:
        if cap is not None:
            cap.close()


def test_permanently_dead_device_stops_after_exhaustion(monkeypatch):
    """#131177: a permanently dead device must refresh-then-retry a finite number of
    times and then stop — never loop forever."""
    detector, engine = _wake_detector()
    events, opens = [], []
    sd, np = _fake_sounddevice(events, opens, fail_opens=999)
    monkeypatch.setattr(ww, "_import_audio", lambda: (sd, np))

    with pytest.raises(OSError, match="invalid device"):
        detector._open_capture(engine.frame_length)

    assert detector._recover_capture(engine.frame_length, iter([0, 0, 0])) is None
    assert len(opens) == 4, "one initial open + three bounded retries, then stop"
    assert events.count("terminate") == 3 and events.count("initialize") == 3
