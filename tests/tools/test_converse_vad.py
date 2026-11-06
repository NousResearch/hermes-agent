"""Speech detection contracts for the converse loop (tools/voice_converse_vad.py).

Silero's inference is replaced by a scripted probability per block; the hysteresis, timing and
echo rules around it run for real.
"""

import pytest

np = pytest.importorskip("numpy")

from tools import voice_mode as vm
from tools.voice_converse_vad import (
    MIN_SPEECH_MS, SPEECH_OFF, VAD_ENERGY, VAD_SILERO, _EnergyDetector, _SileroDetector,
    make_detector, parse_vad_config)

_BLOCK = 480  # 30 ms at 16 kHz
_BLOCKS_TO_TRIP = int(np.ceil(MIN_SPEECH_MS / 30))


class _ScriptedStream:
    """Stands in for SileroStream: returns the next scripted probability per block."""

    def __init__(self, probs):
        self._probs = list(probs)

    def feed(self, block):
        return self._probs.pop(0)


def _block(level):
    return np.full(_BLOCK, level, dtype=np.int16)


def _run(probs, levels=None, *, playing=False, tail=False):
    """Feed one block per probability; return the index and phase of the first trip."""
    levels = levels or [1000] * len(probs)
    det = _SileroDetector(_ScriptedStream(probs), np, vm, input_rate=16000)
    for i, level in enumerate(levels):
        phase = det.feed(_block(level), playing=playing, tail=tail)
        if phase:
            return i, phase
    return None


def test_onset_needs_sustained_speech_regardless_of_level():
    # A whisper-level block is speech if the model says so; a short burst is not.
    assert _run([0.9] * _BLOCKS_TO_TRIP, levels=[150] * _BLOCKS_TO_TRIP) == (
        _BLOCKS_TO_TRIP - 1, "generation")
    assert _run([0.9] * (_BLOCKS_TO_TRIP - 2) + [0.1] * 10) is None


def test_hysteresis_keeps_a_started_utterance_through_dips():
    # Once speaking, a dip between the off (0.35) and on (0.5) points doesn't reset the run.
    probs = [0.9, 0.9, 0.42, 0.42] + [0.9] * _BLOCKS_TO_TRIP
    assert _run(probs) is not None
    first_trip = _run(probs)[0]
    assert first_trip < len(probs) - 1  # tripped before the probe ran out, dips included
    # Starting from silence, the same 0.42 is not speech.
    assert _run([0.42] * 20) is None


def test_reply_echo_never_barges_in_during_playback_or_tail():
    # The model hears the reply's own echo as speech (p ~ 1). At the echo's level it must not
    # trip, during playback AND during the tail after turn_done (the client is still playing).
    echo = [1200, 1500, 900, 1400] * 15
    for window in ({"playing": True}, {"tail": True}):
        assert _run([1.0] * len(echo), levels=echo, **window) is None


def test_speech_over_the_echo_barges_in():
    echo = [1200, 1500, 900, 1400] * 9        # ~1 s of reply: echo level learned
    voice = [4000] * 20                       # the user talking over it, well above the echo
    result = _run([1.0] * (len(echo) + len(voice)), levels=echo + voice, playing=True)
    assert result is not None
    index, phase = result
    assert phase == "playback" and index >= len(echo)


def test_no_barge_in_before_the_echo_is_measured():
    # At the start of playback there is no echo reference yet; loud speech-like audio there is
    # the reply starting, so it can't barge in until the echo has been measured.
    assert _run([1.0] * 12, levels=[6000] * 12, playing=True) is None


def test_endpoint_quiet_is_below_the_off_point():
    det = _SileroDetector(_ScriptedStream([SPEECH_OFF - 0.01, SPEECH_OFF]), np, vm,
                          input_rate=16000)
    assert det.is_quiet(_block(0)) is True
    assert det.is_quiet(_block(0)) is False


def test_silero_falls_back_to_energy_with_a_reason(monkeypatch):
    monkeypatch.setattr("tools.silero_vad.load_silero_session",
                        lambda: (None, "onnxruntime is not installed"))
    det, reason = make_detector(VAD_SILERO, np, vm, input_rate=16000, energy_kwargs=dict(
        mult=3.0, calib_blocks=15, trip_blocks=10, grace_blocks=16, trip_fraction=0.6,
        floor_percentile=50.0, onset_mult=1.5, endpoint_floor_mult=1.3))
    assert isinstance(det, _EnergyDetector) and reason == "onnxruntime is not installed"


def test_vad_config_defaults_to_silero():
    assert parse_vad_config({}) == VAD_SILERO
    assert parse_vad_config({"voice": {"vad": {"model": "Energy"}}}) == VAD_ENERGY
    assert parse_vad_config({"voice": {"vad": {"model": "something-else"}}}) == VAD_SILERO
