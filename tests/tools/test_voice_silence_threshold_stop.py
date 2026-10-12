"""Bug-fix regression tests for #84046: ``AudioRecorder.stop()`` must apply the CONFIGURED
silence threshold (``self._silence_threshold``, wired from ``voice.silence_threshold`` by the
CLI before each recording), not the hardcoded ``SILENCE_RMS_THRESHOLD`` default of 200 —
otherwise a low-threshold setup (e.g. 80 for a quiet mic that peaks at RMS ~160) auto-stops on
speech and then discards the valid recording as "too quiet".

numpy is absent from the test environment, so captured frames are faked as dicts that carry
just the attributes ``stop()`` touches (``__len__``, concatenation). This is a behavior
contract on the threshold, not on any waveform shape.
"""

import pytest

from tools.voice_mode import AudioRecorder, SILENCE_RMS_THRESHOLD


class _Frame:
    """Minimal frame stand-in: length + concatenation + bytes, everything stop()
    and _write_wav touch. A constant-PCM stand-in would need numpy to build;
    plain zero bytes exercise the write path the same way the old silence
    fixture did."""

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def tobytes(self):
        return b"\x00\x00" * self._n


class _FakeAudio:
    def concatenate(self, frames, axis=0):
        assert axis == 0
        assert len(frames) == 1
        return frames[0]


@pytest.fixture
def mock_audio(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.voice_mode._import_audio", lambda: (None, _FakeAudio()))
    monkeypatch.setattr("tools.voice_mode._TEMP_DIR", str(tmp_path))
    return _FakeAudio()


def _arm(recorder, *, frames, peak_rms):
    """Put the recorder into the state a live capture leaves behind, without
    opening a real InputStream: recording flag on, frames + peak RMS loaded."""
    recorder._recording = True
    recorder._frames = list(frames)
    recorder._peak_rms = peak_rms


class TestStopUsesConfiguredSilenceThreshold:

    def test_stop_keeps_recording_above_configured_threshold(self, mock_audio):
        """peak RMS above the CONFIGURED threshold (but below the 200 default) is
        valid speech and must be returned, not discarded as too quiet."""
        recorder = AudioRecorder()
        recorder._silence_threshold = 80  # the user's voice.silence_threshold
        _arm(recorder, frames=[_Frame(16000)], peak_rms=160)  # above 80, below 200

        assert recorder.stop() is not None

    def test_stop_discards_recording_below_configured_threshold(self, mock_audio):
        """peak RMS below the CONFIGURED threshold is still silence and must be
        discarded — lowering the floor never lets noise through."""
        recorder = AudioRecorder()
        recorder._silence_threshold = 80
        _arm(recorder, frames=[_Frame(16000)], peak_rms=40)  # below 80

        assert recorder.stop() is None

    def test_stop_threshold_tracks_reconfigured_threshold_between_recordings(self, mock_audio):
        """The discard gate follows the live attribute: the same peak RMS that was
        discarded under threshold 80 is kept after the CLI re-wires it to 60 before
        the next recording (per-recording reconfiguration, cli_voice_mixin)."""
        recorder = AudioRecorder()
        recorder._silence_threshold = 80
        _arm(recorder, frames=[_Frame(16000)], peak_rms=70)
        assert recorder.stop() is None  # below 80: discarded

        recorder._silence_threshold = 60
        _arm(recorder, frames=[_Frame(16000)], peak_rms=70)
        assert recorder.stop() is not None  # above 60: kept

    def test_stop_still_discards_below_default_threshold_without_config(self, mock_audio):
        """With no config override the constructor default (200) still applies: RMS
        160 stays a silent recording. Pins the default against the custom floor."""
        recorder = AudioRecorder()
        assert recorder._silence_threshold == SILENCE_RMS_THRESHOLD
        _arm(recorder, frames=[_Frame(16000)], peak_rms=160)

        assert recorder.stop() is None
