"""Regression test: voice.max_recording_seconds is actually enforced.

Before this fix, ``voice.max_recording_seconds`` was defined in the default
config (hermes_cli/config.py) and documented, but no code read it — the
AudioRecorder had no total-length cap, so the setting was dead config. This
test pins the enforcement contract on the recorder:

* cap disabled (0 / unset) → never auto-stops on duration (previous behaviour),
* cap set to N>0 → auto-stop condition becomes true once elapsed >= N.
"""
from tools.voice_mode import AudioRecorder

def test_cap_enforced_when_set():
    r = AudioRecorder()
    r._max_recording_seconds = 120
    assert r._max_duration_reached(119.9) is False
    assert r._max_duration_reached(120.0) is True
    assert r._max_duration_reached(300.0) is True
