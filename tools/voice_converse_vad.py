"""Speech detection for the converse voice loop: one interface, two detectors.

:class:`ConverseSession` asks a detector two things per 30 ms block: did speech start
(:meth:`feed`), and, while capturing an utterance, is this block quiet (:meth:`is_quiet`).

* :class:`_SileroDetector` (default) decides speech with the Silero model, using the model
  authors' published operating points rather than room-level constants.
* :class:`_EnergyDetector` is the RMS-threshold detector the loop used before; it is the fallback
  when Silero can't load and behaves exactly as before.

Both see ``playing`` (reply audio is streaming) and ``tail`` (the reply was sent but the client
is still playing it). Silero hears the reply's own echo as speech, so in either window it also
requires the mic level to clear the echo level measured in that window — see
:meth:`_SileroDetector.feed`.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Silero's published defaults: speech starts at P >= 0.5 and continues until P < 0.35
# (threshold - 0.15), and a speech run shorter than 250 ms is not speech.
SPEECH_ON = 0.5
SPEECH_OFF = 0.35
MIN_SPEECH_MS = 250.0
# While echo is possible, a block counts as the user's speech only if the mic level over the last
# ~300 ms clears the echo's own level by this ratio (~3.5 dB). Levels are 300 ms window RMS, not
# per block, so the dips between syllables (of the user and of the echo) don't break the run; the
# echo level is the 90th percentile of the echo's window RMS. With client-side echo cancellation
# the echo level is just the room, and any speech clears it.
ECHO_MARGIN = 1.5
_ECHO_PERCENTILE = 90.0
_LEVEL_WINDOW_BLOCKS = 10  # ~300 ms
_ECHO_WINDOW_BLOCKS = 100  # ~3 s of window levels
# Blocks of echo to measure before barge-in is possible at the start of playback (~500 ms).
_ECHO_MIN_BLOCKS = 16
# The echo level is measured on blocks older than this (~300 ms, longer than MIN_SPEECH_MS), so
# the user's own barge-in can't raise the level it has to clear before it trips.
_ECHO_LAG_BLOCKS = 10

VAD_SILERO = "silero"
VAD_ENERGY = "energy"


class _EnergyDetector:
    """RMS-threshold detection: the loop's original behavior, kept as the fallback."""

    name = VAD_ENERGY

    def __init__(self, np: Any, vm: Any, *, mult: float, calib_blocks: int, trip_blocks: int,
                 grace_blocks: int, trip_fraction: float, floor_percentile: float,
                 onset_mult: float, endpoint_floor_mult: float) -> None:
        self._np, self._vm = np, vm
        self._endpoint_floor_mult = endpoint_floor_mult
        self._detector = vm._BargeDetector(
            np, mult=mult, calib_blocks=calib_blocks, trip_blocks=trip_blocks,
            grace_blocks=grace_blocks, trip_fraction=trip_fraction,
            floor_percentile=floor_percentile, onset_mult=onset_mult, onset_ceiling=None)

    def feed(self, block: Any, *, playing: bool, tail: bool) -> Optional[str]:
        return self._detector.feed(self._vm._rms(self._np, block), playing)

    def is_quiet(self, block: Any) -> bool:
        # The utterance ends when the level falls back toward the ambient floor, not below an
        # absolute 200, so with a TV on it closes when the user stops instead of at the cap.
        silence_rms = max(float(self._vm.SILENCE_RMS_THRESHOLD),
                          self._detector.quiet_floor * self._endpoint_floor_mult)
        return self._vm._rms(self._np, block) < silence_rms


class _SileroDetector:
    """Model-based detection with Silero's hysteresis, plus an echo-relative level check while
    the reply may be audible at the mic."""

    name = VAD_SILERO

    def __init__(self, stream: Any, np: Any, vm: Any, *, input_rate: int) -> None:
        self._stream, self._np, self._vm = stream, np, vm
        self._input_rate = int(input_rate)
        self._in_speech = False
        self._speech_ms = 0.0
        self._energy: deque = deque(maxlen=_LEVEL_WINDOW_BLOCKS)  # per-block mean square
        self._echo: deque = deque(maxlen=_ECHO_WINDOW_BLOCKS + _ECHO_LAG_BLOCKS)

    def feed(self, block: Any, *, playing: bool, tail: bool) -> Optional[str]:
        prob = self._stream.feed(block)
        echo_possible = playing or tail
        voiced = prob >= (SPEECH_OFF if self._in_speech else SPEECH_ON)
        if echo_possible:
            self._energy.append(float(self._np.mean(block.astype(self._np.float64) ** 2)))
            level = float(self._np.sqrt(self._np.mean(self._energy)))
            echo_level = self._echo_level()
            self._echo.append(level)
            voiced = voiced and echo_level is not None and level >= echo_level * ECHO_MARGIN
        else:
            self._energy.clear()
            self._echo.clear()
        self._in_speech = voiced
        self._speech_ms = self._speech_ms + block.size * 1000.0 / self._input_rate if voiced else 0.0
        if self._speech_ms < MIN_SPEECH_MS:
            return None
        phase = "playback" if echo_possible else "generation"
        self._vm._vad_log(f"silero TRIPPED ({phase}): p={prob:.2f} speech_ms={self._speech_ms:.0f}")
        self._in_speech, self._speech_ms = False, 0.0
        return phase

    def _echo_level(self) -> Optional[float]:
        settled = list(self._echo)[:-_ECHO_LAG_BLOCKS]
        if len(settled) < _ECHO_MIN_BLOCKS:
            return None
        return float(self._np.percentile(settled, _ECHO_PERCENTILE))

    def is_quiet(self, block: Any) -> bool:
        return self._stream.feed(block) < SPEECH_OFF


def parse_vad_config(cfg: Any) -> str:
    """``voice.vad.model`` from a loaded config dict: ``"silero"`` (default) or ``"energy"``."""
    voice = cfg.get("voice") if isinstance(cfg, dict) else None
    vad = voice.get("vad") if isinstance(voice, dict) else None
    model = str(vad.get("model") or "").strip().lower() if isinstance(vad, dict) else ""
    return VAD_ENERGY if model == VAD_ENERGY else VAD_SILERO


def make_detector(model: str, np: Any, vm: Any, *, input_rate: int,
                  energy_kwargs: dict) -> tuple[Any, Optional[str]]:
    """Build the configured detector → ``(detector, fallback_reason)``. Silero falls back to the
    energy detector when it can't load; the reason is returned so the caller can say so."""
    if model == VAD_SILERO:
        from tools.silero_vad import SileroStream, load_silero_session

        session, reason = load_silero_session()
        if session is not None:
            stream = SileroStream(session, np, input_rate=input_rate)
            return _SileroDetector(stream, np, vm, input_rate=input_rate), None
        return _EnergyDetector(np, vm, **energy_kwargs), reason
    return _EnergyDetector(np, vm, **energy_kwargs), None
