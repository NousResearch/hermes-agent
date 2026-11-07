"""Tracing seam for the converse voice loop — a no-op by default.

:mod:`tools.voice_converse_loop` reports every turn's phase boundaries through this module,
and every call here does nothing. It exists so that adding tracing (temporarily, to debug
latency on a real device, or permanently, wired to a backend) means replacing the bodies in
this one file — keep the names and signatures — without touching the voice loop.

Phases, in order, per turn:

* ``voice.capture`` — VAD trip → utterance endpointed. Receives ``endpoint.mode``
  (``"fixed"`` or ``"smart_turn"``).
* ``voice.stt`` — transcription. Receives ``stt.success``, ``stt.model``,
  ``stt.transcript_chars``.
* ``voice.agent`` — agent turn, start → last delta. Receives ``llm.ttft_ms``,
  ``reply.chars``, ``llm.deltas``, ``error``, ``llm.timed_out``.
* ``voice.tts`` — first sentence handed to TTS → last PCM sent. Receives
  ``tts.pcm_chunks``, ``tts.first_pcm_ms``.

Capture and STT run on the VAD worker and are recorded live into a :class:`PhaseRecorder`
that travels with that utterance's transcript (so a driver backed up behind a long turn still
finishes the right utterance's trace). The driver then opens the turn with :func:`start_turn`,
appends the agent/TTS phases with wall-clock (epoch-ns) start/end via
:meth:`TurnTrace.add_phase`, sets ``outcome`` (``ok`` / ``error`` / ``timeout`` /
``interrupted`` / ``stop_word``) plus ``interrupted`` and ``expects_more``, and calls
:meth:`TurnTrace.end`. An utterance discarded after a sign-off turn is handed to
:func:`emit_turn_trace` with ``outcome="discarded_after_signoff"``.

If :attr:`TurnTrace.traceparent` returns a W3C traceparent, the loop forwards it on the
``transcript`` and ``speaking`` frames so a client can nest its own spans under the turn.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional


class _Phase:
    """Handle for one live phase; :meth:`set` attaches attributes to it."""

    def set(self, **attrs: Any) -> None:
        pass


class PhaseRecorder:
    """Collects one utterance's capture/STT phases (and turn-level attributes)."""

    def __init__(self, **turn_attrs: Any) -> None:
        pass

    def set(self, **attrs: Any) -> None:
        pass

    @contextmanager
    def phase(self, name: str, **attrs: Any) -> Iterator[_Phase]:
        yield _Phase()


class TurnTrace:
    """One live turn, opened by :func:`start_turn` and closed by :meth:`end`."""

    traceparent: Optional[str] = None

    def add_phase(self, name: str, start_ns: Optional[int], end_ns: Optional[int],
                  **attrs: Any) -> None:
        pass

    def set(self, **attrs: Any) -> None:
        pass

    def end(self) -> None:
        pass


def start_turn(*, recorder: Optional[PhaseRecorder], turn_id: str, **attrs: Any) -> TurnTrace:
    """Open the turn for *recorder*'s utterance (``recorder`` is None for injected transcripts)."""
    return TurnTrace()


def emit_turn_trace(recorder: Optional[PhaseRecorder]) -> None:
    """Emit a recorder that never became a turn (e.g. discarded after sign-off)."""
