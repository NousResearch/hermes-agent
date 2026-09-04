"""Framework-agnostic off-device realtime voice loop primitives.

This is the neutral home of the VAD/STT/mic-shim core shared by every surface
that hosts a live "converse" WebSocket (the FastAPI dashboard router
:mod:`hermes_cli.web_routers._converse_loop` and the aiohttp gateway module
:mod:`gateway.platforms.api_server_converse`). Nothing here touches a socket, an
audio device, a live model or a specific web framework, so it can be unit-tested
in isolation.

Pieces:

* :class:`_NetworkMicStream` — a ``sounddevice``-shaped shim whose ``.read()``
  pulls int16 blocks from a thread-safe queue fed by a WebSocket. It lets the
  existing endpointer (:func:`tools.voice_mode._capture_until_quiet`) run
  unchanged against a network source instead of a local microphone.
* :class:`ConverseSession` — drives the reused VAD/STT loop on a worker thread:
  read 30 ms blocks, feed :class:`tools.voice_mode._BargeDetector`, and on a
  trip (speech onset) or silence endpoint capture the utterance, transcribe it
  and hand the transcript to the handler. It also owns the ``playing`` flag and
  barge-in (a trip while playing cuts TTS).
* :func:`split_text_for_tts_stream` — a provider-cap-aware sentence splitter, so
  a host that has no dashboard dependency can chunk a synthesized sentence
  without importing :mod:`hermes_cli.web_server_gateway`.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Optional, Tuple

_log = logging.getLogger("hermes_cli.web_server")

# ── DSP constants — mirror tools.voice_mode.full_duplex_listen exactly ──
# Inbound audio is PCM16 mono @16 kHz (Whisper-native, matches voice_mode.SAMPLE_RATE);
# 30 ms blocks = 480 frames.  These knobs mirror full_duplex_listen's defaults so the
# network loop behaves identically to the local mic loop.
_SUSTAINED_MS = 300
_CALIBRATION_MS = 450
_GRACE_MS = 500
_PRE_ROLL_MS = 1200
_ENDPOINT_SILENCE_MS = 1250
_MAX_UTTERANCE_MS = 30_000


class _NetworkMicStream:
    """A ``sounddevice.InputStream``-shaped shim over a queue of inbound PCM.

    The WebSocket handler calls :meth:`feed` with raw PCM16 bytes as they arrive;
    the VAD/endpointer worker calls :meth:`read` for exact-size int16 blocks. The
    shim concatenates and splits inbound chunks so ``read(block)`` always returns
    a ``(np.ndarray[int16] shape (block,), overflow_bool)`` tuple exactly like the
    real stream, blocking (with a stop check) until ``block`` samples are ready.
    """

    def __init__(self, np: Any, *, stop: threading.Event, poll_seconds: float = 0.1) -> None:
        self._np = np
        self._stop = stop
        self._poll_seconds = poll_seconds
        self._chunks: "queue.Queue[Optional[Any]]" = queue.Queue()
        # Leftover samples from a chunk that overshot the requested block size.
        self._carry = np.zeros(0, dtype=np.int16)
        # A lone trailing byte from an odd-length feed: buffered so a sample split
        # across two frames survives (clients may frame on arbitrary byte counts).
        self._byte_carry = b""
        self._feed_lock = threading.Lock()

    def feed(self, pcm_bytes: bytes) -> None:
        """Append inbound PCM16 bytes (little-endian mono) as an int16 block.

        A sample split across two feeds is preserved via a one-byte carry, so a
        client that frames on arbitrary byte boundaries never loses or misaligns
        audio.
        """
        if not pcm_bytes:
            return
        with self._feed_lock:
            buf = self._byte_carry + pcm_bytes
            # Keep any lone trailing byte for the next feed to complete.
            if len(buf) % 2:
                buf, self._byte_carry = buf[:-1], buf[-1:]
            else:
                self._byte_carry = b""
        if buf:
            self._chunks.put(self._np.frombuffer(buf, dtype=self._np.int16).copy())

    def close(self) -> None:
        """Unblock any reader waiting for more samples."""
        self._stop.set()
        # A sentinel wakes a reader parked on the queue's timeout-free path.
        self._chunks.put(None)

    def read(self, block: int) -> Tuple[Any, bool]:
        """Return exactly *block* int16 samples as ``(np.ndarray, overflow=False)``.

        Blocks until enough samples arrive or the stop event is set; on stop,
        returns whatever is buffered zero-padded up to *block* so the endpointer
        drains and exits cleanly instead of raising.
        """
        np = self._np
        while len(self._carry) < block:
            if self._stop.is_set():
                # Drain anything already queued before giving up (a client's last
                # frames may have landed before close), then zero-pad the tail: a
                # partial final block reads as silence, which the endpointer treats
                # as quiet and stops on.
                self._drain_pending()
                if len(self._carry) >= block:
                    break
                pad = np.zeros(block - len(self._carry), dtype=np.int16)
                out = np.concatenate([self._carry, pad])
                self._carry = np.zeros(0, dtype=np.int16)
                return out, False
            try:
                chunk = self._chunks.get(timeout=self._poll_seconds)
            except queue.Empty:
                continue
            if chunk is None:  # close() sentinel
                continue
            self._carry = np.concatenate([self._carry, chunk])
        out, self._carry = self._carry[:block], self._carry[block:]
        return out, False

    def _drain_pending(self) -> None:
        """Pull every queued chunk into the carry without blocking."""
        while True:
            try:
                chunk = self._chunks.get_nowait()
            except queue.Empty:
                return
            if chunk is not None:
                self._carry = self._np.concatenate([self._carry, chunk])


class ConverseSession:
    """Drives the reused VAD → STT loop against a :class:`_NetworkMicStream`.

    A worker thread reads 30 ms blocks, computes RMS and feeds a
    :class:`~tools.voice_mode._BargeDetector`.  On a trip (speech onset) it runs
    the shared endpointer (:func:`~tools.voice_mode._capture_until_quiet`) →
    ``_write_wav`` → ``transcribe_recording`` and puts the transcript on
    :attr:`transcripts` for the handler.  The handler flips :meth:`set_playing`
    around TTS playback so the detector rejects speaker bleed; a trip while
    playing is a barge-in (TTS is cut and the interrupt latch is set).
    """

    def __init__(
        self, np: Any, *, stt_model: Optional[str] = None,
        barge_multiplier: Optional[float] = None,
    ) -> None:
        from tools import voice_mode as _vm

        self._np = np
        self._vm = _vm
        self._stt_model = stt_model
        self._stop = threading.Event()
        self._playing = threading.Event()
        # Set by the handler while TTS is streaming so a barge-in can cut it.
        self._tts_stop: Optional[threading.Event] = None
        self._interrupted = threading.Event()
        self.stream = _NetworkMicStream(np, stop=self._stop)
        # Transcripts ready for a turn (or the None sentinel on shutdown).
        self.transcripts: "queue.Queue[Optional[str]]" = queue.Queue()

        self._block = int(_vm.SAMPLE_RATE * 0.03)  # 480 frames @16 kHz
        mult = float(barge_multiplier) if barge_multiplier else _vm.DEFAULT_BARGE_MULTIPLIER
        self._detector = _vm._BargeDetector(
            np, mult=mult,
            calib_blocks=max(1, _CALIBRATION_MS // 30),
            trip_blocks=max(1, _SUSTAINED_MS // 30),
            grace_blocks=max(0, _GRACE_MS // 30),
        )
        from collections import deque

        self._pre_roll: deque = deque(maxlen=max(1, _PRE_ROLL_MS // 30))
        self._endpoint_blocks = max(1, _ENDPOINT_SILENCE_MS // 30)
        self._max_blocks = max(1, _MAX_UTTERANCE_MS // 30)
        self._worker: Optional[threading.Thread] = None
        # Called with the trip phase name ("generation"/"playback") on every trip.
        self.on_trip: Optional[Callable[[str], None]] = None

    # ── playback / barge-in coordination ──
    def playing(self) -> bool:
        return self._playing.is_set()

    def set_playing(self, value: bool, *, tts_stop: Optional[threading.Event] = None) -> None:
        """Mark playback active/idle; while active a VAD trip cuts *tts_stop*."""
        self._tts_stop = tts_stop if value else None
        if value:
            self._interrupted.clear()
            self._playing.set()
        else:
            self._playing.clear()
            self._tts_stop = None

    def take_interrupted(self) -> bool:
        """Pop the barge-in flag; True when a trip cut playback since the last check."""
        if self._interrupted.is_set():
            self._interrupted.clear()
            return True
        return False

    def stop(self) -> None:
        """End the loop and unblock the reader and any transcript waiter."""
        self._stop.set()
        self.stream.close()
        self.transcripts.put(None)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def commit(self) -> None:
        """Force the current utterance to endpoint now (client pressed 'commit')."""
        # A run of silence blocks reaches the endpointer's quiet threshold; the
        # simplest cross-thread nudge is a stop of the network source, but that
        # would kill the whole loop.  Instead feed enough zero blocks to satisfy
        # the endpoint-silence window so _capture_until_quiet returns promptly.
        silence = self._np.zeros(self._block, dtype=self._np.int16).tobytes()
        for _ in range(self._endpoint_blocks + 1):
            self.stream.feed(silence)

    # ── worker loop ──
    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name="converse-vad", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        np, vm = self._np, self._vm
        try:
            while not self._stop.is_set():
                data, _ = self.stream.read(self._block)
                if self._stop.is_set():
                    break
                self._pre_roll.append(data.copy())
                playing = self.playing()
                phase = self._detector.feed(vm._rms(np, data), playing)
                if phase is None:
                    continue
                # Barge-in: a trip during playback cuts the reply mid-stream.
                if playing:
                    self._trigger_barge_in()
                if self.on_trip is not None:
                    try:
                        self.on_trip(phase)
                    except Exception:  # noqa: BLE001 - callback must not kill the loop
                        _log.debug("converse on_trip callback failed", exc_info=True)
                transcript = self._capture_and_transcribe()
                if transcript:
                    self.transcripts.put(transcript)
        except Exception:  # noqa: BLE001 - a loop crash must not wedge the socket
            _log.warning("converse VAD loop failed", exc_info=True)
        finally:
            self.transcripts.put(None)

    def _trigger_barge_in(self) -> None:
        """Cut the in-flight reply: latch the interrupt note and stop TTS."""
        try:
            from tools.tts_streaming import mark_speech_interrupted

            mark_speech_interrupted()
        except Exception:  # noqa: BLE001
            _log.debug("mark_speech_interrupted failed", exc_info=True)
        if self._tts_stop is not None:
            self._tts_stop.set()
        self._playing.clear()
        self._interrupted.set()

    def _capture_and_transcribe(self) -> str:
        """Endpoint the utterance from the pre-roll and return its transcript."""
        vm, np = self._vm, self._np
        wav_path = vm._capture_until_quiet(
            self.stream, np, self._block, self._pre_roll,
            endpoint_blocks=self._endpoint_blocks, max_blocks=self._max_blocks,
        )
        # _capture_until_quiet drained the pre-roll into the WAV; start fresh.
        self._pre_roll.clear()
        result = vm.transcribe_recording(wav_path, model=self._stt_model)
        vm._unlink_quietly(wav_path)
        if not result.get("success"):
            _log.debug("converse transcription failed: %s", result.get("error"))
            return ""
        return str(result.get("transcript") or "").strip()


def split_text_for_tts_stream(text: str, cap: int) -> list:
    """Split *text* into provider-cap-sized pieces on sentence boundaries.

    Mirror of :func:`hermes_cli.web_server_gateway._split_text_for_speak_stream`,
    lifted here so a host with no dashboard dependency (e.g. the aiohttp gateway)
    can chunk synthesized sentences without importing the FastAPI web server.
    Reflows whitespace (sentences re-joined with single spaces); no fence
    semantics — deliberately NOT unified with the fence-aware splitter.
    """
    from tools.tts_streaming import SENTENCE_BOUNDARY_RE as _SENTENCE_BOUNDARY_RE

    cap = cap if cap and cap > 0 else 4000
    pieces, buf = [], ""
    for sentence in filter(str.strip, _SENTENCE_BOUNDARY_RE.split(text)):
        while len(sentence) > cap:
            pieces.append(sentence[:cap])
            sentence = sentence[cap:]
        if buf and len(buf) + len(sentence) + 1 > cap:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
    if buf:
        pieces.append(buf)
    return pieces
