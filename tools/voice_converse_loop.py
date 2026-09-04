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

import json
import logging
import queue
import threading
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

_log = logging.getLogger("hermes_cli.web_server")

# One-shot fallback synthesis always decodes to this rate (matches the built-in
# streamers' 24 kHz so the wire format — and the `ready` frame's output.sample_rate —
# is identical whichever path serves a turn).
_FALLBACK_SAMPLE_RATE = 24000
# Split fallback PCM into ~32 KiB frames so a long sentence doesn't land as one
# giant WS frame (matches the chunk-sized cadence of the streaming path).
_FALLBACK_PCM_CHUNK_BYTES = 32 * 1024

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


# ── converse synthesizer: one uniform "text -> int16 PCM" seam for both paths ──
#
# The converse loop needs a synthesizer that ALWAYS works, mirroring Hermes
# Desktop: when the configured TTS provider has a chunked/streaming API we use it
# (low latency, playback starts on sentence one); when it doesn't (edge, the
# default), we fall back to one-shot synthesis of the whole sentence and transcode
# the resulting audio file to raw int16 PCM server-side. Both expose the same
# ``.sample_rate: int`` + ``.synth(text) -> Iterator[bytes]`` contract, so the
# handler code is identical whichever path serves a turn.


def _decode_audio_file_to_pcm16(path: str, target_rate: int = _FALLBACK_SAMPLE_RATE) -> bytes:
    """Decode an audio file to raw little-endian int16 mono PCM at *target_rate*.

    Uses PyAV to open/decode any container the one-shot providers emit (mp3, wav,
    opus/ogg, …) and resample to s16/mono/*target_rate*. On any failure logs and
    returns ``b""`` so a bad file degrades to "no audio", never an exception into
    the synthesis thread.
    """
    try:
        import av

        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=target_rate)
        out = bytearray()

        def _emit(frame) -> None:
            # PyAV 18: resample() returns a LIST of frames (may be empty).
            for rs in resampler.resample(frame):
                out.extend(bytes(rs.planes[0]))

        with av.open(path) as container:
            for frame in container.decode(audio=0):
                _emit(frame)
        _emit(None)  # flush the resampler's internal buffer
        return bytes(out)
    except Exception:  # noqa: BLE001 - a decode failure is "no audio", not a crash
        _log.warning("converse fallback: failed to decode %s", path, exc_info=True)
        return b""


class _StreamingConverseSynth:
    """Adapter over a streaming TTS provider (the low-latency path)."""

    def __init__(self, streamer: Any) -> None:
        self._streamer = streamer
        self.sample_rate: int = streamer.sample_rate

    def synth(self, text: str) -> Iterator[bytes]:
        return self._streamer.stream(text)


class _OneShotConverseSynth:
    """One-shot fallback: synth to a temp file, transcode to int16 PCM, yield it.

    Works with ANY provider (including edge, which has no chunked API): call the
    sync ``text_to_speech_tool``, read the file it wrote, decode it to raw PCM at
    the fixed converse rate, then unlink. A provider that reports failure or writes
    no readable file yields nothing (the loop treats that as a silent turn).
    """

    sample_rate: int = _FALLBACK_SAMPLE_RATE

    def synth(self, text: str) -> Iterator[bytes]:
        from tools import tts_tool, voice_mode

        result_json = tts_tool.text_to_speech_tool(text)
        try:
            result = json.loads(result_json) if isinstance(result_json, str) else result_json
        except Exception:  # noqa: BLE001
            _log.debug("converse fallback: TTS envelope was not valid JSON")
            return
        if not isinstance(result, dict) or not result.get("success"):
            _log.debug("converse fallback: TTS reported no audio (%s)",
                       (result or {}).get("error") if isinstance(result, dict) else result)
            return
        file_path = result.get("file_path")
        if not file_path:
            _log.debug("converse fallback: TTS envelope had no file_path")
            return
        try:
            pcm = _decode_audio_file_to_pcm16(file_path, self.sample_rate)
        finally:
            voice_mode._unlink_quietly(file_path)
        for start in range(0, len(pcm), _FALLBACK_PCM_CHUNK_BYTES):
            yield pcm[start:start + _FALLBACK_PCM_CHUNK_BYTES]


def resolve_converse_synthesizer(tts_config: Dict) -> Any:
    """Return a synthesizer for the converse loop — NEVER ``None``.

    Prefers the configured streaming provider (low latency); falls back to one-shot
    synthesis + server-side transcode when the provider has no chunked API. The
    returned object always exposes ``.sample_rate: int`` and
    ``.synth(text) -> Iterator[bytes]`` yielding int16 mono PCM.

    The streaming provider is resolved via the MODULE attribute
    (``tts_streaming.resolve_streaming_provider``) so a test's monkeypatch applies.
    """
    from tools import tts_streaming

    streamer = tts_streaming.resolve_streaming_provider(tts_config)
    if streamer is not None:
        return _StreamingConverseSynth(streamer)
    return _OneShotConverseSynth()
