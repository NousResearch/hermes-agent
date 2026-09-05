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

import asyncio
import contextlib
import json
import logging
import queue
import threading
from typing import (
    Any, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple)

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
        barge_multiplier: Optional[float] = None, input_rate: int = 16000,
    ) -> None:
        from tools import voice_mode as _vm

        self._np = np
        self._vm = _vm
        self._stt_model = stt_model
        # Per-connection capture rate. A single-clock device (ESP32) sets this so the
        # capture WAV is written at the rate the client actually sends; Whisper resamples
        # internally, so STT works at any rate. Block size is 30 ms worth of samples at it.
        self._input_rate = int(input_rate)
        self._stop = threading.Event()
        self._playing = threading.Event()
        # Set by the handler while TTS is streaming so a barge-in can cut it.
        self._tts_stop: Optional[threading.Event] = None
        self._interrupted = threading.Event()
        self.stream = _NetworkMicStream(np, stop=self._stop)
        # Transcripts ready for a turn (or the None sentinel on shutdown).
        self.transcripts: "queue.Queue[Optional[str]]" = queue.Queue()

        self._block = int(self._input_rate * 0.03)  # 30 ms of samples at the input rate
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
            sample_rate=self._input_rate,
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


# Bound on the per-connection conversation history kept in memory: at most this
# many messages (~20 user+assistant turns). A long-lived socket that never closes
# would otherwise grow ``history`` without limit; this simple tail-cap (no
# summarization) keeps it bounded while preserving recent context.
_HISTORY_MAX_MESSAGES = 40


_IDLE_INTERVAL_MAX = 3600.0

# Per-connection sample-rate defaults + clamp range. A single-clock device (ESP32) can set
# input_rate == output_rate; browsers keep the 16 kHz-in / 24 kHz-out split. Rates outside
# this range are clamped to a sane telephony..studio window.
DEFAULT_INPUT_RATE = 16000
DEFAULT_OUTPUT_RATE = 24000
_SAMPLE_RATE_MIN = 8000
_SAMPLE_RATE_MAX = 48000


def clamp_sample_rate(raw: Any, default: int) -> int:
    """Coerce *raw* to an int sample rate in [8000, 48000]; *default* when unusable."""
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    return max(_SAMPLE_RATE_MIN, min(val, _SAMPLE_RATE_MAX))


# Default converse identity — the name a client gets when the start frame omits one.
DEFAULT_CONVERSE_NAME = "Sakura"


def parse_start_config(frame: Dict[str, Any]) -> Tuple[int, int, float, str, Optional[str]]:
    """Resolve ``(input_rate, output_rate, idle_interval, name, profile)`` from a ``start`` frame.

    All fields optional; defaults: input_rate=16000, output_rate=24000, idle_interval=0
    (continuous), name="Sakura", profile=None. Rates are clamped to [8000, 48000] and
    idle_interval via :func:`parse_idle_interval`. Shared by both WS hosts.
    """
    input_rate = clamp_sample_rate(frame.get("input_rate"), DEFAULT_INPUT_RATE)
    output_rate = clamp_sample_rate(frame.get("output_rate"), DEFAULT_OUTPUT_RATE)
    idle_interval = parse_idle_interval(frame.get("idle_interval"))
    raw_name = frame.get("name")
    name = str(raw_name).strip() if raw_name is not None and str(raw_name).strip() \
        else DEFAULT_CONVERSE_NAME
    profile = frame.get("profile") or None
    return input_rate, output_rate, idle_interval, name, profile


# Voice replies are spoken aloud — keep them short and speakable. Built per-turn as the
# ephemeral system prompt (mirrors the CLI voice mode's brevity prefix), so spoken replies
# don't balloon to full chat length. When a name is known it also carries wake handling.
_VOICE_BREVITY_PROMPT = (
    "You are in a live voice conversation and your reply is spoken aloud. Answer concisely and "
    "conversationally — at most 2-3 short sentences of plain spoken text, with no code blocks, "
    "markdown, lists, or URLs. If the request is unclear or you only caught fragments, ask one "
    "short clarifying question instead of guessing or rambling."
)


def voice_system_prompt(name: Optional[str] = None) -> str:
    """Ephemeral system prompt for one converse turn.

    When *name* is given, prepend an identity + wake-word preamble (so the model knows what
    it's called and treats a leading name / "hey <name>" as being addressed, not part of the
    request) before the spoken-brevity rules. When *name* is ``None``/empty, just the brevity
    rules. The default converse name is ``"Sakura"``, so most turns carry the identity block.
    """
    name = (str(name).strip() if name is not None else "")
    if name:
        identity = (
            f"Your name is {name}. People talk to you by voice and get your attention by saying "
            f"your name (or 'hey {name}') — treat that as being addressed, not part of the "
            "request, and don't repeat it back. ")
        return identity + _VOICE_BREVITY_PROMPT
    return _VOICE_BREVITY_PROMPT
# Safety cap on how much of ONE reply is ever synthesized to speech, so a runaway reply (a
# model ignoring the brevity prompt, or a tool result read aloud) can't play for minutes.
# Normal replies sit far under this; it only bounds the pathological case.
_MAX_TTS_CHARS_PER_TURN = 1500


def parse_idle_interval(raw: Any) -> float:
    """Clamp an ``idle_interval`` VALUE (from the ``start`` frame) → seconds of quiet between
    turns before an ``{"type":"idle"}`` notification (which also enables stop-phrase →
    ``{"type":"stop_word"}``). Accepts a number, numeric string, or ``None``.

    This is the SESSION-mode opt-in. Absent / invalid / ``<= 0`` → ``0.0`` = continuous mode:
    no idle pings and no stop-word handling — the original always-listening behavior, so
    existing clients are unaffected. A positive value enables session mode (clamped to a sane
    max); a wake-word client passes e.g. ``15``."""
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if val <= 0 else min(val, _IDLE_INTERVAL_MAX)


async def drive_converse_turns(
    *,
    session: "ConverseSession",
    synth: Any,
    cap: int,
    loop: asyncio.AbstractEventLoop,
    send_json: Callable[[dict], Awaitable[Any]],
    send_bytes: Callable[[bytes], Awaitable[Any]],
    run_turn: Callable[..., Awaitable[Tuple[str, Optional[str]]]],
    history: List[Dict[str, str]],
    idle_interval: float = 0.0,
) -> None:
    """Run the per-transcript incremental-TTS turn loop shared by both WS hosts.

    One turn per transcript pulled off ``session.transcripts``: announce it
    (``{"type":"transcript"}``), run the real agent turn via ``run_turn`` (deltas
    stream out as they land), and speak the reply INCREMENTALLY — each sentence is
    synthesized and streamed the moment it is ready (``SentenceChunker`` +
    ``synth.synth`` → ``send_bytes``), so the user hears sentence 1 while sentence 2
    is still being generated. Control-frame order per turn is:
    ``transcript`` → (``speaking`` + PCM frames) → optional ``interrupted``/``error``
    → ``turn_done``.

    The host adapts by passing:

    * ``send_json`` / ``send_bytes`` — awaitables wrapping the host ws (aiohttp and
      starlette both expose async ``send_json``/``send_bytes``, so a host can pass
      ``ws.send_json``/``ws.send_bytes`` directly).
    * ``run_turn(transcript, on_delta, *, interrupted) -> (reply_text, err)`` — runs
      one agent turn, calling ``on_delta`` with streaming text, and returns the reply
      text (for history) and an error string (or ``None``). The driver awaits it as a
      task and owns closing the synthesis pipeline when it ends.
    * ``synth`` — a converse synthesizer (``.sample_rate`` + ``.synth(text) ->
      Iterator[bytes]``); ``cap`` — the provider's max text length.
    * ``history`` — the mutable conversation-history list; the driver appends the
      user + assistant message each turn and caps it to a bounded tail.

    BARGE-IN / v1 LIMITATION: a VAD trip while playing stops TTS PLAYBACK
    (``session`` sets ``tts_stop`` + ``mark_speech_interrupted``) and we emit
    ``{"type":"interrupted"}``, but the in-flight agent turn (``run_turn`` /
    ``_run_agent`` / ``prompt.submit``) is NOT cancelled in v1 — it is allowed to run
    to completion, so a barged turn may still finish and fire tools, and the next
    utterance queues behind it. Cancelling the in-flight turn is a deliberate
    follow-up, not implemented here.
    """
    from tools.tts_streaming import SentenceChunker
    from tools.tts_text_normalize import _strip_markdown_for_tts
    from tools.voice_mode_transcript import is_voice_stop_phrase

    quiet = 0.0  # seconds of no NEW utterance while waiting between turns (session mode)
    while not session.stopped:
        if idle_interval > 0:
            # Session mode (wake-word clients). Timed wait for the next utterance; on a
            # quiet interval, notify the client and KEEP the socket open (streaming
            # continues) — a periodic {"type":"idle"} hint a wake-word client uses to
            # re-arm/sleep, and a continuous client ignores. Repeats every idle_interval
            # of quiet; `quiet_seconds` is the cumulative quiet since the last utterance.
            try:
                transcript = await loop.run_in_executor(
                    None, lambda: session.transcripts.get(timeout=idle_interval))
            except queue.Empty:
                quiet = round(quiet + idle_interval, 1)
                await send_json({"type": "idle", "quiet_seconds": quiet})
                continue
        else:
            transcript = await loop.run_in_executor(None, session.transcripts.get)
        if transcript is None:  # shutdown sentinel
            break
        if not transcript:
            continue
        quiet = 0.0  # a real utterance resets the quiet clock
        await send_json({"type": "transcript", "text": transcript})
        # Session mode: a spoken stop phrase ("goodbye"/"stop"/…) ends the exchange —
        # tell the client and skip the agent turn (the client decides to re-arm/sleep).
        if idle_interval > 0 and is_voice_stop_phrase(transcript):
            await send_json({"type": "stop_word", "text": transcript})
            continue

        # Clear any stale barge-in latch before this turn; capture it as the per-turn
        # interrupted note so a host that plumbs barge-in parity (the dashboard) can
        # prepend it to the model-bound message.
        interrupted_in = session.take_interrupted()
        text_q: "queue.Queue[Optional[str]]" = queue.Queue()  # deltas; None = turn done
        pcm_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()  # PCM out; None = done
        tts_stop = threading.Event()
        reply_parts: List[str] = []
        turn_result: dict = {}

        def _on_delta(delta: str) -> None:
            # Called from run_turn's execution context (main-loop coroutine or a
            # worker thread); text_q is thread-safe either way.
            if delta:
                reply_parts.append(delta)
                text_q.put(delta)

        async def _run_turn_task(t=transcript, note=interrupted_in) -> None:
            # Drive one agent turn via the host adapter; the None sentinel closes the
            # synthesis pipeline when the turn ends (whichever way it ends).
            try:
                reply, err = await run_turn(t, _on_delta, interrupted=note)
                turn_result["reply"] = reply
                if err:
                    turn_result["err"] = err
            except Exception as exc:  # noqa: BLE001 - surface, don't wedge the loop
                turn_result["err"] = f"voice turn failed: {exc}"
            finally:
                text_q.put(None)

        def _produce() -> None:
            # Cut streaming deltas into sentences and synthesize each as it lands, so
            # playback overlaps generation (mirrors the speak-stream producer).
            chunker = SentenceChunker()
            idle_poll_seconds = 0.5
            idle_polls_before_force_flush = 4  # ~2s of silence -> speak the tail

            def _sentences():
                idle_polls = 0
                while not (tts_stop.is_set() or session.stopped):
                    try:
                        delta = text_q.get(timeout=idle_poll_seconds)
                    except queue.Empty:
                        idle_polls += 1
                        buffered = chunker.buf.strip()
                        if not buffered or (
                                "<think" in chunker.buf and "</think>" not in chunker.buf):
                            continue
                        if buffered.endswith((".", "!", "?", "…", ":")) or (
                                idle_polls >= idle_polls_before_force_flush):
                            yield from chunker.flush()
                        continue
                    idle_polls = 0
                    if delta is None:
                        yield from chunker.flush()
                        return
                    yield from chunker.feed(delta)

            spoken_chars = 0
            try:
                for sentence in _sentences():
                    cleaned = _strip_markdown_for_tts(sentence)
                    if not cleaned:
                        continue
                    for piece in split_text_for_tts_stream(cleaned, cap):
                        for chunk in synth.synth(piece):
                            if tts_stop.is_set() or session.stopped:
                                return
                            loop.call_soon_threadsafe(pcm_q.put_nowait, chunk)
                    spoken_chars += len(cleaned)
                    if spoken_chars >= _MAX_TTS_CHARS_PER_TURN:
                        # Safety cap: stop speaking a runaway reply. The agent turn still
                        # completes and the full reply is recorded in history; only the
                        # spoken audio is bounded (see _MAX_TTS_CHARS_PER_TURN).
                        _log.debug("converse: TTS output capped at %d chars", spoken_chars)
                        break
            except Exception as exc:  # noqa: BLE001
                _log.warning("converse synthesis failed: %s", exc)
            finally:
                loop.call_soon_threadsafe(pcm_q.put_nowait, None)

        turn_task = asyncio.ensure_future(_run_turn_task())
        threading.Thread(target=_produce, name="converse-tts", daemon=True).start()

        # Consumer: stream PCM out; flip `playing` on only when real audio starts
        # (kept off during generation so a mid-thought interjection stays VAD-sensitive).
        speaking = False
        while True:
            chunk = await pcm_q.get()
            if chunk is None:
                break
            if not speaking:
                session.set_playing(True, tts_stop=tts_stop)
                await send_json({"type": "speaking"})
                speaking = True
            await send_bytes(chunk)
        if speaking:
            session.set_playing(False)

        # The turn task set the None sentinel that ended synthesis, so it is
        # effectively done; await it to surface errors and settle turn_result.
        with contextlib.suppress(Exception):
            await turn_task

        # Persist the turn so history carries across the connection. Prefer the reply
        # the adapter returned (the agent's final response); fall back to the streamed
        # deltas. Cap the history to a bounded tail so a long-lived socket doesn't grow
        # `history` without limit (simple slice cap, no summarization).
        reply = turn_result.get("reply") or "".join(reply_parts)
        history.append({"role": "user", "content": transcript})
        if reply:
            history.append({"role": "assistant", "content": reply})
        if len(history) > _HISTORY_MAX_MESSAGES:
            del history[:-_HISTORY_MAX_MESSAGES]

        # Barge-in stops PLAYBACK only (see the v1 limitation above): report it and
        # skip the error frame (a barged turn's error is noise), else surface any
        # turn error. Always end the turn with `turn_done`.
        if session.take_interrupted() or tts_stop.is_set():
            await send_json({"type": "interrupted"})
        elif turn_result.get("err"):
            await send_json({"type": "error", "error": turn_result["err"]})
        await send_json({"type": "turn_done"})


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


class _ResampledConverseSynth:
    """Wrap a converse synth so ``.synth(text)`` yields int16 mono PCM at *output_rate*.

    The inner synth yields int16 mono PCM at ``inner.sample_rate`` (e.g. 24 kHz); this feeds
    each chunk through a stateful :class:`av.audio.resampler.AudioResampler` and yields the
    resampled bytes, flushing at end-of-text. Used to serve a single-clock client (ESP32)
    output at its own rate. A fresh resampler per ``synth`` call keeps turns independent.
    """

    def __init__(self, inner: Any, output_rate: int) -> None:
        self._inner = inner
        self.sample_rate: int = int(output_rate)
        self._src_rate: int = int(inner.sample_rate)

    def synth(self, text: str) -> Iterator[bytes]:
        import av
        import numpy as np

        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=self.sample_rate)

        def _feed(frame) -> Iterator[bytes]:
            # PyAV 18: resample() returns a LIST of frames (may be empty). Use to_ndarray()
            # (exact sample count) rather than bytes(planes[0]) — the plane buffer is
            # over-allocated/padded, so raw plane bytes would append garbage per chunk.
            for rs in resampler.resample(frame):
                data = rs.to_ndarray().tobytes()
                if data:
                    yield data

        carry = b""
        for chunk in self._inner.synth(text):
            if not chunk:
                continue
            carry += chunk
            n = len(carry) - (len(carry) % 2)  # feed whole int16 samples only
            if n == 0:
                continue
            buf, carry = carry[:n], carry[n:]
            arr = np.frombuffer(buf, dtype=np.int16).reshape(1, -1)  # (channels, samples)
            frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")
            frame.sample_rate = self._src_rate
            yield from _feed(frame)
        yield from _feed(None)  # flush the resampler's internal buffer


def resample_synth(synth: Any, output_rate: int) -> Any:
    """Return a synth whose ``.synth`` yields PCM at *output_rate*.

    No-op (returns *synth* unchanged) when ``synth.sample_rate == output_rate``; otherwise
    wraps it in :class:`_ResampledConverseSynth`. The result always exposes
    ``.sample_rate == output_rate`` and the same ``.synth(text) -> Iterator[bytes]`` contract.
    """
    if int(output_rate) == int(synth.sample_rate):
        return synth
    return _ResampledConverseSynth(synth, output_rate)
