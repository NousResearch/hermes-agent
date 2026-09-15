"""xAI Grok realtime (S2S WebSocket) session for voice mode.

Two brains (``voice.realtime.brain``):

* ``ears`` — input only. Server VAD + transcription; every utterance becomes
  a normal Hermes turn; replies speak via the regular TTS pipeline.
* ``supervisor`` — grok-voice converses instantly and delegates real work to
  Hermes through ``consult_hermes`` / ``steer_hermes``.

Config, tool schemas, and ``session.update`` live in
:mod:`tools.voice_realtime_config`. This module is the live session:
mic → socket → VAD events → optional supervisor playback.

Heavy deps (websockets, sounddevice, numpy, credentials) import lazily:
tools/*.py must stay cheap to import and voice is an optional extra.
"""

from __future__ import annotations

import base64
import contextvars
import json
import logging
import math
import queue
import random
import struct
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

from tools.voice_realtime_config import (
    FRAME_MS,
    FRAME_SAMPLES,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    RECONNECT_DELAYS,
    RealtimeConfig,
    RealtimeVoiceError,
    _ACK_PHRASES,
    build_session_update,
    check_realtime_requirements,
)

logger = logging.getLogger(__name__)

_PREBUFFER_MAX_FRAMES = 20
_QUIET_WAIT_TIMEOUT_S = 20.0
# A connection that stays up at least this long resets the reconnect budget.
_STABLE_CONNECTION_S = 5.0

def _default_connect(url: str, headers: Dict[str, str]):
    """Open a synchronous WebSocket connection (thread-based client)."""
    from websockets.sync.client import connect  # lazy: keep module import cheap

    return connect(url, additional_headers=headers, open_timeout=10, max_size=2**22)


class _SounddevicePlayoutSink:
    """Default speech sink: local speakers @ 24 kHz mono int16.

    Marks tools.voice_mode's audio-output refcount so the CLI's gate/idle/
    barge logic sees supervisor speech like any other playback. Non-local
    surfaces (Discord VC) inject their own sink and skip the refcount.
    """

    def __init__(self):
        import numpy as np  # lazy: optional voice extra
        import sounddevice as sd

        from tools.voice_mode import mark_audio_output_active

        self._np = np
        self._mark = mark_audio_output_active
        self._stream = sd.OutputStream(
            samplerate=OUTPUT_SAMPLE_RATE, channels=1, dtype="int16"
        )
        self._stream.start()

    def write(self, chunk: bytes) -> None:
        self._stream.write(self._np.frombuffer(chunk, dtype=self._np.int16))

    def set_active(self, active: bool) -> None:
        self._mark(active)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class _SounddeviceMic:
    """Default mic source: 16 kHz mono int16 InputStream → frame callback."""

    def __init__(self, on_frame: Callable[[bytes], None]):
        import sounddevice as sd  # lazy: optional voice extra

        def _callback(indata, frames, _time, status):
            if status:
                logger.debug("realtime mic status: %s", status)
            on_frame(bytes(indata.tobytes()))

        self._stream = sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=_callback,
        )
        self._stream.start()

    def close(self) -> None:
        try:
            self._stream.stop()
        finally:
            self._stream.close()


class RealtimeVoiceSession:
    """xAI realtime session: mic → server VAD → transcript / supervisor speech.

    Callbacks fire on session threads — keep them fast/thread-safe:
    * ``on_transcript(text)`` — finished utterance (armed + gate-open only)
    * ``on_speech_started()`` / ``on_speech_stopped()`` — server VAD edges;
      speech_started is the CLI's barge-in trigger
    * ``on_state(state, detail)`` — "connected" | "reconnecting" | "dead"
    * ``on_idle_pause()`` — idle timer fired; session disarmed itself first
    * ``input_gate()`` — False drops mic frames and suppresses speech events
    * ``activity_hold()`` — True while the user is correctly silent (agent
      busy / TTS live); idle timer pauses
    * ``on_function_call(name, call_id, args_json)`` — supervisor brain only
    """

    def __init__(
        self,
        cfg: RealtimeConfig,
        *,
        on_transcript: Callable[[str], None],
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_stopped: Optional[Callable[[], None]] = None,
        on_state: Optional[Callable[[str, str], None]] = None,
        on_idle_pause: Optional[Callable[[], None]] = None,
        input_gate: Optional[Callable[[], bool]] = None,
        activity_hold: Optional[Callable[[], bool]] = None,
        on_function_call: Optional[Callable[[str, str, str], None]] = None,
        connect_fn: Optional[Callable[[str, Dict[str, str]], Any]] = None,
        mic_factory: Optional[Callable[[Callable[[bytes], None]], Any]] = None,
        playout_sink_factory: Optional[Callable[[], Any]] = None,
        require_local_audio: bool = True,
    ):
        self._cfg = cfg
        self._require_local_audio = require_local_audio
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._on_speech_stopped = on_speech_stopped
        self._on_state = on_state
        self._on_idle_pause = on_idle_pause
        self._input_gate = input_gate
        self._activity_hold = activity_hold
        self._on_function_call = on_function_call
        self._connect_fn = connect_fn or _default_connect
        self._mic_factory = mic_factory if mic_factory is not None else _SounddeviceMic
        self._playout_sink_factory = (
            playout_sink_factory if playout_sink_factory is not None
            else _SounddevicePlayoutSink
        )

        self._armed = threading.Event()
        self._stop = threading.Event()
        self._dead = threading.Event()
        self._frames: "queue.Queue[bytes]" = queue.Queue(maxsize=100)
        self._prebuffer: deque = deque(maxlen=_PREBUFFER_MAX_FRAMES)
        self._ws: Any = None
        self._send_lock = threading.Lock()
        self._mic: Any = None
        self._net_thread: Optional[threading.Thread] = None
        self._pump_thread: Optional[threading.Thread] = None
        self._minimal_retry_done = False
        self._last_voice_activity = time.monotonic()
        self._current_rms = 0
        # Supervisor speech playback (lazy — never started in ears mode).
        self._playout_q: "queue.Queue[bytes]" = queue.Queue(maxsize=400)
        self._playout_thread: Optional[threading.Thread] = None
        self._playout_sink: Any = None
        self._playing = False
        self._active_response = False
        self._response_had_audio = False
        # Loud-barge (half-duplex supervisor): rolling speaker-bleed floor,
        # consecutive hot frames, open-mic window after a trigger, and a
        # short tail of gated frames replayed so the utterance start isn't
        # clipped.
        self._bleed_floor = 0.0
        self._barge_hot_frames = 0
        self._barge_until = 0.0
        self._gated_tail: deque = deque(maxlen=5)
        self._np: Any = None
        self._tool_results: "queue.Queue[Tuple[str, str, bool]]" = queue.Queue()
        self._tool_result_thread: Optional[threading.Thread] = None
        self._tool_result_lock = threading.Lock()

    # -- public surface ----------------------------------------------------

    @property
    def alive(self) -> bool:
        return (
            self._net_thread is not None
            and self._net_thread.is_alive()
            and not self._dead.is_set()
            and not self._stop.is_set()
        )

    @property
    def connected(self) -> bool:
        return self.alive and self._ws is not None

    @property
    def current_rms(self) -> int:
        """Mic level for the CLI's audio meter (same contract as AudioRecorder)."""
        return self._current_rms

    def start(self) -> None:
        """Open the mic and start connecting (non-blocking; results via
        ``on_state``). Raises only on mic/requirements failure."""
        ok, detail = check_realtime_requirements(
            require_local_audio=self._require_local_audio
        )
        if not ok:
            raise RealtimeVoiceError(detail)
        try:
            self._mic = self._mic_factory(self._enqueue_frame)
        except Exception as exc:
            raise RealtimeVoiceError(f"microphone open failed: {exc}") from exc
        self._last_voice_activity = time.monotonic()
        # Threads do not inherit ContextVars. The net loop resolves xAI
        # credentials (on connect and every reconnect); under multiplexed
        # profiles that read must happen inside the caller's profile secret
        # scope or it silently falls back to the default profile's key.
        ctx = contextvars.copy_context()
        self._net_thread = threading.Thread(
            target=ctx.run, args=(self._net_loop,), name="voice-rt-net", daemon=True
        )
        self._pump_thread = threading.Thread(
            target=self._pump_loop, name="voice-rt-pump", daemon=True
        )
        self._net_thread.start()
        self._pump_thread.start()

    def stop(self) -> None:
        """Tear down mic, socket, and threads. Idempotent."""
        self._stop.set()
        self._armed.clear()
        self.clear_playout()
        mic, self._mic = self._mic, None
        if mic is not None:
            try:
                mic.close()
            except Exception:
                pass
        self._close_ws()
        for t in (self._net_thread, self._pump_thread, self._playout_thread, self._tool_result_thread):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=3)

    def set_armed(self, armed: bool) -> None:
        """Arm/disarm transcript delivery; disarming clears the server buffer
        so a paused mic can't produce a stale utterance."""
        if armed:
            self._last_voice_activity = time.monotonic()
            self._armed.set()
        else:
            self._armed.clear()
            self._send_event({"type": "input_audio_buffer.clear"})
            self.clear_playout()

    @property
    def armed(self) -> bool:
        return self._armed.is_set()

    @property
    def speaking(self) -> bool:
        """True while supervisor speech is queued, playing, or still draining
        inside a buffering sink (``pending()`` — e.g. the Discord mixer)."""
        if self._playing or not self._playout_q.empty():
            return True
        pending = getattr(self._playout_sink, "pending", None)
        if pending is not None:
            try:
                return bool(pending())
            except Exception:
                return False
        return False

    @property
    def barge_active(self) -> bool:
        """True right after a loud-barge trigger — the gate lets mic frames
        through even though speech was just playing."""
        return time.monotonic() < self._barge_until

    def speak_verbatim(self, text: str, *, interruptible: bool = True) -> bool:
        """Inject exact text as spoken audio (xAI ``force_message``).
        Supervisor brain only — the ears brain never plays server audio."""
        text = (text or "").strip()
        if not text or not self._cfg.supervisor:
            return False
        return self._send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "force_message",
                "role": "assistant",
                "interruptible": interruptible,
                "content": [{"type": "output_text", "text": text}],
            },
        })

    @property
    def last_response_had_audio(self) -> bool:
        """Whether the current/most recent response produced any speech."""
        return self._response_had_audio

    def speak_acknowledgment(self) -> None:
        """Instantly speak a rotating "on it" line (force_message, no model
        turn). Used when a consult arrived silently — the model skipped its
        mandated filler and the user must not get dead air."""
        self.speak_verbatim(random.choice(_ACK_PHRASES), interruptible=True)

    def send_function_output(
        self, call_id: str, output: str, *, respond: bool = True,
    ) -> None:
        """Return a tool result and optionally ask for the follow-up response.

        ``respond=False`` completes the tool call without ``response.create``
        (wake-name misses must not speak). ``response.create`` otherwise waits
        until current speech finishes (bounded) so the follow-up never talks
        over an in-flight answer. One worker thread serializes deliveries.
        """
        self._tool_results.put((call_id, output, respond))
        # Locked check-then-start: concurrent callers (recv thread failing a
        # consult, turn thread completing one) must not spawn two workers —
        # duplicate workers race concurrent response.create sends.
        with self._tool_result_lock:
            if self._tool_result_thread is None or not self._tool_result_thread.is_alive():
                self._tool_result_thread = threading.Thread(
                    target=self._tool_result_loop, name="voice-rt-tool-result", daemon=True
                )
                self._tool_result_thread.start()

    def _tool_result_loop(self) -> None:
        while not self._stop.is_set():
            try:
                call_id, output, respond = self._tool_results.get(timeout=0.25)
            except queue.Empty:
                continue
            self._send_event({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            })
            if not respond:
                continue
            deadline = time.monotonic() + _QUIET_WAIT_TIMEOUT_S
            while time.monotonic() < deadline and not self._stop.is_set():
                if not self.speaking and not self._active_response:
                    break
                time.sleep(0.2)
            if self._stop.is_set():
                return
            self._send_event({"type": "response.create"})

    def cancel_response(self) -> None:
        """Stop in-flight supervisor speech (wake-name miss / barge)."""
        self.clear_playout()
        self._active_response = False
        self._send_event({"type": "response.cancel"})

    def clear_playout(self) -> None:
        """Drop queued supervisor speech (barge-in / pause)."""
        try:
            while True:
                self._playout_q.get_nowait()
        except queue.Empty:
            pass
        # Buffering sinks (Discord mixer) hold already-written audio too.
        clear = getattr(self._playout_sink, "clear", None)
        if clear is not None:
            try:
                clear()
            except Exception:
                logger.debug("playout sink clear failed", exc_info=True)

    # -- internals ----------------------------------------------------------

    def _gate_open(self) -> bool:
        if self._input_gate is None:
            return True
        try:
            return bool(self._input_gate())
        except Exception:
            return True

    def _emit_state(self, state: str, detail: str = "") -> None:
        if self._on_state is not None:
            try:
                self._on_state(state, detail)
            except Exception:
                logger.debug("realtime on_state callback failed", exc_info=True)

    def _enqueue_frame(self, frame: bytes) -> None:
        try:
            self._frames.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._frames.get_nowait()
        except queue.Empty:
            return
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            pass

    def _pump_loop(self) -> None:
        """Mic frame consumer: RMS meter, gating, loud-barge, idle pause.

        Discord (and any source that only emits when RTP is present) goes
        quiet by sending *nothing* when the user stops talking. Server VAD
        needs actual silence PCM (or a commit) to fire speech_stopped — an
        empty queue is not silence, so we keep the socket fed with zeros.
        """
        silence = b"\x00" * (FRAME_SAMPLES * 2)  # int16 mono, FRAME_MS of zeros
        frame_s = FRAME_MS / 1000.0
        # Virtual input clock: a local mic delivers one frame per FRAME_MS
        # with ordinary callback jitter, so a single empty poll is NOT
        # silence — splicing zeros there puts holes in words and runs the
        # stream ahead of real time. Only a source that has been quiet for
        # two full frame periods (Discord RTP stopped) gets keep-alive
        # zeros, paced at one frame per FRAME_MS from then on.
        last_fed = time.monotonic()
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=frame_s)
            except queue.Empty:
                self._check_idle_pause()
                now = time.monotonic()
                if now - last_fed < 2 * frame_s:
                    continue
                # Never prebuffer silence: _send_frame with _ws is None would
                # delay the first real utterance once the socket connects.
                if (
                    self._armed.is_set()
                    and self._gate_open()
                    and self._ws is not None
                ):
                    self._send_frame(silence)
                last_fed = max(last_fed + frame_s, now - 2 * frame_s)
                continue
            last_fed = time.monotonic()
            self._update_rms(frame)
            self._update_barge_detector()
            if not self._armed.is_set() or not self._gate_open():
                if self._playing:
                    # Keep a short tail so a barge doesn't clip the start
                    # of the user's utterance.
                    self._gated_tail.append(frame)
                self._check_idle_pause()
                continue
            self._check_idle_pause()
            if self._gated_tail:
                # The tail is speaker bleed captured behind the closed gate.
                # Replay it only when a barge opened the gate (it holds the
                # onset of the user's words); after normal playback end it
                # is just the assistant's own voice — never feed that back.
                if self.barge_active:
                    while self._gated_tail:
                        self._send_frame(self._gated_tail.popleft())
                else:
                    self._gated_tail.clear()
            self._send_frame(frame)

    def _send_frame(self, frame: bytes) -> None:
        ws = self._ws
        if ws is None:
            self._prebuffer.append(frame)
            return
        payload = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(frame).decode("ascii"),
        })
        if not self._send_raw(payload):
            self._prebuffer.append(frame)

    def _update_barge_detector(self) -> None:
        """Loud-barge for half-duplex supervisor speech: the user talking
        clearly OVER the playback (RMS well above the tracked speaker-bleed
        floor) cuts playout and opens the mic. Bleed itself can't trigger —
        the floor is calibrated from it."""
        if not self._cfg.supervisor or self._cfg.full_duplex:
            return
        if not self._playing:
            self._barge_hot_frames = 0
            self._bleed_floor = 0.0  # recalibrate on the next playback
            return
        rms = float(self._current_rms)
        floor = self._bleed_floor
        if floor <= 0:
            self._bleed_floor = max(rms, 200.0)
            return
        # Track bleed: rise slowly (a shout must not become the floor),
        # fall quickly (quiet passages lower the trigger point).
        alpha = 0.05 if rms > floor else 0.3
        self._bleed_floor = floor + alpha * (rms - floor)
        if rms > max(self._bleed_floor, 200.0) * self._cfg.barge_multiplier:
            self._barge_hot_frames += 1
            if self._barge_hot_frames >= 2:  # ~200 ms sustained
                self._barge_hot_frames = 0
                self._barge_until = time.monotonic() + 4.0
                self.clear_playout()
                if self._active_response:
                    self._send_event({"type": "response.cancel"})
        else:
            self._barge_hot_frames = max(0, self._barge_hot_frames - 1)

    def _update_rms(self, frame: bytes) -> None:
        np = self._np
        if np is False:
            self._update_rms_without_numpy(frame)
            return
        if np is None:
            try:
                import numpy as np  # lazy: optional voice extra
                self._np = np
            except Exception:
                self._np = False
                self._update_rms_without_numpy(frame)
                return
        try:
            arr = np.frombuffer(frame, dtype=np.int16)
            if arr.size:
                self._current_rms = int(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
        except Exception:
            self._current_rms = 0

    def _update_rms_without_numpy(self, frame: bytes) -> None:
        """Keep metering/barge-in alive when the optional accelerator is absent."""
        sample_bytes = len(frame) - (len(frame) % 2)
        if not sample_bytes:
            self._current_rms = 0
            return
        try:
            samples = struct.iter_unpack("<h", memoryview(frame)[:sample_bytes])
            square_sum = math.fsum(sample * sample for (sample,) in samples)
            self._current_rms = int(math.sqrt(square_sum / (sample_bytes // 2)))
        except (TypeError, ValueError, struct.error):
            self._current_rms = 0

    def _check_idle_pause(self) -> None:
        idle_limit = self._cfg.idle_pause_seconds
        if idle_limit <= 0 or not self._armed.is_set():
            return
        if not self._gate_open():
            # Suppressed input (agent turn / TTS with barge off) is not idle.
            self._last_voice_activity = time.monotonic()
            return
        if self._activity_hold is not None:
            try:
                if self._activity_hold():
                    self._last_voice_activity = time.monotonic()
                    return
            except Exception:
                pass
        if time.monotonic() - self._last_voice_activity >= idle_limit:
            self.set_armed(False)
            if self._on_idle_pause is not None:
                try:
                    self._on_idle_pause()
                except Exception:
                    logger.debug("realtime on_idle_pause failed", exc_info=True)

    def _net_loop(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                ws = self._open_session()
            except Exception as exc:
                if self._stop.is_set():
                    return
                if attempt >= len(RECONNECT_DELAYS):
                    logger.warning("realtime voice connection failed permanently: %s", exc)
                    self._dead.set()
                    self._emit_state("dead", str(exc))
                    return
                delay = RECONNECT_DELAYS[attempt]
                attempt += 1
                self._emit_state("reconnecting", f"retry in {delay:.0f}s: {exc}")
                if self._stop.wait(delay):
                    return
                continue

            connected_at = time.monotonic()
            try:
                self._recv_loop(ws)
            except Exception as exc:
                logger.debug("realtime recv loop ended: %s", exc)
            finally:
                self._ws = None
                try:
                    ws.close()
                except Exception:
                    pass
            if self._stop.is_set():
                return
            # A connection that lived long enough was healthy: reconnect at
            # once with a fresh retry budget. One that the server accepted
            # and closed right away (rejected session, quota) counts as a
            # failed attempt — otherwise the loop redials with no delay,
            # forever, and never reports the session dead.
            if time.monotonic() - connected_at >= _STABLE_CONNECTION_S:
                attempt = 0
                self._emit_state("reconnecting", "connection lost")
                continue
            if attempt >= len(RECONNECT_DELAYS):
                logger.warning(
                    "realtime voice connection keeps closing right after connect; giving up"
                )
                self._dead.set()
                self._emit_state("dead", "connection closed repeatedly after connect")
                return
            delay = RECONNECT_DELAYS[attempt]
            attempt += 1
            self._emit_state("reconnecting", f"connection lost; retry in {delay:.0f}s")
            if self._stop.wait(delay):
                return

    def _open_session(self) -> Any:
        from tools.xai_http import resolve_xai_http_credentials  # lazy: heavy

        creds = resolve_xai_http_credentials()
        api_key = str(creds.get("api_key") or "").strip()
        if not api_key:
            raise RealtimeVoiceError("no xAI credentials available")
        url = f"{self._cfg.url}?model={self._cfg.model}"
        ws = self._connect_fn(url, {"Authorization": f"Bearer {api_key}"})
        # A blocking connect() cannot be interrupted, so stop() may have run while we dialled.
        # Configuring and publishing this socket would fire "connected" (and bill a session)
        # on a surface that already tore down.
        if self._stop.is_set():
            try:
                ws.close()
            except Exception:
                pass
            raise RealtimeVoiceError("session stopped during connect")
        with self._send_lock:
            ws.send(json.dumps(build_session_update(self._cfg)))
        # Flush disconnect-buffered audio before publishing the socket.
        while self._prebuffer:
            frame = self._prebuffer.popleft()
            with self._send_lock:
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(frame).decode("ascii"),
                }))
        self._ws = ws
        self._emit_state("connected", "")
        return ws

    def _recv_loop(self, ws: Any) -> None:
        while not self._stop.is_set():
            raw = ws.recv()
            if isinstance(raw, (bytes, bytearray, memoryview)):
                continue  # audio arrives as base64 JSON deltas, not binary frames
            try:
                event = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if isinstance(event, dict):
                self._handle_event(event)

    def _send_raw(self, payload: str) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            with self._send_lock:
                ws.send(payload)
            return True
        except Exception as exc:
            logger.debug("realtime send failed: %s", exc)
            return False

    def _send_event(self, event: Dict[str, Any]) -> bool:
        return self._send_raw(json.dumps(event))

    def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _mark_voice_activity(self) -> None:
        self._last_voice_activity = time.monotonic()

    # -- supervisor speech playback ------------------------------------------

    def _enqueue_playout(self, chunk: bytes) -> None:
        # is_alive check mirrors the tool-result worker: a sink exception
        # kills _playout_loop, and a dead-but-referenced thread would
        # otherwise silently drop all supervisor speech for the session.
        if self._playout_thread is None or not self._playout_thread.is_alive():
            self._playout_thread = threading.Thread(
                target=self._playout_loop, name="voice-rt-playout", daemon=True
            )
            self._playout_thread.start()
        try:
            self._playout_q.put_nowait(chunk)
        except queue.Full:
            pass  # sustained overrun — dropping late audio beats blocking recv

    def _playout_loop(self) -> None:
        """Feed queued PCM (24 kHz mono int16) into the playout sink."""
        try:
            sink = self._playout_sink_factory()
        except Exception as exc:
            logger.warning("supervisor playback unavailable: %s", exc)
            return
        self._playout_sink = sink
        set_active = getattr(sink, "set_active", None)
        pending = getattr(sink, "pending", None)
        marked = False

        def _mark(active: bool) -> None:
            nonlocal marked
            if marked == active:
                return
            marked = active
            self._playing = active
            if set_active is not None:
                try:
                    set_active(active)
                except Exception:
                    pass

        def _sink_pending() -> bool:
            if pending is None:
                return False
            try:
                return bool(pending())
            except Exception:
                return False

        try:
            while not self._stop.is_set():
                try:
                    chunk = self._playout_q.get(timeout=0.25)
                except queue.Empty:
                    # Buffering sinks are still audible after the queue
                    # drains — stay "playing" until they report empty.
                    if not _sink_pending():
                        _mark(False)
                    continue
                _mark(True)
                sink.write(chunk)
        except Exception as exc:
            logger.warning("supervisor playback stopped: %s", exc)
        finally:
            _mark(False)
            self._playout_sink = None
            try:
                sink.close()
            except Exception:
                pass

    # -- server events (dispatched by ``type`` through _EVENT_HANDLERS) -----------

    def _handle_event(self, event: Dict[str, Any]) -> None:
        handler = self._EVENT_HANDLERS.get(str(event.get("type") or ""))
        if handler is not None:
            handler(self, event)

    def _evt_speech_started(self, event: Dict[str, Any]) -> None:
        self._mark_voice_activity()
        if self._cfg.supervisor:
            # Barge-in: the server interrupts its own response in VAD
            # mode; drop the locally queued remainder to match.
            self.clear_playout()
        if self._armed.is_set() and self._gate_open() and self._on_speech_started:
            try:
                self._on_speech_started()
            except Exception:
                logger.debug("realtime on_speech_started failed", exc_info=True)

    def _evt_speech_stopped(self, event: Dict[str, Any]) -> None:
        self._mark_voice_activity()
        if self._armed.is_set() and self._gate_open() and self._on_speech_stopped:
            try:
                self._on_speech_stopped()
            except Exception:
                logger.debug("realtime on_speech_stopped failed", exc_info=True)

    def _evt_transcription_completed(self, event: Dict[str, Any]) -> None:
        self._mark_voice_activity()
        transcript = str(event.get("transcript") or "").strip()
        if transcript and self._armed.is_set() and self._gate_open():
            try:
                self._on_transcript(transcript)
            except Exception:
                logger.warning("realtime transcript handler failed", exc_info=True)

    def _evt_response_created(self, event: Dict[str, Any]) -> None:
        self._active_response = True
        self._response_had_audio = False
        if not self._cfg.supervisor:
            # Ears relay stays silent: cancel anything the server creates.
            self._send_event({"type": "response.cancel"})

    def _evt_response_ended(self, event: Dict[str, Any]) -> None:
        self._active_response = False

    def _evt_audio_delta(self, event: Dict[str, Any]) -> None:
        self._response_had_audio = True
        b64 = event.get("delta") or event.get("audio") or ""
        if self._cfg.supervisor and self._armed.is_set() and b64:
            try:
                self._enqueue_playout(base64.b64decode(b64))
            except (ValueError, TypeError):
                logger.debug("realtime: bad base64 audio delta")

    def _evt_function_call_done(self, event: Dict[str, Any]) -> None:
        name = str(event.get("name") or "")
        call_id = str(event.get("call_id") or "")
        args = event.get("arguments")
        args_json = args if isinstance(args, str) else json.dumps(args or {})
        if self._on_function_call and name and call_id:
            try:
                self._on_function_call(name, call_id, args_json)
            except Exception:
                logger.warning("realtime function-call handler failed", exc_info=True)

    def _evt_error(self, event: Dict[str, Any]) -> None:
        detail = event.get("error") or event.get("message") or event
        logger.warning("realtime voice server error: %s", detail)
        if not self._minimal_retry_done:
            # Full config may carry unsupported extras — downgrade once.
            self._minimal_retry_done = True
            self._send_event(build_session_update(self._cfg, minimal=True))

    _EVENT_HANDLERS: Dict[str, Callable[["RealtimeVoiceSession", Dict[str, Any]], None]] = {
        "input_audio_buffer.speech_started": _evt_speech_started,
        "input_audio_buffer.speech_stopped": _evt_speech_stopped,
        "conversation.item.input_audio_transcription.completed": _evt_transcription_completed,
        "response.created": _evt_response_created,
        "response.done": _evt_response_ended,
        "response.completed": _evt_response_ended,
        "response.cancelled": _evt_response_ended,
        "response.output_audio.delta": _evt_audio_delta,
        "response.audio.delta": _evt_audio_delta,
        "response.function_call_arguments.done": _evt_function_call_done,
        "error": _evt_error,
    }


__all__ = ["RealtimeVoiceSession"]
