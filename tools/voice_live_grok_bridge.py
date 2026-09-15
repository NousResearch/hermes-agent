"""Grok-Live realtime bridge: the backend-held xAI websocket session manager.

This is the ``grok-live`` engine's server-side half (docs/grok-live-voice/SPEC.md §4.2, §7,
§9, §10). Unlike gpt-live (renderer-negotiated WebRTC), the gateway process holds the xAI
realtime websocket exclusively and relays audio both ways over the existing JSON-RPC channel
(base64 PCM16 chunks, SPEC §4.1 — the ``wake.feed`` precedent, NOT raw binary WS frames).

The bridge is transport-agnostic: it never imports ``tui_gateway``. Events reach the renderer
through the ``emit`` callable the methods layer injects (``tui_gateway/methods_voice_grok.py``
binds it to ``server._emit`` addressed at the owning session), and the delegation seam (SPEC §5)
is the injectable ``delegation_sink`` — the actual ``prompt.submit`` call is sibling card
t_f7cae076's; this module owns accumulation, the settle flush and the ``voice.grok.delegation``
event only.

Ported from the proven standalone reference (``/home/ericio/grok_live_voice.py``, live on this
box 2026-09-15): same wire contract, VAD/session config, playback-echo gate and 5s reconnect —
minus PortAudio (the renderer owns mic/speaker) and minus the local function-tool bridge
(decision #6: v1 ships zero local function tools).

Threading model: one daemon thread owns one asyncio loop; the RPC-facing methods
(:meth:`GrokLiveBridge.send_mic`, :meth:`set_muted`, :meth:`stop`, :meth:`status`) are
thread-safe and funnel onto that loop. ``--check`` mirrors the reference script's smoke:
connect, ``session.update``, wait for ``session.updated``, print, exit.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Audio contract (SPEC §4): PCM16 mono 24 kHz, ~100ms chunks both directions.
SAMPLE_RATE = 24_000
CHUNK_FRAMES = 2_400                       # 100 ms at 24 kHz
CHUNK_BYTES = CHUNK_FRAMES * 2             # PCM16 little-endian
UPSTREAM_BACKPRESSURE_CHUNKS = 3           # >300ms queued upstream => drop, never queue stale mic
DEGRADED_DROP_STREAK = 5                   # consecutive drops before a "degraded" state event
PLAYBACK_ECHO_TAIL_S = 0.45                # AEC tail after the last playback chunk (SPEC §7)
RECONNECT_DELAY_S = 5.0                    # matches the reference script's outer loop (SPEC §10)
UTTERANCE_SETTLE_S = 1.5                   # mirrors UTTERANCE_SETTLE_MS in use-voice-live-conversation.ts
HANDSHAKE_TIMEOUT_S = 20.0
MAX_EXCHANGE_FRAGMENTS = 12                # recent spoken exchange kept for delegation context

XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime"

# Event types this bridge emits through the injected sink (contract: contracts/prompt_voice_grok.py).
EVENT_AUDIO = "voice.grok.audio"
EVENT_TRANSCRIPT = "voice.grok.transcript"
EVENT_STATE = "voice.grok.state"
EVENT_DELEGATION = "voice.grok.delegation"


def realtime_url(model: str) -> str:
    return f"{XAI_REALTIME_URL}?model={model}"


def session_update_payload(live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ``session.update`` frame, built from the ``voice.grok_live`` config section
    (SPEC §4.2): server VAD, PCM16 24kHz binary transport in/out, persona instructions — and
    NO ``tools`` array (decision #6: zero local function tools in v1)."""
    from tools.voice_live_grok import (
        DEFAULT_GROK_LIVE_MODEL, DEFAULT_GROK_LIVE_VOICE, _grok_live_section, live_instructions)
    live = live if live is not None else _grok_live_section()
    return {
        "type": "session.update",
        "session": {
            "reasoning": {"effort": "none"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": float(live.get("vad_threshold") or 0.75),
                "silence_duration_ms": int(live.get("silence_ms") or 700),
                "prefix_padding_ms": int(live.get("prefix_ms") or 333),
            },
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transport": "binary",
                    # Input transcription feeds the delegation seam (SPEC §5); the model name
                    # is the one the reference script proved live.
                    "transcription": {"model": "grok-transcribe", "language_hint": "en"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transport": "binary",
                    "speed": float(live.get("speed") or 1.0),
                },
            },
            "voice": str(live.get("voice") or DEFAULT_GROK_LIVE_VOICE),
            "instructions": live_instructions(live),
        },
    }


class _PlaybackGate:
    """Half-duplex AEC guard (SPEC §7, ported from the reference script): while playback audio
    is flowing (plus a tail), upstream mic chunks are withheld so the speaker does not re-enter
    the mic as a false user turn. The clock is injectable for tests."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 tail_s: float = PLAYBACK_ECHO_TAIL_S) -> None:
        self._until = 0.0
        self._clock = clock
        self._tail_s = tail_s

    def bump(self, extra: Optional[float] = None) -> None:
        self._until = max(self._until, self._clock() + (self._tail_s if extra is None else extra))

    def blocked(self) -> bool:
        return self._clock() < self._until


def _merge_turns(fragments: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Consecutive same-speaker fragments merge — mirrors ``delegationPrompt()`` in
    apps/desktop/.../use-voice-live-conversation.ts exactly (SPEC §5 step 1)."""
    turns: List[Dict[str, str]] = []
    for fragment in fragments:
        if turns and turns[-1]["speaker"] == fragment["speaker"]:
            turns[-1]["text"] += fragment["text"]
        else:
            turns.append({"speaker": fragment["speaker"], "text": fragment["text"]})
    return turns


def build_delegation(fragments: List[Dict[str, str]]) -> Dict[str, str]:
    """``{context, prompt}`` from the recent spoken exchange — the server-side mirror of the
    renderer's ``delegationPrompt()``: last user turn is the prompt, the full recent exchange
    (whitespace-collapsed, ``User:`` / ``Voice assistant:`` lines) is the context."""
    turns = _merge_turns(fragments)
    prompt = ""
    for turn in reversed(turns):
        if turn["speaker"] == "user":
            prompt = " ".join(turn["text"].split())
            break
    lines = []
    for turn in turns:
        text = " ".join(turn["text"].split())
        if text:
            lines.append(f"{'User' if turn['speaker'] == 'user' else 'Voice assistant'}: {text}")
    context = "\n".join(lines)
    return {"context": context, "prompt": prompt or context[-400:]}


class GrokLiveBridge:
    """One logical grok-live conversation: owns the xAI websocket, the audio relay both ways,
    the AEC gate, the reconnect loop and the delegation accumulation for one Hermes session.

    Parameters
    ----------
    session_id:
        The Hermes session this conversation belongs to (event routing + reconnect identity).
    emit:
        Thread-safe ``(event_type, payload) -> None`` sink for ``voice.grok.*`` events; the
        methods layer binds it to ``server._emit`` addressed at ``session_id``.
    delegation_sink:
        Optional ``(session_id, delegation_id, prompt, context) -> None`` called after the
        ``voice.grok.delegation`` event fires. None in v1 — t_f7cae076 wires the actual
        ``prompt.submit`` here (SPEC §5 / seam inventory).
    live_config:
        Snapshot of the ``voice.grok_live`` config section taken at start; reconnects re-send
        it verbatim (xAI requires a fresh ``session.updated`` handshake every time).
    connect_factory:
        Injectable async websocket connector ``(url, headers) -> async context manager``
        (production: ``websockets.connect``; tests: fakes, no network).
    """

    def __init__(
        self,
        *,
        session_id: str,
        emit: Callable[[str, Dict[str, Any]], None],
        delegation_sink: Optional[Callable[[str, str, str, str], None]] = None,
        live_config: Optional[Dict[str, Any]] = None,
        connect_factory: Optional[Callable[..., Any]] = None,
        reconnect_delay_s: float = RECONNECT_DELAY_S,
        settle_s: float = UTTERANCE_SETTLE_S,
        echo_tail_s: float = PLAYBACK_ECHO_TAIL_S,
        handshake_timeout_s: float = HANDSHAKE_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_id = session_id
        self._emit = emit
        self._delegation_sink = delegation_sink
        self._live = dict(live_config or {})
        self._connect_factory = connect_factory or self._default_connect
        self._reconnect_delay_s = reconnect_delay_s
        self._settle_s = settle_s
        self._handshake_timeout_s = handshake_timeout_s
        self._gate = _PlaybackGate(clock=clock, tail_s=echo_tail_s)

        self._state_lock = threading.Lock()
        self._state = "idle"             # last state emitted (incl. degraded overlay)
        self._base_state = "idle"        # listening | speaking | thinking — degraded overlays this
        self._established = False        # session.updated seen at least once this attempt
        self._ever_established = False   # ...on any attempt (start-failure vs reconnect, SPEC §9)
        self._muted = False
        self._reconnects = 0
        self._dropped_chunks = 0
        self._consecutive_drops = 0

        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._mic_q: Optional[asyncio.Queue] = None
        self._ws: Any = None

        # Delegation accumulation (SPEC §5): utterance items of the current user turn +
        # the recent merged exchange for context.
        self._utterance_items: Dict[str, str] = {}
        self._exchange: List[Dict[str, str]] = []
        self._settle_task: Optional[asyncio.Task] = None

    # ── public, thread-safe API (called from RPC threads) ─────────────────────

    def start(self) -> None:
        """Spawn the bridge thread + asyncio loop and begin the connect sequence. State
        transitions arrive as ``voice.grok.state`` events (``connecting`` → ``listening`` …)."""
        if self._thread is not None:
            raise RuntimeError("grok-live bridge already started")
        self._thread = threading.Thread(
            target=self._thread_main, name=f"grok-live-{self.session_id[:8]}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> bool:
        """User-initiated stop: cancel any in-flight session or pending reconnect, close the
        websocket, emit ``idle``. Symmetric with gpt-live's ``end()`` (SPEC §9)."""
        self._stop_requested.set()
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._cancel_task)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

    def send_mic(self, pcm: bytes) -> Dict[str, Any]:
        """Relay one upstream mic chunk (~100ms PCM16 24kHz) toward xAI. Drop-not-queue when
        the send side is backed up (>300ms of audio pending — stale live audio would desync
        the server VAD, SPEC §4.1 backpressure)."""
        loop, mic_q = self._loop, self._mic_q
        if loop is None or mic_q is None or self._stop_requested.is_set():
            return {"accepted": False, "reason": "not_running"}
        if not self._established:
            return {"accepted": False, "reason": "connecting"}
        if mic_q.qsize() >= UPSTREAM_BACKPRESSURE_CHUNKS:
            with self._state_lock:
                self._dropped_chunks += 1
                self._consecutive_drops += 1
                streak = self._consecutive_drops
                already_degraded = self._state == "degraded"
            if streak >= DEGRADED_DROP_STREAK and not already_degraded:
                self._set_state("degraded", "upstream backpressure: mic audio dropped", overlay=True)
            return {"accepted": False, "dropped": True, "reason": "backpressure"}
        loop.call_soon_threadsafe(mic_q.put_nowait, pcm)
        return {"accepted": True}

    def set_muted(self, muted: bool) -> bool:
        """Explicit mute toggle — independent of the AEC gate (SPEC §4.1)."""
        self._muted = bool(muted)
        return self._muted

    def speak_reply(self, text: str) -> bool:
        """Make xAI speak ``text`` verbatim (SPEC §5 step 5 / risk #6 — the implementation spike):
        a Hermes-generated reply the voice model never produced itself needs to be read aloud
        without re-prompting the model (which would risk paraphrasing or refusing to repeat it).

        xAI's realtime API resolves this with an extension the reference script never needed
        (it let xAI both hear and answer): ``conversation.item.create`` with ``item.type ==
        "force_message"`` makes the agent speak a hard-coded, TTS-synthesized line with NO model
        involvement — no ``response.create`` follows (the force message IS the turn), so it never
        re-enters function-calling / reasoning (docs.x.ai/developers/model-capabilities/audio/
        speech-to-speech#force-message, confirmed 2026-09-15). ``interruptible: True`` (default)
        so the AEC gate's normal barge-in behavior still applies."""
        loop, ws = self._loop, self._ws
        text = text.strip()
        if not text or loop is None or ws is None or self._stop_requested.is_set():
            return False
        frame = json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "force_message",
                "role": "assistant",
                "interruptible": True,
                "content": [{"type": "output_text", "text": text}],
            },
        })
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._send_speak_frame(frame)))
        return True

    async def _send_speak_frame(self, frame: str) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(frame)
        except Exception:
            logger.debug("grok-live: speak_reply send failed", exc_info=True)

    def status(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "session_id": self.session_id,
                "state": self._state,
                "muted": self._muted,
                "established": self._ever_established,
                "reconnects": self._reconnects,
                "dropped_chunks": self._dropped_chunks,
            }

    @property
    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop_requested.is_set()

    # ── event emission helpers ────────────────────────────────────────────────

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            self._emit(event_type, payload)
        except Exception:
            logger.debug("grok-live: emit %s failed", event_type, exc_info=True)

    def _set_state(self, state: str, reason: Optional[str] = None, *, overlay: bool = False) -> None:
        """Emit a ``voice.grok.state`` transition. ``overlay`` states (degraded) do not replace
        the base listening/speaking/thinking state recovery returns to."""
        with self._state_lock:
            if not overlay:
                if state in ("listening", "speaking", "thinking"):
                    self._base_state = state
                if state == self._state and reason is None:
                    return  # suppress duplicate transition spam
            self._state = state
        payload: Dict[str, Any] = {"session_id": self.session_id, "state": state}
        if reason is not None:
            payload["reason"] = reason
        self._emit_event(EVENT_STATE, payload)

    def _recover_from_overlay(self) -> None:
        with self._state_lock:
            if self._state != "degraded":
                return
            base = self._base_state
        self._set_state(base)

    # ── thread / loop plumbing ────────────────────────────────────────────────

    @staticmethod
    def _default_connect(url: str, headers: Dict[str, str]):
        import websockets
        return websockets.connect(url, additional_headers=headers,
                                  ping_interval=20, ping_timeout=20)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._mic_q = asyncio.Queue()
        try:
            loop.run_until_complete(self._run())
        except Exception:
            logger.exception("grok-live: bridge loop crashed")
            try:
                self._set_state("error", "bridge loop crashed")
            except Exception:
                pass
        finally:
            self._loop = None
            try:
                loop.close()
            except Exception:
                pass

    def _cancel_task(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()

    # ── the session state machine (loop thread) ───────────────────────────────

    async def _run(self) -> None:
        """Outer lifecycle (SPEC §9/§10): a start failure is reported and NOT retried; an
        established session that drops reconnects after 5s until the user stops."""
        self._set_state("connecting")
        self._task = asyncio.current_task()
        first_attempt = True
        try:
            while not self._stop_requested.is_set():
                try:
                    await self._run_session()
                    break  # clean exit only happens on explicit stop
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if self._stop_requested.is_set():
                        break
                    reason = self._reason_of(exc)
                    if first_attempt and not self._ever_established:
                        # Explicit start failure: surface it, do NOT silently retry (SPEC §9).
                        logger.warning("grok-live: start failed: %s", reason)
                        self._set_state("error", reason)
                        return
                    logger.info("grok-live: session dropped (%s); reconnecting in %ss",
                                reason, self._reconnect_delay_s)
                    with self._state_lock:
                        self._reconnects += 1
                    self._set_state("reconnecting", reason)
                    await self._interruptible_sleep(self._reconnect_delay_s)
                finally:
                    first_attempt = False
                    self._established = False
        except asyncio.CancelledError:
            pass
        finally:
            await self._teardown()
            if self._stop_requested.is_set():
                self._set_state("idle")

    async def _interruptible_sleep(self, seconds: float) -> None:
        # Poll the stop flag so a user stop during the reconnect wait cancels the pending
        # retry promptly (SPEC §10) without relying on task cancellation alone.
        deadline = time.monotonic() + seconds
        while not self._stop_requested.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.1))

    @staticmethod
    def _reason_of(exc: BaseException) -> str:
        text = str(exc).strip()
        return f"{type(exc).__name__}: {text}" if text else type(exc).__name__

    def _resolve_credential(self) -> str:
        from tools.voice_live_grok import _resolve_grok_credentials
        return _resolve_grok_credentials(self._live)

    async def _run_session(self) -> None:
        """One xAI realtime session: credential → connect → session.update → handshake →
        relay until the socket dies (fatal) or the user stops."""
        from tools.voice_live_grok import DEFAULT_GROK_LIVE_MODEL, NO_CREDENTIAL_REASON
        credential = self._resolve_credential()
        if not credential:
            raise RuntimeError(NO_CREDENTIAL_REASON)
        model = str(self._live.get("model") or DEFAULT_GROK_LIVE_MODEL)
        url = realtime_url(model)
        headers = {"Authorization": f"Bearer {credential}"}
        handshake = asyncio.Event()
        self._start_error: Optional[str] = None

        connect_cm = self._connect_factory(url, headers)
        async with connect_cm as ws:
            self._ws = ws
            await ws.send(json.dumps(session_update_payload(self._live)))
            mic_task = asyncio.create_task(self._mic_pump(ws))
            recv_task = asyncio.create_task(self._receiver(ws, handshake))
            try:
                handshake_waiter = asyncio.create_task(handshake.wait())
                try:
                    # Racing the handshake against the pump/receiver so a socket that dies
                    # mid-handshake fails fast instead of hanging until the timeout.
                    await asyncio.wait_for(
                        self._first_of({handshake_waiter, recv_task, mic_task}),
                        timeout=self._handshake_timeout_s)
                finally:
                    handshake_waiter.cancel()
                if self._start_error is not None:
                    raise RuntimeError(self._start_error)
                if not handshake.is_set():
                    raise ConnectionError("xAI realtime connection closed during handshake")
                self._established = self._ever_established = True
                self._set_state("listening")
                # Mirrors the reference script's FIRST_COMPLETED wait: a dead mic pump
                # (ConnectionClosed on send) is just as fatal as a dead receiver (§9).
                await self._first_of({mic_task, recv_task})
            finally:
                for task in (mic_task, recv_task):
                    task.cancel()
                await asyncio.gather(mic_task, recv_task, return_exceptions=True)
                self._ws = None

    @staticmethod
    async def _first_of(tasks) -> None:
        """Wait until any task finishes; re-raise its exception when it failed. Does NOT
        cancel the survivors — the caller decides what the first completion means."""
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc

    async def _teardown(self) -> None:
        if self._settle_task is not None:
            self._settle_task.cancel()
            self._settle_task = None
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    # ── upstream relay (renderer → xAI) ───────────────────────────────────────

    async def _mic_pump(self, ws: Any) -> None:
        """Drain the upstream queue into the websocket. The AEC gate and the explicit mute
        withhold forwarding (SPEC §7: the renderer keeps streaming; the backend decides)."""
        mic_q = self._mic_q
        assert mic_q is not None
        while True:
            pcm = await mic_q.get()
            if self._stop_requested.is_set():
                return
            if self._muted or self._gate.blocked():
                continue  # withheld — NOT counted as a backpressure drop
            await ws.send(pcm)  # binary PCM16 little-endian on the backend leg
            recovered = False
            with self._state_lock:
                self._consecutive_drops = 0
                recovered = self._state == "degraded"
            if recovered:
                self._recover_from_overlay()

    # ── downstream relay + event handling (xAI → renderer / delegation) ───────

    async def _receiver(self, ws: Any, handshake: asyncio.Event) -> None:
        """Consume xAI events. Reaching the end of the iterator without a user stop means the
        remote session ended — FATAL (SPEC §9): raise so the outer loop reconnects."""
        async for msg in ws:
            if self._stop_requested.is_set():
                return
            if isinstance(msg, (bytes, bytearray)):
                self._relay_downstream(bytes(msg))
                continue
            try:
                event = json.loads(msg)
            except Exception:
                logger.debug("grok-live: unparseable event frame: %r", msg[:200])
                continue
            self._handle_event(event, handshake)
        if not self._stop_requested.is_set():
            raise ConnectionError("xAI realtime receiver ended")

    def _relay_downstream(self, pcm: bytes) -> None:
        """One ~100ms of Grok's spoken reply → ``voice.grok.audio`` event. Downstream chunks
        are never dropped (SPEC §4.1); the playback gate feeds off this flow."""
        self._gate.bump()
        seq = getattr(self, "_out_seq", 0) + 1
        self._out_seq = seq
        self._emit_event(EVENT_AUDIO, {
            "session_id": self.session_id,
            "pcm_b64": base64.b64encode(pcm).decode("ascii"),
            "seq": seq,
        })
        if self._base_state != "speaking":
            self._set_state("speaking")

    def _handle_event(self, event: Dict[str, Any], handshake: asyncio.Event) -> None:
        etype = str(event.get("type") or "")
        if etype == "session.created":
            logger.info("grok-live: session.created (session=%s)", self.session_id)
        elif etype == "session.updated":
            handshake.set()
        elif etype == "error":
            detail = json.dumps(event.get("error") or event)[:400]
            if not handshake.is_set():
                self._start_error = f"xAI realtime error: {detail}"
                handshake.set()
            else:
                # Established-session error event: surface it; the socket close that follows
                # drives the reconnect (SPEC §9 error taxonomy).
                logger.warning("grok-live: xAI error event: %s", detail)
                self._set_state("error", f"xAI realtime error: {detail}")
        elif etype == "input_audio_buffer.speech_stopped":
            self._schedule_settle_flush()
        elif etype == "conversation.item.input_audio_transcription.updated":
            item_id = str(event.get("item_id") or "")
            delta = str(event.get("delta") or event.get("transcript") or event.get("text") or "")
            if delta:
                self._utterance_items[item_id] = self._utterance_items.get(item_id, "") + delta
                self._emit_transcript("user", delta, item_id)
        elif etype == "conversation.item.input_audio_transcription.completed":
            item_id = str(event.get("item_id") or "")
            transcript = str(event.get("transcript") or event.get("text") or "")
            if transcript:
                # The completed transcript finalizes the accumulated deltas for that item.
                self._utterance_items[item_id] = transcript
                self._emit_transcript("user", transcript, item_id)
        elif etype == "response.output_audio_transcript.done":
            text = str(event.get("transcript") or event.get("text") or "")
            if text:
                self._exchange.append({"speaker": "assistant", "text": text})
                self._exchange = self._exchange[-MAX_EXCHANGE_FRAGMENTS:]
                self._emit_transcript("assistant", text, str(event.get("item_id") or ""))
        elif etype in ("response.output_audio.delta", "response.audio.delta"):
            delta = event.get("delta") or event.get("audio")
            if delta:
                try:
                    self._relay_downstream(base64.b64decode(delta))
                except Exception:
                    logger.debug("grok-live: bad audio delta payload", exc_info=True)
        elif etype in ("response.output_audio.done", "response.audio.done"):
            self._gate.bump()  # extend the echo tail past the final chunk
        elif etype == "response.done":
            self._set_state("listening")
        elif etype == "response.function_call_arguments.done":
            # v1 registers no tools, so xAI should never emit this (SPEC §4.2): log, don't crash.
            logger.warning("grok-live: unexpected function call %r ignored (no tools registered)",
                           event.get("name"))
        elif etype in ("conversation.created", "input_audio_buffer.speech_started"):
            pass  # known, no action
        else:
            logger.debug("grok-live: unhandled xAI event type %r", etype)

    def _emit_transcript(self, speaker: str, text: str, item_id: str) -> None:
        self._emit_event(EVENT_TRANSCRIPT, {
            "session_id": self.session_id, "speaker": speaker, "text": text,
            "item_id": item_id or None,
        })

    # ── delegation (SPEC §5 — event + seam; prompt.submit lands in t_f7cae076) ──

    def _schedule_settle_flush(self) -> None:
        """Mirror gpt-live's utterance settle: flush the accumulated user turn ~1.5s after
        speech_stopped so late transcription deltas still land in the same delegation."""
        if self._settle_task is not None and not self._settle_task.done():
            self._settle_task.cancel()
        self._settle_task = asyncio.create_task(self._settle_then_flush())

    async def _settle_then_flush(self) -> None:
        try:
            await asyncio.sleep(self._settle_s)
        except asyncio.CancelledError:
            return
        self._flush_delegation()

    def _flush_delegation(self) -> None:
        items, self._utterance_items = self._utterance_items, {}
        user_text = " ".join(t for t in (x.strip() for x in items.values()) if t)
        if not user_text:
            return
        self._exchange.append({"speaker": "user", "text": user_text})
        self._exchange = self._exchange[-MAX_EXCHANGE_FRAGMENTS:]
        delegation = build_delegation(self._exchange)
        if not delegation["prompt"]:
            return
        delegation_id = f"grok-{uuid.uuid4().hex[:12]}"
        self._emit_event(EVENT_DELEGATION, {
            "session_id": self.session_id,
            "delegation_id": delegation_id,
            "context": delegation["context"],
        })
        self._set_state("thinking")
        if self._delegation_sink is not None:
            try:
                self._delegation_sink(self.session_id, delegation_id,
                                      delegation["prompt"], delegation["context"])
            except Exception:
                logger.exception("grok-live: delegation sink failed")


# ── --check smoke (mirrors the reference script's proven probe) ─────────────────


async def _check_connection() -> int:
    from tools.voice_live_grok import (
        DEFAULT_GROK_LIVE_MODEL, DEFAULT_GROK_LIVE_VOICE, NO_CREDENTIAL_REASON,
        _grok_live_section, _resolve_grok_credentials)
    live = _grok_live_section()
    credential = _resolve_grok_credentials(live)
    if not credential:
        print(NO_CREDENTIAL_REASON, file=sys.stderr)
        return 2
    model = str(live.get("model") or DEFAULT_GROK_LIVE_MODEL)
    voice = str(live.get("voice") or DEFAULT_GROK_LIVE_VOICE)
    headers = {"Authorization": f"Bearer {credential}"}
    async with GrokLiveBridge._default_connect(realtime_url(model), headers) as ws:
        await ws.send(json.dumps(session_update_payload(live)))
        for _ in range(20):
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            if isinstance(msg, (bytes, bytearray)):
                continue
            event = json.loads(msg)
            print(event.get("type"), json.dumps(event)[:500])
            if event.get("type") == "session.updated":
                print(f"OK: connected to xAI Grok Voice API model={model}, voice={voice}")
                return 0
            if event.get("type") == "error":
                print(f"ERROR: {event}", file=sys.stderr)
                return 1
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Grok-Live realtime bridge (Hermes voice engine)")
    parser.add_argument("--check", action="store_true",
                        help="connect + session.update + wait for session.updated, then exit")
    args = parser.parse_args()
    if args.check:
        return asyncio.run(_check_connection())
    parser.error("only --check is supported standalone; the gateway drives live sessions")
    return 2


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
