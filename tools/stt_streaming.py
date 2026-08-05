"""Provider-neutral streaming speech-to-text sessions and coordination.

Capture surfaces may enqueue copied audio frames through the coordinator's
non-blocking frame sink. WebSocket I/O and sample-rate conversion run on the
coordinator's daemon thread.

Streaming is deliberately transcription-only.  Hermes continues to own turn
submission, agent execution, TTS, and barge-in handling.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# Two seconds absorbs ordinary scheduling and network jitter while failing over
# before a live transcript falls noticeably behind the speaker. This is an
# internal backpressure policy, not a user-facing quality or endpointing knob.
_MAX_BUFFERED_AUDIO_SECONDS = 2.0
_AUDIO_SEND_BATCH_SIZE = 16
# ElevenLabs enforces an account-dependent Realtime STT session lifetime but
# does not include that limit in session_started metadata. A 60-second ceiling
# leaves room for the next utterance below the shortest observed limit while
# still reusing each physical connection across several turns.
_ELEVENLABS_SESSION_RECYCLE_SECONDS = 60.0


@dataclass(frozen=True)
class StreamingTranscriptEvent:
    """Normalized transcript event emitted by a provider session."""

    text: str
    is_final: bool
    event_id: Optional[str] = None
    segment_id: Optional[str] = None


@dataclass
class _UtteranceState:
    generation: int = 0
    active: bool = False
    accepting_audio: bool = False
    has_sent_audio: bool = False
    has_activity: bool = False
    failure_pending: bool = False
    error_reported: bool = False
    resampler: Optional["_PCM16StreamResampler"] = None


class StreamingSTTSession(ABC):
    """One persistent transcription-only provider connection."""

    sample_rate: int

    @abstractmethod
    async def send_audio(self, pcm16: bytes) -> None:
        """Send mono, little-endian PCM16 at ``sample_rate``."""

    @abstractmethod
    async def commit(self) -> None:
        """Force the current utterance to commit without closing the session."""

    @abstractmethod
    async def receive_event(self) -> Optional[StreamingTranscriptEvent]:
        """Wait for and normalize the next provider event."""

    @abstractmethod
    async def close(self) -> None:
        """Close the physical provider connection."""


class StreamingSTTProvider(ABC):
    """Factory for one provider's persistent streaming STT session."""

    name: str
    session_recycle_seconds: Optional[float] = None

    @property
    def configuration_key(self) -> tuple[Any, ...]:
        """Return the connection settings that determine session reuse."""
        return (self.name,)

    @abstractmethod
    async def open_session(self) -> StreamingSTTSession:
        """Open and configure a transcription-only session."""


async def _connect(url: str, headers: Dict[str, str]):
    import websockets

    return await websockets.connect(
        url,
        additional_headers=headers,
        open_timeout=10,
        close_timeout=3,
        max_size=2 * 1024 * 1024,
    )


def _websocket_endpoint(base_url: str, endpoint: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/")
    endpoint_path = f"/{endpoint.strip('/')}"
    if not path.endswith(endpoint_path):
        path = f"{path}{endpoint_path}"
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, ""))


def _with_query(url: str, params: Dict[str, Any]) -> str:
    parsed = urlsplit(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.extend(params.items())
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urlencode(query, doseq=True),
        "",
    ))


class _JSONWebSocketSession(StreamingSTTSession):
    """Shared JSON send/receive and close handling for provider adapters."""

    def __init__(self, websocket: Any):
        self.websocket = websocket

    async def _receive_json(self) -> Dict[str, Any]:
        raw = await self.websocket.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise RuntimeError("Streaming STT provider returned a non-object event")
        return message

    async def _send_json(self, message: Dict[str, Any]) -> None:
        await self.websocket.send(json.dumps(message))

    async def close(self) -> None:
        await self.websocket.close()


class OpenAIStreamingSTTSession(StreamingSTTSession):
    sample_rate = 24000

    def __init__(self, client: Any, connection: Any):
        self.client = client
        self.connection = connection

    async def send_audio(self, pcm16: bytes) -> None:
        await self.connection.input_audio_buffer.append(
            audio=base64.b64encode(pcm16).decode("ascii"),
        )

    async def commit(self) -> None:
        await self.connection.input_audio_buffer.commit()

    async def receive_event(self) -> Optional[StreamingTranscriptEvent]:
        event = await self.connection.recv()
        event_type = event.type
        if event_type == "conversation.item.input_audio_transcription.delta":
            return StreamingTranscriptEvent(
                text=str(event.delta or ""),
                is_final=False,
                event_id=getattr(event, "event_id", None),
                segment_id=getattr(event, "item_id", None),
            )
        if event_type == "conversation.item.input_audio_transcription.completed":
            return StreamingTranscriptEvent(
                text=str(event.transcript or ""),
                is_final=True,
                event_id=getattr(event, "event_id", None),
                segment_id=getattr(event, "item_id", None),
            )
        if event_type == "conversation.item.input_audio_transcription.failed":
            error = getattr(event, "error", None)
            raise RuntimeError(
                str(
                    getattr(error, "message", None)
                    or "OpenAI Realtime transcription failed"
                )
            )
        if event_type == "error":
            message = str(
                event.error.message or "OpenAI Realtime transcription failed"
            )
            if (
                "input audio buffer" in message.lower()
                and "buffer too small" in message.lower()
            ):
                # Semantic VAD may commit immediately before a manual commit
                # reaches the server. Keep receiving: the matching final can
                # still be in flight, and the controller timeout remains the
                # bounded fallback when it is not.
                return None
            raise RuntimeError(message)
        return None

    async def close(self) -> None:
        try:
            await self.connection.close()
        finally:
            await self.client.close()


class OpenAIStreamingSTTProvider(StreamingSTTProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        language: str,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.language = language

    @property
    def configuration_key(self) -> tuple[Any, ...]:
        return (
            self.name,
            self.api_key,
            self.base_url,
            self.model,
            self.language,
        )

    async def open_session(self) -> StreamingSTTSession:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url or None,
        )
        try:
            connection = await client.realtime.connect(
                extra_query={"intent": "transcription"},
            ).enter()
        except Exception:
            await client.close()
            raise

        session = OpenAIStreamingSTTSession(client, connection)
        transcription: Dict[str, Any] = {"model": self.model}
        if self.language:
            transcription["language"] = self.language
        try:
            await connection.session.update(
                session={
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": transcription,
                            "turn_detection": {
                                "type": "semantic_vad",
                                "eagerness": "auto",
                            },
                        },
                    },
                }
            )
            while True:
                event = await asyncio.wait_for(connection.recv(), timeout=10)
                event_type = event.type
                if event_type == "session.updated":
                    return session
                if event_type == "error":
                    raise RuntimeError(
                        str(
                            event.error.message
                            or "OpenAI Realtime session setup failed"
                        )
                    )
        except Exception:
            await session.close()
            raise


class ElevenLabsStreamingSTTSession(_JSONWebSocketSession):
    sample_rate = 16000

    def __init__(self, websocket: Any):
        super().__init__(websocket)
        self._segment_index = 0

    async def send_audio(self, pcm16: bytes) -> None:
        await self._send_json(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm16).decode("ascii"),
            }
        )

    async def commit(self) -> None:
        # ElevenLabs attaches the commit flag to an audio chunk rather than a
        # standalone message. A short silence chunk gives manual PTT a valid
        # commit payload without replaying or omitting captured speech.
        await self._send_json(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(b"\0" * 640).decode("ascii"),
                "commit": True,
            }
        )

    async def receive_event(self) -> Optional[StreamingTranscriptEvent]:
        message = await self._receive_json()
        message_type = message.get("message_type")
        segment_id = f"elevenlabs-{self._segment_index}"
        if message_type in {"partial_transcript", "final_transcript"}:
            return StreamingTranscriptEvent(
                text=str(message.get("text") or ""),
                is_final=False,
                event_id=message.get("event_id"),
                segment_id=segment_id,
            )
        if message_type == "committed_transcript":
            event = StreamingTranscriptEvent(
                text=str(message.get("text") or ""),
                is_final=True,
                event_id=message.get("event_id") or segment_id,
                segment_id=segment_id,
            )
            self._segment_index += 1
            return event
        if message_type in {
            "auth_error",
            "chunk_size_exceeded",
            "commit_throttled",
            "error",
            "input_error",
            "insufficient_audio_activity",
            "queue_overflow",
            "quota_exceeded",
            "rate_limited",
            "resource_exhausted",
            "session_time_limit_exceeded",
            "transcriber_error",
            "unaccepted_terms",
        }:
            raise RuntimeError(str(message.get("error") or message_type))
        return None


class ElevenLabsStreamingSTTProvider(StreamingSTTProvider):
    name = "elevenlabs"
    session_recycle_seconds = _ELEVENLABS_SESSION_RECYCLE_SECONDS

    def __init__(
        self,
        api_key: str,
        base_url: str,
        section: Dict[str, Any],
        language: str,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.section = section
        self.language = language

    @property
    def configuration_key(self) -> tuple[Any, ...]:
        return (
            self.name,
            self.api_key,
            self.base_url,
            self.language,
            json.dumps(self.section, sort_keys=True, default=str),
        )

    async def open_session(self) -> StreamingSTTSession:
        params = {
            "model_id": str(self.section["streaming_model_id"]),
            "audio_format": "pcm_16000",
            "commit_strategy": "vad",
        }
        if self.language:
            params["language_code"] = self.language
        for key in (
            "vad_threshold",
            "vad_silence_threshold_secs",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
        ):
            if self.section.get(key) is not None:
                params[key] = self.section[key]
        websocket_base_url = (
            str(self.section.get("wss_url") or self.base_url).strip().rstrip("/")
        )
        websocket = await _connect(
            _with_query(
                _websocket_endpoint(websocket_base_url, "speech-to-text/realtime"),
                params,
            ),
            {"xi-api-key": self.api_key},
        )
        session = ElevenLabsStreamingSTTSession(websocket)
        try:
            while True:
                message = await asyncio.wait_for(session._receive_json(), timeout=10)
                message_type = message.get("message_type")
                if message_type == "session_started":
                    return session
                if message_type and message_type != "partial_transcript":
                    raise RuntimeError(str(message.get("error") or message_type))
        except Exception:
            await websocket.close()
            raise


def resolve_streaming_stt_provider(
    stt_config: Optional[Dict[str, Any]] = None,
) -> Optional[StreamingSTTProvider]:
    """Resolve the explicitly selected streaming provider, or ``None``.

    A different cloud provider is never selected merely because it supports
    streaming.
    """
    if stt_config is None:
        try:
            from hermes_cli.config import load_config

            raw = load_config().get("stt", {})
            stt_config = raw if isinstance(raw, dict) else {}
        except Exception:
            return None
    if not isinstance(stt_config, dict):
        return None

    provider_name = str(stt_config.get("provider") or "").strip().lower()
    from tools.transcription_tools import resolve_stt_provider_config

    resolved = resolve_stt_provider_config(provider_name, stt_config)
    if resolved is None or not resolved.streaming_enabled or not resolved.api_key:
        return None

    if provider_name == "openai":
        return OpenAIStreamingSTTProvider(
            resolved.api_key,
            resolved.base_url,
            resolved.streaming_model_id,
            resolved.language,
        )

    if provider_name == "elevenlabs":
        return ElevenLabsStreamingSTTProvider(
            resolved.api_key,
            resolved.base_url,
            resolved.section,
            resolved.language,
        )

    return None


class _PCM16StreamResampler:
    """Stateful chunk resampler for live mono PCM16 provider input."""

    def __init__(self, source_rate: int, target_rate: int):
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._tail = None
        self._next_position = 0.0

    def convert(self, frame: Any) -> bytes:
        np = __import__("numpy")
        samples = np.asarray(frame, dtype=np.int16).reshape(-1)
        if not len(samples):
            return b""
        if self.source_rate == self.target_rate:
            return samples.astype("<i2", copy=False).tobytes()

        if self._tail is not None:
            samples = np.concatenate((self._tail, samples))
        ratio = self.source_rate / self.target_rate
        if len(samples) < 2:
            self._tail = samples[-1:].copy()
            return b""
        positions = np.arange(self._next_position, len(samples) - 1, ratio)
        output = np.interp(positions, np.arange(len(samples)), samples)
        next_absolute = (
            (positions[-1] + ratio) if len(positions) else self._next_position
        )
        self._next_position = next_absolute - (len(samples) - 1)
        self._tail = samples[-1:].copy()
        return np.rint(output).astype("<i2").tobytes()


class StreamingSTTCoordinator:
    """Own provider I/O, ordered callbacks, and capture backpressure."""

    def __init__(
        self,
        provider: StreamingSTTProvider,
        on_event: Callable[[int, StreamingTranscriptEvent], None],
        on_error: Callable[[int, Exception], None],
        *,
        continuous: bool = False,
    ):
        self._provider = provider
        self._on_event = on_event
        self._on_error = on_error
        self._continuous = continuous
        self._audio_queue: queue.Queue = queue.Queue()
        self._command_queue: queue.Queue = queue.Queue()
        self._callback_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._callback_thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._reconnect = threading.Event()
        self._reset_session = threading.Event()
        self._closing = threading.Event()
        self._session: Optional[StreamingSTTSession] = None
        self._session_opened_at = 0.0
        self._connected = False
        self._queued_audio_seconds = 0.0
        self._audio_sequence = 0
        self._deferred_audio: Optional[tuple[int, int, Any, int]] = None
        self._revoked_generation = 0
        self._utterance = _UtteranceState()

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def provider_configuration_key(self) -> tuple[Any, ...]:
        return self._provider.configuration_key

    def start_utterance(self, generation: int, timeout: float = 10.0) -> bool:
        """Ensure a session exists and arm one capture turn."""
        self._ensure_thread()
        with self._lock:
            if self._closing.is_set() or generation <= self._revoked_generation:
                return False
            recycle_seconds = self._provider.session_recycle_seconds
            recycle_session = bool(
                self._connected
                and recycle_seconds
                and self._session_opened_at
                and time.monotonic() - self._session_opened_at
                >= recycle_seconds
            )
            if recycle_session:
                self._connected = False
                self._ready.clear()
                self._reset_session.set()
                self._reconnect.set()
            elif not self._connected:
                self._ready.clear()
                self._reconnect.set()
        if not self._ready.wait(timeout):
            return False

        # Nothing can legitimately carry across utterance boundaries. Drain
        # before arming so a newly attached capture sink cannot race cleanup.
        self._drain_audio_queue()
        self._drain_command_queue()
        with self._lock:
            if (
                not self._connected
                or self._closing.is_set()
                or generation <= self._revoked_generation
            ):
                return False
            self._utterance = _UtteranceState(
                generation=generation,
                active=True,
                accepting_audio=True,
            )
        return True

    def enqueue_audio(self, frame: Any, sample_rate: int) -> bool:
        """Accept a copied PCM frame without blocking the capture callback."""
        try:
            source_rate = int(sample_rate)
            frame_count = len(frame)
        except (TypeError, ValueError):
            return False
        if source_rate <= 0 or frame_count <= 0:
            return False
        frame_seconds = frame_count / source_rate

        with self._lock:
            utterance = self._utterance
            if (
                not utterance.active
                or not utterance.accepting_audio
                or utterance.failure_pending
                or not self._connected
            ):
                return False
            generation = utterance.generation
            if (
                self._queued_audio_seconds + frame_seconds
                > _MAX_BUFFERED_AUDIO_SECONDS
            ):
                overflow = True
            else:
                overflow = False
                self._queued_audio_seconds += frame_seconds
                self._audio_sequence += 1
                self._audio_queue.put_nowait(
                    (
                        generation,
                        self._audio_sequence,
                        frame,
                        source_rate,
                        frame_seconds,
                    )
                )
        if overflow:
            with self._lock:
                utterance = self._utterance
                if (
                    utterance.active
                    and generation == utterance.generation
                    and not utterance.failure_pending
                ):
                    utterance.failure_pending = True
                    self._command_queue.put_nowait(
                        (
                            "fail",
                            generation,
                            RuntimeError("Streaming STT audio queue overflow"),
                        )
                    )
            return False
        return True

    def commit(self, generation: int) -> bool:
        with self._lock:
            utterance = self._utterance
            if (
                not utterance.active
                or not utterance.accepting_audio
                or generation != utterance.generation
            ):
                return False
            # This lock is also used by enqueue_audio(). Once accepting_audio
            # is cleared, every frame already accepted is ahead of this commit.
            if not self._continuous:
                utterance.accepting_audio = False
            self._command_queue.put_nowait(
                ("commit", generation, self._audio_sequence)
            )
        return True

    def pause(self, generation: Optional[int] = None) -> None:
        reset_session = False
        with self._lock:
            if generation is not None:
                self._revoked_generation = max(
                    self._revoked_generation,
                    generation,
                )
            if (
                generation is not None
                and generation != self._utterance.generation
            ):
                return
            reset_session = self._utterance.active
            self._utterance.active = False
            self._utterance.accepting_audio = False
            self._utterance.resampler = None
            if reset_session:
                self._connected = False
                self._ready.clear()
                self._reset_session.set()
        self._drain_audio_queue()

    def has_activity(self, generation: int) -> bool:
        with self._lock:
            return (
                generation == self._utterance.generation
                and self._utterance.has_activity
            )

    def close(self, timeout: float = 3.0) -> None:
        with self._lock:
            self._utterance.active = False
            self._utterance.accepting_audio = False
            self._connected = False
            self._session_opened_at = 0.0
        self._closing.set()
        self._reconnect.set()
        self._command_queue.put(("close", 0, None))
        self._drain_audio_queue()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._callback_queue.put_nowait(None)
        callback_thread = self._callback_thread
        if (
            callback_thread is not None
            and callback_thread is not threading.current_thread()
        ):
            callback_thread.join(timeout=timeout)

    def _ensure_thread(self) -> None:
        with self._lock:
            if (
                self._callback_thread is None
                or not self._callback_thread.is_alive()
            ):
                self._callback_thread = threading.Thread(
                    target=self._run_callbacks,
                    name=f"stt-callback-{self._provider.name}",
                    daemon=True,
                )
                self._callback_thread.start()
            if self._thread is not None and self._thread.is_alive():
                return
            self._closing.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"stt-stream-{self._provider.name}",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._worker())
        except Exception:
            logger.debug("Streaming STT worker exited unexpectedly", exc_info=True)
        finally:
            with self._lock:
                self._connected = False
                self._session = None
                self._session_opened_at = 0.0
            self._ready.set()

    async def _worker(self) -> None:
        event_queue: Optional[asyncio.Queue] = None
        reader_task: Optional[asyncio.Task] = None
        while not self._closing.is_set():
            if self._reset_session.is_set():
                self._reset_session.clear()
                await self._cancel_reader(reader_task)
                reader_task = None
                event_queue = None
                session = self._session
                self._session = None
                if session is not None:
                    await self._close_session(session)
                with self._lock:
                    self._connected = False
                    self._session_opened_at = 0.0
            if self._session is None:
                if not self._reconnect.is_set():
                    await asyncio.sleep(0.02)
                    continue
                self._reconnect.clear()
                try:
                    session = await self._provider.open_session()
                except Exception as exc:
                    with self._lock:
                        self._connected = False
                    logger.warning(
                        "Streaming STT connection to %s failed: %s",
                        self._provider.name,
                        exc,
                    )
                    self._ready.set()
                    continue
                event_queue = asyncio.Queue()
                reader_task = asyncio.create_task(
                    self._read_events(session, event_queue)
                )
                self._reconnect.clear()
                with self._lock:
                    self._session = session
                    self._session_opened_at = time.monotonic()
                    self._connected = True
                logger.debug(
                    "Streaming STT session opened: provider=%s",
                    self._provider.name,
                )
                self._ready.set()

            try:
                if event_queue is not None:
                    while True:
                        try:
                            event = event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        self._handle_event(event)
                if reader_task is not None and reader_task.done():
                    error = reader_task.exception()
                    if error is None:
                        error = RuntimeError(
                            "Streaming STT provider event stream ended"
                        )
                    reader_task = None
                    event_queue = None
                    await self._drop_session(error)
                    continue
                await self._send_pending()
            except Exception as exc:
                await self._cancel_reader(reader_task)
                reader_task = None
                event_queue = None
                await self._drop_session(exc)
                continue
            await asyncio.sleep(0.01)

        await self._cancel_reader(reader_task)
        session = self._session
        self._session = None
        if session is not None:
            await self._close_session(session)

    @staticmethod
    async def _read_events(
        session: StreamingSTTSession,
        event_queue: asyncio.Queue,
    ) -> None:
        while True:
            event = await session.receive_event()
            if event is not None:
                await event_queue.put(event)

    @staticmethod
    async def _cancel_reader(reader_task: Optional[asyncio.Task]) -> None:
        if reader_task is None:
            return
        if reader_task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                reader_task.exception()
            return
        reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader_task

    async def _send_pending(self) -> None:
        pending_commit = None
        pending_failure = None
        while True:
            try:
                command, generation, value = self._command_queue.get_nowait()
            except queue.Empty:
                break
            if command == "close":
                return
            with self._lock:
                valid = (
                    self._utterance.active
                    and generation == self._utterance.generation
                )
            if command == "commit" and valid:
                pending_commit = (generation, value)
            elif command == "fail" and valid:
                pending_failure = (generation, value)

        if pending_failure is not None:
            generation, error = pending_failure
            self._fail_turn(generation, error, reset_session=True)
            return

        # A manual/max-duration commit must not overtake frames accepted before
        # the request. Continuous capture may enqueue later frames, so the
        # sequence fence establishes the provider-side audio boundary.
        send_limit = (
            self._audio_queue.qsize()
            if pending_commit is not None
            else _AUDIO_SEND_BATCH_SIZE
        )
        if self._deferred_audio is not None:
            send_limit += 1
        for _ in range(send_limit):
            queued = self._dequeue_audio()
            if queued is None:
                break
            generation, sequence, frame, source_rate = queued
            if pending_commit is not None and sequence > pending_commit[1]:
                self._deferred_audio = queued
                break
            with self._lock:
                utterance = self._utterance
                valid = utterance.active and generation == utterance.generation
                session = self._session
                if session is None:
                    return
                target_rate = session.sample_rate
                if valid and (
                    utterance.resampler is None
                    or utterance.resampler.source_rate != source_rate
                    or utterance.resampler.target_rate != target_rate
                ):
                    utterance.resampler = _PCM16StreamResampler(
                        source_rate,
                        target_rate,
                    )
                resampler = utterance.resampler
            if not valid or resampler is None:
                continue
            pcm16 = resampler.convert(frame)
            if pcm16:
                await session.send_audio(pcm16)
                with self._lock:
                    if (
                        self._utterance.active
                        and generation == self._utterance.generation
                    ):
                        self._utterance.has_sent_audio = True

        if pending_commit is not None:
            commit_generation = pending_commit[0]
            with self._lock:
                valid = (
                    self._utterance.active
                    and commit_generation == self._utterance.generation
                )
                session = self._session
            if valid and session is not None:
                await session.commit()

    def _handle_event(self, event: StreamingTranscriptEvent) -> None:
        with self._lock:
            utterance = self._utterance
            if not utterance.active or not utterance.has_sent_audio:
                logger.debug(
                    "Streaming STT event ignored: provider=%s active=%s "
                    "audio_sent=%s final=%s text_chars=%d",
                    self._provider.name,
                    utterance.active,
                    utterance.has_sent_audio,
                    event.is_final,
                    len(event.text.strip()),
                )
                return
            generation = utterance.generation
            text = event.text.strip()
            logger.debug(
                "Streaming STT event: provider=%s generation=%d final=%s "
                "text_chars=%d event_id=%s segment_id=%s",
                self._provider.name,
                generation,
                event.is_final,
                len(text),
                event.event_id or "-",
                event.segment_id or "-",
            )
            if text:
                utterance.has_activity = True
            if event.is_final and not self._continuous:
                utterance.active = False
                utterance.accepting_audio = False
        if event.is_final and not self._continuous:
            self._drain_audio_queue()
        self._dispatch_callback(
            self._on_event,
            generation,
            event,
            failure_message="Streaming STT event callback failed",
        )

    async def _drop_session(self, exc: Exception) -> None:
        session = self._session
        if session is not None:
            await self._close_session(session)
        with self._lock:
            self._session = None
            self._connected = False
            self._session_opened_at = 0.0
            generation = self._utterance.generation
            active = self._utterance.active
        if active:
            self._fail_turn(generation, exc)

    @staticmethod
    async def _close_session(session: StreamingSTTSession) -> None:
        try:
            await session.close()
        except Exception:
            logger.debug("Streaming STT session close failed", exc_info=True)

    def _fail_turn(
        self,
        generation: int,
        exc: Exception,
        *,
        reset_session: bool = False,
    ) -> None:
        with self._lock:
            utterance = self._utterance
            if (
                utterance.error_reported
                or not utterance.active
                or generation != utterance.generation
            ):
                return
            utterance.error_reported = True
            utterance.active = False
            utterance.accepting_audio = False
            utterance.resampler = None
            if reset_session:
                self._connected = False
                self._reset_session.set()
        self._drain_audio_queue()
        self._dispatch_callback(
            self._on_error,
            generation,
            exc,
            failure_message="Streaming STT error callback failed",
        )

    def _dispatch_callback(
        self,
        callback: Callable[[int, Any], None],
        generation: int,
        value: Any,
        *,
        failure_message: str,
    ) -> None:
        self._callback_queue.put_nowait(
            (callback, generation, value, failure_message)
        )

    def _run_callbacks(self) -> None:
        while True:
            item = self._callback_queue.get()
            if item is None:
                return
            callback, generation, value, failure_message = item
            try:
                callback(generation, value)
            except Exception:
                logger.error(failure_message, exc_info=True)

    def _dequeue_audio(self) -> Optional[tuple[int, int, Any, int]]:
        if self._deferred_audio is not None:
            queued = self._deferred_audio
            self._deferred_audio = None
            return queued
        try:
            generation, sequence, frame, source_rate, frame_seconds = (
                self._audio_queue.get_nowait()
            )
        except queue.Empty:
            return None
        with self._lock:
            self._queued_audio_seconds = max(
                0.0,
                self._queued_audio_seconds - frame_seconds,
            )
        return generation, sequence, frame, source_rate

    def _drain_audio_queue(self) -> None:
        self._deferred_audio = None
        while self._dequeue_audio() is not None:
            pass

    def _drain_command_queue(self) -> None:
        while True:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                return


class StreamingSTTTurnController:
    """Present streaming capture without exposing coordinator generations."""

    def __init__(
        self,
        provider: StreamingSTTProvider,
        on_event: Callable[[StreamingTranscriptEvent], None],
        on_error: Callable[[Exception], None],
        *,
        commit_timeout: float = 8.0,
        continuous: bool = False,
    ):
        self._on_event = on_event
        self._on_error = on_error
        self._commit_timeout = commit_timeout
        self._continuous = continuous
        self._lock = threading.Lock()
        self._generation = 0
        self._active = False
        self._delivering = False
        self._closed = False
        self._commit_requested = False
        self._commit_timer: Optional[threading.Timer] = None
        self._coordinator = StreamingSTTCoordinator(
            provider,
            self._handle_event,
            self._handle_error,
            continuous=continuous,
        )

    @property
    def provider_name(self) -> str:
        return self._coordinator.provider_name

    @property
    def provider_configuration_key(self) -> tuple[Any, ...]:
        return self._coordinator.provider_configuration_key

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def delivering(self) -> bool:
        """Whether a provider result is currently entering its callback."""
        with self._lock:
            return self._delivering

    @property
    def commit_pending(self) -> bool:
        """Whether a provider commit response is already outstanding."""
        with self._lock:
            return self._commit_requested

    def start_utterance(self, timeout: float = 10.0) -> bool:
        with self._lock:
            if self._closed or self._delivering:
                return False
            if self._active:
                return self._continuous
            self._generation += 1
            generation = self._generation
            self._active = True
            self._commit_requested = False
        started = self._coordinator.start_utterance(generation, timeout=timeout)
        with self._lock:
            still_active = (
                self._active
                and generation == self._generation
                and not self._closed
            )
            if not started and generation == self._generation:
                self._active = False
        if started and still_active:
            return True
        if started:
            self._coordinator.pause(generation)
        return False

    def enqueue_audio(self, frame: Any, sample_rate: int) -> bool:
        return self._coordinator.enqueue_audio(frame, sample_rate)

    def commit(self) -> bool:
        with self._lock:
            if not self._active or self._commit_requested:
                return False
            generation = self._generation
            self._commit_requested = True
        if not self._coordinator.commit(generation):
            self._fail_turn(
                generation,
                RuntimeError("Streaming STT commit was not accepted"),
            )
            return False

        timer = threading.Timer(
            self._commit_timeout,
            lambda: self._fail_turn(
                generation,
                RuntimeError("Streaming STT commit timed out"),
            ),
        )
        timer.daemon = True
        with self._lock:
            if not self._active or generation != self._generation:
                return True
            self._commit_timer = timer
        timer.start()
        return True

    def pause(self) -> None:
        with self._lock:
            if not self._active:
                return
            generation = self._generation
            self._active = False
            self._commit_requested = False
            timer = self._take_commit_timer_locked()
        if timer is not None:
            timer.cancel()
        self._coordinator.pause(generation)

    def close(self, timeout: float = 3.0) -> None:
        with self._lock:
            self._closed = True
            self._active = False
            self._commit_requested = False
            timer = self._take_commit_timer_locked()
        if timer is not None:
            timer.cancel()
        self._coordinator.close(timeout=timeout)

    def _handle_event(
        self,
        generation: int,
        event: StreamingTranscriptEvent,
    ) -> None:
        self._deliver(
            generation,
            self._on_event,
            event,
            keep_active=self._continuous or not event.is_final,
            completes_commit=event.is_final,
        )

    def _handle_error(self, generation: int, error: Exception) -> None:
        self._deliver(
            generation,
            self._on_error,
            error,
            keep_active=False,
            completes_commit=True,
        )

    def _fail_turn(self, generation: int, error: Exception) -> None:
        self._coordinator.pause(generation)
        self._handle_error(generation, error)

    def _deliver(
        self,
        generation: int,
        callback: Callable[[Any], None],
        value: Any,
        *,
        keep_active: bool,
        completes_commit: bool,
    ) -> None:
        with self._lock:
            if (
                not self._active
                or generation != self._generation
                or self._delivering
            ):
                return
            if not keep_active:
                self._active = False
            self._delivering = True
            if completes_commit:
                self._commit_requested = False
                timer = self._take_commit_timer_locked()
            else:
                timer = None
        if timer is not None:
            timer.cancel()
        try:
            callback(value)
        finally:
            with self._lock:
                self._delivering = False

    def _take_commit_timer_locked(self) -> Optional[threading.Timer]:
        timer = self._commit_timer
        self._commit_timer = None
        return timer
