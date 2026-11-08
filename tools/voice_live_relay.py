"""Server-side relay between a voice-live browser/desktop client and the Azure Realtime API.

``/api/audio/voice-live-stream`` (``hermes_cli/web_routers/audio_voice_live.py``) accepts the
client's WebSocket; this module owns the vendor socket and the event translation, so the Azure
URL and resource key never reach the browser:

* the client opens with ``{"type": "session.start", "history": [...]}``; the route runs
  :func:`tools.voice_live.open_azure_realtime_session` (connect → ``session.created`` →
  ``session.update`` → ``session.updated``), then emits a synthetic ``session.started`` and
  starts :class:`VoiceLiveRelay`;
* vendor events pump to the client through :func:`translate_vendor_event` — except
  ``session.created``, which the handshake already consumed — with secrets redacted;
* delegation is function calling: ``response.function_call_arguments.done`` is forwarded (its
  ``call_id`` is what the desktop turns into ``onDelegation``/``prompt.submit``), while the
  client's ``session.commentary.append`` / ``session.thinking.append`` frames accumulate in a
  :class:`DelegationTracker` and ``session.commentary.complete`` produces ONE
  ``conversation.item.create`` ``function_call_output`` + ``response.create``. ``think`` text is
  the fallback result when no commentary arrived;
* ``session.close`` (or a dropped client/vendor socket) closes everything exactly once.

The relay is synchronous and runs on one thread that owns the vendor socket; the route feeds it
client frames through a queue and receives client-bound events through a thread-safe callback.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
from typing import Any, Callable, NamedTuple, Optional

logger = logging.getLogger(__name__)

# Client → relay control events (the route handles session.start itself, before the pump starts).
CLIENT_SESSION_START = "session.start"
CLIENT_SESSION_CLOSE = "session.close"
CLIENT_COMMENTARY_APPEND = "session.commentary.append"
CLIENT_COMMENTARY_COMPLETE = "session.commentary.complete"
CLIENT_THINKING_APPEND = "session.thinking.append"
#: Client event types forwarded to the vendor verbatim (mic audio + barge-in cancel).
CLIENT_PASSTHROUGH_EVENTS = frozenset({
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "response.cancel",
})
#: Accepted but not forwarded: Azure Realtime has no such client event, so the relay
#: swallows it instead of earning a vendor error (the persona steering is advisory).
CLIENT_DROP_EVENTS = frozenset({"session.instructions.append"})

# Vendor → client events.
VENDOR_SESSION_CREATED = "session.created"
VENDOR_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"

# Relay-synthesized client events.
RELAY_SESSION_STARTED = "session.started"
RELAY_SESSION_CLOSED = "session.closed"
RELAY_ERROR = "error"

#: Sentinel put on the relay's client queue when the browser side is done.
RELAY_CLOSE = object()

#: Cap on the buffered delegation text per call_id, so a client that never completes cannot grow
#: memory without bound.
MAX_DELEGATION_CHARS = 256 * 1024


def redact_secrets(value: Any, secrets: tuple) -> Any:
    """Deep-copy *value* with every occurrence of a non-empty *secrets* string replaced by
    ``[redacted]``. Applied to everything sent to the browser."""
    secrets = tuple(s for s in secrets if s)
    if not secrets:
        return value
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {key: redact_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item, secrets) for item in value]
    return value


def _error(message: str) -> dict[str, Any]:
    return {"type": RELAY_ERROR, "message": message}


class DelegationTracker:
    """Per-``call_id`` accumulation of the client's commentary/think text for one vendor function
    call. Thread-safe (the relay thread is its only user, but the route may inspect it)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, dict[str, Any]] = {}

    def note_function_call(self, call_id: Any, name: Any, arguments: Any = "") -> None:
        """Record a vendor ``response.function_call_arguments.done`` so a later
        ``session.commentary.complete`` for this call_id maps onto it."""
        if not isinstance(call_id, str) or not call_id:
            return
        with self._lock:
            self._calls.setdefault(call_id, {
                "name": str(name or ""), "arguments": str(arguments or ""),
                "commentary": [], "think": [], "chars": 0,
            })

    def append(self, delegation_id: str, kind: str, content: str) -> None:
        """Buffer one ``session.commentary.append`` (*kind* ``"commentary"``) or
        ``session.thinking.append`` (*kind* ``"think"``) frame for *delegation_id*."""
        with self._lock:
            entry = self._calls.setdefault(str(delegation_id), {
                "name": "", "arguments": "", "commentary": [], "think": [], "chars": 0,
            })
            if entry["chars"] >= MAX_DELEGATION_CHARS:
                return
            entry[kind].append(content)
            entry["chars"] += len(content)

    def complete(self, delegation_id: str) -> tuple[str, list[dict[str, Any]]]:
        """Pop *delegation_id* and return ``(output_text, vendor_events)``: the accumulated
        commentary joined (``think`` text as the fallback when there is none), plus ONE
        ``conversation.item.create`` function_call_output and a ``response.create``."""
        with self._lock:
            entry = self._calls.pop(str(delegation_id), None)
        commentary = [c for c in (entry or {}).get("commentary", []) if c]
        think = [t for t in (entry or {}).get("think", []) if t]
        text = " ".join(commentary) or " ".join(think) or ""
        call_id = str(delegation_id)
        return text, [
            {"type": "conversation.item.create",
             "item": {"type": "function_call_output", "call_id": call_id, "output": text}},
            {"type": "response.create"},
        ]


class ClientEventResult(NamedTuple):
    """Translation of one client frame: messages for the vendor socket, events for the browser,
    and whether the session should close."""
    vendor: list[dict[str, Any]]
    to_browser: list[dict[str, Any]]
    close: bool


def translate_client_event(frame: Any, tracker: DelegationTracker) -> ClientEventResult:
    """Translate one browser frame. Relay-owned control/commentary events never reach the vendor
    as-is; audio/cancel frames pass through verbatim; malformed frames get an error event back."""
    if not isinstance(frame, dict):
        return ClientEventResult([], [_error("frames must be JSON objects")], False)
    kind = frame.get("type")
    if kind == CLIENT_SESSION_CLOSE:
        return ClientEventResult([], [], True)
    if kind == CLIENT_SESSION_START:
        return ClientEventResult([], [_error("session.start is only valid as the first frame")], False)
    if kind in CLIENT_PASSTHROUGH_EVENTS:
        return ClientEventResult([frame], [], False)
    if kind in CLIENT_DROP_EVENTS:
        return ClientEventResult([], [], False)
    if kind in (CLIENT_COMMENTARY_APPEND, CLIENT_THINKING_APPEND):
        delegation_id = frame.get("delegation_id")
        content = frame.get("content")
        if not isinstance(delegation_id, str) or not delegation_id:
            # think()/speak() legitimately carry delegation_id null when no call is active.
            return ClientEventResult([], [], False)
        if not isinstance(content, str):
            return ClientEventResult([], [_error(f"{kind} needs string content")], False)
        tracker.append(delegation_id, "commentary" if kind == CLIENT_COMMENTARY_APPEND else "think", content)
        return ClientEventResult([], [], False)
    if kind == CLIENT_COMMENTARY_COMPLETE:
        delegation_id = frame.get("delegation_id")
        if not isinstance(delegation_id, str) or not delegation_id:
            return ClientEventResult([], [_error("session.commentary.complete needs a delegation_id")], False)
        _text, events = tracker.complete(delegation_id)
        return ClientEventResult(events, [], False)
    return ClientEventResult([], [_error(f"unsupported event type {kind!r}")], False)


def translate_vendor_event(event: Any, tracker: DelegationTracker,
                           secrets: tuple = ()) -> Optional[dict[str, Any]]:
    """Translate one vendor event for the browser: ``None`` suppresses it (``session.created`` is
    consumed by the open handshake), everything else is forwarded with *secrets* redacted.
    ``response.function_call_arguments.done`` is additionally recorded in *tracker*."""
    if not isinstance(event, dict):
        return None
    kind = event.get("type")
    if kind == VENDOR_SESSION_CREATED:
        return None
    if kind == VENDOR_FUNCTION_CALL_ARGUMENTS_DONE:
        tracker.note_function_call(event.get("call_id"), event.get("name"), event.get("arguments"))
    return redact_secrets(event, secrets)


def _recv_poll(ws: Any, timeout: float) -> Optional[str]:
    """``ws.recv(timeout=...)`` where a poll timeout returns None and anything else propagates
    (the vendor socket is gone)."""
    try:
        return ws.recv(timeout=timeout)
    except Exception as exc:
        if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
            return None
        raise


class VoiceLiveRelay:
    """Owns the vendor WebSocket on one thread: pumps vendor events to the browser via
    *send_to_browser* (must be thread-safe; the route passes a ``call_soon_threadsafe`` hop) and
    drains client frames from *client_q* in :meth:`run`."""

    def __init__(self, ws: Any, session_id: str, *,
                 send_to_browser: Callable[[dict[str, Any]], None], secrets: tuple = ()) -> None:
        self._ws = ws
        self.session_id = session_id
        self._send_to_browser = send_to_browser
        self._secrets = tuple(secrets)
        self.tracker = DelegationTracker()
        self.closed_reason: Optional[str] = None
        self._closed = False

    def _emit(self, event: dict[str, Any]) -> None:
        # The browser sink may already be gone at shutdown; a dead queue is not an error.
        with contextlib.suppress(Exception):
            self._send_to_browser(event)

    def _send_vendor(self, message: dict[str, Any]) -> bool:
        try:
            self._ws.send(json.dumps(message))
            return True
        except Exception:  # health: allow BLE001 -- any vendor socket error means the peer is gone; closed_reason records it
            self.closed_reason = "vendor_socket_closed"
            return False

    def handle_client_frame(self, frame: Any) -> ClientEventResult:
        return translate_client_event(frame, self.tracker)

    def _drain_client(self, client_q: queue.Queue) -> bool:
        """Send queued client frames; returns False when the session should end."""
        while True:
            try:
                frame = client_q.get_nowait()
            except queue.Empty:
                return True
            if frame is RELAY_CLOSE:
                self.closed_reason = "client_close"
                return False
            result = self.handle_client_frame(frame)
            for event in result.to_browser:
                self._emit(event)
            for message in result.vendor:
                if not self._send_vendor(message):
                    return False
            if result.close:
                self.closed_reason = "client_close"
                return False

    def run(self, client_q: queue.Queue, stop: threading.Event) -> None:
        """Pump until *stop* is set, the client closes, or the vendor socket dies; always closes
        the vendor socket and emits exactly one terminal ``session.closed``."""
        try:
            while not stop.is_set():
                if not self._drain_client(client_q):
                    break
                try:
                    raw = _recv_poll(self._ws, 0.2)
                except Exception:  # health: allow BLE001 -- any vendor socket error ends the pump; closed_reason records it
                    self.closed_reason = self.closed_reason or "vendor_socket_closed"
                    break
                if raw is None:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                out = translate_vendor_event(event, self.tracker, self._secrets)
                if out is not None:
                    self._emit(out)
        except Exception:
            self.closed_reason = self.closed_reason or "relay_error"
            logger.exception("voice-live relay pump crashed")
        finally:
            self.close()
            self._emit({"type": RELAY_SESSION_CLOSED, "reason": self.closed_reason or "closed"})

    def close(self) -> None:
        """Close the vendor socket, once."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._ws.close()
