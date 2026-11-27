"""Read-only transcript feed for an existing OpenAI GPT-Live WebRTC call.

The caller supplies the configured project endpoint and credential used to create
that call. This module doesn't resolve profiles, send Live control events, retry,
deduplicate replay, or persist anything.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal, Protocol
from urllib.parse import quote, urlsplit, urlunsplit

from tools.voice_live import DEFAULT_LIVE_BASE_URL

__all__ = [
    "ClientDelegation",
    "SessionClosed",
    "SidebandConnector",
    "SidebandError",
    "SidebandEvent",
    "SidebandSocket",
    "TranscriptDelta",
    "open_sideband",
]

_Speaker = Literal["user", "assistant"]
_ErrorReason = Literal[
    "attach_failed", "disconnected", "invalid_event", "provider_error", "close_failed"
]
_SPEAKERS: dict[str, _Speaker] = {
    "session.input_transcript.delta": "user",
    "session.output_transcript.delta": "assistant",
}


@dataclass(frozen=True, slots=True)
class TranscriptDelta:
    """Exact provider text in arrival order; absent identity or timing remains None."""

    call_id: str = field(repr=False)
    event_id: str | None = field(repr=False)
    speaker: _Speaker
    text: str = field(repr=False)
    start_ms: int | float | None
    end_ms: int | float | None


@dataclass(frozen=True, slots=True)
class SessionClosed:
    """The provider's terminal event, distinct from a WebSocket EOF."""

    call_id: str = field(repr=False)
    event_id: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class ClientDelegation:
    """A provider marker in stream order, not an authoritative utterance boundary."""

    call_id: str = field(repr=False)
    event_id: str | None = field(repr=False)
    delegation_id: str = field(repr=False)
    offset_ms: int | float | None


SidebandEvent = TranscriptDelta | SessionClosed | ClientDelegation


class SidebandError(RuntimeError):
    """A capture failure with a safe, stable reason and no wire diagnostics."""

    def __init__(self, reason: _ErrorReason) -> None:
        super().__init__(f"GPT-Live sideband: {reason}")
        self.reason = reason


class SidebandSocket(Protocol):
    """Receive-only view of an attached socket."""

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


SidebandConnector = Callable[[str, Mapping[str, str]], Awaitable[SidebandSocket]]


def _attach_url(base_url: str, call_id: str) -> str:
    if (
        not call_id.isprintable()
        or not call_id.strip()
        or call_id in {".", ".."}
        or any(char in call_id for char in "/\\%")
    ):
        raise ValueError("GPT-Live sideband requires a valid call ID")
    if any(
        char.isspace() or not char.isprintable() or char in "?#\\" for char in base_url
    ):
        raise ValueError("GPT-Live sideband requires a trusted HTTPS API base URL")
    try:
        base = urlsplit(base_url)
        port = base.port  # Also validates the supplied port, if any.
    except ValueError:
        raise ValueError(
            "GPT-Live sideband requires a trusted HTTPS API base URL"
        ) from None
    if (
        base.scheme != "https"
        or not base.hostname
        or base.username is not None
        or port == 0
    ):
        raise ValueError("GPT-Live sideband requires a trusted HTTPS API base URL")
    path = f"{base.path.rstrip('/')}/live/sessions/{quote(call_id, safe='')}/attach"
    return urlunsplit(("wss", base.netloc, path, "", ""))


async def _connect(url: str, headers: Mapping[str, str]) -> SidebandSocket:
    from websockets.asyncio.client import connect

    class _DirectConnect(connect):
        def process_redirect(self, exc: Exception) -> Exception:
            # websockets would otherwise reuse the bearer header on the new URL.
            return exc

    # websockets' debug logs include HTTP headers and media/transcript frames.
    quiet = logging.Logger(__name__, level=logging.CRITICAL + 1)
    return await _DirectConnect(
        url,
        additional_headers=headers,
        proxy=None,
        open_timeout=5,
        close_timeout=1,
        max_size=1024 * 1024,
        max_queue=16,
        logger=quiet,
    )


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise SidebandError("invalid_event")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        # Escaped lone surrogates are accepted by json.loads, but not durable UTF-8.
        raise SidebandError("invalid_event") from None
    return value


def _milliseconds(value: object) -> int | float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise SidebandError("invalid_event")
    return value


def _parse_event(raw: str, call_id: str) -> SidebandEvent | None:
    try:
        event = json.loads(raw)
    except (ValueError, RecursionError):
        raise SidebandError("invalid_event") from None
    if (
        not isinstance(event, dict)
        or not isinstance(kind := event.get("type"), str)
        or not kind
    ):
        raise SidebandError("invalid_event")
    if kind == "error":
        raise SidebandError("provider_error")
    speaker = _SPEAKERS.get(kind)
    delegation = (
        event.get("delegation") if kind == "session.delegation.created" else None
    )
    is_client_delegation = (
        isinstance(delegation, dict) and delegation.get("target") == "client"
    )
    if speaker is None and kind != "session.closed" and not is_client_delegation:
        return None

    event_id = event.get("event_id")
    if event_id is not None:
        event_id = _text(event_id)
        if not event_id.strip():
            raise SidebandError("invalid_event")
    if isinstance(delegation, dict) and is_client_delegation:
        delegation_id = _text(delegation.get("id"))
        if not delegation_id.strip():
            raise SidebandError("invalid_event")
        return ClientDelegation(
            call_id, event_id, delegation_id, _milliseconds(event.get("offset_ms"))
        )
    if speaker is None:
        return SessionClosed(call_id, event_id)

    text = _text(event.get("delta"))
    start = _milliseconds(event.get("start_ms"))
    end = _milliseconds(event.get("end_ms"))
    if start is not None and end is not None and end < start:
        raise SidebandError("invalid_event")
    return TranscriptDelta(call_id, event_id, speaker, text, start, end)


async def _read_events(
    socket: SidebandSocket, call_id: str
) -> AsyncGenerator[SidebandEvent, None]:
    while True:
        try:
            raw = await socket.recv()
        except Exception:  # health: allow BLE001 -- never surface raw transport diagnostics
            raise SidebandError("disconnected") from None
        if not isinstance(raw, str):
            continue
        event = _parse_event(raw, call_id)
        if event is not None:
            yield event
            if isinstance(event, SessionClosed):
                return


@asynccontextmanager
async def open_sideband(
    *,
    api_key: str,
    call_id: str,
    base_url: str = DEFAULT_LIVE_BASE_URL,
    connect: SidebandConnector | None = None,
) -> AsyncIterator[AsyncIterator[SidebandEvent]]:
    """Attach once and yield a single-consumer iterator on handshake success.

    Use the same trusted HTTPS API base URL and project credential that created
    the call, not a URL or credential from a renderer. No event is needed to
    report ready: install the SDP answer after entering the context. The iterator
    ends normally only after delivering ``SessionClosed``; any disconnect before
    that is incomplete. If the caller stops reading or cancels, exit the context
    to close the socket without a session-control event. No replay is initiated
    here; any provider-replayed deltas are delivered unchanged.
    """
    if re.fullmatch(r"[!-~]+", api_key) is None:
        raise ValueError("GPT-Live sideband requires a valid project API key")
    url = _attach_url(base_url, call_id)
    connector = _connect if connect is None else connect
    try:
        socket = await connector(url, {"Authorization": f"Bearer {api_key}"})
    except Exception:  # health: allow BLE001 -- handshake errors can carry secret headers
        raise SidebandError("attach_failed") from None

    completed = False
    try:
        async with aclosing(_read_events(socket, call_id)) as events:
            yield events
        completed = True
    finally:
        try:
            await socket.close()
        except Exception:  # health: allow BLE001 -- close reasons may contain private data
            if completed:
                raise SidebandError("close_failed") from None
