"""Offline contracts for the GPT-Live WebRTC sideband reader."""

import asyncio
import json
import logging
from collections import deque
from collections.abc import Mapping

import pytest

from tools.gpt_live_sideband import (
    ClientDelegation,
    SessionClosed,
    SidebandError,
    TranscriptDelta,
    open_sideband,
)

KEY = "synthetic-project-credential"
CALL = "sess_synthetic"
INPUT = "session.input_transcript.delta"
OUTPUT = "session.output_transcript.delta"


def _delta(kind=INPUT, **fields):
    return json.dumps({"type": kind, "delta": "sí — ", **fields})


class _Socket:
    """Only recv/close: any attempt to send application events fails the test."""

    def __init__(self, *frames):
        self.frames = deque(frames)
        self.closed = 0
        self.close_error: Exception | None = None

    async def recv(self):
        frame = self.frames.popleft()
        if isinstance(frame, Exception):
            raise frame
        return frame

    async def close(self):
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


class _Connector:
    def __init__(self, socket):
        self.socket = socket
        self.requests = []

    async def __call__(self, url: str, headers: Mapping[str, str]):
        self.requests.append((url, dict(headers)))
        return self.socket


def test_attach_is_ready_on_entry_and_only_typed_transcript_and_close_escape():
    replay = _delta(event_id="evt_1", start_ms=100.5, end_ms=300, audio="discard")
    socket = _Socket(
        '{"type":"session.started"}',
        b'{"type":"session.closed"}',  # binary/media isn't a text protocol event
        '{"type":"session.input_audio.append","audio":"never delivered"}',
        replay,
        '{"type":"session.output_audio.delta","delta":"never delivered"}',
        _delta(OUTPUT, event_id="evt_2", start_ms=50, end_ms=120, delta="Mm-hmm."),
        '{"type":"session.delegation.created","text":"never act on this"}',
        replay,  # attached replay is for the consumer to deduplicate, not this transport
        _delta(event_id=None, start_ms=301, delta="sí — "),
        '{"type":"session.closed","event_id":"evt_3","usage":{"seconds":15}}',
        _delta(delta="do not read after terminal marker"),
    )
    connector = _Connector(socket)
    call_id = "sess_synthetic:1?x#y"

    async def run():
        async with open_sideband(
            api_key=KEY,
            call_id=call_id,
            base_url="https://project.example/prefix/v1/",
            connect=connector,
        ) as events:
            assert connector.requests == [
                (
                    "wss://project.example/prefix/v1/live/sessions/sess_synthetic%3A1%3Fx%23y/attach",
                    {"Authorization": f"Bearer {KEY}"},
                )
            ]
            assert socket.closed == 0
            assert (
                len(socket.frames) == 11
            )  # no first-frame wait before SDP can be installed
            assert [event async for event in events] == [
                TranscriptDelta(call_id, "evt_1", "user", "sí — ", 100.5, 300),
                TranscriptDelta(call_id, "evt_2", "assistant", "Mm-hmm.", 50, 120),
                TranscriptDelta(call_id, "evt_1", "user", "sí — ", 100.5, 300),
                TranscriptDelta(call_id, None, "user", "sí — ", 301, None),
                SessionClosed(call_id, "evt_3"),
            ]
        assert socket.closed == 1
        assert len(socket.frames) == 1

    asyncio.run(run())


def test_client_delegation_keeps_wire_order_without_a_timestamp_cutoff():
    socket = _Socket(
        _delta(event_id="input", end_ms=5600, start_ms=5400, delta="Denver."),
        '{"type":"session.delegation.created","event_id":"del-event","offset_ms":5400,'
        '"delegation":{"id":"del-id","type":"delegation","target":"client"},"text":"ignored"}',
        '{"type":"session.closed"}',
    )

    async def run():
        async with open_sideband(
            api_key=KEY, call_id=CALL, connect=_Connector(socket)
        ) as events:
            received = [event async for event in events]
        assert received == [
            TranscriptDelta(CALL, "input", "user", "Denver.", 5400, 5600),
            ClientDelegation(CALL, "del-event", "del-id", 5400),
            SessionClosed(CALL, None),
        ]

    asyncio.run(run())


def test_disconnect_preserves_delivered_tail_but_is_not_session_closed(caplog):
    socket = _Socket(
        _delta(delta="private spoken words"),
        OSError(f"private close reason {KEY}"),
    )
    connector = _Connector(socket)

    async def run():
        with pytest.raises(SidebandError) as caught:
            async with open_sideband(
                api_key=KEY, call_id=CALL, connect=connector
            ) as events:
                delta = await anext(events)
                assert delta == TranscriptDelta(
                    CALL, None, "user", "private spoken words", None, None
                )
                assert "private spoken words" not in repr(delta)
                await anext(events)
        assert caught.value.reason == "disconnected"
        assert "private close reason" not in str(caught.value)
        assert KEY not in str(caught.value)
        assert socket.closed == 1
        assert connector.requests == [
            (
                f"wss://api.openai.com/v1/live/sessions/{CALL}/attach",
                {"Authorization": f"Bearer {KEY}"},
            )
        ]

    with caplog.at_level(logging.DEBUG):
        asyncio.run(run())
    assert "private spoken words" not in caplog.text
    assert "private close reason" not in caplog.text
    assert KEY not in caplog.text


@pytest.mark.parametrize(
    "frame, reason",
    [
        ('{"type":"error","error":{"message":"private diagnostic"}}', "provider_error"),
        ("not JSON; private diagnostic", "invalid_event"),
        ("[]", "invalid_event"),
        ('{"type":[]}', "invalid_event"),
        (_delta(delta=None), "invalid_event"),
        (_delta(delta="\ud800"), "invalid_event"),
        (_delta(event_id=False), "invalid_event"),
        (_delta(start_ms=True, end_ms=2), "invalid_event"),
        (_delta(start_ms=2, end_ms=1), "invalid_event"),
        (_delta(start_ms=-1), "invalid_event"),
        (_delta(end_ms=float("inf")), "invalid_event"),
    ],
)
def test_bad_events_fail_visibly_without_returning_provider_diagnostics(frame, reason):
    socket = _Socket(frame, '{"type":"session.closed"}')

    async def run():
        with pytest.raises(SidebandError) as caught:
            async with open_sideband(
                api_key=KEY, call_id=CALL, connect=_Connector(socket)
            ) as events:
                await anext(events)
        assert caught.value.reason == reason
        assert "private diagnostic" not in str(caught.value)
        assert socket.closed == 1
        assert len(socket.frames) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "options",
    [
        {"base_url": "http://project.example/v1"},
        {"base_url": "https://user:secret@project.example/v1"},
        {"base_url": "https://project.example/v1?token=secret"},
        {"base_url": "https://project.example/v1#"},
        {"base_url": "https://project.example:bogus/v1"},
        {"base_url": "https://project.example/v1\n"},
        {"base_url": "https:///v1"},
        {"call_id": ".."},
        {"call_id": "../other"},
        {"call_id": ""},
        {"api_key": ""},
        {"api_key": "synthetic\r\nX-Header: injection"},
    ],
)
def test_invalid_attach_input_never_reaches_connector(options):
    connector = _Connector(_Socket())

    async def run():
        args = {"api_key": KEY, "call_id": CALL, **options}
        with pytest.raises(ValueError):
            async with open_sideband(**args, connect=connector):
                pytest.fail("invalid attach reported ready")
        assert connector.requests == []

    asyncio.run(run())


def test_handshake_failure_does_not_report_ready_or_echo_vendor_response():
    async def fail(url, headers):
        raise OSError(f"private response from {url}: {headers['Authorization']}")

    async def run():
        with pytest.raises(SidebandError) as caught:
            async with open_sideband(api_key=KEY, call_id=CALL, connect=fail):
                pytest.fail("failed handshake reported ready")
        assert caught.value.reason == "attach_failed"
        assert KEY not in str(caught.value)
        assert CALL not in str(caught.value)

    asyncio.run(run())


def test_cancellation_during_receive_closes_socket_and_propagates():
    async def run():
        receiving = asyncio.Event()

        class WaitingSocket(_Socket):
            async def recv(self) -> str:
                receiving.set()
                pending: asyncio.Future[str] = asyncio.Future()
                return await pending

        socket = WaitingSocket()

        async def consume():
            async with open_sideband(
                api_key=KEY, call_id=CALL, connect=_Connector(socket)
            ) as events:
                await anext(events)

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(receiving.wait(), 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert socket.closed == 1

    asyncio.run(run())


def test_early_exit_closes_socket_without_draining_or_exposing_a_close_marker():
    socket = _Socket(_delta(), '{"type":"session.closed"}')

    async def run():
        async with open_sideband(
            api_key=KEY, call_id=CALL, connect=_Connector(socket)
        ) as events:
            assert await anext(events) == TranscriptDelta(
                CALL, None, "user", "sí — ", None, None
            )
        assert socket.closed == 1
        assert len(socket.frames) == 1
        with pytest.raises(StopAsyncIteration):
            await anext(events)

    asyncio.run(run())


def test_cleanup_failure_does_not_mask_consumer_error():
    socket = _Socket()
    socket.close_error = OSError("private close reason")
    sink_error = LookupError("sink failed")

    async def run():
        with pytest.raises(LookupError) as caught:
            async with open_sideband(
                api_key=KEY, call_id=CALL, connect=_Connector(socket)
            ):
                raise sink_error
        assert caught.value is sink_error
        assert socket.closed == 1

    asyncio.run(run())


def test_cleanup_failure_is_reported_without_the_wire_reason():
    socket = _Socket('{"type":"session.closed"}')
    socket.close_error = OSError("private close reason")

    async def run():
        with pytest.raises(SidebandError) as caught:
            async with open_sideband(
                api_key=KEY, call_id=CALL, connect=_Connector(socket)
            ) as events:
                assert [event async for event in events] == [SessionClosed(CALL, None)]
        assert caught.value.reason == "close_failed"
        assert "private close reason" not in str(caught.value)

    asyncio.run(run())


def test_default_connector_never_follows_a_redirect_with_the_bearer_key(monkeypatch):
    from websockets.asyncio.client import connect
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    attempts = []

    class Redirect:
        @property
        def transport(self):
            return self

        def abort(self):
            pass

        async def handshake(self, headers, user_agent_header):
            assert headers == {"Authorization": f"Bearer {KEY}"}
            raise InvalidStatus(
                Response(
                    307,
                    "Redirect",
                    Headers(
                        {
                            "Location": "wss://other.example/credential-sink",
                        }
                    ),
                )
            )

    async def create_connection(client):
        attempts.append(client.uri)
        return Redirect()

    monkeypatch.setattr(connect, "create_connection", create_connection)

    async def run():
        with pytest.raises(SidebandError) as caught:
            async with open_sideband(api_key=KEY, call_id=CALL):
                pytest.fail("redirect reported ready")
        assert caught.value.reason == "attach_failed"
        assert attempts == [f"wss://api.openai.com/v1/live/sessions/{CALL}/attach"]

    asyncio.run(run())


@pytest.mark.parametrize("provider_closed", [True, False])
def test_default_connector_reads_loopback_without_logging_or_sending_events(
    monkeypatch,
    caplog,
    provider_closed,
):
    from websockets.asyncio.server import serve

    received = []

    async def run():
        async def provider(socket):
            received.append(
                (socket.request.path, socket.request.headers["Authorization"])
            )
            await socket.send(
                '{"type":"session.output_audio.delta","delta":"private audio"}'
            )
            await socket.send(_delta(delta="private transcript"))
            if provider_closed:
                await socket.send('{"type":"session.closed"}')
            else:
                await socket.close(reason="private close reason")
            received.extend([frame async for frame in socket])

        # Exercise the real HTTP Upgrade and data framing, but force TCP to loopback.
        quiet_server = logging.Logger("synthetic-sideband-server", logging.CRITICAL + 1)
        async with serve(provider, "127.0.0.1", 0, logger=quiet_server) as server:
            loopback_port = next(iter(server.sockets)).getsockname()[1]
            loop = asyncio.get_running_loop()
            dial = loop.create_connection

            async def dial_loopback(factory, *, host, port: int, **kwargs):
                assert (host, port) == ("api.openai.com", 443)
                assert kwargs.pop("ssl") is True
                assert kwargs.pop("server_hostname") == "api.openai.com"
                return await dial(factory, "127.0.0.1", loopback_port, **kwargs)

            monkeypatch.setattr(loop, "create_connection", dial_loopback)
            async with open_sideband(api_key=KEY, call_id=CALL) as events:
                assert await anext(events) == TranscriptDelta(
                    CALL,
                    None,
                    "user",
                    "private transcript",
                    None,
                    None,
                )
                if provider_closed:
                    assert [event async for event in events] == [
                        SessionClosed(CALL, None)
                    ]
                else:
                    with pytest.raises(SidebandError) as caught:
                        await anext(events)
                    assert caught.value.reason == "disconnected"

    with caplog.at_level(logging.DEBUG):
        asyncio.run(asyncio.wait_for(run(), 3))
    assert received == [(f"/v1/live/sessions/{CALL}/attach", f"Bearer {KEY}")]
    for private in (KEY, "private audio", "private transcript", "private close reason"):
        assert private not in caplog.text
