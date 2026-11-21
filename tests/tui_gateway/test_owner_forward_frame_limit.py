"""#145215: a forwarded prompt must accept the same frame the client path does.

A non-owner backend forwards a Desktop prompt to the gateway that owns the session
(``tui_gateway/owner_forward.py``). That dial opened its socket with the ``websockets``
default ``max_size`` (1 MiB), while ``connect_gateway`` accepts 8 MiB. A long Bot Chat's
``session.resume`` answer passes 1 MiB, so the forward died with
``frame exceeds limit of 1048576 bytes`` and the user's message was lost.

Real: the ``connect_owner`` dial and the ``GatewayClient`` RPC it performs. Replaced: only
the mux ticket mint (an in-process constant) and ``discover_owner``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tui_gateway import owner_forward


def _serve(handler, subprotocols):
    """A real loopback WebSocket owner; returns (endpoint, stop, thread)."""
    from websockets.asyncio.server import serve

    ready, state = threading.Event(), {}

    async def run():
        state["stop"] = asyncio.get_running_loop().create_future()
        async with serve(handler, "127.0.0.1", 0, subprotocols=subprotocols) as server:
            state["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await state["stop"]

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_until_complete, args=(run(),), daemon=True)
    thread.start()
    assert ready.wait(10), "loopback owner never started"

    def stop():
        loop.call_soon_threadsafe(state["stop"].set_result, None)
        thread.join(10)

    return SimpleNamespace(api_origin=f"http://127.0.0.1:{state['port']}"), stop


def test_a_forwarded_prompt_accepts_an_answer_past_the_one_mib_default(tmp_path, monkeypatch):
    """The owner's ``session.resume`` answer is a >1 MiB snapshot; the forward must take it.

    On the unfixed code the socket dies at 1 MiB (``client_disconnect(code=1009)``) and
    ``connect_owner`` raises instead of yielding the client.
    """
    from hermes_cli import gateway_client

    snapshot = {"session_id": "long-chat", "messages": ["x" * 1024 for _ in range(1400)]}
    payload = json.dumps(snapshot)
    assert len(payload) > 2 ** 20, "the fixture must exceed websockets' default frame cap"

    async def answer(ws):
        async for raw in ws:
            request = json.loads(raw)
            assert request["method"] == "session.resume"
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": snapshot}))

    endpoint, stop = _serve(answer, [gateway_client.GATEWAY_WS_PROTOCOL])
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *_a, **_k: "ticket")
    try:
        async def resume():
            async with owner_forward.connect_owner(Path(tmp_path), endpoint) as client:
                return await client.rpc("session.resume", session_id="long-chat")

        attached = asyncio.run(resume())
    finally:
        stop()

    assert attached == snapshot


def test_the_forward_and_client_paths_share_one_frame_bound(monkeypatch):
    """The two dials must not drift again: the forward default IS the client default."""
    from hermes_cli import gateway_client

    bound = gateway_client.DEFAULT_MAX_FRAME_BYTES
    assert bound == 8 * 1024 * 1024
    seen = {}

    @__import__("contextlib").asynccontextmanager
    async def fake_connect(url, **kwargs):
        seen.update(kwargs)
        yield SimpleNamespace(subprotocol=gateway_client.GATEWAY_WS_PROTOCOL)

    import websockets.asyncio.client as ws_client
    monkeypatch.setattr(ws_client, "connect", fake_connect)
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *_a, **_k: "ticket")

    async def dial():
        async with owner_forward.connect_owner(Path("."), SimpleNamespace(api_origin="http://127.0.0.1:1")):
            pass

    asyncio.run(dial())
    assert seen["max_size"] == bound


def test_an_explicit_frame_bound_is_honoured(tmp_path, monkeypatch):
    """``forward_rpc`` sizes its socket for the largest attachment the room store accepts."""
    import contextlib
    from hermes_cli import gateway_client
    import websockets.asyncio.client as ws_client

    seen = {}

    @contextlib.asynccontextmanager
    async def fake_connect(url, **kwargs):
        seen.update(kwargs)
        yield SimpleNamespace(subprotocol=gateway_client.GATEWAY_WS_PROTOCOL)

    monkeypatch.setattr(ws_client, "connect", fake_connect)
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *_a, **_k: "ticket")

    from gateway.hosted_room_attachments import MAX_ATTACHMENT_BYTES
    expected = (MAX_ATTACHMENT_BYTES + 2) // 3 * 4 + 2 ** 20
    assert expected > gateway_client.DEFAULT_MAX_FRAME_BYTES

    async def call():
        async with owner_forward.connect_owner(Path(tmp_path), SimpleNamespace(api_origin="http://127.0.0.1:1"),
                                               scope="host", max_size=expected):
            pass

    asyncio.run(call())
    assert seen["max_size"] == expected


def test_resume_omits_the_stored_transcript(monkeypatch):
    """A persisted chat past the frame cap must not be loaded on the forward path.

    ``_Relay`` only reads ``session_id``. Requesting the whole transcript closed the
    socket with ``1009 frame exceeds limit`` before ``prompt.submit``.
    """
    seen = {}

    class _Client:
        async def rpc(self, method, **params):
            seen.setdefault(method, params)
            if method == "session.resume":
                return {"session_id": "long-chat"}
            return {"status": "queued", "admission_id": "adm-1", "voice_stopped": True}

    class _Conn:
        async def __aenter__(self):
            return _Client()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(owner_forward, "connect_owner", lambda *a, **k: _Conn())
    relay = owner_forward._Relay("sid", {}, "long-chat", Path("."), None, "hello", None)
    asyncio.run(relay._run())
    assert seen["session.resume"]["omit_messages"] is True
    assert seen["session.resume"]["session_id"] == "long-chat"
