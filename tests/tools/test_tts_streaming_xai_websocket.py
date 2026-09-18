"""Regression coverage for XAIStreamer against the pinned websockets client API."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any, cast

import pytest
import websockets

from tools.tts_streaming import XAIStreamer


@pytest.mark.asyncio
async def test_xai_streamer_opens_local_socket_with_authorization(monkeypatch):
    """Exercise a real connect() call so websocket keyword drift cannot be mocked away."""
    received: dict[str, object] = {}

    async def handle(ws):
        received["authorization"] = ws.request.headers["Authorization"]
        received["request"] = json.loads(await ws.recv())
        await ws.send(b"\x01\x00")
        await ws.send('{"type":"done"}')

    server = await websockets.serve(handle, "127.0.0.1", 0)
    socket = next(iter(server.sockets))
    streaming_url = f"ws://127.0.0.1:{socket.getsockname()[1]}"
    xai_http = ModuleType("tools.xai_http")
    xai_http.resolve_xai_http_credentials = lambda: {"api_key": "xai-test-key"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.xai_http", xai_http)

    try:
        streamer = XAIStreamer({}, {"streaming_url": streaming_url})
        frames = [frame async for frame in cast(Any, streamer)._async_frames("Test sentence.")]
    finally:
        server.close()
        await server.wait_closed()

    assert frames == [b"\x01\x00"]
    assert received["authorization"] == "Bearer xai-test-key"
    assert received["request"] == {
        "text": "Test sentence.", "voice_id": "eve", "response_format": "pcm",
    }
