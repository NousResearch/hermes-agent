"""The dashboard sidecar publisher dials its loopback socket directly, never through HTTP(S)_PROXY."""

import asyncio
import threading

import pytest

from tests.waiting import wait_until

pytest.importorskip("websockets")


def test_sidecar_publisher_reaches_loopback_with_a_proxy_set(monkeypatch):
    from websockets.asyncio.server import serve

    from tui_gateway.event_publisher import WsPublisherTransport

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:9")  # nothing listens: a proxied dial fails
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)

    frames, ready, port = [], threading.Event(), {}
    loop_box = {}

    async def handler(ws):
        async for message in ws:
            frames.append(message)

    async def main():
        stop = asyncio.Event()
        loop_box["stop"] = (asyncio.get_running_loop(), stop)
        async with serve(handler, "127.0.0.1", 0) as srv:
            port["p"] = next(iter(srv.sockets)).getsockname()[1]
            ready.set()
            await stop.wait()

    worker = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    worker.start()
    assert ready.wait(5)
    try:
        publisher = WsPublisherTransport(f"ws://127.0.0.1:{port['p']}/sidecar")
        assert publisher.write({"jsonrpc": "2.0", "method": "event", "params": {"type": "probe"}})
        wait_until(lambda: frames, timeout=5, what="the loopback server received the frame")
        publisher.close()
    finally:
        loop, stop = loop_box["stop"]
        loop.call_soon_threadsafe(stop.set)
        worker.join(5)
