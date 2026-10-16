"""Behavior contract for ``SUPERVISOR_REGISTRY.capture`` (public CDP seam for trusted plugins).

A real ``CDPSupervisor`` talks to a real local WebSocket server that speaks just enough CDP
(getTargets / attachToTarget / echo). Each connection gets its own page session id
(``S1``, ``S2``, ...) so the tests can tell which socket a frame travelled on.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from tools.browser_supervisor import _SupervisorRegistry
from tools.browser_supervisor_capture import CapturedCDPInvalid


class _FakeCDP:
    def __init__(self) -> None:
        self.frames: list[tuple[int, str, str | None]] = []  # (connection no, method, sessionId)
        self.connections: list = []
        self.url = ""
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._serve, args=(ready,), daemon=True).start()
        assert ready.wait(5)

    def _serve(self, ready: threading.Event) -> None:
        from websockets.asyncio.server import serve

        async def main():
            async with serve(self._handle, "127.0.0.1", 0) as server:
                self.url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}/devtools/browser/x"
                ready.set()
                await asyncio.Event().wait()

        self.loop.run_until_complete(main())

    async def _handle(self, ws) -> None:
        self.connections.append(ws)
        conn = len(self.connections)
        async for raw in ws:
            msg = json.loads(raw)
            method, sid = msg["method"], msg.get("sessionId")
            self.frames.append((conn, method, sid))
            if method == "Test.hang":
                continue
            if method == "Test.error":
                await ws.send(json.dumps({"id": msg["id"], "error": {"code": -32000, "message": "boom"}}))
                continue
            result = {"echo": method, "sessionId": sid, "params": msg.get("params")}
            if method == "Target.getTargets":
                result = {"targetInfos": [{"targetId": "T1", "type": "page", "url": "about:blank"}]}
            elif method == "Target.attachToTarget":
                result = {"sessionId": f"S{conn}"}
            await ws.send(json.dumps({"id": msg["id"], "result": result}))

    def drop_current(self) -> None:
        asyncio.run_coroutine_threadsafe(self.connections[-1].close(), self.loop).result(5)

    def methods_on(self, conn: int) -> list[str]:
        return [m for c, m, _ in self.frames if c == conn]


@pytest.fixture
def cdp():
    return _FakeCDP()


@pytest.fixture
def registry():
    reg = _SupervisorRegistry()
    yield reg
    reg.stop_all()


def _wait(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def test_call_round_trips_on_captured_connection(cdp, registry):
    registry.get_or_start("t", cdp.url, start_timeout=5)
    handle = registry.capture("t")
    assert handle.is_valid() and handle.page_session_id == "S1"

    reply = handle.call("Runtime.evaluate", {"expression": "1"}, session_id=handle.page_session_id)
    assert reply["result"] == {"echo": "Runtime.evaluate", "sessionId": "S1", "params": {"expression": "1"}}
    assert handle.call("Target.getTargets")["result"]["targetInfos"][0]["targetId"] == "T1"  # browser-level

    with pytest.raises(RuntimeError, match="boom"):
        handle.call("Test.error")
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        handle.call("Test.hang", timeout=0.3)
    assert time.monotonic() - started < 2
    assert handle.is_valid()  # neither an error reply nor a timeout invalidates the handle

    with pytest.raises(CapturedCDPInvalid):
        registry.capture("no-such-task")


def test_handle_never_follows_a_reconnect_or_replacement(cdp, registry):
    sup = registry.get_or_start("t", cdp.url, start_timeout=5)
    old = registry.capture("t")

    # A call in flight when the socket drops fails promptly as invalidation, not at its timeout.
    errors: list = []
    caller = threading.Thread(target=lambda: errors.append(_raises(lambda: old.call("Test.hang", timeout=30))))
    caller.start()
    _wait(lambda: "Test.hang" in cdp.methods_on(1))
    started = time.monotonic()
    cdp.drop_current()
    caller.join(10)
    assert errors and isinstance(errors[0], CapturedCDPInvalid), errors
    assert time.monotonic() - started < 5

    _wait(lambda: len(cdp.connections) == 2 and sup.snapshot().active)
    assert not old.is_valid()
    with pytest.raises(CapturedCDPInvalid):
        old.call("Probe.afterReconnect", session_id=old.page_session_id)
    assert "Probe.afterReconnect" not in cdp.methods_on(2)  # never silently retargeted

    fresh = registry.capture("t")
    assert fresh.page_session_id == "S2"
    assert fresh.call("Probe.fresh", session_id=fresh.page_session_id)["result"]["sessionId"] == "S2"

    registry.stop("t")  # stop / replacement in the registry invalidates too
    assert not fresh.is_valid()
    with pytest.raises(CapturedCDPInvalid):
        fresh.call("Probe.afterStop")


def _raises(fn) -> BaseException | None:
    try:
        fn()
    except BaseException as exc:
        return exc
    return None
