"""Protocol tests must not leave process-wide change broadcasts behind (#147198)."""

import asyncio
import json
import threading

from tui_gateway import server
from tui_gateway import ws as ws_mod


def test_websocket_protocol_test_leaves_no_change_watcher(monkeypatch):
    frames = []
    monkeypatch.setattr(server, "resolve_skin", dict)
    monkeypatch.setattr(server, "_ensure_lease_watcher", lambda: None)
    monkeypatch.setattr(ws_mod, "_start_backend_liveness", lambda: None)

    class Socket:
        async def accept(self):
            pass

        async def send_text(self, line):
            frames.append(json.loads(line))

        async def receive_text(self):
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(Socket()))

    assert frames[0]["params"]["type"] == "gateway.ready"
    assert not any(
        thread.name == "hermes-change-watcher" and thread.is_alive()
        for thread in threading.enumerate()
    ), "a protocol test started a permanent watcher that can broadcast into later tests"


def test_protocol_test_does_not_inherit_a_stdio_broadcast_sink(monkeypatch):
    frames = []
    monkeypatch.setattr(server, "_live_transports", set())
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)

    server._broadcast_global_event("skills.changed")

    assert frames == [], "an earlier stdio entry test left its broadcast channel active"
