"""Tests for dashboard WebSocket broadcast helpers."""


class _FakeTransport:
    def __init__(self, ok=True):
        self.ok = ok
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return self.ok


def test_broadcast_event_reaches_connected_transports_and_prunes_dead_ones():
    from tui_gateway import ws

    live = _FakeTransport(ok=True)
    dead = _FakeTransport(ok=False)

    with ws._TRANSPORTS_LOCK:
        ws._TRANSPORTS.clear()
        ws._TRANSPORTS.add(live)
        ws._TRANSPORTS.add(dead)

    try:
        result = ws.broadcast_event("skin.changed", {"name": "mono"})

        assert result == {"connected": 2, "sent": 1}
        assert live.frames == [
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "skin.changed",
                    "session_id": "",
                    "payload": {"name": "mono"},
                },
            }
        ]
        assert dead.frames
        with ws._TRANSPORTS_LOCK:
            assert live in ws._TRANSPORTS
            assert dead not in ws._TRANSPORTS
    finally:
        with ws._TRANSPORTS_LOCK:
            ws._TRANSPORTS.clear()


def test_broadcast_event_is_safe_with_no_transports():
    from tui_gateway import ws

    with ws._TRANSPORTS_LOCK:
        ws._TRANSPORTS.clear()

    assert ws.broadcast_event("skin.changed", {"name": "default"}) == {
        "connected": 0,
        "sent": 0,
    }
