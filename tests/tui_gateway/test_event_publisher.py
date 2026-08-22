import json
import threading
import time

from tui_gateway import event_publisher as publisher


class _FakeWS:
    def __init__(self):
        self.closed = False
        self.sent: list[str] = []
        self.sent_event = threading.Event()

    def send(self, line: str) -> None:
        self.sent.append(line)
        self.sent_event.set()

    def close(self) -> None:
        self.closed = True


def test_constructor_does_not_block_on_connect(monkeypatch):
    gate = threading.Event()
    ws = _FakeWS()

    def fake_connect(url, open_timeout=None, max_size=None):
        gate.wait(timeout=1.0)
        return ws

    monkeypatch.setattr(publisher, "ws_connect", fake_connect)

    start = time.perf_counter()
    transport = publisher.WsPublisherTransport("ws://example.test/pub")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.2
    gate.set()
    transport.close()


def test_write_before_connect_is_buffered_until_background_connect(monkeypatch):
    gate = threading.Event()
    ws = _FakeWS()

    def fake_connect(url, open_timeout=None, max_size=None):
        gate.wait(timeout=1.0)
        return ws

    monkeypatch.setattr(publisher, "ws_connect", fake_connect)

    transport = publisher.WsPublisherTransport("ws://example.test/pub")
    assert transport.write({"hello": "world"}) is True

    gate.set()
    assert ws.sent_event.wait(timeout=1.0)
    assert json.loads(ws.sent[0]) == {"hello": "world"}

    transport.close()
