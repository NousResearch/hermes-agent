"""EmbeddedTransport: an in-process host (e.g. a mobile app embedding CPython) drives the gateway
through a callback instead of stdio or a WebSocket.

- a request submitted as a JSON line is answered through the host callback, on the real dispatch path
- a malformed line is answered with a JSON-RPC parse error, not an exception into the host
- an inline handler that crashes is answered with an internal error and the pump keeps serving
- a long handler answers from the dispatch pool through the same host callback
- the host's ``False`` return means the peer is gone for good; a closed transport never calls the host
- ``connect`` greets the host and attaches it for global broadcasts; ``close`` detaches it
"""

import json
import threading

from tests.waiting import wait_until
from tui_gateway import server
from tui_gateway.embedded import EmbeddedTransport


class Host:
    def __init__(self, accept: bool = True):
        self.frames: list[dict] = []
        self.accept = accept

    def emit(self, line: str) -> bool:
        self.frames.append(json.loads(line))
        return self.accept


def test_a_request_is_answered_through_the_host_callback():
    request = {"jsonrpc": "2.0", "id": 7, "method": "gateway.capabilities", "params": {}}
    host = Host()
    transport = EmbeddedTransport(host.emit)

    transport.submit(json.dumps(request))

    assert host.frames == [server.dispatch(dict(request))]


def test_a_malformed_line_is_answered_with_a_parse_error():
    host = Host()

    EmbeddedTransport(host.emit).submit("{not json")

    assert [(f["id"], f["error"]["code"]) for f in host.frames] == [(None, -32700)]


def test_a_crashing_inline_handler_is_answered_and_the_pump_keeps_serving(monkeypatch):
    def boom(rid, params):
        raise RuntimeError("handler bug")

    monkeypatch.setitem(server._methods, "test.boom", boom)
    host = Host()
    transport = EmbeddedTransport(host.emit)

    transport.submit('{"jsonrpc": "2.0", "id": 1, "method": "test.boom", "params": {}}')
    transport.submit('{"jsonrpc": "2.0", "id": 2, "method": "gateway.capabilities", "params": {}}')

    assert [(f["id"], f.get("error", {}).get("code")) for f in host.frames] == [(1, -32603), (2, None)]


def test_a_long_handler_answers_from_the_pool_through_the_same_host(monkeypatch):
    handler_threads = []

    def slow(rid, params):
        handler_threads.append(threading.current_thread())
        return server._ok(rid, {"done": True})

    monkeypatch.setitem(server._methods, "test.slow", slow)
    monkeypatch.setattr(server, "_LONG_HANDLERS", server._LONG_HANDLERS | {"test.slow"})
    host = Host()

    EmbeddedTransport(host.emit).submit('{"jsonrpc": "2.0", "id": 3, "method": "test.slow", "params": {}}')

    assert wait_until(lambda: host.frames, what="pool response") == [{"jsonrpc": "2.0", "id": 3, "result": {"done": True}}]
    assert handler_threads and handler_threads[0] is not threading.current_thread()


def test_a_host_that_rejects_a_frame_is_treated_as_gone_from_then_on():
    host = Host(accept=False)
    transport = EmbeddedTransport(host.emit)

    assert transport.write({"jsonrpc": "2.0", "method": "event"}) is False
    assert transport.write({"jsonrpc": "2.0", "method": "event"}) is False

    assert len(host.frames) == 1
    assert server._transport_is_dead(transport)


def test_a_closed_transport_dispatches_nothing(monkeypatch):
    calls = []
    monkeypatch.setitem(server._methods, "test.record", lambda rid, params: calls.append(rid) or server._ok(rid, {}))
    transport = EmbeddedTransport(Host().emit)
    transport.close()

    transport.submit('{"jsonrpc": "2.0", "id": 1, "method": "test.record", "params": {}}')

    assert calls == []


def test_connect_greets_the_host_and_receives_global_broadcasts():
    host = Host()
    transport = EmbeddedTransport.connect(host.emit)
    try:
        server._broadcast_global_event("skin.changed", {"probe": True})
    finally:
        transport.close()

    assert host.frames[0]["params"]["type"] == "gateway.ready"
    assert host.frames[-1]["params"] == {"type": "skin.changed", "session_id": "", "payload": {"probe": True}}


def test_close_detaches_the_host_from_the_gateway():
    host = Host()
    transport = EmbeddedTransport.connect(host.emit)
    transport.close()
    greeted = len(host.frames)

    server._broadcast_global_event("skin.changed", {"probe": True})

    assert len(host.frames) == greeted
    assert transport not in server._live_transports  # no leaked peer per closed host


def test_a_closed_transport_never_calls_the_host():
    host = Host()
    transport = EmbeddedTransport(host.emit)

    transport.close()

    assert transport.write({"jsonrpc": "2.0", "method": "event"}) is False
    assert host.frames == []
