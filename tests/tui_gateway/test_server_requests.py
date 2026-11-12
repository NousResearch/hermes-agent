"""Server→client request ordering on the wire (hermes_agent/tui_gateway/server_requests.py).

A request is published in ``_open`` before its frame is written, so a cancel (session.interrupt / close /
shutdown on an RPC thread) can withdraw it in between. The renderer must still see the request before its
``request.cancel`` — a cancel for an id it never saw is dropped, and the late frame becomes a prompt card
the backend already withdrew.
"""

from __future__ import annotations

import threading

import pytest

from hermes_agent.tui_gateway import server_requests


@pytest.fixture()
def wire(monkeypatch):
    frames: list[tuple[str, str]] = []
    monkeypatch.setattr(server_requests, "_write", lambda frame: frames.append(("request", frame["id"])))
    monkeypatch.setattr(server_requests, "_emit",
                        lambda event, sid, payload: frames.append((event, payload["id"])))
    monkeypatch.setattr(server_requests, "_answerable", lambda sid: True)
    server_requests.reset_for_tests()
    yield frames
    server_requests.reset_for_tests()


def test_cancel_racing_the_request_write_reaches_the_wire_after_the_request(wire, monkeypatch):
    write_request = server_requests._write
    cancelled = []

    def write_after_an_interrupt(frame):
        # session.interrupt on another thread lands after registration, before the frame goes out.
        interrupt = threading.Thread(target=lambda: cancelled.append(server_requests.cancel("s1")))
        interrupt.start()
        interrupt.join(timeout=10)
        write_request(frame)

    monkeypatch.setattr(server_requests, "_write", write_after_an_interrupt)

    assert server_requests.send("sudo", "s1", {}, timeout=10) is None
    assert cancelled == [1]
    assert [event for event, _ in wire] == ["request", "request.cancel"]
    assert wire[0][1] == wire[1][1]
    assert server_requests.open_requests("s1") == []


def test_cancel_after_the_write_is_emitted_once(wire):
    pending = server_requests.send_async("approval", "s1", {"request_id": "r1", "command": "rm", "description": ""},
                                         lambda result: None)

    assert server_requests.cancel("s1") == 1
    pending("resolved")
    assert [event for event, _ in wire] == ["request", "request.cancel"]


class _Socket:
    def __init__(self):
        self.frames: list[dict] = []

    def write(self, frame):
        self.frames.append(frame)
        return True


def test_a_pinned_request_goes_to_and_is_settled_only_by_the_socket_that_asked(wire):
    asker, bystander = _Socket(), _Socket()
    params = {"action": "vault", "vault": {"operation": "close", "target": "t"}}
    out: list = []
    worker = threading.Thread(target=lambda: out.append(
        server_requests.send("preview.act", "", params, timeout=10, transport=asker)))
    worker.start()
    for _ in range(1000):
        if asker.frames:
            break
        threading.Event().wait(0.01)
    rid = asker.frames[0]["id"]
    assert wire == [] and bystander.frames == []  # never broadcast

    forged = {"id": rid, "result": {"value": "from another window"}}
    assert not server_requests.resolve_response(forged, bystander)
    assert not server_requests.resolve_pinned_response(forged, bystander)
    assert server_requests.open_request_count() == 1

    assert server_requests.resolve_pinned_response({"id": rid, "result": {"value": "ok"}}, asker)
    worker.join(timeout=10)
    assert out == [{"value": "ok"}]


def _answer_with_error(code, message):
    result: list = []
    asker = threading.Thread(target=lambda: result.append(server_requests.send("sudo", "s1", {}, timeout=10)))
    asker.start()
    from tests.waiting import wait_until
    request = wait_until(lambda: server_requests.open_requests("s1"))[0]
    server_requests.resolve_response({"jsonrpc": "2.0", "id": request["id"], "error": {"code": code, "message": message}})
    asker.join(timeout=10)
    return result


@pytest.mark.parametrize("code", [-32603, -32601])
def test_an_error_answer_is_logged_with_its_code(wire, caplog, code):
    """An error answer still reads as unanswered for the caller, but its code and message reach the
    log at warning: a client handler crash (-32603) or a missing handler (-32601) is no longer silent."""
    with caplog.at_level("WARNING", logger="hermes_agent.tui_gateway.server_requests"):
        assert _answer_with_error(code, "handler crashed: TypeError") == [None]

    assert any(str(code) in r.getMessage() and "handler crashed" in r.getMessage() for r in caplog.records)
