"""Server→client request ordering on the wire (tui_gateway/server_requests.py).

A request is published in ``_open`` before its frame is written, so a cancel (session.interrupt / close /
shutdown on an RPC thread) can withdraw it in between. The renderer must still see the request before its
``request.cancel`` — a cancel for an id it never saw is dropped, and the late frame becomes a prompt card
the backend already withdrew.
"""

from __future__ import annotations

import threading

import pytest

from tui_gateway import server_requests


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
