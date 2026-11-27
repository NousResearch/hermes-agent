"""A settlement by one client must reach the session's other clients (#136709).

``resolve_response`` and the final ``clarify.lock`` settled a request without notifying the other
attached clients, so their approval/clarify cards stayed open and a later tap was dropped. The
timeout / interrupt / ``approval.respond`` paths already emit ``request.cancel``; settlement now
emits it too (reason ``resolved``) — but only when another client may still be showing the card,
keeping the single-client promise of #112548: the answerer itself never gets a bogus cancel.
"""

import pytest

from tui_gateway import server_requests


@pytest.fixture
def events(monkeypatch):
    """Captured session events, with two answering clients attached to s1."""
    out: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server_requests,
        "_emit",
        lambda event, sid, payload: out.append((event, sid, payload)),
    )
    monkeypatch.setattr(server_requests, "_clients", lambda sid: [object(), object()])
    yield out
    server_requests.reset_for_tests()


def _cancels(events) -> list[dict]:
    return [payload for event, _sid, payload in events if event == "request.cancel"]


def _open_request(
    sid: str = "s1", method: str = "approval", qids: list[str] | None = None
):
    req = server_requests.ServerRequest(sid, method, {}, qids=qids)
    with server_requests._lock:
        server_requests._open[req.id] = req
    return req


def test_a_response_frame_settle_notifies_the_other_clients(events):
    req = _open_request()

    assert (
        server_requests.resolve_response({"id": req.id, "result": {"choice": "once"}})
        is True
    )

    assert _cancels(events) == [
        {"id": req.id, "method": "approval", "reason": "resolved"}
    ]


def test_an_error_response_settle_notifies_the_other_clients(events):
    """An error reply settles the request just the same; the other clients' cards are stale too."""
    req = _open_request(method="vault")

    assert (
        server_requests.resolve_response({"id": req.id, "error": {"code": -32601}})
        is True
    )

    assert _cancels(events) == [{"id": req.id, "method": "vault", "reason": "resolved"}]


def test_the_final_clarify_lock_notifies_the_other_clients(events):
    """Locking the last question resolves the batch: that settlement notifies too, an early
    lock does not."""
    req = _open_request(method="clarify", qids=["q1", "q2"])

    assert server_requests.lock_answer(req.id, "q1", "yes") == ["q2"]
    assert _cancels(events) == []
    assert server_requests.lock_answer(req.id, "q2", "no") == []

    assert _cancels(events) == [
        {"id": req.id, "method": "clarify", "reason": "resolved"}
    ]


def test_a_lone_client_never_hears_a_cancel_for_its_own_answer(monkeypatch):
    """One attached client is the settler itself: no cancel, or #112548's guard is back (#136709
    must not reintroduce the bogus notice for the card the user just answered)."""
    out: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server_requests,
        "_emit",
        lambda event, sid, payload: out.append((event, sid, payload)),
    )
    monkeypatch.setattr(server_requests, "_clients", lambda sid: [object()])

    req = _open_request()
    assert (
        server_requests.resolve_response({"id": req.id, "result": {"choice": "once"}})
        is True
    )
    batch = _open_request(method="clarify", qids=["q1"])
    assert server_requests.lock_answer(batch.id, "q1", "yes") == []

    assert out == []
    server_requests.reset_for_tests()


def test_the_settle_notice_is_not_repeated_when_the_queue_settles_later(
    events, monkeypatch
):
    """The approval queue's ``settle("resolved")`` finds the request already gone and must not
    emit a second cancel."""
    frames: list[dict] = []
    monkeypatch.setattr(server_requests, "_write", frames.append)
    seen: list = []
    settle = server_requests.send_async(
        "approval",
        "s1",
        {
            "request_id": "r1",
            "command": "rm -rf /tmp/demo",
            "description": "demo",
            "pattern_key": "",
            "pattern_keys": [],
        },
        seen.append,
    )
    rid = frames[-1]["id"]

    assert (
        server_requests.resolve_response({"id": rid, "result": {"choice": "once"}})
        is True
    )
    assert seen == [{"choice": "once"}]
    settle("resolved")

    assert _cancels(events) == [{"id": rid, "method": "approval", "reason": "resolved"}]
