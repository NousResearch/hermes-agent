"""Browser observe/takeover/fencing contract for live TUI gateway sessions."""

from __future__ import annotations

import threading

import pytest

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class BrowserTransport:
    supports_session_takeover = True

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self._closed = False

    def write(self, frame: dict) -> bool:
        self.frames.append(frame)
        return True


@pytest.fixture(autouse=True)
def clean_gateway_state(monkeypatch):
    server._sessions.clear()
    server._pending.clear()
    server._pending_prompt_payloads.clear()
    server._answers.clear()
    monkeypatch.setattr(server, "_cancel_ws_orphan_reap", lambda _sid: None)
    yield
    server._sessions.clear()
    server._pending.clear()
    server._pending_prompt_payloads.clear()
    server._answers.clear()


def session(owner: str, epoch: int, transport: object) -> dict:
    return {
        "agent": None,
        "created_at": 1.0,
        "history": [{"role": "assistant", "content": "already here"}],
        "history_lock": threading.Lock(),
        "owner_id": owner,
        "ownership_epoch": epoch,
        "owner_transport": transport,
        "transport": transport,
        "session_key": "durable-1",
        "running": False,
    }


def call(method: str, params: dict, transport: object) -> dict:
    token = bind_transport(transport)
    try:
        return server._methods[method]("request-1", params)
    finally:
        reset_transport(token)


def test_browser_resume_of_live_tui_session_stays_read_only(monkeypatch):
    class StdioTransport:
        supports_session_takeover = False

    tui = StdioTransport()
    browser = BrowserTransport()
    live = {
        "agent": None,
        "created_at": 1.0,
        "history": [{"role": "assistant", "content": "tui live"}],
        "history_lock": threading.Lock(),
        "session_key": "durable-1",
        "running": True,
        "transport": tui,
    }
    server._sessions["runtime-1"] = live
    monkeypatch.setattr(server, "_session_db", lambda _session: _NullContext(None))

    ctx = server._Resume("resume-1", {"owner_id": "phone-tab"}, "durable-1")
    token = bind_transport(browser)
    try:
        response = server._resume_reuse_live(ctx, "runtime-1", live)
    finally:
        reset_transport(token)

    assert response["result"]["read_only"] is True
    assert not live.get("owner_id")
    assert live["transport"] is tui
    assert browser in live.get("observers", set())


def test_foreign_browser_resume_observes_without_rebinding_live_transport(monkeypatch):
    owner_transport = BrowserTransport()
    observer_transport = BrowserTransport()
    live = session("tab-owner", 4, owner_transport)
    server._sessions["runtime-1"] = live
    monkeypatch.setattr(server, "_session_db", lambda _session: _NullContext(None))

    ctx = server._Resume("resume-1", {"owner_id": "tab-observer"}, "durable-1")
    token = bind_transport(observer_transport)
    try:
        response = server._resume_reuse_live(ctx, "runtime-1", live)
    finally:
        reset_transport(token)

    assert response["result"]["read_only"] is True
    assert response["result"]["owner_id"] == "tab-owner"
    assert response["result"]["ownership_epoch"] == 4
    assert response["result"]["messages"][0]["text"] == "already here"
    assert live["transport"] is owner_transport
    assert live["owner_transport"] is owner_transport
    assert observer_transport in live["observers"]


def test_observer_receives_session_live_events_without_becoming_owner():
    owner_transport = BrowserTransport()
    observer_transport = BrowserTransport()
    live = session("tab-owner", 2, owner_transport)
    live["observers"] = {observer_transport}
    server._sessions["runtime-1"] = live

    server.write_json(server._event_frame("message.delta", "runtime-1", {"text": "x"}))

    assert [frame["params"]["type"] for frame in owner_transport.frames] == ["message.delta"]
    assert [frame["params"]["type"] for frame in observer_transport.frames] == ["message.delta"]
    assert live["owner_id"] == "tab-owner"
    assert live["ownership_epoch"] == 2


def test_confirmed_takeover_is_atomic_increments_epoch_and_revokes_old_owner():
    old_transport = BrowserTransport()
    new_transport = BrowserTransport()
    live = session("tab-old", 7, old_transport)
    live["observers"] = {new_transport}
    server._sessions["runtime-1"] = live

    response = call("session.takeover", {
        "session_id": "runtime-1",
        "owner_id": "tab-new",
        "ownership_epoch": 7,
        "confirmed": True,
    }, new_transport)

    assert response["result"] == {
        "session_id": "runtime-1",
        "owner_id": "tab-new",
        "ownership_epoch": 8,
        "read_only": False,
    }
    assert live["transport"] is new_transport
    assert live["owner_transport"] is new_transport
    assert old_transport in live["observers"]
    assert new_transport not in live["observers"]
    revoked = old_transport.frames[-1]["params"]
    assert revoked["type"] == "session.revoked"
    assert revoked["session_id"] == "runtime-1"
    assert revoked["payload"]["ownership_epoch"] == 8


def test_takeover_rejects_non_revocable_owner_and_missing_confirmation():
    old_transport = object()
    new_transport = BrowserTransport()
    live = session("tab-old", 3, old_transport)
    server._sessions["runtime-1"] = live

    unconfirmed = call("session.takeover", {
        "session_id": "runtime-1", "owner_id": "tab-new", "ownership_epoch": 3,
    }, new_transport)
    unsafe = call("session.takeover", {
        "session_id": "runtime-1", "owner_id": "tab-new", "ownership_epoch": 3, "confirmed": True,
    }, new_transport)

    assert unconfirmed["error"]["code"] == 4094
    assert unsafe["error"]["code"] == 4095
    assert live["owner_id"] == "tab-old"
    assert live["ownership_epoch"] == 3
    assert live["transport"] is old_transport


@pytest.mark.parametrize("method,value_key", [
    ("clarify.respond", "answer"),
    ("sudo.respond", "password"),
    ("secret.respond", "value"),
])
def test_stale_owner_response_is_fenced_before_pending_prompt_side_effect(method, value_key):
    transport = BrowserTransport()
    live = session("tab-current", 9, transport)
    server._sessions["runtime-1"] = live
    event = threading.Event()
    server._pending["pending-1"] = ("runtime-1", event)

    response = call(method, {
        "session_id": "runtime-1",
        "request_id": "pending-1",
        "owner_id": "tab-stale",
        "ownership_epoch": 8,
        value_key: "must-not-land",
    }, transport)

    assert response["error"]["code"] == 4093
    assert response["error"]["data"] == {
        "reason": "SESSION_NOT_OWNED",
        "owner_id": "tab-current",
        "ownership_epoch": 9,
        "read_only": True,
    }
    assert not event.is_set()
    assert "pending-1" not in server._answers


def test_stale_owner_prompt_submit_is_fenced_before_any_turn_side_effect(monkeypatch):
    current_transport = BrowserTransport()
    live = session("tab-current", 5, current_transport)
    server._sessions["runtime-1"] = live
    touched = []
    monkeypatch.setattr(server, "_typed_stop_phrase_response", lambda *_args: touched.append(True))

    response = call("prompt.submit", {
        "session_id": "runtime-1", "text": "stop", "owner_id": "tab-old", "ownership_epoch": 4,
    }, current_transport)

    assert response["error"]["code"] == 4093
    assert touched == []
    assert live["running"] is False
    assert live["history"] == [{"role": "assistant", "content": "already here"}]


def test_current_owner_can_mutate_only_with_matching_transport_and_epoch():
    owner_transport = BrowserTransport()
    other_transport = BrowserTransport()
    live = session("tab-current", 5, owner_transport)
    server._sessions["runtime-1"] = live

    assert server._browser_mutation_fence(
        "r", {"owner_id": "tab-current", "ownership_epoch": 5}, live,
        transport=owner_transport,
    ) is None
    wrong_transport = server._browser_mutation_fence(
        "r", {"owner_id": "tab-current", "ownership_epoch": 5}, live,
        transport=other_transport,
    )
    assert wrong_transport["error"]["code"] == 4093


def test_disconnect_removes_observer_without_clobbering_new_owner_generation(monkeypatch):
    old_transport = BrowserTransport()
    new_transport = BrowserTransport()
    observer_transport = BrowserTransport()
    live = session("tab-new", 12, new_transport)
    live["observers"] = {old_transport, observer_transport}
    live["viewers"] = {old_transport: 1.0, new_transport: 2.0}
    server._sessions["runtime-1"] = live
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda _sid: None)

    reaped, detached = server._close_sessions_for_transport(old_transport)

    assert (reaped, detached) == (0, 0)
    assert old_transport not in live["observers"]
    assert live["owner_id"] == "tab-new"
    assert live["ownership_epoch"] == 12
    assert live["owner_transport"] is new_transport
    assert live["transport"] is new_transport


def test_live_events_carry_session_id_seq_and_ownership_epoch():
    from tui_gateway import event_replay
    event_replay.reset_replay_state()
    owner_transport = BrowserTransport()
    observer_transport = BrowserTransport()
    live = session("tab-owner", 4, owner_transport)
    live["observers"] = {observer_transport}
    server._sessions["runtime-1"] = live

    server._emit("message.delta", "runtime-1", {"text": "x"})

    for frames in (owner_transport.frames, observer_transport.frames):
        params = frames[-1]["params"]
        assert params["type"] == "message.delta"
        assert params["session_id"] == "runtime-1"
        assert params["seq"] == 1
        assert params["ownership_epoch"] == 4
    replayed = event_replay.events_since("runtime-1", 0)
    assert replayed[0]["seq"] == 1
    assert replayed[0]["ownership_epoch"] == 4


def test_takeover_admits_exactly_one_subsequent_owner_submit(monkeypatch):
    old_transport = BrowserTransport()
    new_transport = BrowserTransport()
    live = session("tab-old", 7, old_transport)
    server._sessions["runtime-1"] = live
    admitted = []
    monkeypatch.setattr(server, "_typed_stop_phrase_response", lambda *_args: admitted.append("stop") or {"result": "stopped"})

    call("session.takeover", {
        "session_id": "runtime-1", "owner_id": "tab-new", "ownership_epoch": 7, "confirmed": True,
    }, new_transport)
    stale = call("prompt.submit", {
        "session_id": "runtime-1", "text": "stop", "owner_id": "tab-old", "ownership_epoch": 7,
    }, old_transport)
    accepted = call("prompt.submit", {
        "session_id": "runtime-1", "text": "stop", "owner_id": "tab-new", "ownership_epoch": 8,
    }, new_transport)

    assert stale["error"]["code"] == 4093
    assert accepted.get("result") == "stopped"
    assert admitted == ["stop"]


def test_spent_request_id_does_not_overwrite_the_first_answer():
    transport = BrowserTransport()
    live = session("tab-current", 9, transport)
    server._sessions["runtime-1"] = live
    event = threading.Event()
    server._pending["pending-1"] = ("runtime-1", event)

    first = call("clarify.respond", {
        "session_id": "runtime-1", "request_id": "pending-1",
        "owner_id": "tab-current", "ownership_epoch": 9, "answer": "one",
    }, transport)
    second = call("clarify.respond", {
        "session_id": "runtime-1", "request_id": "pending-1",
        "owner_id": "tab-current", "ownership_epoch": 9, "answer": "two",
    }, transport)

    assert first["result"]["status"] == "ok"
    assert second["result"]["status"] == "expired"
    assert server._answers["pending-1"] == "one"
    assert event.is_set()


def test_session_history_keeps_the_live_tail_the_pty_uses(monkeypatch):
    transport = BrowserTransport()
    live = session("tab-owner", 1, transport)
    live["session_key"] = "durable-1"
    live["history_lock"] = threading.Lock()
    live["history"] = [
        {"role": "user", "content": "alt"},
        {"role": "assistant", "content": "aktueller PTY-Schwanz"},
    ]
    server._sessions["runtime-1"] = live

    class StaleDb:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_messages_as_conversation(self, *_args, **_kwargs):
            return [{"role": "user", "content": "alt"}]

    monkeypatch.setattr(server, "_session_db", lambda _session: StaleDb())
    response = call("session.history", {"session_id": "runtime-1"}, transport)
    assert [row.get("text") for row in response["result"]["messages"]] == ["alt", "aktueller PTY-Schwanz"]


class _NullContext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *_args):
        return False
