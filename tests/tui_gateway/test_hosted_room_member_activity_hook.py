"""on_room_member_activity: a room member's hidden-session runtime events reach plugins with room
coordinates; an ordinary session's identical events never do."""

from __future__ import annotations

import threading
import time

import pytest

from tui_gateway import server
from tui_gateway.hosted_room_member_activity import HOOK_NAME


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def observer(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    seen: list[dict] = []
    lock = threading.Lock()

    def on_activity(**kwargs):
        with lock:
            seen.append(kwargs)

    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks", lambda name: (on_activity,) if name == HOOK_NAME else ())
    monkeypatch.setattr(server, "_stdio_transport", type("Sink", (), {"write": staticmethod(lambda _o: True)})())
    yield seen
    shutdown_plugin_stream_hook_dispatcher()


HOSTED_TASK = {
    "room_id": "room-a", "task_id": "task-1", "thread_id": "thread-1", "turn_id": "turn-1",
    "execution_generation": 2, "member_id": "reviewer"}


def _session(sid: str, hosted: bool):
    entry = {"session_key": sid, "transport": None, "agent": None, "created_at": time.time()}
    if hosted:
        entry["_hosted_room_task"] = dict(HOSTED_TASK)
    with server._sessions_lock:
        server._sessions[sid] = entry


def test_room_member_session_events_reach_plugins_with_room_coordinates(observer):
    _session("room-sid", hosted=True)
    try:
        server._emit("tool.start", "room-sid", {"tool_id": "call-1", "name": "terminal", "args": {"command": "ls"}})
        server._emit("approval.request", "room-sid", {"request_id": "req-1", "command": "rm -rf build"})
        server._emit("session.info", "room-sid", {"title": "chrome, not member activity"})
        server._emit("tool.complete", "room-sid", {"tool_id": "call-1", "name": "terminal", "result": "ok"})
    finally:
        with server._sessions_lock:
            server._sessions.pop("room-sid", None)

    _wait_for(lambda: len(observer) == 3)
    kinds = [event["kind"] for event in observer]
    assert kinds == ["tool.started", "request.opened", "tool.completed"]
    started = observer[0]
    assert {k: started[k] for k in HOSTED_TASK} == HOSTED_TASK
    assert started["payload"]["tool_id"] == "call-1" and started["payload"]["args"] == {"command": "ls"}
    assert observer[1]["payload"]["request_id"] == "req-1"
    # Per-session replay seq travels with the event so consumers can order/dedupe.
    assert [event["seq"] for event in observer] == sorted(event["seq"] for event in observer)


def test_ordinary_session_events_never_fire_the_room_hook(observer):
    _session("plain-sid", hosted=False)
    try:
        server._emit("tool.start", "plain-sid", {"tool_id": "call-1", "name": "terminal"})
        server._emit("approval.request", "plain-sid", {"request_id": "req-1", "command": "ls"})
        server._emit("message.delta", "plain-sid", {"text": "hi"})
    finally:
        with server._sessions_lock:
            server._sessions.pop("plain-sid", None)

    time.sleep(0.2)
    assert observer == []
