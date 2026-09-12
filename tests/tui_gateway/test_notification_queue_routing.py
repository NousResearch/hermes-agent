"""Regression coverage for session-targeted background completion dequeue."""

from __future__ import annotations

import queue
import threading
from typing import Callable

from tools.process_registry import _NotificationQueue, process_registry
from tui_gateway import server


class _StopAfterOnePoll:
    def __init__(self) -> None:
        self._checks = 0

    def is_set(self) -> bool:
        self._checks += 1
        return self._checks > 1


class _InstrumentedQueue:
    def __init__(self, events: list[dict]) -> None:
        self.events = list(events)
        self.get_calls = 0
        self.matching_calls = 0

    def get(self, timeout: float | None = None) -> dict:
        self.get_calls += 1
        raise queue.Empty

    def get_matching(
        self,
        predicate: Callable[[dict], bool],
        timeout: float | None = None,
    ) -> dict:
        self.matching_calls += 1
        for index, event in enumerate(self.events):
            if predicate(event):
                return self.events.pop(index)
        raise queue.Empty

    def get_nowait(self) -> dict:
        if not self.events:
            raise queue.Empty
        return self.events.pop(0)

    def put(self, event: dict) -> None:
        self.events.append(event)

    def empty(self) -> bool:
        return not self.events


def _session(session_key: str) -> dict:
    return {
        "session_key": session_key,
        "history_lock": threading.Lock(),
        "running": False,
        "_finalized": False,
    }


def test_notification_queue_removes_only_the_matching_event() -> None:
    routed_queue = _NotificationQueue()
    foreign = {"session_key": "foreign"}
    owned = {"session_key": "owned"}
    routed_queue.put(foreign)
    routed_queue.put(owned)

    assert (
        routed_queue.get_matching(
            lambda event: event.get("session_key") == "owned", timeout=0
        )
        is owned
    )
    assert routed_queue.get_nowait() is foreign


def test_notification_poller_dequeues_its_event_without_rotating_foreign_event(
    monkeypatch,
) -> None:
    foreign = {
        "type": "completion",
        "session_id": "proc-foreign",
        "session_key": "foreign-key",
        "command": "sleep 1",
        "exit_code": 0,
        "output": "foreign",
    }
    owned = {
        "type": "completion",
        "session_id": "proc-owned",
        "session_key": "owned-key",
        "command": "sleep 1",
        "exit_code": 0,
        "output": "owned",
    }
    routed_queue = _InstrumentedQueue([foreign, owned])
    owner = _session("owned-key")
    other = _session("foreign-key")
    delivered: list[str] = []

    monkeypatch.setattr(process_registry, "completion_queue", routed_queue)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda _rid, _sid, _session, text, **_kwargs: delivered.append(text),
    )
    server._sessions.update({"owner-sid": owner, "foreign-sid": other})
    process_registry._completion_consumed.discard("proc-owned")

    try:
        getattr(server, "_notification_poller_loop")(
            _StopAfterOnePoll(), "owner-sid", owner
        )

        assert routed_queue.matching_calls == 1
        assert routed_queue.get_calls == 0
        assert len(delivered) == 1
        assert "proc-owned completed normally" in delivered[0]
        assert routed_queue.events == [foreign]
    finally:
        server._sessions.pop("owner-sid", None)
        server._sessions.pop("foreign-sid", None)
        process_registry._completion_consumed.discard("proc-owned")
