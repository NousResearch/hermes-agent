"""groups.state names the member answering, so Desktop can show per-bot activity.

Regression for the gateway-owned group view losing its working indicators, and #102196.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from tests.waiting import wait_until
from hermes_agent.gateway import hosted_room_driver as driver
from hermes_agent.tui_gateway.hosted_room_member_activity import project_room_member_activity
from hermes_agent.tui_gateway.hosted_room_service import HostedRoomService


class _HeldRPC:
    """Real driver path; the member's turn stays running until the test releases it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.sessions = {}

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(self, *, profile, on_terminal, **_kwargs):
        self.started.set()
        assert self.release.wait(timeout=5)
        on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}

    def history(self, *, profile, session_id, source):
        return []

    def info(self, *, profile, session_id, source):
        return {"active": False, "task_id": None}


def test_status_names_the_member_answering_while_its_turn_runs(tmp_path: Path):
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=tmp_path / "state.db")
    service.rpc = service.runtime.rpc = _HeldRPC()
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.start()
    try:
        service.send(
            room_id="room-1",
            event_id="user-1",
            payload={"text": "@ops inspect the release", "thread_id": "thread-1"},
        )
        assert service.rpc.started.wait(timeout=5)
        status = service.status("room-1")
        assert status["working"] is True
        [activity] = status["member_activity"]
        assert (activity["member_id"], activity["status"]) == ("ops", "running")
        assert activity["started_at"] > 0 and "tool" not in activity
        assert status["max_rounds"] == service.discussion_limits.max_rounds

        # The member's own session frames say which tool it is in, and that it is still making progress.
        [task] = driver.list_tasks(service.db_path, room_id="room-1")
        sessions = {"member-sid": {"_hosted_room_task": {
            "room_id": "room-1", "task_id": task["identity"].task_id, "member_id": "ops",
            "execution_generation": task["execution_generation"]}}}

        def frame(kind: str) -> dict:
            return {"method": "event", "params": {
                "type": kind, "session_id": "member-sid", "payload": {"tool_id": "c1", "name": "terminal"}}}

        project_room_member_activity(frame("tool.start"), sessions)
        [activity] = service.status("room-1")["member_activity"]
        assert activity["tool"] == "terminal" and activity["progress_at"] >= activity["started_at"]
        project_room_member_activity(frame("tool.complete"), sessions)
        [activity] = service.status("room-1")["member_activity"]
        assert "tool" not in activity and activity["progress_at"] > 0

        service.rpc.release.set()
        wait_until(lambda: service.status("room-1")["member_activity"] == [], timeout=5)
        assert service.status("room-1")["working"] is False
    finally:
        service.rpc.release.set()
        assert service.stop(timeout=5.0)
