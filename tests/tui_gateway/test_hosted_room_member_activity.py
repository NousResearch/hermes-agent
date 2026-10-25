"""groups.state names the member answering, so Desktop can show per-bot activity.

Regression for the gateway-owned group view losing its working indicators, and #102196.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from tests.waiting import wait_until
from tui_gateway.hosted_room_service import HostedRoomService


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
        assert status["member_activity"] == [{"member_id": "ops", "status": "running"}]

        service.rpc.release.set()
        wait_until(lambda: service.status("room-1")["member_activity"] == [], timeout=5)
        assert service.status("room-1")["working"] is False
    finally:
        service.rpc.release.set()
        assert service.stop(timeout=5.0)
