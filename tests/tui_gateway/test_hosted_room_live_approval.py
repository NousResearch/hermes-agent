"""Observed live approvals must resolve without reopening their transient IDs."""

import threading

import pytest

from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry
from tui_gateway import server
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC
from tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.hosted_room_service_fixtures import _server
from tests.tui_gateway.test_hosted_room_messaging_approvals import _create_local_room
from tests.tui_gateway.test_session_resume_db_ownership import profile_dbs


@pytest.mark.parametrize("choice", ["once", "deny"])
def test_observed_named_profile_approval_resolves_only_its_live_request(
    tmp_path, monkeypatch, profile_dbs, choice,
):
    target = _ApprovalEntry({"request_id": "request-1", "command": "rm -rf ./obsolete-draft"})
    other = _ApprovalEntry({"request_id": "request-2", "command": "rm -rf ./obsolete-draft"})
    monkeypatch.setattr(approval, "_gateway_queues", {
        "stored-ops": [target], "other-session": [other],
    })
    monkeypatch.setattr(server, "_sessions", {
        "live-ops": {
            "session_key": "stored-ops", "profile_home": str(tmp_path / "work"),
            "agent": object(), "history_lock": threading.Lock(), "running": True,
        },
    })
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    service.rpc = service.runtime.rpc = HostedRoomServerRPC(server)
    _create_local_room(service)
    service._set_pending_action("room-1", "ops", {
        "kind": "approval", "task_id": "task-1", "execution_generation": 1,
        "profile": "ops", "session_id": "live-ops", "request_id": "request-1",
        "approval": {"choices": ["once", "deny"]},
    })
    result = service.approve_room_task(
        "room-1", member_id="ops", task_id="task-1", execution_generation=1,
        request_id="request-1", choice=choice,
    )
    assert result == {"resolved": 1}
    assert target.event.is_set() and target.result == choice
    assert not other.event.is_set() and other.result is None
    assert service.status("room-1")["pending_actions"] == []
    with pytest.raises(RuntimeError, match="no longer pending"):
        service.approve_room_task(
            "room-1", member_id="ops", task_id="task-1", execution_generation=1,
            request_id="request-1", choice=choice,
        )
    assert not other.event.is_set()
