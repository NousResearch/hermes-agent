"""The actual room worker consumes scoped grants and keeps denied/revoked requests manual."""

import pytest

from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_rooms
from gateway import hosted_room_driver as driver
from tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.hosted_room_service_fixtures import _FakeRPC, _server
from tests.gateway.test_hosted_room_approval_rules import pending, finish, transaction
from gateway import hosted_room_approval_rules as rules


def observe(service, request, attempt):
    service.runtime._leases[request["room_id"]] = attempt.lease
    service._set_pending_action(request["room_id"], request["member_id"], {
        **request, "kind": "approval", "session_id": "live-writer",
    })


@pytest.mark.parametrize("later", ["same", "other-room", "revoked", "new-operation"])
def test_worker_restart_reuses_only_the_same_group_bot_and_operation(tmp_path, monkeypatch, later):
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: "home")
    db = tmp_path / "state.db"
    first, attempt = pending(db)
    service = HostedRoomService(_server(), db_path=db)
    service.rpc = service.runtime.rpc = _FakeRPC()
    observe(service, first, attempt)
    result = approvals.submit_approval(db, service=service, command_id="owner-confirmed", pending=first, choice="remember")
    assert result["applied"] is True and result["queued"] is False
    assert service.rpc.approvals == [{"session_id": "live-writer", "request_id": first["request_id"], "choice": "once"}]
    finish(db, attempt)
    if later == "revoked":
        with transaction(db) as conn:
            grant = rules.list_rules(conn, "group-a")[0]
            assert rules.revoke_rule(conn, "group-a", grant["rule_id"])
    second, second_attempt = pending(db, suffix="2", room_id="group-b" if later == "other-room" else "group-a",
                                     key="b" * 64 if later == "new-operation" else "a" * 64)
    restarted = HostedRoomService(_server(), db_path=db)
    restarted.rpc = restarted.runtime.rpc = _FakeRPC()
    observe(restarted, second, second_attempt)
    if later == "same":
        assert restarted.rpc.approvals == [{"session_id": "live-writer", "request_id": second["request_id"], "choice": "once"}]
        assert approvals.list_pending_approvals(db, room_id=second["room_id"]) == []
    else:
        assert restarted.rpc.approvals == []
        assert approvals.list_pending_approvals(db, room_id=second["room_id"])[0]["request_id"] == second["request_id"]


@pytest.mark.parametrize("override,eligible", [
    ({}, True), ({"choices": ["once", "deny"]}, False), ({"choices": "always"}, False),
    ({"allow_permanent": False}, False), ({"allow_permanent": "true"}, False),
    ({"allow_session": False}, False), ({"smart_denied": True}, False),
    ({"remember_context": ""}, False), ({"remember_key": "invalid"}, False),
])
def test_observer_carries_only_supported_explicit_remember_metadata(tmp_path, monkeypatch, override, eligible):
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: "home")
    db = tmp_path / "state.db"
    _, attempt = pending(db)
    service = HostedRoomService(_server(), db_path=db)
    service.rpc = service.runtime.rpc = _FakeRPC()
    service.runtime.process_generation = "worker"
    service.runtime._leases["group-a"] = attempt.lease
    metadata = {"request_id": "observed", "command": "rm -rf ./build", "description": "Remove build",
                "choices": ["once", "deny", "always", "session"], "remember_key": "c" * 64,
                "remember_context": "Local, folder /workspace", "allow_permanent": True, "allow_session": True,
                **override}
    service.runtime._report_pending_action(service.bindings()[0], driver.get_task(db, attempt.identity),
                                          session_id="live", info={"pending_approval": metadata})
    stored = approvals.list_pending_approvals(db, room_id="group-a")[0]
    assert stored["request_id"] == "observed"
    assert "always" not in stored["approval"]["choices"] and "session" not in stored["approval"]["choices"]
    assert ("remember" in stored["approval"]["choices"]) is eligible
    assert bool(stored["approval"].get("remember_context")) is eligible
    assert service.rpc.approvals == []
