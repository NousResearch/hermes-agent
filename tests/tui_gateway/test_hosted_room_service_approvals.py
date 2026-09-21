"""Hosted-room approvals retain exact local and peer task coordinates."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


class _FakeRPC:
    def __init__(self) -> None:
        self.approvals = []

    def approve(self, **kwargs):
        self.approvals.append(dict(kwargs))
        return {"resolved": 1}


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


class _ApprovalPeerClient:
    def __init__(self) -> None:
        self.dispatches = []
        self.approvals = []

    def status(self, **kwargs):
        task_id = self.dispatches[-1]["task_id"] if self.dispatches else "task-1"
        return {
            "status": "waiting_for_approval",
            "active": True,
            "task_id": task_id,
            "execution_generation": 2,
            "run_id": "run-peer-1",
            "session_id": "peer-group-session",
            "request_id": "req-peer-1",
            "approval": {
                "description": "Run the focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
        }

    def approve_receipt(self, **kwargs):
        self.approvals.append(dict(kwargs))
        return {"resolved": 1}


def _start_approval_task(service, *, task_id, member_id, profile, generation=1):
    room = hosted_rooms.room_state(service.db_path, room_id="room-1")
    identity = driver.TaskIdentity("room-1", task_id, "thread-1", "turn-1")
    driver.admit_task(
        service.db_path, identity,
        payload={"target_profile": profile, "target_member_id": member_id,
                 "prompt": "Run focused tests", "source_event_seq": 1}, clock=time.time)
    lease = driver.acquire_lease(
        service.db_path, room_id="room-1", gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"], process_generation="approval-worker",
        ttl_seconds=30, clock=time.time)
    for index in range(generation):
        attempt = driver.start_task(
            service.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
        if index + 1 < generation:
            driver.requeue_not_admitted_task(service.db_path, attempt, clock=time.time)
    return identity


def test_peer_approval_is_scoped_visible_and_resolvable(tmp_path: Path):
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=catalog.catalog_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed.room.grant",
    )
    peer = _ApprovalPeerClient()
    service = HostedRoomService(_server(), db_path=db)
    service.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=peer,
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    service.create_room(
        room_id="room-1",
        name="Peer room",
        members=[
            {
                "member_id": "default",
                "profile": "default",
                "handle": "hermes",
            },
            {
                "member_id": "member-peer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "peer-review",
                    "installation_id": "install-peer",
                    "profile": "reviewer",
                    "capability_digest": catalog.catalog_digest,
                },
            }
        ],
    )
    identity = _start_approval_task(
        service, task_id="task-1", member_id="member-peer", profile="reviewer", generation=2)
    transport = service._resolve_member_transport(
        service.bindings()[0],
        {
            "identity": identity,
            "execution_generation": 2,
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 1,
            },
        },
    )

    status = transport.info(
        profile="reviewer",
        session_id="peer-group-session",
        source="bot_room",
    )
    assert status["status"] == "waiting_for_approval"
    service._set_pending_action(
        "room-1",
        "member-peer",
        {
            "kind": "approval",
            "task_id": status["task_id"],
            "execution_generation": status["execution_generation"],
            "run_id": status["run_id"],
            "session_id": "peer-group-session",
            "request_id": "req-peer-1",
            "approval": status["approval"],
        },
    )
    pending = service.status("room-1")["pending_actions"]
    assert pending == [
        {
            "kind": "approval",
            "task_id": "task-1",
            "execution_generation": 2,
            "run_id": "run-peer-1",
            "session_id": "peer-group-session",
            "request_id": "req-peer-1",
            "approval": {
                "description": "Run the focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
            "member_id": "member-peer",
        }
    ]

    assert service.approve_room_task(
        "room-1",
        member_id="member-peer",
        task_id="task-1",
        execution_generation=2,
        choice="once",
        request_id="req-peer-1",
    ) == {"resolved": 1}
    assert peer.approvals == [
        {
            "task_id": "task-1",
            "execution_generation": 2,
            "request_id": "req-peer-1",
            "choice": "once",
            "grant": "signed.room.grant",
        }
    ]
    assert service.status("room-1")["pending_actions"] == []


def test_local_room_approval_uses_the_exact_hidden_session(tmp_path: Path):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    hosted_rooms.create_room(
        service.db_path, room_id="room-1", name="Approval room",
        members=[{"member_id": "local", "profile": "default", "handle": "hermes"}],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    _start_approval_task(service, task_id="task-local-1", member_id="local", profile="default")
    service._set_pending_action(
        "room-1",
        "local",
        {
            "kind": "approval",
            "task_id": "task-local-1",
            "execution_generation": 1,
            "session_id": "local-session",
            "request_id": "approval-local-1",
            "approval": {
                "description": "Run focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
        },
    )

    assert service.approve_room_task(
        "room-1",
        member_id="local",
        task_id="task-local-1",
        execution_generation=1,
        choice="once",
        request_id="approval-local-1",
    ) == {"resolved": 1}
    assert rpc.approvals == [
        {
            "session_id": "local-session",
            "request_id": "approval-local-1",
            "choice": "once",
        }
    ]
    assert service.status("room-1")["pending_actions"] == []


def test_stale_local_approval_cannot_resolve_replacement_request(tmp_path: Path):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    hosted_rooms.create_room(
        service.db_path, room_id="room-1", name="Approval room",
        members=[{"member_id": "local", "profile": "default", "handle": "hermes"}],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    _start_approval_task(service, task_id="task-local-1", member_id="local", profile="default")
    action = {
        "kind": "approval",
        "task_id": "task-local-1",
        "execution_generation": 1,
        "session_id": "local-session",
        "approval": {"choices": ["once", "deny"]},
    }
    service._set_pending_action(
        "room-1", "local", {**action, "request_id": "approval-A"}
    )
    service._set_pending_action(
        "room-1", "local", {**action, "request_id": "approval-B"}
    )

    with pytest.raises(RuntimeError, match="no longer pending"):
        service.approve_room_task(
            "room-1",
            member_id="local",
            task_id="task-local-1",
            execution_generation=1,
            choice="once",
            request_id="approval-A",
        )

    assert rpc.approvals == []
    assert service.status("room-1")["pending_actions"][0]["request_id"] == (
        "approval-B"
    )
