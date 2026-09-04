"""Cross-process RoomLink route synchronization tests."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_link_records
from gateway import hosted_room_driver as driver
from gateway import hosted_room_links, hosted_rooms
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    HostedMemberDispatch,
    PROTOCOL_VERSION,
    catalog_mapping,
    issue_room_grant,
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
from tui_gateway.hosted_room_peer_transport import (
    PeerHostedRoomTransport,
    PeerMemberRoute,
)
from tui_gateway.hosted_room_service import HostedRoomService
from tui_gateway.hosted_room_peer_status import _RouteStatusPeerClient


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


class _FakePeerClient:
    def __init__(self) -> None:
        self.dispatches: list[dict] = []
        self.revoked: list[str] = []
        self.session = {"session_id": "peer-group-session"}

    def prepare(self, **kwargs):
        return (
            self.session
            if kwargs["create"] or kwargs.get("expected_session_id")
            else None
        )

    def dispatch(self, **kwargs):
        self.dispatches.append(kwargs["dispatch"])
        return {"status": "accepted", "task_id": kwargs["dispatch"]["task_id"]}

    def history(self, **_kwargs):
        if not self.dispatches:
            return []
        dispatch = self.dispatches[-1]
        return [
            {
                "role": "assistant",
                "task_id": dispatch["task_id"],
                "execution_generation": dispatch["execution_generation"],
                "status": "settled",
                "message_id": f"peer:{dispatch['task_id']}",
                "content": "Remote review complete.",
            }
        ]

    def status(self, **_kwargs):
        task_id = self.dispatches[-1]["task_id"] if self.dispatches else None
        return {"active": False, "task_id": task_id}

    def stop(self, **_kwargs):
        return {"status": "cancelled"}

    def revoke_grant(self, **kwargs):
        self.revoked.append(kwargs["grant"])
        return {"revoked": True}

    def revoke_grant_exact(self, **kwargs):
        return {"revoked": True}


class _RefreshingPeerClient(_FakePeerClient):
    def __init__(self, replacement: str) -> None:
        super().__init__()
        self.replacement = replacement
        self.exact_revoked = []

    def revoke_grant_exact(self, *, grant):
        self.exact_revoked.append(grant)
        return {"revoked": True}

    def refresh_grant(self, **_kwargs):
        return {"grant": self.replacement}


class _UnavailablePeerClient(_FakePeerClient):
    def prepare(self, **_kwargs):
        raise RuntimeError("peer is offline before admission")


class _FakeRPC:
    def __init__(self) -> None:
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
        on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}

    def history(self, **_kwargs):
        return []

    def info(self, **_kwargs):
        return {"active": False, "task_id": None}

    def interrupt(self, **_kwargs):
        return {"interrupted": True}


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def test_headless_room_publishes_peer_member_reply_without_desktop_transport(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    peer = _FakePeerClient()
    route = PeerMemberRoute(
        home_install_id="install-home",
        member_id="member-reviewer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest="a" * 64,
        execution_policy_digest="b" * 64,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed-room-grant",
    )
    service = HostedRoomService(_server(), db_path=db)
    # Process-local transport injection is a test seam, not a runtime API.
    service.peer_routes[("room-1", "member-reviewer")] = route
    service.peer_clients[("room-1", "member-reviewer")] = peer
    service.rpc = _FakeRPC()
    service.runtime.rpc = service.rpc
    service.local_profiles = lambda: ("default",)
    room = service.create_room(
        room_id="room-1",
        name="Review room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "local"},
            {
                "member_id": "member-reviewer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "peer-review",
                    "installation_id": "install-peer",
                    "profile": "reviewer",
                    "capability_digest": "a" * 64,
                },
            },
        ],
    )
    assert room["members"][1]["target"]["kind"] == "peer"

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-peer-1",
        payload={"text": "@reviewer inspect this", "thread_id": "thread-1"},
    )
    _wait_for(
        lambda: any(
            event["kind"] == "message.member" for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)

    events = service._events("room-1")
    reply = next(event for event in events if event["kind"] == "message.member")
    assert reply["payload"]["member_id"] == "member-reviewer"
    assert reply["payload"]["text"] == "Remote review complete."
    assert reply["actor"]["connection_id"] == "peer-review"
    assert peer.dispatches[0]["target_profile"] == "reviewer"


def test_unadmitted_peer_failure_does_not_block_next_healthy_member(tmp_path: Path):
    db = tmp_path / "state.db"
    route = PeerMemberRoute(
        home_install_id="install-home",
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest="a" * 64,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed-room-grant",
    )
    service = HostedRoomService(_server(), db_path=db)
    service.peer_routes[("room-1", "member-peer")] = route
    service.peer_clients[("room-1", "member-peer")] = _UnavailablePeerClient()
    service.rpc = _FakeRPC()
    service.runtime.rpc = service.rpc
    service.local_profiles = lambda: ("local",)
    service.create_room(
        room_id="room-1",
        name="Fallback room",
        members=[
            {
                "member_id": "member-peer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "peer-review",
                    "installation_id": "install-peer",
                    "profile": "reviewer",
                    "capability_digest": "a" * 64,
                },
            },
            {"member_id": "local", "profile": "local", "handle": "local"},
        ],
    )

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-fallback-1",
        payload={"text": "Review this together", "thread_id": "thread-1"},
    )
    _wait_for(
        lambda: any(
            event["kind"] == "message.member"
            and event["payload"]["member_id"] == "local"
            for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)

    events = service._events("room-1")
    assert any(
        event["kind"] == "turn.failed"
        and event["payload"]["member_id"] == "member-peer"
        for event in events
    )
    assert any(
        event["kind"] == "message.member" and event["payload"]["member_id"] == "local"
        for event in events
    )


def test_existing_room_database_adds_route_retirement_fence(tmp_path: Path):
    db = tmp_path / "state.db"
    hosted_rooms.list_rooms(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_disband_fences")
        conn.execute(
            """CREATE TABLE hosted_room_disband_fences (
                room_id TEXT PRIMARY KEY,
                authority_gateway_id TEXT NOT NULL,
                authority_epoch INTEGER NOT NULL,
                started_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_disband_fences(
                   room_id, authority_gateway_id, authority_epoch, started_at
               ) VALUES (?, ?, ?, ?)""",
            ("room-legacy-fence", "install:legacy", 1, 1.0),
        )

    fence = hosted_room_link_records.begin_room_link_retirement(
        db,
        room_id="room-before-create",
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
    )

    assert fence["room_id"] == "room-before-create"
    with sqlite3.connect(db) as conn:
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(hosted_room_disband_fences)")
        }
        assert "revocation_complete_at" in columns
        assert conn.execute(
            "SELECT 1 FROM hosted_room_disband_fences WHERE room_id=?",
            ("room-before-create",),
        ).fetchone() == (1,)
        assert conn.execute(
            """SELECT revocation_complete_at
                 FROM hosted_room_disband_fences WHERE room_id=?""",
            ("room-legacy-fence",),
        ).fetchone() == (None,)


def test_sqlite_fence_rejects_old_process_route_writes(tmp_path: Path):
    db = tmp_path / "state.db"
    hosted_rooms.list_rooms(db)
    values = (
        "room-1",
        "member-existing",
        "https://peer.example.test",
        "reviewer",
        "signed.room.grant",
        "{}",
        "cancel-room-1",
        "trace-room-1",
        "https",
        "ready",
        1.0,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            """INSERT INTO hosted_room_links(
                   room_id, member_id, target_url, target_profile, grant,
                   catalog_json, cancellation_scope_id, trace_id,
                   transport_security, status, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
    with sqlite3.connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="not revoked"):
            conn.execute(
                "DELETE FROM hosted_room_links WHERE room_id=?",
                ("room-1",),
            )
    hosted_room_link_records.begin_room_link_retirement(
        db,
        room_id="room-1",
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
    )

    with sqlite3.connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="not revoked"):
            conn.execute(
                "DELETE FROM hosted_room_links WHERE room_id=?",
                ("room-1",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="registration is fenced"):
            conn.execute(
                "UPDATE hosted_room_links SET grant=? WHERE room_id=?",
                ("replacement.grant", "room-1"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="registration is fenced"):
            conn.execute(
                """INSERT INTO hosted_room_links(
                       room_id, member_id, target_url, target_profile, grant,
                       catalog_json, cancellation_scope_id, trace_id,
                       transport_security, status, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("room-1", "member-new", *values[2:]),
            )
    hosted_room_link_records.complete_room_link_retirement(
        db,
        room_id="room-1",
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
    )
    assert hosted_room_link_records.delete_room_link_records(db, room_id="room-1") == 1


def test_retired_room_rejects_old_route_writer_after_fence_cleanup(tmp_path: Path):
    db = tmp_path / "state.db"
    gateway_id = hosted_rooms.local_authority_gateway_id()
    hosted_rooms.create_room(
        db,
        room_id="room-retired",
        name="Retired room",
        members=[],
        authority_gateway_id=gateway_id,
    )
    hosted_room_link_records.begin_room_link_retirement(
        db,
        room_id="room-retired",
        authority_gateway_id=gateway_id,
        authority_epoch=1,
    )
    hosted_rooms.disband_room(
        db,
        room_id="room-retired",
        expected_gateway_id=gateway_id,
        expected_epoch=1,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "DELETE FROM hosted_room_disband_fences WHERE room_id=?",
            ("room-retired",),
        )
        with pytest.raises(sqlite3.IntegrityError, match="registration is fenced"):
            conn.execute(
                """INSERT INTO hosted_room_links(
                       room_id, member_id, target_url, target_profile, grant,
                       catalog_json, cancellation_scope_id, trace_id,
                       transport_security, status, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "room-retired",
                    "member-old-worker",
                    "https://peer.example.test",
                    "reviewer",
                    "signed.room.grant",
                    "{}",
                    "cancel-room-retired",
                    "trace-room-retired",
                    "https",
                    "ready",
                    1.0,
                ),
            )


def test_worker_hydrates_and_revokes_route_registered_by_another_process(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    worker = HostedRoomService(_server(), db_path=db)
    revoker = HostedRoomService(_server(), db_path=db)
    writer = HostedRoomService(_server(), db_path=db)
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed.room.grant",
    )
    writer.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=_FakePeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    worker.create_room(
        room_id="room-1",
        name="Cross-process route",
        members=[
            {"member_id": "default", "profile": "default", "handle": "default"},
            {
                "member_id": "member-peer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "install-peer",
                    "installation_id": "install-peer",
                    "profile": "reviewer",
                    "capability_digest": catalog.catalog_digest,
                },
            },
        ],
    )
    transport = worker._resolve_member_transport(
        worker.bindings()[0],
        {
            "identity": driver.TaskIdentity("room-1", "task-1", "thread-1", "turn-1"),
            "status": "queued",
            "execution_generation": 0,
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 1,
                "prompt": "Review this.",
            },
        },
    )

    assert isinstance(transport, PeerHostedRoomTransport)
    assert isinstance(
        worker.peer_clients[("room-1", "member-peer")], PeerRunsHTTPClient
    )
    replacement_route = replace(route, grant="replacement.room.grant")
    writer.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=replacement_route,
        client=_FakePeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    worker._resolve_member_transport(
        worker.bindings()[0],
        {
            "identity": driver.TaskIdentity("room-1", "task-2", "thread-1", "turn-2"),
            "status": "queued",
            "execution_generation": 0,
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 2,
                "prompt": "Review the update.",
            },
        },
    )
    assert worker.peer_routes[("room-1", "member-peer")].grant == (
        "replacement.room.grant"
    )
    replacement_identity = replace(
        replacement_route,
        cancellation_scope_id="cancel-room-2",
        trace_id="trace-room-2",
    )
    writer.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=replacement_identity,
        client=_FakePeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    worker._resolve_member_transport(
        worker.bindings()[0],
        {
            "identity": driver.TaskIdentity(
                "room-1", "task-identity", "thread-1", "turn-identity"
            ),
            "status": "queued",
            "execution_generation": 0,
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 3,
                "prompt": "Use the current route identity.",
            },
        },
    )
    assert worker.peer_routes[("room-1", "member-peer")].cancellation_scope_id == (
        "cancel-room-2"
    )
    assert worker.peer_routes[("room-1", "member-peer")].trace_id == "trace-room-2"
    monkeypatch.setattr(
        PeerRunsHTTPClient,
        "revoke_grant",
        lambda _self, *, grant: {"revoked": bool(grant)},
    )
    assert revoker.revoke_room_routes("room-1") == 1
    assert hosted_room_links.load_room_links(db) == ()
    with pytest.raises(RuntimeError, match="peer room route is unavailable"):
        worker._resolve_member_transport(
            worker.bindings()[0],
            {
                "identity": driver.TaskIdentity(
                    "room-1", "task-4", "thread-1", "turn-4"
                ),
                "status": "queued",
                "execution_generation": 0,
                "payload": {
                    "target_member_id": "member-peer",
                    "target_profile": "reviewer",
                    "source_event_seq": 4,
                    "prompt": "This must not dispatch.",
                },
            },
        )
    with pytest.raises(
        hosted_rooms.HostedRoomError, match="route registration is fenced"
    ):
        writer.register_peer_route(
            room_id="room-1",
            member_id="member-peer",
            route=replacement_route,
            client=_FakePeerClient(),
            target_url="https://peer.example.test",
            catalog=catalog,
        )


def test_failed_remote_retirement_keeps_fence_and_retries(
    tmp_path: Path,
    monkeypatch,
):
    db = tmp_path / "state.db"
    writer = HostedRoomService(_server(), db_path=db)
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed.room.grant",
    )
    writer.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=_FakePeerClient(),
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    revoker = HostedRoomService(_server(), db_path=db)
    attempts = []

    def flaky_revoke(_self, *, grant):
        attempts.append(grant)
        if len(attempts) == 1:
            raise RuntimeError("peer is temporarily unavailable")
        return {"revoked": True}

    monkeypatch.setattr(PeerRunsHTTPClient, "revoke_grant", flaky_revoke)
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        revoker.revoke_room_routes("room-1")
    assert len(hosted_room_links.load_room_links(db)) == 1
    with pytest.raises(hosted_rooms.HostedRoomError, match="registration is fenced"):
        writer.register_peer_route(
            room_id="room-1",
            member_id="member-peer",
            route=route,
            client=_FakePeerClient(),
            target_url="https://peer.example.test",
            catalog=catalog,
        )

    assert revoker.revoke_room_routes("room-1") == 1
    assert hosted_room_links.load_room_links(db) == ()
    assert attempts == ["signed.room.grant", "signed.room.grant"]


def test_runtime_route_registration_requires_persistence_identity(tmp_path: Path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id="cancel-room-ephemeral",
        trace_id="trace-room-ephemeral",
        grant="signed.room.grant",
    )

    with pytest.raises(ValueError, match="persistence identity is required"):
        service.register_peer_route(
            room_id="room-ephemeral",
            member_id="member-peer",
            route=route,
            client=_FakePeerClient(),
        )
    assert service.peer_routes == {}


def test_unpublished_refreshed_grant_is_revoked_before_dispatch():
    now = time.time()
    secret = b"s" * 32
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    grant_kwargs = {
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "install-home",
        "authority_epoch": 1,
        "member_id": "member-peer",
        "target_install_id": "install-peer",
        "target_profile": "reviewer",
        "execution_policy_digest": catalog.execution_policy.policy_digest,
        "ttl_seconds": 3600,
        "status_expires_at": now + 10_000,
    }
    old_grant = issue_room_grant(
        secret,
        grant_id="grant-old",
        issued_at=now - 3700,
        **grant_kwargs,
    )
    new_grant = issue_room_grant(
        secret,
        grant_id="grant-new",
        issued_at=now,
        **grant_kwargs,
    )
    prompt = "Review this"
    dispatch = HostedMemberDispatch(
        protocol_version=PROTOCOL_VERSION,
        room_id="room-1",
        home_install_id="install-home",
        authority_gateway_id="install-home",
        authority_epoch=1,
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        task_id="task-1",
        execution_generation=1,
        source_event_seq=1,
        cancellation_scope_id="cancel-room-1",
        prompt=prompt,
        prompt_digest=hashlib.sha256(prompt.encode()).hexdigest(),
        capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        trace_id="trace-room-1",
    )
    peer = _RefreshingPeerClient(new_grant)
    route_status = []
    tracked = _RouteStatusPeerClient(
        peer,
        on_ready=lambda: None,
        on_reauthorization=lambda: route_status.append("needs_reauthorization"),
        on_unavailable=lambda: None,
        on_refreshed=lambda *_args: (_ for _ in ()).throw(
            hosted_rooms.HostedRoomError("route registration is fenced")
        ),
    )

    with pytest.raises(hosted_rooms.HostedRoomError, match="fenced"):
        tracked.dispatch(dispatch=dispatch.as_mapping(), grant=old_grant)

    assert peer.exact_revoked == [new_grant]
    assert peer.revoked == []
    assert peer.dispatches == []
    assert route_status == ["needs_reauthorization"]
