"""Integration tests for the hosted Discussion coordinator."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import MAX_ACTIVE_POLICY_EVENTS
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    HostedMemberDispatch,
    PROTOCOL_VERSION,
    catalog_mapping,
    issue_room_grant,
)
from gateway.hosted_room_artifacts import (
    RoomArtifactOutbox,
    RoomArtifactScope,
    terminal_artifact_manifest,
)
from tui_gateway.hosted_room_service import (
    HostedRoomService,
    _grant_revoke_is_terminal,
)
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


from tests.tui_gateway.hosted_room_service_fixtures import (
    _FakePeerClient,
    _append_room_event,
    _server,
    _wait_for,
)
from tests.tui_gateway.test_hosted_room_attachment_service import _FakeRPC


class _ArtifactRPC(_FakeRPC):
    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self.db_path = db_path
        self.scope = None
        self.profile = "default"
        self.thread_id = ""
        self.shared_threads: set[str] = set()
        self.output = db_path.parent / "handoff.md"
        self.output.write_text("# Review handoff\n", encoding="utf-8")

    def bind_artifact_scope(self, **kwargs):
        installation = hosted_rooms.local_authority_gateway_id()
        self.scope = RoomArtifactScope.from_mapping({
            "room_id": kwargs["task"].room_id,
            "task_id": kwargs["task"].task_id,
            "execution_generation": kwargs["execution_generation"],
            "member_id": kwargs["member_id"],
            "target_profile": kwargs["profile"],
            "home_install_id": installation,
            "target_install_id": installation,
            "authority_gateway_id": kwargs["authority_gateway_id"],
            "authority_epoch": kwargs["authority_epoch"],
        })
        self.profile = kwargs["profile"]
        self.thread_id = kwargs["task"].thread_id

    def submit(self, **kwargs):
        assert self.scope is not None
        if self.thread_id in self.shared_threads:
            kwargs["on_terminal"]({"status": "settled", "text": "(pass)"})
            return {"accepted": True}
        self.shared_threads.add(self.thread_id)
        outbox_db = (
            self.db_path
            if self.profile == "default"
            else self.db_path.parent / "profiles" / self.profile / "state.db"
        )
        outbox_db.parent.mkdir(parents=True, exist_ok=True)
        RoomArtifactOutbox(outbox_db).put_path(
            scope=self.scope,
            path=self.output,
        )
        kwargs["on_terminal"]({
            "status": "settled",
            "text": "Please review the attached handoff.",
            "artifacts": terminal_artifact_manifest(outbox_db, self.scope),
        })
        return {"accepted": True}


class _CorruptManifestArtifactRPC(_ArtifactRPC):
    def submit(self, **kwargs):
        on_terminal = kwargs["on_terminal"]

        def corrupt_manifest(result):
            manifest = result.get("artifacts")
            if isinstance(manifest, dict) and manifest.get("items"):
                manifest = dict(manifest)
                manifest["items"] = [dict(item) for item in manifest["items"]]
                manifest["items"][0]["sha256"] = "0" * 64
                manifest["manifest_digest"] = hashlib.sha256(
                    json.dumps(
                        manifest["items"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                result = {**result, "artifacts": manifest}
            on_terminal(result)

        return super().submit(**{**kwargs, "on_terminal": corrupt_manifest})


class _ArtifactPeerClient(_FakePeerClient):
    def __init__(self) -> None:
        super().__init__()
        self.data = b"# Remote handoff\n"
        self.item = {
            "artifact_id": "rart_66666666666666666666666666666666",
            "kind": "file",
            "name": "remote.md",
            "size": len(self.data),
            "mime": "text/markdown",
            "sha256": __import__("hashlib").sha256(self.data).hexdigest(),
        }
        import json

        self.manifest = {
            "version": 1,
            "manifest_digest": __import__("hashlib").sha256(
                json.dumps([self.item], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "items": [self.item],
        }
        self.acks = []
        self.discards = []

    def history(self, **kwargs):
        if not self.dispatches:
            return []
        dispatch = self.dispatches[-1]
        return [{
            "role": "assistant",
            "task_id": dispatch["task_id"],
            "execution_generation": dispatch["execution_generation"],
            "status": "settled",
            "message_id": f"peer:{dispatch['task_id']}",
            "content": "Remote draft attached.",
            "artifacts": self.manifest,
            "run_id": "run-remote-1",
        }]

    def read_artifact(self, **kwargs):
        assert kwargs["run_id"] == "run-remote-1"
        assert kwargs["artifact_id"] == self.item["artifact_id"]
        return self.data

    def acknowledge_artifacts(self, **kwargs):
        self.acks.append(dict(kwargs))
        return {"acknowledged": True}

    def discard_artifacts(self, **kwargs):
        self.discards.append(dict(kwargs))
        return {"discarded": True, "removed": 1}


class _LostAckArtifactPeerClient(_ArtifactPeerClient):
    def acknowledge_artifacts(self, **kwargs):
        self.acks.append(dict(kwargs))
        if len(self.acks) == 1:
            raise PeerRunsHTTPError(
                "acknowledgement response was lost",
                retryable=True,
                ambiguous=True,
            )
        return {"acknowledged": True, "changed": 0}


class _ReauthAckArtifactPeerClient(_ArtifactPeerClient):
    def acknowledge_artifacts(self, **kwargs):
        self.acks.append(dict(kwargs))
        raise PeerRunsHTTPError(
            "room grant expired",
            status_code=401,
            error_code="invalid_room_grant",
        )


class _CorruptArtifactPeerClient(_ArtifactPeerClient):
    def read_artifact(self, **kwargs):
        super().read_artifact(**kwargs)
        return b"corrupt peer bytes"


class _InvalidDiscardReceiptPeerClient(_ArtifactPeerClient):
    def discard_artifacts(self, **kwargs):
        self.discards.append(dict(kwargs))
        return {}


def test_bot_file_is_imported_before_member_message_and_reaches_next_bot(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _ArtifactRPC(db)
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("viewer", "ops")
    service.create_room(
        room_id="room-1",
        name="Handoff room",
        members=[
            {"member_id": "viewer", "profile": "viewer", "handle": "viewer"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-handoff-1",
        payload={"text": "Prepare the handoff", "thread_id": "thread-1"},
    )
    _wait_for(
        lambda: any(
            event["kind"] == "message.member"
            and event["payload"].get("attachments")
            for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)

    message = next(
        event
        for event in service._events("room-1")
        if event["kind"] == "message.member" and event["payload"].get("attachments")
    )
    attachment = message["payload"]["attachments"][0]
    assert attachment["name"] == "handoff.md"
    stored = service.read_attachment(
        room_id="room-1",
        attachment_id=attachment["attachment_id"],
        recipient_member_id="ops",
    )
    assert stored.data == b"# Review handoff\n"
    assert service.read_attachment(
        room_id="room-1",
        attachment_id=attachment["attachment_id"],
        recipient_member_id="viewer",
    ).data == b"# Review handoff\n"
    assert service.read_attachment(
        room_id="room-1",
        attachment_id=attachment["attachment_id"],
        recipient_member_id=None,
        viewer=True,
    ).data == b"# Review handoff\n"
    assert RoomArtifactOutbox(db).list(rpc.scope) == []


def test_local_verification_failure_discards_source_before_failed_publication(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _CorruptManifestArtifactRPC(db)
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("reviewer", "ops")
    service.create_room(
        room_id="room-1",
        name="Verification room",
        members=[
            {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-corrupt-file",
        payload={"text": "@reviewer prepare the handoff", "thread_id": "thread-1"},
    )
    _wait_for(
        lambda: any(
            event["kind"] == "turn.failed"
            and event["payload"].get("error")
            == "A Group Chat file could not be verified."
            for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)
    assert rpc.scope is not None
    profile_db = db.parent / "profiles" / "reviewer" / "state.db"
    assert RoomArtifactOutbox(profile_db).list(rpc.scope) == []
    assert not any(
        event["kind"] == "message.member"
        and event["payload"].get("attachments")
        for event in service._events("room-1")
    )


def test_upgrade_replays_legacy_terminal_task_then_admits_new_work(tmp_path: Path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id="legacy-user",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
        payload={"text": "@ops inspect", "thread_id": "legacy-thread"},
    )
    room = hosted_rooms.room_state(db, room_id="room-1")
    events = service._events("room-1")
    planned = discussion.plan_next_task(
        room,
        events,
        local_profiles=("default", "ops"),
    ).task
    assert planned is not None
    legacy_payload = dict(planned.payload)
    legacy_payload.pop("recipient_member_ids")
    admitted = driver.admit_task(
        db,
        planned.identity,
        payload=legacy_payload,
        clock=time.time,
    )
    binding = service.bindings()[0]
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch,
        process_generation="pre-upgrade",
        ttl_seconds=30,
        clock=time.time,
    )
    attempt = driver.start_task(
        db,
        admitted["identity"],
        lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    driver.settle_task(
        db,
        attempt,
        settlement_id="legacy-reply",
        status="settled",
        result={"text": "legacy done"},
        clock=time.time,
    )

    service.prepare_room(binding)
    assert any(
        row["kind"] == "message.member"
        and row["payload"].get("text") == "legacy done"
        for row in service._events("room-1")
    )
    hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id="post-upgrade-user",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
        payload={"text": "@ops continue", "thread_id": "post-upgrade-thread"},
    )
    service.prepare_room(binding)

    queued = driver.list_tasks(db, room_id="room-1", status="queued")
    assert len(queued) == 1
    assert queued[0]["payload"]["recipient_member_ids"] == ["default", "ops"]


def test_upgrade_fails_legacy_file_output_closed_then_continues(tmp_path: Path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _ArtifactRPC(db)
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Handoff room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id="legacy-file-user",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
        payload={"text": "Prepare the file", "thread_id": "legacy-thread"},
    )
    room = hosted_rooms.room_state(db, room_id="room-1")
    planned = discussion.plan_next_task(
        room,
        service._events("room-1"),
        local_profiles=("default", "ops"),
    ).task
    assert planned is not None
    legacy_payload = dict(planned.payload)
    legacy_payload.pop("recipient_member_ids")
    driver.admit_task(
        db,
        planned.identity,
        payload=legacy_payload,
        clock=time.time,
    )

    service.start()
    _wait_for(
        lambda: any(
            row["kind"] == "turn.failed"
            and row["payload"].get("task_id") == planned.identity.task_id
            for row in service._events("room-1")
        )
    )
    assert not any(
        row["kind"] == "message.member"
        and row["payload"].get("attachments")
        and row["payload"].get("thread_id") == "legacy-thread"
        for row in service._events("room-1")
    )
    assert RoomArtifactOutbox(db).list(rpc.scope) == []
    _wait_for(
        lambda: any(
            row["kind"] == "room.activity"
            and row["payload"].get("thread_id") == "legacy-thread"
            for row in service._events("room-1")
        )
    )

    service.send(
        room_id="room-1",
        event_id="post-upgrade-file-user",
        payload={"text": "Prepare another file", "thread_id": "new-thread"},
    )
    _wait_for(
        lambda: any(
            row["kind"] == "message.member"
            and row["payload"].get("attachments")
            and row["payload"].get("thread_id") == "new-thread"
            for row in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)


def test_upgrade_retires_legacy_peer_file_before_failing_closed(tmp_path: Path):
    db = tmp_path / "state.db"
    peer = _ReauthAckArtifactPeerClient()
    local_id = hosted_rooms.local_authority_gateway_id()
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        installation_id="install-peer",
        persistent_process=True,
        attachments=True,
        target_profile="reviewer",
    ))
    grant = issue_room_grant(
        b"source-route-fixture-secret" * 2,
        grant_id="legacy-peer-repair",
        room_id="room-1",
        home_install_id=local_id,
        authority_gateway_id=local_id,
        authority_epoch=1,
        member_id="member-reviewer",
        target_install_id="install-peer",
        target_profile="reviewer",
        execution_policy_digest=catalog.execution_policy.policy_digest,
        ttl_seconds=3600,
    )
    route = PeerMemberRoute(
        home_install_id=local_id,
        member_id="member-reviewer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=catalog.catalog_digest,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant=grant,
        attachments=True,
    )
    service = HostedRoomService(
        _server(),
        db_path=db,
        peer_routes={("room-1", "member-reviewer"): route},
        peer_clients={"install-peer": peer},
    )
    service.local_profiles = lambda: ("ops",)
    service.create_room(
        room_id="room-1",
        name="Cross-gateway handoff",
        members=[
            {
                "member_id": "member-reviewer",
                "profile": "reviewer",
                "handle": "reviewer",
                "target": {
                    "kind": "peer",
                    "peer_id": "peer-review",
                    "installation_id": "install-peer",
                    "profile": "reviewer",
                    "capability_digest": catalog.catalog_digest,
                },
            },
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id="legacy-peer-user",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
        payload={"text": "@reviewer prepare the file", "thread_id": "thread-1"},
    )
    room = hosted_rooms.room_state(db, room_id="room-1")
    planned = discussion.plan_next_task(
        room,
        service._events("room-1"),
        local_profiles=("ops",),
    ).task
    assert planned is not None
    legacy_payload = dict(planned.payload)
    legacy_payload.pop("recipient_member_ids")
    driver.admit_task(db, planned.identity, payload=legacy_payload, clock=time.time)

    service.start()
    _wait_for(lambda: bool(peer.acks))
    _wait_for(
        lambda: service.status("room-1")["peer_routes"][0]["status"]
        == "needs_reauthorization"
    )
    assert not any(
        row["kind"] == "turn.failed"
        and row["payload"].get("task_id") == planned.identity.task_id
        for row in service._events("room-1")
    )
    assert service._peer_route_status[("room-1", "member-reviewer")] == (
        "needs_reauthorization"
    )
    with service._artifact_retry_connection() as conn:
        assert conn.execute(
            "SELECT blocked FROM hosted_room_artifact_retries"
        ).fetchone()["blocked"] == 1

    replacement = _ArtifactPeerClient()
    service.register_peer_route(
        room_id="room-1",
        member_id="member-reviewer",
        route=route,
        client=replacement,
        target_url="https://peer.example.test",
        catalog=catalog,
    )
    _wait_for(lambda: bool(replacement.acks))

    def artifact_retry_count():
        with service._artifact_retry_connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM hosted_room_artifact_retries"
            ).fetchone()[0]

    _wait_for(lambda: artifact_retry_count() == 0)
    _wait_for(
        lambda: any(
            row["kind"] == "turn.failed"
            and row["payload"].get("task_id") == planned.identity.task_id
            for row in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)
    assert replacement.acks[0]["artifact_ids"] == (peer.item["artifact_id"],)
    assert replacement.discards == [
        {"run_id": "run-remote-1", "grant": grant}
    ]
    assert not any(
        row["kind"] == "message.member" and row["payload"].get("attachments")
        for row in service._events("room-1")
    )


def test_peer_bot_file_is_canonicalized_before_next_local_bot_receives_it(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    peer = _ArtifactPeerClient()
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-reviewer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest="a" * 64,
        execution_policy_digest="b" * 64,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed-room-grant",
        attachments=True,
    )
    service = HostedRoomService(
        _server(),
        db_path=db,
        peer_routes={("room-1", "member-reviewer"): route},
        peer_clients={"install-peer": peer},
    )
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("ops",)
    service.create_room(
        room_id="room-1",
        name="Cross-gateway handoff",
        members=[
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
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-cross-file",
        payload={"text": "Prepare and review a handoff", "thread_id": "thread-1"},
    )
    _wait_for(lambda: bool(peer.acks))
    _wait_for(
        lambda: any(
            method == "stage_attachment" and params["profile"] == "ops"
            for method, params in rpc.calls
        )
    )
    assert service.stop(timeout=1.0)

    reply = next(
        event
        for event in service._events("room-1")
        if event["kind"] == "message.member"
        and event["payload"]["member_id"] == "member-reviewer"
    )
    attachment = reply["payload"]["attachments"][0]
    assert attachment["name"] == "remote.md"
    assert peer.acks[0]["message_event_id"] == reply["event_id"]
    staged = next(
        params
        for method, params in rpc.calls
        if method == "stage_attachment" and params["profile"] == "ops"
    )
    assert staged["data"] == peer.data


def test_peer_verification_failure_discards_source_before_failed_publication(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    peer = _CorruptArtifactPeerClient()
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-reviewer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest="a" * 64,
        execution_policy_digest="b" * 64,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed-room-grant",
        attachments=True,
    )
    service = HostedRoomService(
        _server(),
        db_path=db,
        peer_routes={("room-1", "member-reviewer"): route},
        peer_clients={"install-peer": peer},
    )
    service.local_profiles = lambda: ("ops",)
    service.create_room(
        room_id="room-1",
        name="Peer verification room",
        members=[
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
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-corrupt-peer-file",
        payload={"text": "@reviewer prepare the handoff", "thread_id": "thread-1"},
    )
    _wait_for(lambda: bool(peer.discards))
    _wait_for(
        lambda: any(
            event["kind"] == "turn.failed"
            and event["payload"].get("error")
            == "A Group Chat file could not be verified."
            for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)
    assert peer.discards == [
        {"run_id": "run-remote-1", "grant": "signed-room-grant"}
    ]
    assert peer.acks == []
    assert not any(
        event["kind"] == "message.member"
        and event["payload"].get("attachments")
        for event in service._events("room-1")
    )


def test_peer_retirement_requires_typed_success_receipt(tmp_path: Path):
    peer = _InvalidDiscardReceiptPeerClient()
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-reviewer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest="a" * 64,
        execution_policy_digest="b" * 64,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed-room-grant",
        attachments=True,
    )
    service = HostedRoomService(
        _server(),
        db_path=tmp_path / "state.db",
        peer_routes={("room-1", "member-reviewer"): route},
        peer_clients={"install-peer": peer},
    )
    plan = SimpleNamespace(
        identity=SimpleNamespace(task_id="dtask:one"),
        member=SimpleNamespace(
            member_id="member-reviewer",
            profile="reviewer",
            target={"kind": "peer"},
        ),
    )

    with pytest.raises(RuntimeError, match="retirement needs a repaired route"):
        service._retire_failed_terminal_artifacts(
            room={
                "room_id": "room-1",
                "authority_gateway_id": hosted_rooms.local_authority_gateway_id(),
                "authority_epoch": 1,
            },
            task={
                "execution_generation": 1,
                "result": {"run_id": "run-remote-1", "artifacts": {"version": 1}},
            },
            plan=plan,
        )
    assert service.status("room-1")["peer_routes"][0]["status"] == (
        "needs_reauthorization"
    )


def test_lost_peer_artifact_ack_retries_without_duplicate_member_message(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    now = [0.0]
    peer = _LostAckArtifactPeerClient()
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-reviewer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest="a" * 64,
        execution_policy_digest="b" * 64,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed-room-grant",
        attachments=True,
    )
    service = HostedRoomService(
        _server(),
        db_path=db,
        peer_routes={("room-1", "member-reviewer"): route},
        peer_clients={"install-peer": peer},
        artifact_clock=lambda: now[0],
        artifact_retry_min_seconds=10,
        artifact_retry_max_seconds=40,
    )
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("ops",)
    service.create_room(
        room_id="room-1",
        name="Lost ACK room",
        members=[
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
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.start()
    service.send(
        room_id="room-1",
        event_id="user-lost-ack",
        payload={"text": "Share the result", "thread_id": "thread-1"},
    )
    _wait_for(lambda: len(peer.acks) == 1)
    assert service.stop(timeout=1.0)
    with service._artifact_retry_connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_artifact_retries"
        ).fetchone()[0] == 1
    resumed = HostedRoomService(
        _server(),
        db_path=db,
        peer_routes={("room-1", "member-reviewer"): route},
        peer_clients={"install-peer": peer},
        artifact_clock=lambda: now[0],
        artifact_retry_min_seconds=10,
        artifact_retry_max_seconds=40,
    )
    resumed.rpc = rpc
    resumed.runtime.rpc = rpc
    resumed.local_profiles = lambda: ("ops",)
    binding = resumed.bindings()[0]
    for _ in range(10):
        resumed.prepare_room(binding)
    assert len(peer.acks) == 1
    now[0] = 10.0
    resumed.prepare_room(binding)
    _wait_for(lambda: len(peer.acks) == 2)
    events = resumed._events("room-1")
    assert sum(
        event["kind"] == "message.member"
        and event["payload"].get("member_id") == "member-reviewer"
        for event in events
    ) == 1
    assert sum(
        event["kind"] == "turn.settled"
        and event["payload"].get("member_id") == "member-reviewer"
        for event in events
    ) == 1
    with resumed._artifact_retry_connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_artifact_retries"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_artifact_completions"
        ).fetchone()[0] == 1
    assert peer.dispatches[0]["target_profile"] == "reviewer"


def test_artifact_retry_store_migrates_and_backfills_member_id(tmp_path: Path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Migration room",
        members=[
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
            {"member_id": "qa", "profile": "qa", "handle": "qa"},
        ],
        authority_gateway_id="gateway-a",
    )
    identity = driver.TaskIdentity("room-1", "task-old", "thread-1", "turn-1")
    driver.admit_task(
        db,
        identity,
        payload={
            "target_member_id": "ops",
            "target_profile": "ops",
            "prompt": "Review",
            "source_event_seq": 1,
        },
        clock=time.time,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE hosted_room_artifact_retries (
                room_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                execution_generation INTEGER NOT NULL,
                attempts INTEGER NOT NULL,
                next_attempt_at REAL NOT NULL,
                blocked INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY(room_id, task_id, execution_generation)
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_artifact_retries
               VALUES ('room-1', 'task-old', 1, 1, 10, 1, 1)"""
        )

    service = HostedRoomService(_server(), db_path=db)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """SELECT member_id FROM hosted_room_artifact_retries
               WHERE room_id='room-1' AND task_id='task-old'"""
        ).fetchone()
    assert row == ("ops",)

    with sqlite3.connect(db) as conn:
        conn.execute(
            "DELETE FROM hosted_room_driver_tasks WHERE task_id='task-old'"
        )
    service._prune_artifact_metadata()
    assert service._artifact_retry_keys("room-1") == set()


def test_authoritative_stop_discards_crash_stranded_local_output(tmp_path: Path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "ops")
    room = service.create_room(
        room_id="room-1",
        name="Crash room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "default"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-crash",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
        payload={"text": "Prepare a file", "thread_id": "thread-1"},
    )
    binding = service.bindings()[0]
    service.prepare_room(binding)
    task = driver.list_tasks(db, room_id="room-1", status="queued")[0]
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET status='indeterminate', execution_generation=1
               WHERE room_id='room-1' AND task_id=?""",
            (task["identity"].task_id,),
        )
    scope = RoomArtifactScope.from_mapping({
        "room_id": "room-1",
        "task_id": task["identity"].task_id,
        "execution_generation": 1,
        "member_id": task["payload"]["target_member_id"],
        "target_profile": task["payload"]["target_profile"],
        "home_install_id": hosted_rooms.local_authority_gateway_id(),
        "target_install_id": hosted_rooms.local_authority_gateway_id(),
        "authority_gateway_id": room["authority_gateway_id"],
        "authority_epoch": room["authority_epoch"],
    })
    output = tmp_path / "crash.md"
    output.write_text("stranded\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(db)
    outbox.put_path(scope=scope, path=output)

    assert service.stop_room("room-1", cancel_id="stop-crash") == 1
    stopping = driver.get_task(db, task["identity"])
    driver.complete_task_cancel(
        db,
        task["identity"],
        cancel_id="stop-crash",
        expected_cancel_generation=stopping["cancel_generation"],
        clock=time.time,
    )
    service.prepare_room(binding)
    assert outbox.list(scope) == []


def test_room_retirement_deletes_terminal_local_artifacts_and_retry_metadata(tmp_path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-retire",
        name="Retire local files",
        members=[
            {"member_id": "default", "profile": "default", "handle": "default"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    hosted_rooms.append_event(
        db,
        room_id="room-retire",
        event_id="retire-user",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
        payload={"text": "Prepare a file", "thread_id": "thread-retire"},
    )
    service.prepare_room(service.bindings()[0])
    task = driver.list_tasks(db, room_id="room-retire", status="queued")[0]
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_driver_tasks
                  SET status='settled', execution_generation=1,
                      settlement_id='settled-retire', settlement_status='settled',
                      result_json='{}', terminal_at=1
                WHERE room_id='room-retire' AND task_id=?""",
            (task["identity"].task_id,),
        )
        conn.commit()
    terminal = driver.get_task(db, task["identity"])
    room = hosted_rooms.room_state(db, room_id="room-retire")
    scope = RoomArtifactScope.from_mapping({
        "room_id": "room-retire",
        "task_id": task["identity"].task_id,
        "execution_generation": 1,
        "member_id": "default",
        "target_profile": "default",
        "home_install_id": hosted_rooms.local_authority_gateway_id(),
        "target_install_id": hosted_rooms.local_authority_gateway_id(),
        "authority_gateway_id": room["authority_gateway_id"],
        "authority_epoch": room["authority_epoch"],
    })
    outbox = RoomArtifactOutbox(db)
    outbox.put_bytes(scope=scope, data=b"private result", source_name="result.md")
    service._defer_artifact_retry(terminal, RuntimeError("retry later"))

    assert outbox.list(scope)
    assert service._artifact_retry_keys("room-retire")
    service.retire_room_artifacts("room-retire")
    assert outbox.list(scope) == []
    assert service._artifact_retry_keys("room-retire") == set()
    first = service.retire_and_disband_room(
        "room-retire",
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]),
    )
    repeated = service.retire_and_disband_room(
        "room-retire",
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]),
    )
    assert repeated["disbanded_at"] == first["disbanded_at"]
