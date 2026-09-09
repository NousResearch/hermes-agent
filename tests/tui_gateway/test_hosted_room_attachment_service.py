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
from tui_gateway.hosted_room_service import (
    HostedRoomService,
    _RouteStatusPeerClient,
    _grant_revoke_is_terminal,
)
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


from tests.tui_gateway.hosted_room_service_fixtures import (
    _append_room_event,
    _server,
    _wait_for,
)


class _FakeRPC:
    def __init__(self) -> None:
        self.sessions = {}
        self.approvals = []
        self.calls = []
        self.fail_attachment_profiles = set()

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(
        self,
        *,
        profile,
        session_id,
        prompt,
        source,
        task,
        execution_generation,
        on_terminal,
    ):
        self.calls.append(("submit", {"profile": profile, "prompt": prompt}))
        on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}

    def stage_attachment(
        self,
        *,
        profile,
        session_id,
        source,
        attachment,
        data,
        execution_generation,
    ):
        self.calls.append((
            "stage_attachment",
            {
                "profile": profile,
                "attachment": dict(attachment),
                "data": data,
                "execution_generation": execution_generation,
            },
        ))
        if profile in self.fail_attachment_profiles:
            raise RuntimeError("attachment staging unavailable")
        return {
            "attached": True,
            **(
                {"ref_text": f"@file:attachments/{attachment['name']}"}
                if attachment["kind"] == "file"
                else {}
            ),
        }

    def begin_attachment_staging(
        self, *, profile, session_id, source, execution_generation
    ):
        self.calls.append((
            "begin_attachment_staging",
            {"profile": profile, "execution_generation": execution_generation},
        ))

    def commit_attachment_staging(
        self, *, profile, session_id, source, execution_generation
    ):
        self.calls.append((
            "commit_attachment_staging",
            {"profile": profile, "execution_generation": execution_generation},
        ))

    def rollback_attachment_staging(
        self, *, profile, session_id, source, execution_generation
    ):
        self.calls.append((
            "rollback_attachment_staging",
            {"profile": profile, "execution_generation": execution_generation},
        ))

    def history(self, *, profile, session_id, source):
        return []

    def info(self, *, profile, session_id, source):
        return {"active": False, "task_id": None}

    def interrupt(self, *, profile, session_id, source, expected_task_id):
        return {"interrupted": True}

    def approve(self, **kwargs):
        self.approvals.append(dict(kwargs))
        return {"resolved": 1}


class _FakePeerClient:
    def __init__(self) -> None:
        self.dispatches = []
        self.staged = []
        self.revoked = []
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

    def stage_attachments(self, **kwargs):
        self.staged.append(dict(kwargs))
        return {"complete": True, "count": len(kwargs["attachments"])}

    def history(self, **kwargs):
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

    def status(self, **kwargs):
        task_id = self.dispatches[-1]["task_id"] if self.dispatches else None
        return {"active": False, "task_id": task_id}

    def stop(self, **kwargs):
        return {"status": "cancelled"}

    def revoke_grant(self, **kwargs):
        self.revoked.append(kwargs["grant"])
        return {"revoked": True}


class _RecoveringPeerClient(_FakePeerClient):
    def __init__(self) -> None:
        super().__init__()
        self.recoveries = []

    def recover_dispatch(self, **kwargs):
        dispatch = dict(kwargs["dispatch"])
        self.recoveries.append({**kwargs, "dispatch": dispatch})
        self.dispatches.append(dispatch)
        return {
            "status": "accepted",
            "task_id": dispatch["task_id"],
            "execution_generation": dispatch["execution_generation"],
            "run_id": "run-recovered",
        }


class _UnavailableAttachmentPeerClient(_FakePeerClient):
    def stage_attachments(self, **kwargs):
        raise PeerRunsHTTPError(
            "peer is offline before attachment admission",
            retryable=True,
            not_admitted=True,
        )


class _AmbiguousAttachmentPeerClient(_FakePeerClient):
    def stage_attachments(self, **kwargs):
        raise PeerRunsHTTPError(
            "attachment upload response was lost",
            retryable=True,
            ambiguous=True,
        )


def test_attachment_send_freezes_roster_stages_bytes_and_logs_metadata_only(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _FakeRPC()
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    stored = service.put_attachment(
        room_id="room-1",
        upload_id="upload-1",
        kind="image",
        name="diagram.png",
        mime="image/png",
        data=b"\x89PNG\r\n\x1a\nimage",
    )
    manifest = {
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-attachment-1",
        payload={
            "text": "@ops inspect",
            "thread_id": "thread-1",
            "attachments": [manifest],
        },
    )
    _wait_for(lambda: any(method == "stage_attachment" for method, _ in rpc.calls))
    _wait_for(
        lambda: any(
            event["kind"] == "message.member" for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)

    stage_index = next(index for index, call in enumerate(rpc.calls) if call[0] == "stage_attachment")
    submit_index = next(index for index, call in enumerate(rpc.calls) if call[0] == "submit")
    assert stage_index < submit_index
    assert rpc.calls[stage_index][1]["data"] == b"\x89PNG\r\n\x1a\nimage"
    user_event = service._events("room-1")[0]
    assert user_event["payload"]["attachments"] == [manifest]
    assert "PNG" not in repr(user_event)
    assert "base64" not in repr(user_event)
    try:
        service.read_attachment(
            room_id="room-1",
            attachment_id=stored["attachment_id"],
            recipient_member_id="late-member",
        )
    except ValueError:
        pass
    else:  # pragma: no cover - ownership must fail closed
        raise AssertionError("late member unexpectedly received historic attachment")


def test_conflicting_event_id_releases_an_invisible_attachment_commit(tmp_path: Path):
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
        event_id="duplicate-event",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "original", "thread_id": "thread-1"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
    )
    stored = service.put_attachment(
        room_id="room-1",
        upload_id="upload-conflict",
        kind="file",
        name="notes.txt",
        mime="text/plain",
        data=b"release notes",
    )
    manifest = [{
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }]

    with pytest.raises(hosted_rooms.EventConflictError):
        service.send(
            room_id="room-1",
            event_id="duplicate-event",
            payload={
                "text": "different",
                "thread_id": "thread-1",
                "attachments": manifest,
            },
        )

    with sqlite3.connect(db) as conn:
        state, event_id, expires_at = conn.execute(
            """SELECT state, event_id, expires_at
                 FROM hosted_room_attachments WHERE attachment_id=?""",
            (stored["attachment_id"],),
        ).fetchone()
    assert state == "uploaded"
    assert event_id is None
    assert expires_at is not None


def test_conflicting_replay_preserves_the_original_events_attachment(tmp_path: Path):
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
    stored = service.put_attachment(
        room_id="room-1",
        upload_id="upload-original",
        kind="file",
        name="notes.txt",
        mime="text/plain",
        data=b"release notes",
    )
    manifest = [{
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }]
    service.send(
        room_id="room-1",
        event_id="same-event",
        payload={"text": "original", "thread_id": "thread-1", "attachments": manifest},
    )

    with pytest.raises(hosted_rooms.EventConflictError):
        service.send(
            room_id="room-1",
            event_id="same-event",
            payload={"text": "changed", "thread_id": "thread-1", "attachments": manifest},
        )

    assert service.read_attachment(
        room_id="room-1",
        attachment_id=stored["attachment_id"],
        recipient_member_id="default",
        event_id="same-event",
    ).data == b"release notes"


def test_viewer_access_does_not_collide_with_a_bot_named_viewer(tmp_path: Path):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("viewer", "ops")
    service.create_room(
        room_id="room-1",
        name="Review room",
        members=[
            {"member_id": "viewer", "profile": "viewer", "handle": "viewer"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    stored = service.put_attachment(
        room_id="room-1",
        upload_id="upload-viewer",
        kind="file",
        name="notes.txt",
        mime="text/plain",
        data=b"release notes",
    )
    manifest = [{
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }]
    service.send(
        room_id="room-1",
        event_id="event-viewer",
        payload={"text": "Review", "thread_id": "thread-1", "attachments": manifest},
    )

    assert service.read_attachment(
        room_id="room-1",
        attachment_id=stored["attachment_id"],
        recipient_member_id="viewer",
        event_id="event-viewer",
    ).data == b"release notes"
    assert service.read_attachment(
        room_id="room-1",
        attachment_id=stored["attachment_id"],
        recipient_member_id=None,
        event_id="event-viewer",
        viewer=True,
    ).data == b"release notes"


def test_pdf_upload_is_refused_before_room_admission_without_poppler(
    tmp_path: Path,
    monkeypatch,
):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="PDF room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "default"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    monkeypatch.setattr(
        "tui_gateway.hosted_room_service.shutil.which",
        lambda _name: None,
    )

    with pytest.raises(hosted_rooms.HostedRoomError, match="Poppler"):
        service.put_attachment(
            room_id="room-1",
            upload_id="pdf-1",
            kind="pdf",
            name="brief.pdf",
            mime="application/pdf",
            data=b"%PDF-1.7\n%%EOF\n",
        )


def test_partial_attachment_failure_never_submits_text_only_and_other_member_continues(
    tmp_path: Path,
):
    db = tmp_path / "state.db"
    service = HostedRoomService(_server(), db_path=db)
    rpc = _FakeRPC()
    rpc.fail_attachment_profiles.add("default")
    service.rpc = rpc
    service.runtime.rpc = rpc
    service.local_profiles = lambda: ("default", "ops")
    service.create_room(
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    stored = service.put_attachment(
        room_id="room-1",
        upload_id="upload-1",
        kind="file",
        name="notes.txt",
        mime="text/plain",
        data=b"release notes",
    )
    manifest = {
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-attachment-1",
        payload={
            "text": "Inspect this",
            "thread_id": "thread-1",
            "attachments": [manifest],
        },
    )
    _wait_for(
        lambda: any(
            event["kind"] == "message.member" for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)

    events = service._events("room-1")
    failed = next(event for event in events if event["kind"] == "turn.failed")
    assert failed["payload"]["member_id"] == "default"
    assert any(
        event["kind"] == "message.member" and event["payload"]["member_id"] == "ops"
        for event in events
    )
    assert not any(
        method == "submit" and params["profile"] == "default"
        for method, params in rpc.calls
    )


@pytest.mark.parametrize(
    "peer_client",
    [_UnavailableAttachmentPeerClient, _AmbiguousAttachmentPeerClient],
)
def test_peer_attachment_failure_does_not_block_healthy_member(
    tmp_path: Path,
    peer_client,
):
    db = tmp_path / "state.db"
    route = PeerMemberRoute(
        home_install_id="install-home",
        member_id="member-peer",
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
        peer_routes={("room-1", "member-peer"): route},
        peer_clients={"install-peer": peer_client()},
    )
    service.rpc = _FakeRPC()
    service.runtime.rpc = service.rpc
    service.local_profiles = lambda: ("local",)
    service.create_room(
        room_id="room-1",
        name="Attachment fallback room",
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
    data = b"release notes"
    stored = service.put_attachment(
        room_id="room-1",
        upload_id="upload-fallback",
        kind="file",
        name="notes.txt",
        mime="text/plain",
        data=data,
    )
    manifest = [{
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }]

    service.start()
    service.send(
        room_id="room-1",
        event_id="user-fallback-file",
        payload={
            "text": "Review the file together",
            "thread_id": "thread-1",
            "attachments": manifest,
        },
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
    assert not any(
        event["kind"] == "turn.failed"
        and event["payload"]["member_id"] == "member-peer"
        for event in events
    )
    peer_tasks = [
        task
        for task in driver.list_tasks(db, room_id="room-1")
        if task["payload"].get("target_member_id") == "member-peer"
    ]
    assert peer_tasks[0]["status"] == "deferred"
    assert any(
        event["kind"] == "turn.deferred"
        and event["payload"]["member_id"] == "member-peer"
        for event in events
    )
    assert any(
        event["kind"] == "message.member"
        and event["payload"]["member_id"] == "local"
        for event in events
    )


def test_mixed_version_file_peer_is_deferred_without_wedging_the_room(tmp_path: Path):
    db = tmp_path / "state.db"
    route = PeerMemberRoute(
        home_install_id="install-home",
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest="a" * 64,
        execution_policy_digest="b" * 64,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed-room-grant",
        attachments=False,
    )
    service = HostedRoomService(
        _server(),
        db_path=db,
        peer_routes={("room-1", "member-peer"): route},
        peer_clients={"install-peer": _FakePeerClient()},
    )
    service.rpc = _FakeRPC()
    service.runtime.rpc = service.rpc
    service.local_profiles = lambda: ("local",)
    service.create_room(
        room_id="room-1",
        name="Mixed version room",
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
    stored = service.put_attachment(
        room_id="room-1",
        upload_id="mixed-version-file",
        kind="file",
        name="notes.txt",
        mime="text/plain",
        data=b"release notes",
    )
    manifest = [{
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }]

    service.start()
    service.send(
        room_id="room-1",
        event_id="mixed-version-event",
        payload={"text": "Review", "thread_id": "thread-1", "attachments": manifest},
    )
    _wait_for(
        lambda: any(
            event["kind"] == "message.member"
            and event["payload"]["member_id"] == "local"
            for event in service._events("room-1")
        )
    )
    assert service.stop(timeout=1.0)
    peer_task = next(
        task
        for task in driver.list_tasks(db, room_id="room-1")
        if task["payload"].get("target_member_id") == "member-peer"
    )
    assert peer_task["status"] == "deferred"
    assert any(
        event["kind"] == "turn.deferred"
        and event["payload"]["member_id"] == "member-peer"
        for event in service._events("room-1")
    )


@pytest.mark.parametrize(
    ("stored_support", "live_support"), [(False, True), (True, False)]
)
def test_attachment_catalog_reprobe_persists_peer_upgrade_or_downgrade(
    tmp_path: Path,
    stored_support: bool,
    live_support: bool,
):
    db = tmp_path / "state.db"
    stored_catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            attachments=stored_support,
        )
    )
    live_catalog = catalog_mapping(
        installation_id="install-peer",
        persistent_process=True,
        attachments=live_support,
    )
    route = PeerMemberRoute(
        home_install_id=hosted_rooms.local_authority_gateway_id(),
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        capability_digest=stored_catalog.catalog_digest,
        cancellation_scope_id="cancel-room-1",
        trace_id="trace-room-1",
        grant="signed.room.grant",
        attachments=stored_support,
    )

    class ProbedPeer(_FakePeerClient):
        def probe(self, **_kwargs):
            return {"catalog": live_catalog}

    peer = ProbedPeer()
    service = HostedRoomService(_server(), db_path=db)
    service.register_peer_route(
        room_id="room-1",
        member_id="member-peer",
        route=route,
        client=peer,
        target_url="https://peer.example.test",
        catalog=stored_catalog,
    )

    refreshed = service._refresh_peer_attachment_catalog(
        "room-1",
        "member-peer",
        route,
        peer,
    )

    assert refreshed.attachments is live_support
    assert refreshed.capability_digest == live_catalog["catalog_digest"]
    restarted = HostedRoomService(_server(), db_path=db)
    persisted = restarted.peer_routes[("room-1", "member-peer")]
    assert persisted.attachments is live_support
    assert persisted.capability_digest == live_catalog["catalog_digest"]


def test_peer_attachment_recovery_replays_the_same_execution_generation(tmp_path: Path):
    db = tmp_path / "state.db"
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            attachments=True,
        )
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
        attachments=True,
    )
    peer = _RecoveringPeerClient()
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
            {"member_id": "default", "profile": "default", "handle": "hermes"},
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
            },
        ],
    )
    identity = driver.TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
    data = b"release notes"
    stored = service.put_attachment(
        room_id="room-1",
        upload_id="upload-recovery",
        kind="file",
        name="notes.txt",
        mime="text/plain",
        data=data,
    )
    manifest = [{
        key: stored[key]
        for key in ("attachment_id", "kind", "name", "size", "mime")
    }]
    service.attachments.commit_message(
        room_id="room-1",
        event_id="user-attachment",
        manifest=manifest,
        recipient_member_ids=("member-peer",),
    )

    service._resolve_member_transport(
        service.bindings()[0],
        {
            "identity": identity,
            "status": "indeterminate",
            "execution_generation": 1,
            "payload": {
                "target_member_id": "member-peer",
                "target_profile": "reviewer",
                "source_event_seq": 9,
                "prompt": "Recover the accepted review.",
                "attachments": manifest,
            },
        },
    )

    assert len(peer.recoveries) == 1
    recovered = peer.recoveries[0]["dispatch"]
    assert recovered["task_id"] == "task-1"
    assert recovered["execution_generation"] == 1
    assert recovered["prompt"] == "Recover the accepted review."
    assert peer.staged[0]["attachments"][0]["data"] == data
    assert (
        peer.staged[0]["dispatch"]["attachment_manifest_digest"]
        == recovered["attachment_manifest_digest"]
    )
