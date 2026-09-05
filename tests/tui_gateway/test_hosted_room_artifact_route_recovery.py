"""Terminal file retries observe route repairs made by another gateway worker."""

from dataclasses import replace
import time

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_room_links, hosted_rooms
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    catalog_mapping,
    issue_room_grant,
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.test_hosted_room_artifact_service import _ArtifactPeerClient
from tests.tui_gateway.hosted_room_service_fixtures import _server


@pytest.mark.parametrize("force_hydration", [False, True])
def test_artifact_retry_observes_route_repaired_by_other_service(
    tmp_path, monkeypatch, force_hydration
):
    db = tmp_path / "state.db"
    local_id = hosted_rooms.local_authority_gateway_id()
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            attachments=True,
            target_profile="reviewer",
        )
    )
    grant_scope = dict(
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
    old_grant = issue_room_grant(
        b"review-only-secret" * 2,
        grant_id="old",
        issued_at=time.time() - 7200,
        **grant_scope,
    )
    new_grant = issue_room_grant(
        b"review-only-secret" * 2,
        grant_id="new",
        issued_at=time.time(),
        **grant_scope,
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
        grant=old_grant,
        attachments=True,
    )
    writer = HostedRoomService(_server(), db_path=db)
    revoked = []

    def revoke_exact(client, *, grant):
        revoked.append((client.base_url, grant))
        return {"revoked": True}

    monkeypatch.setattr(PeerRunsHTTPClient, "revoke_grant_exact", revoke_exact)
    writer.local_profiles = lambda: ("ops",)
    writer.create_room(
        room_id="room-1",
        name="Review room",
        members=[
            {
                "member_id": "member-reviewer",
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
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )

    def register(grant):
        writer.register_peer_route(
            room_id="room-1",
            member_id="member-reviewer",
            route=replace(route, grant=grant),
            client=PeerRunsHTTPClient(
                base_url="https://peer.example.test",
                api_key="",
                target_profile="reviewer",
                receipt_db_path=db,
            ),
            target_url="https://peer.example.test",
            catalog=catalog,
        )

    register(old_grant)
    worker = HostedRoomService(_server(), db_path=db)
    worker.local_profiles = lambda: ("ops",)
    hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-1",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=local_id,
        authority_epoch=1,
        payload={"text": "@reviewer prepare a file", "thread_id": "thread-1"},
    )
    room = hosted_rooms.room_state(db, room_id="room-1")
    plan = discussion.plan_next_task(
        room,
        worker._events("room-1"),
        local_profiles=("ops",),
    ).task
    assert plan is not None
    driver.admit_task(db, plan.identity, payload=plan.payload, clock=time.time)
    lease = driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=local_id,
        authority_epoch=1,
        process_generation=worker.runtime.process_generation,
        ttl_seconds=30,
        clock=time.time,
    )
    attempt = driver.start_task(
        db,
        plan.identity,
        lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    artifact = _ArtifactPeerClient()
    driver.settle_task(
        db,
        attempt,
        settlement_id="terminal-1",
        status="settled",
        result={
            "text": "Handoff",
            "run_id": "run-remote-1",
            "artifacts": artifact.manifest,
        },
        clock=time.time,
    )
    used = []

    def refresh(_client, **kwargs):
        used.append("new" if kwargs["grant"] == new_grant else "old")
        raise PeerRunsHTTPError(
            "grant expired", status_code=401, error_code="invalid_room_grant"
        )

    def read(_client, **kwargs):
        used.append("new" if kwargs["grant"] == new_grant else "old")
        if kwargs["grant"] != new_grant:
            raise PeerRunsHTTPError(
                "grant expired", status_code=401, error_code="invalid_room_grant"
            )
        return artifact.data

    def discard(_client, **kwargs):
        raise PeerRunsHTTPError(
            "artifact retirement rejected",
            status_code=409,
            error_code="invalid_artifact_retirement",
        )

    monkeypatch.setattr(PeerRunsHTTPClient, "refresh_grant", refresh)
    monkeypatch.setattr(PeerRunsHTTPClient, "read_artifact", read)
    monkeypatch.setattr(PeerRunsHTTPClient, "discard_artifacts", discard)
    monkeypatch.setattr(
        PeerRunsHTTPClient,
        "acknowledge_artifacts",
        lambda _client, **_kw: {"acknowledged": True},
    )
    binding = worker.bindings()[0]
    worker.prepare_room(binding)
    task = driver.get_task(db, plan.identity)
    assert not worker._artifact_retry_due(task)
    assert used and set(used) == {"old"}
    used.clear()

    register(new_grant)
    assert revoked == [("https://peer.example.test", old_grant)]
    assert (
        hosted_room_links.load_room_link(
            db, room_id="room-1", member_id="member-reviewer"
        ).grant
        == new_grant
    )
    assert worker._artifact_retry_due(task)
    if force_hydration:
        worker._hydrate_persisted_peer_route("room-1", "member-reviewer")
    worker.prepare_room(binding)
    assert used == ["new"], "terminal publication retried the stale in-memory grant"
    messages = [e for e in worker._events("room-1") if e["kind"] == "message.member"]
    assert len(messages) == 1
    assert messages[0]["payload"]["attachments"]
    assert worker._artifact_retry_keys("room-1") == set()
