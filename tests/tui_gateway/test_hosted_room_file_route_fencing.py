"""Files consumers preserve the route owner's observed-grant boundary."""

import io
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_room_links, hosted_rooms, hosted_room_link_records
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    catalog_mapping,
    issue_room_grant,
)
from tests.tui_gateway.test_hosted_room_artifact_service import _ArtifactPeerClient
from tests.tui_gateway.hosted_room_service_fixtures import _server
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def routes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        PeerRunsHTTPClient, "revoke_grant_exact", lambda self, **kwargs: {"revoked": True}
    )
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            attachments=True,
            target_profile="reviewer",
        )
    )
    first = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    home = hosted_rooms.local_authority_gateway_id()
    first.create_room(
        room_id="room-1",
        name="File routes",
        members=[
            {"member_id": "default", "profile": "default", "handle": "local"},
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
    now = time.time()
    tokens = {
        name: issue_room_grant(
            b"files-route-test-secret" * 2,
            grant_id=name,
            issued_at=now - age,
            room_id="room-1",
            home_install_id=home,
            authority_gateway_id=home,
            authority_epoch=1,
            member_id="member-peer",
            target_install_id="install-peer",
            target_profile="reviewer",
            execution_policy_digest=catalog.execution_policy.policy_digest,
            ttl_seconds=3600,
            status_expires_at=now + 10000,
        )
        for name, age in (("old", 0), ("winner", 0), ("stale", 0), ("aging", 3550))
    }

    def register(service, grant):
        service.register_peer_route(
            room_id="room-1",
            member_id="member-peer",
            route=PeerMemberRoute(
                home_install_id=home,
                member_id="member-peer",
                target_install_id="install-peer",
                target_profile="reviewer",
                capability_digest=catalog.catalog_digest,
                execution_policy_digest=catalog.execution_policy.policy_digest,
                cancellation_scope_id="cancel-room",
                trace_id="trace-room",
                grant=grant,
                attachments=True,
            ),
            client=PeerRunsHTTPClient(
                base_url="https://peer.example.test",
                api_key="",
                target_profile="reviewer",
                receipt_db_path=first.db_path,
            ),
            target_url="https://peer.example.test",
            catalog=catalog,
        )

    register(first, tokens["old"])
    second = HostedRoomService(_server(), db_path=first.db_path)
    return SimpleNamespace(
        first=first, second=second, register=register, tokens=tokens, catalog=catalog
    )


def _settled(routes, *, legacy=False):
    worker = routes.second
    room = hosted_rooms.room_state(worker.db_path, room_id="room-1")
    hosted_rooms.append_event(
        worker.db_path,
        room_id="room-1",
        event_id="file-request",
        kind="message.user",
        actor={"kind": "user", "id": "local"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
        payload={"text": "@reviewer hand off a file", "thread_id": "work"},
    )
    plan = discussion.plan_next_task(
        room, worker._events("room-1"), local_profiles=("default",)
    ).task
    assert plan is not None and plan.member.member_id == "member-peer"
    if legacy:
        payload = dict(plan.payload)
        payload.pop("recipient_member_ids")
        plan = replace(plan, payload=payload)
    driver.admit_task(
        worker.db_path, plan.identity, payload=plan.payload, clock=time.time
    )
    lease = driver.acquire_lease(
        worker.db_path,
        room_id="room-1",
        gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
        process_generation="file-route",
        ttl_seconds=300,
        clock=time.time,
    )
    attempt = driver.start_task(
        worker.db_path,
        plan.identity,
        lease,
        expected_cancel_generation=0,
        clock=time.time,
    )
    driver.settle_task(
        worker.db_path,
        attempt,
        settlement_id="file-result",
        status="settled",
        result={
            "text": "File",
            "run_id": "run-remote-1",
            "artifacts": _ArtifactPeerClient().manifest,
        },
        clock=time.time,
    )
    return room, driver.get_task(worker.db_path, plan.identity), plan


def test_loaded_and_rehydrated_routes_keep_binary_support_and_target_profile(
    routes, monkeypatch
):
    worker = routes.second
    key = ("room-1", "member-peer")
    urls = []

    def opened(request, **kwargs):
        urls.append(request.full_url)
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr("tui_gateway.hosted_room_peer_http._open_roomlink_url", opened)
    assert worker.peer_routes[key].attachments
    worker.peer_clients[key].probe(grant=routes.tokens["old"])
    worker.peer_routes.pop(key)
    worker.peer_clients.pop(key)
    route, client = worker._hydrate_persisted_peer_route(*key)
    assert route.attachments
    client.probe(grant=routes.tokens["old"])
    assert (
        urls
        == ["https://peer.example.test/p/reviewer/v1/room-members/capabilities"] * 2
    )


@pytest.mark.parametrize("hydrate", [False, True])
def test_retirement_reads_a_route_repaired_by_another_worker(
    routes, monkeypatch, hydrate
):
    room, task, plan = _settled(routes)
    routes.register(routes.first, routes.tokens["winner"])
    if hydrate:
        routes.second._hydrate_persisted_peer_route("room-1", "member-peer")
    used = []

    def discard(_client, **kwargs):
        used.append(kwargs)
        return {"discarded": True, "removed": 1}

    monkeypatch.setattr(PeerRunsHTTPClient, "discard_artifacts", discard)
    routes.second._retire_failed_terminal_artifacts(room=room, task=task, plan=plan)
    assert used == [{"run_id": "run-remote-1", "grant": routes.tokens["winner"]}]
    assert (
        driver.get_task(routes.second.db_path, plan.identity)["result"]
        == task["result"]
    )


@pytest.mark.parametrize("hydrate", [False, True])
@pytest.mark.parametrize("legacy_ack", [False, True])
def test_late_output_cleanup_error_cannot_poison_replacement_health(
    routes, monkeypatch, hydrate, legacy_ack
):
    room, task, plan = _settled(routes, legacy=legacy_ack)

    def fail_late(_client, **kwargs):
        assert kwargs["grant"] == routes.tokens["old"]
        routes.register(routes.first, routes.tokens["winner"])
        if hydrate:
            routes.second.status_with_grant_fingerprints("room-1")
        raise PeerRunsHTTPError(
            "old request failed", status_code=401, error_code="invalid_room_grant"
        )

    monkeypatch.setattr(
        PeerRunsHTTPClient,
        "acknowledge_artifacts" if legacy_ack else "discard_artifacts",
        fail_late,
    )
    with pytest.raises(RuntimeError, match="repaired route"):
        if legacy_ack:
            routes.second._import_terminal_artifacts(
                room=room, task=task, plan=plan, events=routes.second._events("room-1")
            )
        else:
            routes.second._retire_failed_terminal_artifacts(
                room=room, task=task, plan=plan
            )
    stored = hosted_room_link_records.room_link_record(
        routes.first.db_path, room_id="room-1", member_id="member-peer"
    )
    assert stored["grant"] == routes.tokens["winner"]
    assert stored["status"] == "ready"


@pytest.mark.parametrize("hydrate", [False, True])
@pytest.mark.parametrize("changed_catalog", [False, True])
def test_binary_catalog_probe_cannot_overwrite_a_concurrent_registration(
    routes, hydrate, changed_catalog
):
    worker = routes.second
    route = worker.peer_routes[("room-1", "member-peer")]
    catalog = json.loads(
        hosted_room_links.load_room_link(
            worker.db_path,
            room_id="room-1",
            member_id="member-peer",
        ).as_record()["catalog_json"]
    )
    if changed_catalog:
        catalog = catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            attachments=False,
            target_profile="reviewer",
        )

    class Peer:
        def probe(self, **kwargs):
            assert kwargs["grant"] == routes.tokens["old"]
            routes.register(routes.first, routes.tokens["winner"])
            if hydrate:
                worker.status_with_grant_fingerprints("room-1")
            return {"catalog": catalog}

    with pytest.raises(hosted_rooms.HostedRoomError, match="grant changed"):
        worker._refresh_peer_attachment_catalog("room-1", "member-peer", route, Peer())
    stored = hosted_room_links.load_room_link(
        worker.db_path, room_id="room-1", member_id="member-peer"
    )
    assert stored.grant == routes.tokens["winner"]
    assert stored.catalog.attachments
    assert stored.status == "ready"


@pytest.mark.parametrize(
    "operation",
    [
        "read_artifact",
        "acknowledge_artifacts",
        "discard_artifacts",
        "stage_attachments",
    ],
)
@pytest.mark.parametrize("hydrate", [False, True])
@pytest.mark.parametrize("status_write_fails", [False, True])
def test_file_operation_refresh_uses_observed_grant_and_exact_cleanup(
    routes, monkeypatch, operation, hydrate, status_write_fails
):
    routes.register(routes.first, routes.tokens["aging"])
    routes.second._hydrate_persisted_peer_route("room-1", "member-peer")
    catalog = json.loads(
        hosted_room_links.load_room_link(
            routes.first.db_path,
            room_id="room-1",
            member_id="member-peer",
        ).as_record()["catalog_json"]
    )
    cleaned, operated = [], []

    class Peer:
        base_url = None

        def refresh_grant(self, **kwargs):
            assert kwargs["grant"] == routes.tokens["aging"]
            assert kwargs["capability_digest"] == routes.catalog.catalog_digest
            assert (
                kwargs["execution_policy_digest"]
                == routes.catalog.execution_policy.policy_digest
            )
            routes.register(routes.first, routes.tokens["winner"])
            if hydrate:
                routes.second.status_with_grant_fingerprints("room-1")
            return {"grant": routes.tokens["stale"], "catalog": catalog}

        def revoke_grant_exact(self, *, grant):
            cleaned.append(grant)
            return {"revoked": True}

        def revoke_grant(self, **kwargs):
            raise AssertionError("exact cleanup fell back to scope revocation")

        def __getattr__(self, name):
            assert name == operation
            return lambda **kwargs: operated.append(kwargs)

    if status_write_fails:

        def fail_status(*args, **kwargs):
            raise OSError("status write failed")

        monkeypatch.setattr(hosted_room_links, "mark_room_link_status", fail_status)
    tracked = routes.second._tracked_peer_client("room-1", "member-peer", Peer())
    with pytest.raises(hosted_rooms.HostedRoomError, match="changed during reconnect"):
        getattr(tracked, operation)(grant=routes.tokens["aging"])
    stored = hosted_room_link_records.room_link_record(
        routes.first.db_path, room_id="room-1", member_id="member-peer"
    )
    assert stored["grant"] == routes.tokens["winner"] and stored["status"] == "ready"
    assert operated == []
    assert cleaned == [routes.tokens["stale"]]
