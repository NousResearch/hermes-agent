"""Later history can progress without recapturing or sending unsupported v1 work."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from tests.gateway.test_api_server_room_replicas import HOME, TARGET, KEY, setup  # noqa: F401
from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, transfer
from tests.tui_gateway.test_replication_lineage_http import successor_publisher
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient


@pytest.mark.asyncio
async def test_successor_history_leaves_old_pending_work_immutable_and_visibly_unsupported(setup, monkeypatch):
    source, target, app = setup
    with rooms._transaction(source, immediate=True) as conn:
        records.prepare_delivery_locked(conn, room_id="room", target_install_id=TARGET,
            route_generation="original", local_gateway_id=HOME, through_seq=1)
        before = tuple(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone())
    transfer(source)
    work_requests = []

    @web.middleware
    async def observe(request, handler):
        if request.path.endswith("/work-records"):
            work_requests.append(await request.text())
        return await handler(request)

    app.middlewares.append(observe)
    async with TestClient(TestServer(app)) as http:
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key=KEY, timeout_seconds=2)
        issued = await asyncio.to_thread(client.issue_invitation, room_id="room", home_install_id=SUCCESSOR,
            authority_gateway_id=SUCCESSOR, authority_epoch=2, member_id="reviewer", grant_id="successor",
            replication=True, passive_only=True, work_records=True)
        links.save_room_link(source, links.make_stored_link(
            room_id="room", member_id="reviewer", target_url=str(http.make_url("/")), target_profile="default",
            grant=issued["grant"], catalog=peer.GatewayRoomCatalog.from_mapping(issued["catalog"]),
            cancellation_scope_id="cancel", trace_id="trace"))
        entry = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
            endpoint=str(http.make_url("/")), local_gateway_id=SUCCESSOR, secret=peer.gateway_room_grant_secret())
        response = await http.post("/v1/group-replicas/enroll", json={"enrollment": entry,
            **retirement.home_enrollment_history(source, enrollment_id=entry["enrollment_id"])},
            headers={"Authorization": f"Bearer {KEY}"})
        assert response.status == 200, await response.text()
        for turn in range(3):
            publisher = successor_publisher(source, monkeypatch)
            rooms.append_event(source, room_id="room", event_id=f"busy-{turn}", kind="message.user",
                actor={"kind": "user", "id": "owner"}, payload={"text": "after transfer"},
                authority_gateway_id=SUCCESSOR, authority_epoch=2)
            await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
            state = replicas.replica_state(target, room_id="room")
            assert state["last_seq"] == rooms.room_state(source, room_id="room")["latest_seq"]
            assert state["lineage_status"] == "verified"
            status = publisher.status("room")
            assert status["routes"][0]["status"] == "acked"
            assert status["routes"][0]["work_record_status"] == "unsupported_lineage"
            assert status["work_records_error"] == "unsupported_lineage"
            with rooms._transaction(source) as conn:
                assert tuple(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone()) == before
        assert work_requests == []
        assert replicas.replica_state(target, room_id="room")["work_records"]["availability"] == "not_retained"
