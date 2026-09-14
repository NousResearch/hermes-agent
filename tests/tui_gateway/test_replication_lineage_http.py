"""Negotiated passive lineage through native RPC, loopback HTTP and the publisher."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_rooms as rooms
from tests.gateway.test_api_server_room_replicas import HOME, TARGET, KEY, setup  # noqa: F401
from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, FINAL, transfer
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
from tui_gateway.hosted_room_replication import HostedRoomReplicationPublisher


def successor_publisher(source, monkeypatch, identity=SUCCESSOR):
    with monkeypatch.context() as scoped:
        scoped.setattr(rooms, "local_authority_gateway_id", lambda: identity)
        return HostedRoomReplicationPublisher(source)


async def route(http, source, identity=SUCCESSOR, epoch=2):
    client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key=KEY, timeout_seconds=2)
    issued = await asyncio.to_thread(client.issue_invitation, room_id="room", home_install_id=identity,
        authority_gateway_id=identity, authority_epoch=epoch, member_id="reviewer", grant_id=f"fresh-{epoch}",
        replication=True, passive_only=True)
    links.save_room_link(source, links.make_stored_link(
        room_id="room", member_id="reviewer", target_url=str(http.make_url("/")), target_profile="default",
        grant=issued["grant"], catalog=peer.GatewayRoomCatalog.from_mapping(issued["catalog"]),
        cancellation_scope_id="cancel", trace_id="trace"))
    return issued


@pytest.mark.asyncio
async def test_rpc_setup_and_publisher_replay_keep_exact_pending_bytes(setup, monkeypatch):
    source, target, app = setup
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    rooms.request_room_stop(source, room_id="room", cancel_id="historical-stop", expected_gateway_id=SUCCESSOR, expected_epoch=2)
    spans.append(transfer(source, SUCCESSOR, 2, FINAL))
    sent, lose = [], True
    from gateway.platforms.api_server import APIServerAdapter
    read_body = APIServerAdapter._read_json_body

    async def record_body(self, request):
        result = await read_body(self, request)
        if request.path.endswith("/replica"):
            sent.append(await request.text())
        return result

    monkeypatch.setattr(APIServerAdapter, "_read_json_body", record_body)

    @web.middleware
    async def lost_ack(request, handler):
        nonlocal lose
        response = await handler(request)
        if request.path.endswith("/replica") and response.status == 200 and lose:
            lose = False
            request.transport.close()
        return response

    app.middlewares.append(lost_ack)
    async with TestClient(TestServer(app)) as http:
        issued = await route(http, source, FINAL, 3)
        import tui_gateway.server as srv
        with monkeypatch.context() as scoped:
            scoped.setattr(rooms, "default_db_path", lambda: source)
            scoped.setattr(rooms, "local_authority_gateway_id", lambda: FINAL)
            prepared = srv._methods["groups.replication.prepare"](1, dict(
                room_id="room", target_install_id=TARGET, endpoint=str(http.make_url("/")), enrollment_id="v2-enroll"))
        assert "error" not in prepared, prepared
        prepared = prepared["result"]
        assert prepared["authority_history"] == spans
        response = await http.post("/v1/group-replicas/enroll", json=prepared, headers={"Authorization": f"Bearer {KEY}"})
        assert response.status == 200, await response.text()
        from tui_gateway import hosted_room_replication as publishing
        monkeypatch.setattr(publishing, "PAGE_LIMIT", 1)
        publisher = successor_publisher(source, monkeypatch, FINAL)
        await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        state = replicas.replica_state(target, room_id="room")
        assert state["last_seq"] == 1 and state["lineage_status"] == "pending"
        rooms.append_event(source, room_id="room", event_id="later", kind="message.user",
            actor={"kind": "user", "id": "owner"}, payload={"text": "after lost ACK"}, authority_gateway_id=FINAL, authority_epoch=3)
        publisher = successor_publisher(source, monkeypatch, FINAL)
        for _ in range(6):
            await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        assert sent[0] == sent[1]
        assert json.loads(sent[0])["page"]["replica_version"] == 2
        state = replicas.replica_state(target, room_id="room")
        assert state["last_seq"] == 5 and state["lineage_status"] == "verified"
        assert publisher.status("room")["routes"][0]["acked_seq"] == 5
        assert state["authority"] == {"gateway_id": FINAL, "epoch": 3}
        assert peer.decode_room_grant(peer.gateway_room_grant_secret(), issued["grant"], permission="status")["permissions"] == ["replicate", "status"]


@pytest.mark.asyncio
@pytest.mark.parametrize("advertised", [None, {"history_versions": [1], "retirement_versions": [1], "work_record_versions": []},
    {"history_versions": [1, 2, 3], "retirement_versions": [1, 2], "work_record_versions": []},
    {"history_versions": [True, 2], "retirement_versions": [1, 2], "work_record_versions": []}])
async def test_later_epoch_unsupported_is_durable_without_downgrade(setup, monkeypatch, advertised):
    source, target, app = setup
    transfer(source)
    probes, sends = [], []

    @web.middleware
    async def old_peer(request, handler):
        if request.path.endswith("/replica"):
            sends.append(True)
        response = await handler(request)
        if request.path.endswith("/capabilities") and response.status == 200:
            probes.append(True)
            body = json.loads(response.text)
            body.pop("passive_replication", None)
            if advertised is not None:
                body["passive_replication"] = advertised
            return web.json_response(body)
        return response

    app.middlewares.append(old_peer)
    async with TestClient(TestServer(app)) as http:
        await route(http, source)
        publisher = successor_publisher(source, monkeypatch)
        await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        routes = publisher.status("room")["routes"]
        assert routes and routes[0]["status"] == "unsupported_lineage"
        publisher = successor_publisher(source, monkeypatch)
        for _ in range(3):
            await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        assert probes == [True] and sends == []
        assert publisher.status("room")["routes"][0]["acked_seq"] == 0
        with pytest.raises(replicas.ReplicaError):
            replicas.replica_state(target, room_id="room")
        # A capable peer alone does not reset the blocked generation. Only an
        # explicit fresh grant/route plus owner setup resumes this passive copy.
        from gateway.hosted_room_passive_protocol import passive_capabilities
        advertised = passive_capabilities()
        await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        assert probes == [True]
        entry = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
            endpoint=str(http.make_url("/")), local_gateway_id=SUCCESSOR, secret=peer.gateway_room_grant_secret())
        response = await http.post("/v1/group-replicas/enroll", json={"enrollment": entry,
            **retirement.home_enrollment_history(source, enrollment_id=entry["enrollment_id"])},
            headers={"Authorization": f"Bearer {KEY}"})
        assert response.status == 200, await response.text()
        await route(http, source)
        await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        assert probes == [True, True] and sends == [True]
        assert replicas.replica_state(target, room_id="room")["lineage_status"] == "verified"


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["digest", "format", "head", "coverage", "status", "source_transition"])
async def test_v2_ack_cannot_advance_another_scope_or_a_stale_sender(setup, monkeypatch, damage):
    source, target, app = setup
    transfer(source)
    @web.middleware
    async def altered_ack(request, handler):
        response = await handler(request)
        if request.path.endswith("/replica") and response.status == 200:
            body = json.loads(response.text)
            mutations = {
                "digest": {"lineage_sha256": "0" * 64}, "format": {"replica_version": 1},
                "head": {"authority": {"gateway_id": HOME, "epoch": 1}},
                "coverage": {"stored_seq": body["stored_seq"] + 1}, "status": {"lineage_status": "pending"},
            }
            if damage == "source_transition":
                transfer(source, SUCCESSOR, 2, FINAL)
            else:
                body.update(mutations[damage])
            return web.json_response(body)
        return response
    app.middlewares.append(altered_ack)
    async with TestClient(TestServer(app)) as http:
        await route(http, source)
        entry = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
            endpoint=str(http.make_url("/")), local_gateway_id=SUCCESSOR, secret=peer.gateway_room_grant_secret())
        response = await http.post("/v1/group-replicas/enroll", json={"enrollment": entry,
            **retirement.home_enrollment_history(source, enrollment_id=entry["enrollment_id"])},
            headers={"Authorization": f"Bearer {KEY}"})
        assert response.status == 200, await response.text()
        publisher = successor_publisher(source, monkeypatch)
        await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        row = publisher.status("room")["routes"][0]
        assert row["acked_seq"] == 0
        assert row["status"] == ("pending" if damage == "source_transition" else "invalid_ack")
        assert replicas.replica_state(target, room_id="room")["last_seq"] == 2
