"""Later history can progress without recapturing or sending unsupported v1 work."""

import asyncio
import json
import sqlite3

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
@pytest.mark.parametrize("ack_damage", [None, "authority", "lineage_sha256", "version", "old_ack", "epoch_float"])
async def test_lost_a1_ack_then_two_successors_keep_independent_http_work(setup, monkeypatch, ack_damage):
    from tests.gateway.test_api_server_room_work_records import seed, invitation, publisher
    from tests.gateway.test_hosted_room_replica_lineage import FINAL
    source, target, app = setup
    seed(source)
    receipt = {"room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME, "authority_epoch": 1,
        "member_id": "reviewer", "target_install_id": TARGET, "target_profile": "default", "task_id": "task",
        "execution_generation": 7, "run_id": "original-run", "session_id": "original-session"}
    rooms.upsert_remote_run_receipt(source, record=receipt)
    sent, probes = [], []
    @web.middleware
    async def lost_ack(request, handler):
        # Real network work cannot hold a source write transaction.
        with sqlite3.connect(source, timeout=0.1) as conn:
            conn.execute("BEGIN IMMEDIATE")
        response = await handler(request)
        if request.path.endswith("/capabilities"):
            probes.append(True)
        if request.path.endswith("/work-records") and response.status == 200:
            with sqlite3.connect(target) as conn:
                sent.append(json.loads(conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} ORDER BY producer_epoch DESC").fetchone()[0]))
            if len(sent) == 1:
                request.transport.close()
            elif ack_damage:
                body = json.loads(response.text)
                if ack_damage == "old_ack":
                    body = {"room_id": "room", "revision": sent[0]["revision"], "digest": sent[0]["digest"], "passive": True}
                elif ack_damage == "epoch_float":
                    body["authority"]["epoch"] = float(body["authority"]["epoch"])
                else:
                    body[ack_damage] = {"authority": sent[0]["authority"], "lineage_sha256": "0" * 64, "version": 1}[ack_damage]
                return web.json_response(body)
        return response
    app.middlewares.append(lost_ack)
    async with TestClient(TestServer(app)) as http:
        invited = await invitation(http, passive_only=True)
        pub = publisher(source, http, invited, monkeypatch)
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        assert len(sent) == 1
        old = sent[0]
        old_id = None
        for prior, epoch, identity in ((HOME, 1, SUCCESSOR), (SUCCESSOR, 2, FINAL)):
            transfer(source, prior, epoch, identity)
            client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key=KEY, timeout_seconds=2)
            issued = await asyncio.to_thread(client.issue_invitation, room_id="room", home_install_id=identity,
                authority_gateway_id=identity, authority_epoch=epoch+1, member_id="reviewer",
                grant_id=f"work-{epoch+1}", replication=True, passive_only=True, work_records=True)
            links.save_room_link(source, links.make_stored_link(room_id="room", member_id="reviewer",
                target_url=str(http.make_url("/")), target_profile="default", grant=issued["grant"],
                catalog=peer.GatewayRoomCatalog.from_mapping(issued["catalog"]), cancellation_scope_id="cancel", trace_id="trace"))
            entry = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
                endpoint=str(http.make_url("/")), local_gateway_id=identity, secret=peer.gateway_room_grant_secret(),
                replace_enrollment_id=old_id, enrollment_id=f"work-enroll-{epoch+1}")
            response = await http.post("/v1/group-replicas/enroll", json={"enrollment": entry,
                **retirement.home_enrollment_history(source, enrollment_id=entry["enrollment_id"]),
                **({"expected_enrollment_id": old_id} if old_id else {})}, headers={"Authorization": f"Bearer {KEY}"})
            assert response.status == 200, await response.text()
            old_id = entry["enrollment_id"]
            pub = successor_publisher(source, monkeypatch, identity)
            for _ in range(3):
                await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
            assert sent[-1]["authority"] == {"gateway_id": identity, "epoch": epoch+1}
            assert sent[-1]["version"] == 2 and sent[-1]["revision"] == 1
            state = replicas.replica_state(target, room_id="room")["work_records"]
            assert state["producer"] == sent[-1]["authority"]
            assert state["incompleteness"] == ["prior_authority_work_unknown"]
            assert state["task_origins"]["task"] == old["authority"]
            assert state["receipts"] == [receipt]
            assert pub.status()["routes"][0]["work_record_status"] == ("invalid_ack" if ack_damage else "acked")
            assert len(state["scopes"]) == epoch+1
            with rooms._transaction(source) as conn:
                pending = dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE} WHERE producer_epoch=1").fetchone())
                assert pending["record_json"] == records.encode(old)
                assert pending["status"] == "unavailable" and pending["disposition"] == "superseded_authority"
            with rooms._transaction(target) as conn:
                assert conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE producer_epoch=1").fetchone()[0] == records.encode(old)
            # Reopen and quiet retry cannot resend the old producer.
            pub = successor_publisher(source, monkeypatch, identity)
            before, probe_count = len(sent), len(probes)
            await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
            assert len(sent) == before and len(probes) == probe_count
            from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
            for token, payload in ((invited["grant"], sent[-1]), (issued["grant"], old)):
                with pytest.raises(PeerRunsHTTPError):
                    await asyncio.to_thread(client.replicate_work_records, grant=token, target_profile="default", record=payload)
            claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), issued["grant"], permission=records.PERMISSION)
            assert claims["permissions"] == ["replicate", "status", "work_records"]
            rooms.revoke_room_grant_scope(target, claims=claims, expires_at=claims["status_expires_at"])
            with pytest.raises(PeerRunsHTTPError):
                await asyncio.to_thread(client.replicate_work_records, grant=issued["grant"], target_profile="default", record=sent[-1])

        from gateway import hosted_room_link_records as link_records
        closing = dict(room_id="room", authority_gateway_id=FINAL, authority_epoch=3)
        link_records.begin_room_link_retirement(source, **closing)
        link_records.complete_room_link_retirement(source, **closing)
        link_records.delete_room_link_records(source, room_id="room")
        rooms.disband_room(source, room_id="room", expected_gateway_id=FINAL, expected_epoch=3)
        notice = retirement.materialize_notice(source, enrollment_id=old_id,
            local_gateway_id=FINAL, secret_loader=peer.gateway_room_grant_secret)
        result = await asyncio.to_thread(client.retire_replica, notice)
        assert result["retired"] is True
        with sqlite3.connect(target) as raw:
            assert raw.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 0
        assert (await asyncio.to_thread(client.retire_replica, notice)) == result


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["missing", "malformed", "work_only_unsupported"])
async def test_v2_capability_refusal_preserves_current_pending_until_explicit_route_replacement(setup, monkeypatch, capability):
    from dataclasses import replace
    from tests.gateway.test_work_record_v2_admission_closure import enrolled_work
    from tests.gateway.test_api_server_room_work_records import publisher
    source, target, app = setup
    probes, sent = [], []
    supported = False

    @web.middleware
    async def negotiate(request, handler):
        response = await handler(request)
        if request.path.endswith("/capabilities"):
            probes.append(True)
            if not supported:
                body = json.loads(response.text)
                if capability == "missing":
                    body.pop("passive_replication")
                elif capability == "malformed":
                    body["passive_replication"]["work_record_versions"] = [True, 2]
                else:
                    body["passive_replication"]["work_record_versions"] = [1]
                return web.json_response(body)
        if request.path.endswith("/work-records"):
            sent.append(response.status)
        return response

    app.middlewares.append(negotiate)
    async with TestClient(TestServer(app)) as http:
        _, grant, current = await enrolled_work(http, source)
        catalog = peer.catalog_mapping(installation_id=TARGET, target_profile="default", persistent_process=True)
        publisher(source, http, {"grant": grant, "catalog": catalog}, monkeypatch)
        with rooms._transaction(source, immediate=True) as conn:
            records.prepare_delivery_locked(conn, room_id="room", target_install_id=TARGET,
                route_generation="held", local_gateway_id=SUCCESSOR, through_seq=current["history"]["seq"])
            frozen = dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone())
            source_before = [tuple(r) for r in conn.execute(f"SELECT * FROM {records.SOURCE_TABLE}")]
        for turn in range(3):
            pub = successor_publisher(source, monkeypatch)
            await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
            state = pub.status()["routes"][0]
            assert (state["work_record_status"] if capability == "work_only_unsupported" else state["status"]) == "unsupported_lineage"
        assert len(probes) == 1 and sent == []
        with rooms._transaction(source) as conn:
            assert dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone()) == frozen
            assert [tuple(r) for r in conn.execute(f"SELECT * FROM {records.SOURCE_TABLE}")] == source_before
        supported = True
        stored = links.load_room_link(source, room_id="room", member_id="reviewer")
        links.save_room_link(source, replace(stored, trace_id="explicit-compatible-route"))
        pub = successor_publisher(source, monkeypatch)
        for _ in range(3):
            await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        assert len(probes) == 2 and sent == [200]
        assert pub.status()["work_records"][0]["status"] == "acked"
        with rooms._transaction(target) as conn:
            assert conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE}").fetchone()[0] == frozen["record_json"]


@pytest.mark.asyncio
async def test_successor_history_leaves_old_pending_work_immutable_and_visibly_unsupported(setup, monkeypatch):
    source, target, app = setup
    with rooms._transaction(source, immediate=True) as conn:
        records.prepare_delivery_locked(conn, room_id="room", target_install_id=TARGET,
            route_generation="original", local_gateway_id=HOME, through_seq=1)
        before = dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone())
    transfer(source)
    work_requests = []

    @web.middleware
    async def observe(request, handler):
        if request.path.endswith("/work-records"):
            work_requests.append(await request.text())
        response = await handler(request)
        if request.path.endswith("/capabilities") and response.status == 200:
            body = json.loads(response.text)
            body["passive_replication"]["work_record_versions"] = [1]
            return web.json_response(body)
        return response

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
                assert dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone()) == {**before, "disposition": "superseded_authority"}
        assert work_requests == []
        assert replicas.replica_state(target, room_id="room")["work_records"]["availability"] == "not_retained"
