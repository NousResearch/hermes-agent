"""Native publisher threads, real HTTP loss, and independent SQLite stores."""

import asyncio
import multiprocessing
from contextlib import suppress

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from tests.gateway.test_api_server_room_replicas import HOME, invite, setup  # noqa: F401
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
from tui_gateway.hosted_room_replication import HostedRoomReplicationPublisher


async def wait_until(predicate, *, seconds=12):
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("replica did not reach the expected durable state")


def make_publisher(source, monkeypatch):
    # One process models two installation identities; transport and both stores
    # are real. The receiver keeps its own identity outside this constructor.
    with monkeypatch.context() as scoped:
        scoped.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
        return HostedRoomReplicationPublisher(source)


def run_publisher_process(source, control):
    rooms.local_authority_gateway_id = lambda: HOME
    publisher = HostedRoomReplicationPublisher(source)
    publisher.start()
    try:
        if control.poll(20):
            control.recv()
    finally:
        control.close()
        if not publisher.stop(timeout=5):
            raise RuntimeError("test publisher did not stop")


@pytest.mark.asyncio
async def test_lost_http_ack_and_publisher_restart_preserve_history(setup, monkeypatch):
    source, target, app = setup
    accepted = asyncio.Event()
    lose_reply = True

    @web.middleware
    async def lose_after_persistence(request, handler):
        nonlocal lose_reply
        response = await handler(request)
        if request.path.endswith("/replica") and response.status == 200 and lose_reply:
            lose_reply = False
            accepted.set()
            request.transport.close()
        return response

    app.middlewares.append(lose_after_persistence)
    publishers = []
    async with TestClient(TestServer(app)) as http:
        token = await invite(http, replication=True)
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        probe = await asyncio.to_thread(client.probe, grant=token)
        links.save_room_link(source, links.make_stored_link(
            room_id="room", member_id="reviewer", target_url=str(http.make_url("/")),
            target_profile="default", grant=token,
            catalog=peer.GatewayRoomCatalog.from_mapping(probe["catalog"]),
            cancellation_scope_id="test-cancel", trace_id="test-trace",
        ))
        first = make_publisher(source, monkeypatch)
        publishers.append(first)
        try:
            first.start()
            await asyncio.wait_for(accepted.wait(), timeout=10)
            assert await asyncio.to_thread(first.stop, timeout=5)
            assert replicas.replica_state(target, room_id="room")["last_seq"] == 1
            rooms.append_event(
                source, room_id="room", event_id="follow-up", kind="message.user",
                actor={"kind": "user", "id": "owner"}, payload={"text": "Follow up after restart"},
                authority_gateway_id=HOME, authority_epoch=1,
            )
            rooms.rename_room(source, room_id="room", event_id="rename", name="Updated workshop")
            second = make_publisher(source, monkeypatch)
            publishers.append(second)
            second.start()

            def completed():
                states = second.status("room")["routes"]
                return any(s["acked_seq"] == 3 and s["status"] == "acked" for s in states)

            await wait_until(completed)
            state = replicas.replica_state(target, room_id="room")
            assert state["last_seq"] == 3
            assert state["name"] == "Updated workshop"
            assert state["safety_status"] == "passive"
            with rooms._transaction(target) as conn:
                copied = conn.execute(
                    "SELECT event_id FROM hosted_room_replica_events ORDER BY seq",
                ).fetchall()
            assert [r[0] for r in copied] == ["hello", "follow-up", "rename"]
            assert second.status()["source_loss_safe"] is False
        finally:
            for publisher in publishers:
                assert await asyncio.to_thread(publisher.stop, timeout=5)


@pytest.mark.asyncio
async def test_replica_capacity_recovers_without_new_authorization(setup, monkeypatch):
    source, target, app = setup
    async with TestClient(TestServer(app)) as http:
        token = await invite(http, replication=True)
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        probe = await asyncio.to_thread(client.probe, grant=token)
        links.save_room_link(source, links.make_stored_link(
            room_id="room", member_id="reviewer", target_url=str(http.make_url("/")),
            target_profile="default", grant=token,
            catalog=peer.GatewayRoomCatalog.from_mapping(probe["catalog"]),
            cancellation_scope_id="test-cancel", trace_id="test-trace",
        ))
        publisher = make_publisher(source, monkeypatch)
        with monkeypatch.context() as limited:
            limited.setattr(replicas, "MAX_REPLICA_EVENT_BYTES", 1)
            await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        state = publisher.status("room")["routes"][0]
        assert state["status"] == "unavailable"
        assert state["acked_seq"] == 0
        await asyncio.to_thread(publisher._publish_one, ("room", "reviewer"))
        assert publisher.status("room")["routes"][0]["status"] == "acked"
        assert replicas.replica_state(target, room_id="room")["last_seq"] == 1


@pytest.mark.asyncio
async def test_publisher_process_death_after_remote_write_replays_without_duplicates(setup):
    source, target, app = setup
    accepted, release = asyncio.Event(), asyncio.Event()
    hold_first = True

    @web.middleware
    async def hold_after_persistence(request, handler):
        nonlocal hold_first
        response = await handler(request)
        if request.path.endswith("/replica") and response.status == 200 and hold_first:
            hold_first = False
            accepted.set()
            await release.wait()
        return response

    app.middlewares.append(hold_after_persistence)
    context = multiprocessing.get_context("spawn")
    children = []
    async with TestClient(TestServer(app)) as http:
        token = await invite(http, replication=True)
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        probe = await asyncio.to_thread(client.probe, grant=token)
        links.save_room_link(source, links.make_stored_link(
            room_id="room", member_id="reviewer", target_url=str(http.make_url("/")),
            target_profile="default", grant=token,
            catalog=peer.GatewayRoomCatalog.from_mapping(probe["catalog"]),
            cancellation_scope_id="test-cancel", trace_id="test-trace",
        ))
        try:
            receive, control = context.Pipe(duplex=False)
            first = context.Process(target=run_publisher_process, args=(source, receive))
            children.append((first, control))
            first.start()
            receive.close()
            await asyncio.wait_for(accepted.wait(), timeout=12)
            first.terminate()
            await asyncio.to_thread(first.join, 5)
            assert not first.is_alive()
            assert first.exitcode != 0
            release.set()
            assert replicas.replica_state(target, room_id="room")["last_seq"] == 1
            with rooms._transaction(source) as conn:
                assert conn.execute("SELECT acked_seq FROM hosted_room_replication_targets").fetchone()[0] == 0
            rooms.append_event(
                source, room_id="room", event_id="after-crash", kind="message.user",
                actor={"kind": "user", "id": "owner"}, payload={"text": "Resume after process death"},
                authority_gateway_id=HOME, authority_epoch=1,
            )
            receive, control = context.Pipe(duplex=False)
            second = context.Process(target=run_publisher_process, args=(source, receive))
            children.append((second, control))
            second.start()
            receive.close()

            def copied():
                return replicas.replica_state(target, room_id="room")["last_seq"] == 2

            await wait_until(copied)
            with rooms._transaction(target) as conn:
                copied_ids = conn.execute("SELECT event_id FROM hosted_room_replica_events ORDER BY seq").fetchall()
            assert [row[0] for row in copied_ids] == ["hello", "after-crash"]
            assert replicas.replica_state(target, room_id="room")["safety_status"] == "passive"
            control.send("stop")
            await asyncio.to_thread(second.join, 5)
            assert second.exitcode == 0
        finally:
            release.set()
            for child, control in children:
                if child.is_alive():
                    with suppress(BrokenPipeError, OSError):
                        control.send("stop")
                await asyncio.to_thread(child.join, 5)
                if child.is_alive():
                    child.terminate()
                    await asyncio.to_thread(child.join, 5)
                assert not child.is_alive()
                control.close()
                child.close()
