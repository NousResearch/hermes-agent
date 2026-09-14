"""Existing two-worker publisher consumes passive work evidence without starvation."""

import sqlite3
import threading
from contextlib import closing

import pytest
import hermes_state_wal

from gateway import hosted_room_driver as driver
from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_replication import HostedRoomReplicationPublisher


def test_work_record_initializer_reserves_writer_before_reading_schema(tmp_path, monkeypatch):
    from gateway import hosted_room_work_storage as storage

    source = tmp_path / "initialize.db"
    rooms.create_room(source, room_id="room", name="Workshop", members=[], authority_gateway_id="home")
    initialize_locked = storage._initialize_locked
    with closing(sqlite3.connect(source, timeout=0)) as competing_writer:
        def check_writer_reserved(conn):
            conn.execute("SELECT name FROM sqlite_master").fetchall()
            # A deferred read can otherwise lose its write upgrade immediately,
            # even with a busy timeout, when another publisher reserves the writer.
            try:
                with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                    competing_writer.execute("BEGIN IMMEDIATE")
            finally:
                competing_writer.rollback()
            initialize_locked(conn)
        monkeypatch.setattr(storage, "_initialize_locked", check_writer_reserved)
        with closing(sqlite3.connect(source)) as conn:
            conn.row_factory = sqlite3.Row
            storage.initialize(conn)
            assert not conn.in_transaction
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                storage.initialize(conn)
                assert conn.in_transaction  # Never commit the caller's transaction.


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_unavailable_record_target_does_not_block_healthy_target_or_hold_source_sqlite(tmp_path, monkeypatch, journal_mode):
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: journal_mode)
    unexpected_requests = []
    def unexpected_request(client, path, **kwargs):
        unexpected_requests.append(path)
        raise AssertionError(f"unexpected unmocked peer request: {path}")
    monkeypatch.setattr(PeerRunsHTTPClient, "_request", unexpected_request)
    home, secret = "install:home", b"test-work-record-target-secret-not-real"
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: home)
    source = tmp_path / "source.db"
    targets, members = {}, []
    for index in range(2):
        member, installation = f"member-{index}", f"install:target-{index}"
        url = f"http://127.0.0.1:{9800 + index}"
        catalog = peer.GatewayRoomCatalog.from_mapping(peer.catalog_mapping(
            installation_id=installation, target_profile="default", persistent_process=True))
        members.append({"member_id": member, "profile": "default", "handle": member, "target": {
            "kind": "peer", "peer_id": member, "installation_id": installation, "profile": "default",
            "capability_digest": catalog.catalog_digest}})
        token = peer.issue_room_grant(secret, grant_id=f"grant-{index}", room_id="room", home_install_id=home,
                                     authority_gateway_id=home, authority_epoch=1, member_id=member,
                                     target_install_id=installation, target_profile="default",
                                     execution_policy_digest=catalog.execution_policy.policy_digest,
                                     permissions=peer.invitation_permissions(True, True))
        target = tmp_path / f"target-{index}.db"
        claims = peer.decode_room_grant(secret, token, permission=records.PERMISSION)
        rooms.reserve_peer_room(target, claims=claims, expires_at=claims["status_expires_at"])
        targets[url] = (target, installation)
        links.save_room_link(source, links.make_stored_link(room_id="room", member_id=member, target_url=url,
                             target_profile="default", grant=token, catalog=catalog, cancellation_scope_id="cancel", trace_id="trace"))
    rooms.create_room(source, room_id="room", name="Workshop", members=members, authority_gateway_id=home)
    rooms.append_event(source, room_id="room", event_id="hello", kind="message.user",
                       actor={"kind": "user", "id": "owner"}, payload={"text": "Hello"},
                       authority_gateway_id=home, authority_epoch=1)
    driver.admit_task(source, driver.TaskIdentity("room", "task", "thread", "turn"),
                      payload={"target_profile": "default", "target_member_id": "member-0", "prompt": "private", "source_event_seq": 1},
                      clock=lambda: 100)
    def copy_history(client, *, grant, target_profile, **body):
        return replicas.ingest_page(targets[client.base_url][0], **body)
    monkeypatch.setattr(PeerRunsHTTPClient, "replicate_page", copy_history)
    pub = HostedRoomReplicationPublisher(source)
    for member in members:
        pub._publish_one(("room", member["member_id"]))
    with closing(sqlite3.connect(source)) as conn:
        actual_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        # Keep the vulnerable-runtime safety fallback, including in the WAL case.
        expected_mode = "delete" if hermes_state_wal.is_sqlite_wal_reset_vulnerable() else journal_mode
        assert actual_mode == expected_mode
    blocked, release, healthy = threading.Event(), threading.Event(), threading.Event()
    def copy_records(client, *, grant, target_profile, record):
        target, installation = targets[client.base_url]
        if installation.endswith("-1"):
            assert blocked.wait(3)  # Prove progress during the other transport's outage.
        with closing(sqlite3.connect(source, timeout=1)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")  # transport must not hold a source transaction
        if installation.endswith("-0"):
            blocked.set()
            release.wait()  # Only test cleanup releases the unavailable transport.
            raise PeerRunsHTTPError("unavailable", retryable=True, ambiguous=True)
        assert not release.is_set()
        result = records.ingest(target, record=record, token=grant, secret=secret,
                                target_install_id=installation, target_profile=target_profile)
        healthy.set()
        return result
    monkeypatch.setattr(PeerRunsHTTPClient, "replicate_work_records", copy_records)
    pub.start()
    try:
        assert blocked.wait(5), (pub.status(), unexpected_requests)
        assert healthy.wait(3), (pub.status(), unexpected_requests)
        assert pub.status()["workers"] == 2
        assert unexpected_requests == []
    finally:
        release.set()
        assert pub.stop(timeout=5)

    states = {row["target_install_id"]: row["status"] for row in pub.status()["work_records"]}
    assert states == {"install:target-0": "unavailable", "install:target-1": "acked"}


def test_rpc_invitation_work_record_opt_in_is_explicit(tmp_path, monkeypatch):
    import tui_gateway.server as server
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_run_idempotency_store", type("Durable", (), {"durable": True})(), raising=False)
    body = {"room_id": "room", "home_install_id": "home", "authority_gateway_id": "home", "authority_epoch": 1,
            "member_id": "peer", "replication": True, "work_records": True}
    reply = server._methods["groups.peer.invite"](1, body)
    assert "error" not in reply, reply
    claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), reply["result"]["grant"], permission=records.PERMISSION)
    assert records.PERMISSION in claims["permissions"]
    assert reply["result"]["work_records_version"] == records.VERSION
    for value in ("yes", 1, None):
        assert "error" in server._methods["groups.peer.invite"](2, {**body, "work_records": value})
    assert "error" in server._methods["groups.peer.invite"](3, {**body, "replication": False})
