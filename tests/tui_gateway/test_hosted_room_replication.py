"""Real source/replica SQLite stores behind the production HTTP client contract."""

import copy
import io
import json
import multiprocessing
import sqlite3
import threading
import time
import urllib.error
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import hosted_room_link_records
from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from gateway.hosted_room_replica_ingress import ingest_granted_page
from tui_gateway import hosted_room_replication as publisher
from tui_gateway.hosted_room_service import HostedRoomService


SECRET = b"publisher-test-secret-not-a-real-credential"
HOME = "install:publisher"
TARGET = "install:participant"
KEY = ("room", "reviewer")


def append(db, event_id):
    return rooms.append_event(
        db, room_id="room", event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "owner"}, payload={"text": event_id},
        authority_gateway_id=HOME, authority_epoch=1,
    )


def save_link(db, *, member_id="reviewer", profile="reviewer", permissions=("replicate",),
              target=TARGET, url="http://127.0.0.1:9876", **grant_overrides):
    catalog = peer.GatewayRoomCatalog.from_mapping(peer.catalog_mapping(
        installation_id=target, target_profile=profile, persistent_process=True,
    ))
    fields = dict(
        grant_id=f"grant-{member_id}", room_id="room", home_install_id=HOME,
        authority_gateway_id=HOME, authority_epoch=1, member_id=member_id,
        target_install_id=target, target_profile=profile, permissions=permissions,
        execution_policy_digest=catalog.execution_policy.policy_digest,
        issued_at=time.time() - 1, ttl_seconds=600, status_ttl_seconds=3600,
    )
    fields.update(grant_overrides)
    token = peer.issue_room_grant(SECRET, **fields)
    link = links.make_stored_link(
        room_id="room", member_id=member_id, target_url=url, target_profile=profile,
        grant=token, catalog=catalog, cancellation_scope_id="cancel", trace_id="trace",
    )
    links.save_room_link(db, link)
    return link


class HTTP:
    """Mock only urllib I/O; exercise serialization, scoped auth and real ingress."""

    def __init__(self, target):
        self.target = target
        self.requests = []
        self.before = None
        self.lose_ack = False
        self.error = None
        self.reply_transform = None

    def __call__(self, request, *, timeout):
        assert 0 < timeout <= 3
        assert request.method == "POST"
        assert request.full_url.endswith("/v1/room-members/replica")
        authorization = request.get_header("Authorization")
        assert authorization.startswith("HermesRoom ")
        token = authorization.removeprefix("HermesRoom ")
        body = json.loads(request.data)
        self.requests.append((request.full_url, copy.deepcopy(body)))
        if self.before:
            self.before(request, body)
        if self.error:
            code, error_code = self.error
            raise urllib.error.HTTPError(
                request.full_url, code, "controlled failure", {},
                io.BytesIO(json.dumps({"error": {"code": error_code}}).encode()),
            )
        claims = peer.decode_room_grant(SECRET, token, permission="replicate")
        result = ingest_granted_page(
            self.target, token=token, secret=SECRET, target_install_id=claims["target_install_id"],
            target_profile=claims["target_profile"], **body,
        )
        if self.lose_ack:
            self.lose_ack = False
            raise TimeoutError("ack lost after durable ingest")
        if self.reply_transform:
            result = self.reply_transform(result)
        return io.BytesIO(json.dumps({"object": "hermes.room_member.replica", **result}).encode())


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    catalog = peer.catalog_mapping(installation_id=TARGET, target_profile="reviewer", persistent_process=True)
    members = [
        {"member_id": "writer", "handle": "writer", "profile": "default", "target": {"kind": "local", "profile": "default"}},
        {"member_id": "reviewer", "handle": "reviewer", "profile": "reviewer", "target": {
            "kind": "peer", "peer_id": "participant", "installation_id": TARGET,
            "profile": "reviewer", "capability_digest": catalog["catalog_digest"],
        }},
    ]
    rooms.create_room(source, room_id="room", name="Workshop", members=members, authority_gateway_id=HOME)
    append(source, "hello")
    link = save_link(source)
    claims = peer.decode_room_grant(SECRET, link.grant, permission="replicate")
    rooms.reserve_peer_room(target, claims=claims, expires_at=claims["status_expires_at"])
    http = HTTP(target)
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", http)
    return SimpleNamespace(source=source, target=target, link=link, http=http)


def state(pub):
    return pub.status("room")["routes"][0]


def retire_routes(db):
    # Model the source-side completion of successful target revocation.
    scope = dict(room_id="room", authority_gateway_id=HOME, authority_epoch=1)
    hosted_room_link_records.begin_room_link_retirement(db, **scope)
    hosted_room_link_records.complete_room_link_retirement(db, **scope)
    hosted_room_link_records.delete_room_link_records(db, room_id="room")


def add_profile(pair, *, target=TARGET):
    members = rooms.room_state(pair.source, room_id="room")["members"]
    catalog = peer.catalog_mapping(installation_id=target, target_profile="default", persistent_process=True)
    members.append({"member_id": "z-other", "handle": "other", "profile": "default", "target": {
        "kind": "peer", "peer_id": "participant-two", "installation_id": target,
        "profile": "default", "capability_digest": catalog["catalog_digest"],
    }})
    with sqlite3.connect(pair.source) as conn:
        conn.execute("UPDATE hosted_rooms SET members_json=? WHERE room_id='room'", (json.dumps(members),))
    other = save_link(pair.source, member_id="z-other", profile="default", target=target)
    claims = peer.decode_room_grant(SECRET, other.grant, permission="replicate")
    rooms.reserve_peer_room(pair.target, claims=claims, expires_at=claims["status_expires_at"])
    return other


def _child_publish(source, target, entered, release, results):
    """Spawn a separate gateway process, retaining the real OS lock and SQLite."""
    from hermes_cli import urllib_security
    rooms.local_authority_gateway_id = lambda: HOME
    publisher.PAGE_LIMIT = 1
    http = HTTP(target)

    def wait_for_parent(*_):
        entered.set()
        assert release.wait(3)

    http.before, http.lose_ack = wait_for_parent, True
    urllib_security.open_credentialed_url = http
    pub = publisher.HostedRoomReplicationPublisher(source)
    pub._publish_one(KEY)
    results.put({"requests": http.requests, "status": pub.status()})


def test_restart_checkpoint_suppresses_unchanged_http_and_remains_passive(pair):
    first = publisher.HostedRoomReplicationPublisher(pair.source)
    assert first._publish_one(KEY) is False
    assert state(first)["acked_seq"] == 1
    second = publisher.HostedRoomReplicationPublisher(pair.source)
    for _ in range(3):
        assert second._publish_one(KEY) is False
    assert len(pair.http.requests) == 1
    append(pair.source, "later")
    second._publish_one(KEY)
    assert pair.http.requests[-1][1]["page"]["events"][0]["seq"] == 2
    assert replicas.replica_state(pair.target, room_id="room")["safety_status"] == "passive"
    assert rooms.list_rooms(pair.target) == []
    assert second.status()["source_loss_safe"] is False
    assert pair.link.grant not in repr(second.status())
    assert pair.link.grant not in repr(second._load_route(KEY))


def test_ack_loss_restarts_with_identical_page_despite_new_source_events(pair):
    pair.http.lose_ack = True
    first = publisher.HostedRoomReplicationPublisher(pair.source)
    first._publish_one(KEY)
    assert state(first)["acked_seq"] == 0
    assert state(first)["status"] == "unavailable"
    assert replicas.replica_state(pair.target, room_id="room")["last_seq"] == 1
    append(pair.source, "after-lost-ack")
    restarted = publisher.HostedRoomReplicationPublisher(pair.source)
    restarted._publish_one(KEY)
    assert pair.http.requests[0] == pair.http.requests[1]
    restarted._publish_one(KEY)
    assert state(restarted)["acked_seq"] == 2


def test_typed_gap_resets_cursor_but_quarantine_is_permanent(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    append(pair.source, "second")
    pair.http.error = (409, "room_replica_gap")
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 0
    assert state(pub)["status"] == "replica_gap"
    pair.http.error = None
    pub._publish_one(KEY)
    assert [e["seq"] for e in pair.http.requests[-1][1]["page"]["events"]] == [1, 2]
    append(pair.source, "third")
    pair.http.error = (409, "invalid_room_replica")
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 2
    assert state(pub)["status"] == "replica_rejected"
    restarted = publisher.HostedRoomReplicationPublisher(pair.source)
    before = len(pair.http.requests)
    restarted._publish_one(KEY)
    assert len(pair.http.requests) == before


@pytest.mark.parametrize("error", [(401, "invalid_room_grant"), (403, "room_reauthorization_required")])
def test_auth_rejection_stops_until_new_durable_grant(pair, error):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.error = error
    pub._publish_one(KEY)
    assert state(pub)["status"] == "needs_reauthorization"
    pub._publish_one(KEY)
    assert len(pair.http.requests) == 1
    save_link(pair.source, grant_id="replacement")
    pair.http.error = None
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 1


@pytest.mark.parametrize("overrides", [
    {"permissions": ("dispatch", "status", "stop")}, {"member_id": "absent"},
    {"authority_gateway_id": "install:other"}, {"authority_epoch": 2},
    {"home_install_id": "install:other"}, {"target": "install:other"},
])
def test_ineligible_grants_never_send(pair, overrides):
    room = rooms.room_state(pair.source, room_id="room")
    pair.source = pair.source.with_name("ineligible.db")
    rooms.create_room(
        pair.source, room_id="room", name=room["name"], members=room["members"], authority_gateway_id=HOME,
    )
    append(pair.source, "hello")
    link = save_link(pair.source, **overrides)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    assert not pub._publish_one((link.room_id, link.member_id))
    assert pair.http.requests == []


def test_hint_does_not_trust_unsigned_timing(pair):
    encoded, signature = peer._split_token(pair.link.grant)
    hint = json.loads(encoded)
    hint.update(issued_at=0, expires_at=1, status_expires_at=2)
    import base64
    token = base64.urlsafe_b64encode(json.dumps(hint).encode()).decode().rstrip("=")
    token += "." + base64.urlsafe_b64encode(signature).decode().rstrip("=")
    link = replace(pair.link, grant=token)
    room = rooms.room_state(pair.source, room_id="room")
    assert publisher._eligible(link, room, HOME)


@pytest.mark.parametrize("replacement", ["grant", "url", "authority", "remove"])
def test_network_race_cannot_advance_changed_route(pair, replacement):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)

    def change_route(_request, _body):
        pair.http.before = None
        if replacement == "grant":
            save_link(pair.source, grant_id="new-generation")
        elif replacement == "url":
            links.save_room_link(pair.source, replace(pair.link, target_url="http://127.0.0.1:9877"))
        elif replacement == "authority":
            with sqlite3.connect(pair.source) as conn:
                conn.execute("UPDATE hosted_rooms SET authority_gateway_id='install:other' WHERE room_id='room'")
        else:
            retire_routes(pair.source)

    pair.http.before = change_route
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 0
    if replacement in {"grant", "url"}:
        pub._publish_one(KEY)
        assert state(pub)["acked_seq"] == 1
        assert pair.http.requests[-1][1]["page"]["events"][0]["seq"] == 1
    else:
        pub._publish_one(KEY)
        assert len(pair.http.requests) == 1


def test_disband_without_live_route_reports_stopped_not_delivered(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    retire_routes(pair.source)
    rooms.disband_room(pair.source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
    pub._scan(time.monotonic())
    assert state(pub)["status"] == "stopped_route_removed"
    assert state(pub)["acked_seq"] == 1
    assert state(pub)["source_latest_seq"] == 2
    assert state(pub)["delivery_unconfirmed"] is True
    assert replicas.replica_state(pair.target, room_id="room")["disbanded_at"] is None
    assert len(pair.http.requests) == 1


def test_live_route_delivers_real_terminal_event(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    rooms.disband_room(pair.source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
    pub._publish_one(KEY)
    assert pair.http.requests[-1][1]["page"]["events"][-1]["kind"] == "room.disbanded"
    assert replicas.replica_state(pair.target, room_id="room")["disbanded_at"] is not None


def test_two_workers_isolate_unavailable_target_and_stop_without_thread_growth(pair):
    add_profile(pair, target="install:independent-participant")
    blocked, release, healthy = threading.Event(), threading.Event(), threading.Event()

    def unavailable(request, body):
        if "/p/reviewer/" in request.full_url:
            blocked.set()
            assert release.wait(3)
            raise TimeoutError("offline")
        healthy.set()

    pair.http.before = unavailable
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub.start()
    try:
        assert blocked.wait(3)
        assert healthy.wait(3)
        assert pub.status()["workers"] == 2
        assert not pub.stop(timeout=0)
        original = tuple(pub._threads)
        pub.start()
        assert tuple(pub._threads) == original
    finally:
        release.set()
        assert pub.stop(timeout=5)
    profiles = {row["target_profile"] for row in pub.status()["routes"]}
    assert profiles == {"default", "reviewer"}


def test_service_publisher_progresses_while_policy_lock_is_held(pair):
    service = HostedRoomService(SimpleNamespace(), db_path=pair.source)
    received = threading.Event()
    pair.http.before = lambda *_: received.set()
    # Keep this lifecycle test independent of the execution driver's work loop.
    service.runtime = SimpleNamespace(start=lambda: None, stop=lambda **kw: True)
    with service._policy_lock:
        service.start()
        try:
            assert received.wait(3)
        finally:
            assert service.stop(timeout=5)
    assert state(service.replication)["acked_seq"] == 1


def test_fixed_page_budget_and_round_robin_rotate_backlog(pair, monkeypatch):
    monkeypatch.setattr(publisher, "PAGE_LIMIT", 1)
    append(pair.source, "second")
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    assert pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 1
    assert not pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 2
    assert all(len(body["page"]["events"]) == 1 for _, body in pair.http.requests)


def test_source_replica_never_becomes_publishable_authority(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    links.save_room_link(pair.target, pair.link)
    target_publisher = publisher.HostedRoomReplicationPublisher(pair.target)
    assert not target_publisher._publish_one(KEY)
    assert len(pair.http.requests) == 1
    assert rooms.list_rooms(pair.target) == []


def test_empty_page_lost_ack_retries_empty_before_new_events(pair):
    with sqlite3.connect(pair.source) as conn:
        conn.execute("DELETE FROM hosted_room_events WHERE room_id='room'")
        conn.execute("UPDATE hosted_rooms SET next_seq=1, event_bytes=0 WHERE room_id='room'")
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.lose_ack = True
    pub._publish_one(KEY)
    append(pair.source, "first-after-empty-page")
    publisher.HostedRoomReplicationPublisher(pair.source)._publish_one(KEY)
    assert pair.http.requests[0] == pair.http.requests[1]
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 1


@pytest.mark.parametrize("changes", [
    {"room_id": "another"}, {"authority": {"gateway_id": "install:other", "epoch": 1}},
    {"stored_seq": True}, {"stored_seq": 999}, {"stored_seq": 0},
])
def test_malformed_ack_never_advances_or_retries_forever(pair, changes):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.reply_transform = lambda result: {**result, **changes}
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 0
    assert state(pub)["status"] == "invalid_ack"
    pub._publish_one(KEY)
    assert len(pair.http.requests) == 1


def test_source_quarantine_prevents_publishing(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    with sqlite3.connect(pair.source) as conn:
        conn.execute("""INSERT INTO hosted_room_quarantine(room_id, reason, detected_at)
            VALUES ('room', 'test-unverified-lineage', ?)""", (time.time(),))
    assert not pub._publish_one(KEY)
    assert pair.http.requests == []


def test_in_memory_only_service_route_is_not_published(pair):
    retire_routes(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._scan(time.monotonic())
    assert list(pub._routes) == []
    assert not pub._publish_one(KEY)
    assert pair.http.requests == []


def test_removal_diagnostic_tail_stays_bounded(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    with sqlite3.connect(pair.source) as conn:
        conn.executemany(f"""INSERT INTO {publisher._TABLE}
            (room_id,member_id,generation,target_install_id,target_profile,
             authority_gateway_id,authority_epoch,status,updated_at)
            VALUES (?, 'gone', 'old', 'target', 'default', 'home', 1, 'pending', 0)""",
                         [(f"removed-{i}",) for i in range(links.MAX_LINKS + 10)])
    pub._scan(time.monotonic())
    assert len(pub.status()["routes"]) == links.MAX_LINKS + 1
    assert state(pub)["status"] == "acked"


def test_scheduler_honors_due_time_between_scans_and_rotates(pair, monkeypatch):
    other = save_link(pair.source, member_id="second", profile="default")
    other_key = (other.room_id, other.member_id)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    now, waits = [5.0], []
    pub._scan(now[0])
    pub._due = {KEY: 5.2, other_key: 5.2}
    monkeypatch.setattr(publisher.time, "monotonic", lambda: now[0])

    def wait(*, timeout):
        waits.append(timeout)
        now[0] += timeout

    monkeypatch.setattr(pub._condition, "wait", wait)
    assert pub._take() == KEY
    assert waits == [pytest.approx(0.2)]
    pub._inflight.remove(KEY)
    assert pub._take() == other_key


def test_health_status_changes_cannot_reactivate_same_rejected_grant(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.error = (401, "invalid_room_grant")
    pub._publish_one(KEY)
    links.mark_room_link_status(pair.source, room_id="room", member_id="reviewer", status="needs_reauthorization")
    pub._publish_one(KEY)
    links.mark_room_link_status(pair.source, room_id="room", member_id="reviewer", status="ready")
    pub._publish_one(KEY)
    assert len(pair.http.requests) == 1
    assert state(pub)["status"] == "needs_reauthorization"


def test_shared_target_two_processes_lost_acks_growth_and_replacement(pair, monkeypatch):
    other = add_profile(pair)
    other_key = (other.room_id, other.member_id)
    monkeypatch.setattr(publisher, "PAGE_LIMIT", 1)
    second = publisher.HostedRoomReplicationPublisher(pair.source)
    context = multiprocessing.get_context("spawn")
    entered, release, results = context.Event(), context.Event(), context.Queue()
    child = context.Process(target=_child_publish, args=(pair.source, pair.target, entered, release, results))
    child.start()
    try:
        assert entered.wait(8)
        append(pair.source, "source-grows-during-http")
        assert not second._publish_one(other_key)
        assert pair.http.requests == []  # the other process owns this target's lock
        release.set()
        first = results.get(timeout=8)
        child.join(8)
        assert child.exitcode == 0
    finally:
        release.set()
        if child.is_alive():
            child.terminate()
            child.join(5)
        results.close()
        results.join_thread()
    assert first["status"]["routes"][0]["status"] == "unavailable"
    second._publish_one(other_key)
    assert pair.http.requests[-1][1] == first["requests"][0][1]
    assert "/p/reviewer/" not in pair.http.requests[-1][0]  # authorized alternate
    rooms.rename_room(pair.source, room_id="room", event_id="rename", name="Current name")
    pair.http.lose_ack = True
    second._publish_one(other_key)
    old_pending = pair.http.requests[-1][1]
    append(pair.source, "after-second-lost-ack")
    replacement = save_link(pair.source, grant_id="replacement-primary")
    claims = peer.decode_room_grant(SECRET, replacement.grant, permission="replicate")
    rooms.reserve_peer_room(pair.target, claims=claims, expires_at=claims["status_expires_at"])
    restarted = publisher.HostedRoomReplicationPublisher(pair.source)
    restarted._publish_one(KEY)
    assert pair.http.requests[-1][1] == old_pending
    for _ in range(4):
        restarted._publish_one(other_key)
    replica = replicas.replica_state(pair.target, room_id="room")
    assert replica["last_seq"] == rooms.room_state(pair.source, room_id="room")["latest_seq"] == 4
    assert replica["name"] == "Current name"
    assert replica["safety_status"] == "passive"
    rows = restarted.status()["routes"]
    assert all(row["status"] not in publisher._BLOCKED for row in rows)
    assert {row["target_acked_seq"] for row in rows} == {4}
    assert {row["selected_member_id"] for row in rows} == {KEY[1]}
    assert rows[1]["role"] == "alternate"
    before = len(pair.http.requests)
    second._publish_one(other_key)
    restarted._publish_one(KEY)
    assert len(pair.http.requests) == before


def test_revoked_primary_uses_authorized_alternate_shared_pending(pair):
    other = add_profile(pair)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.lose_ack = True
    pub._publish_one(KEY)
    original = pair.http.requests[-1][1]
    claims = peer.decode_room_grant(SECRET, pair.link.grant, permission="replicate")
    rooms.revoke_room_grant_scope(pair.target, claims=claims, expires_at=claims["status_expires_at"])
    links.mark_room_link_status(pair.source, room_id="room", member_id="reviewer", status="needs_reauthorization")
    append(pair.source, "after-revoke")
    pub._publish_one((other.room_id, other.member_id))
    assert pair.http.requests[-1][1] == original
    pub._publish_one((other.room_id, other.member_id))
    assert replicas.replica_state(pair.target, room_id="room")["last_seq"] == 2
    assert pub.status()["routes"][1]["role"] == "selected"


def test_old_grants_filtered_at_scan_without_room_reads_and_removal_still_detected(pair, monkeypatch):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    save_link(pair.source, permissions=("status", "dispatch"))
    reads = []
    monkeypatch.setattr(rooms, "room_state", lambda *a, **k: reads.append(k))
    pub._scan(time.monotonic())
    assert list(pub._routes) == []
    assert pub._load_route(KEY) is None
    assert reads == []
    retire_routes(pair.source)
    pub._scan(time.monotonic())
    assert state(pub)["status"] == "stopped_route_removed"


@pytest.mark.parametrize("error", [sqlite3.OperationalError("database is locked"), ValueError("invalid schema")])
def test_optional_publisher_initialization_failure_does_not_disable_group_controls(pair, monkeypatch, error):
    def fail(_db):
        raise error
    monkeypatch.setattr("tui_gateway.hosted_room_service.HostedRoomReplicationPublisher", fail)
    service = HostedRoomService(SimpleNamespace(), db_path=pair.source)
    assert service.replication is None
    assert service.status()["replication"]["error"] == "publisher_initialization_failed"
    assert service.status()["replication"]["routes"] is None
    created = service.create_room(
        room_id="ordinary", name="Ordinary", members=rooms.room_state(pair.source, room_id="room")["members"])
    assert created["room_id"] == "ordinary"
    assert service.stop(timeout=1)


def test_publisher_status_database_error_is_explicit_not_a_controls_failure(pair, monkeypatch):
    service = HostedRoomService(SimpleNamespace(), db_path=pair.source)
    def fail(*a, **kw):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(publisher, "open_sqlite", fail)
    result = service.status()["replication"]
    assert result["error"] == "publisher_status_unavailable"
    assert result["routes"] is None
    assert result["source_loss_safe"] is False


def test_required_room_store_errors_are_not_hidden_as_optional_replication_failure(pair, monkeypatch):
    def fail(*a, **kw):
        raise sqlite3.DatabaseError("required room store failure")
    monkeypatch.setattr(rooms, "prune_disbanded_rooms", fail)
    with pytest.raises(sqlite3.DatabaseError, match="required room store failure"):
        HostedRoomService(SimpleNamespace(), db_path=pair.source)


@pytest.mark.parametrize("failed_index", [0, 1])
def test_partial_publisher_start_failure_keeps_runtime_and_cleanup_working(tmp_path, monkeypatch, failed_index):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
    service = HostedRoomService(SimpleNamespace(), db_path=tmp_path / "state.db")
    original_start = threading.Thread.start

    def fail_one(thread):
        if thread.name == f"hosted-room-replication-{failed_index}":
            raise RuntimeError("can't start new thread")
        return original_start(thread)

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(threading.Thread, "start", fail_one)
            service.start()
            assert service.runtime.status()["running"] is True
            assert service.replication.status()["error"] == "publisher_start_failed"
            assert service.stop(timeout=2)
        assert not any(thread.is_alive() for thread in service.replication._threads)
        service.start()
        assert service.runtime.status()["running"] is True
        assert service.replication.status()["error"] is None
        assert service.replication.status()["workers"] == publisher.WORKERS
    finally:
        assert service.stop(timeout=2)


@pytest.mark.parametrize("code", [401, 409])
def test_shared_target_blocked_route_can_only_switch_to_another_authorized_route(pair, code):
    other = add_profile(pair)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.error = (code, "invalid_room_grant" if code == 401 else "invalid_room_replica")
    pub._publish_one(KEY)
    original = pair.http.requests[-1][1]
    pair.http.error = None
    pub._publish_one((other.room_id, other.member_id))
    assert pair.http.requests[-1][1] == original
    assert "/p/reviewer/" not in pair.http.requests[-1][0]
    rows = pub.status()["routes"]
    assert rows[0]["role"] == "blocked"
    assert rows[0]["acked_seq"] == 0
    assert rows[1]["target_acked_seq"] == 1
    assert rows[1]["role"] == "selected"
