"""Active conversation traffic must not starve an anchored passive record."""

import io
import json
import sqlite3
import urllib.error

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_peer as peer
from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from tests.tui_gateway.test_hosted_room_replication import (
    HOME, KEY, SECRET, TARGET, add_profile, append, pair, save_link,  # noqa: F401
)
from tui_gateway import hosted_room_replication as publisher


@pytest.fixture
def copying(pair, monkeypatch):
    link = save_link(pair.source, permissions=("replicate", records.PERMISSION))
    claims = peer.decode_room_grant(SECRET, link.grant, permission=records.PERMISSION)
    rooms.reserve_peer_room(pair.target, claims=claims, expires_at=claims["status_expires_at"])
    pair.records = []
    pair.lose_record_ack = False
    pair.record_error = None

    def transport(request, *, timeout):
        # Neither history nor record transport may retain the capture write lock.
        with sqlite3.connect(pair.source, timeout=0.1) as conn:
            conn.execute("BEGIN IMMEDIATE")
        if not request.full_url.endswith("/work-records"):
            return pair.http(request, timeout=timeout)
        record = json.loads(request.data)["record"]
        pair.records.append(record)
        if pair.record_error:
            raise pair.record_error
        result = records.ingest(
            pair.target, record=record,
            token=request.get_header("Authorization").removeprefix("HermesRoom "),
            secret=SECRET, target_install_id=TARGET, target_profile="reviewer",
        )
        if pair.lose_record_ack:
            pair.lose_record_ack = False
            raise TimeoutError("record ACK lost after committed target write")
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", transport)
    pair.pub = publisher.HostedRoomReplicationPublisher(pair.source)
    return pair


def successor_copying(pair, monkeypatch, preferred):
    """Extend the same fairness transport with enrolled v2 and exact profile auth."""
    from gateway import hosted_room_replica_retirement as retirement
    from gateway.hosted_room_passive_protocol import passive_capabilities
    from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, transfer
    record_profiles(pair)
    with rooms._transaction(pair.source, immediate=True) as conn:
        records.prepare_delivery_locked(conn, room_id="room", target_install_id=TARGET,
            route_generation="old", local_gateway_id=HOME, through_seq=1)
        old = dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone())
    transfer(pair.source, HOME)
    for member, profile in (("reviewer", "reviewer"), ("z-other", "default")):
        permissions = ("replicate",) if member == "reviewer" and preferred == "history_only" else ("replicate", records.PERMISSION)
        link = save_link(pair.source, member_id=member, profile=profile, permissions=permissions,
            home_install_id=SUCCESSOR, authority_gateway_id=SUCCESSOR, authority_epoch=2)
        claims = peer.decode_room_grant(SECRET, link.grant, permission="replicate")
        rooms.reserve_peer_room(pair.target, claims=claims, expires_at=claims["status_expires_at"])
    entry = retirement.prepare_home_enrollment(pair.source, room_id="room", target_install_id=TARGET,
        endpoint="http://127.0.0.1:9876", local_gateway_id=SUCCESSOR, secret=SECRET)
    proof = retirement.enroll_target(pair.target, enrollment=entry, target_install_id=TARGET,
        **retirement.home_enrollment_history(pair.source, enrollment_id=entry["enrollment_id"]))
    assert retirement.confirm_home_enrollment(pair.source, enrollment_id=entry["enrollment_id"], proof=proof)
    pair.records, pair.attempts, pair.probes = [], [], []
    pair.recovered = False

    def transport(request, *, timeout):
        with sqlite3.connect(pair.source, timeout=0.1) as conn:
            conn.execute("BEGIN IMMEDIATE")
        token = request.get_header("Authorization").removeprefix("HermesRoom ")
        claims = peer.decode_room_grant(SECRET, token, permission="replicate")
        member = claims["member_id"]
        if request.full_url.endswith("/capabilities"):
            pair.probes.append(member)
            capabilities = passive_capabilities()
            if member == "reviewer" and preferred == "unsupported":
                capabilities["work_record_versions"] = [1]
            body = {"passive_replication": capabilities}
            if member == "reviewer" and preferred == "missing":
                body = {}
            if member == "reviewer" and preferred == "malformed":
                capabilities["work_record_versions"] = "2"
            return io.BytesIO(json.dumps(body).encode())
        if not request.full_url.endswith("/work-records"):
            return pair.http(request, timeout=timeout)
        record = json.loads(request.data)["record"]
        pair.attempts.append(member)
        pair.records.append(record)
        if member == "reviewer" or not pair.recovered:
            code = 403 if member == "reviewer" and preferred == "refused" else 503
            raise urllib.error.HTTPError(request.full_url, code, "unavailable", {},
                io.BytesIO(b'{"error":{"code":"unavailable"}}'))
        reply = records.ingest(pair.target, record=record, token=token, secret=SECRET,
            target_install_id=TARGET, target_profile=claims["target_profile"])
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", transport)
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: SUCCESSOR)
    return old


@pytest.mark.parametrize("preferred", ["unavailable", "refused", "history_only", "unsupported", "missing", "malformed"])
def test_successor_busy_history_recovers_alternate_without_moving_pending_anchor(pair, monkeypatch, preferred):
    from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR
    old = successor_copying(pair, monkeypatch, preferred)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    # One covered immutable current record; both profiles first encounter loss.
    for member in ("reviewer", "z-other", "reviewer", "z-other"):
        route = pub._load_route(("room", member))
        pub._checkpoint(route)
        pub._publish_locked(route)
    with rooms._transaction(pair.source) as conn:
        frozen = dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE} WHERE producer_epoch=2").fetchone())
    assert frozen["status"] != "acked"
    pair.recovered = True
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    before = len(pair.probes)
    for turn in range(6):
        rooms.append_event(pair.source, room_id="room", event_id=f"busy-v2-{turn}", kind="message.user",
            actor={"kind": "user", "id": "owner"}, payload={"text": "growing"},
            authority_gateway_id=SUCCESSOR, authority_epoch=2)
        pub._publish_one(("room", "z-other" if turn % 2 == 0 else "reviewer"))
        status = next(r for r in pub.status()["work_records"] if r["producer_epoch"] == 2)
        if status["status"] == "acked":
            break
    assert status["status"] == "acked", pair.attempts
    assert pair.attempts[-1] == "z-other"
    assert len(pair.probes) == before  # Durable route-local capability cache.
    assert all(records.encode(r) == frozen["record_json"] for r in pair.records)
    with rooms._transaction(pair.source) as conn:
        assert dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE} WHERE producer_epoch=1").fetchone()) == {
            **old, "disposition": "superseded_authority"}
        assert conn.execute(f"SELECT record_json FROM {records.PENDING_TABLE} WHERE producer_epoch=2").fetchone()[0] == frozen["record_json"]


def add_task(source, suffix):
    append(source, f"message-{suffix}")
    seq = rooms.room_state(source, room_id="room")["latest_seq"]
    driver.admit_task(
        source, driver.TaskIdentity("room", f"task-{suffix}", "thread", f"turn-{suffix}"),
        payload={"prompt": "private task body", "source_event_seq": seq,
                 "target_profile": "reviewer", "target_member_id": "reviewer"},
        clock=lambda: 100,
    )


@pytest.mark.parametrize("turns", [5, 20, 100])
def test_busy_history_delivers_current_task_snapshots_without_quiet_turn(copying, turns):
    for turn in range(turns):
        add_task(copying.source, str(turn))
        copying.pub._publish_one(KEY)
        if turn >= 1:
            # With one page per turn, a healthy target must receive a snapshot
            # within the next turn, not wait indefinitely for global quiet.
            latest = rooms.room_state(copying.source, room_id="room")["latest_seq"]
            assert copying.records
            assert latest - copying.records[-1]["history"]["seq"] <= 2
    assert len(copying.records[-1]["tasks"]) >= turns - 2
    assert "private task body" not in json.dumps(copying.records)


def test_history_anchor_survives_new_events_and_lost_ack_across_restart(copying, monkeypatch):
    monkeypatch.setattr(publisher, "PAGE_LIMIT", 1)
    for turn in range(4):
        add_task(copying.source, str(turn))
    copying.http.lose_ack = True
    copying.pub._publish_one(KEY)
    with rooms._transaction(copying.source) as conn:
        pending = conn.execute(f"SELECT record_json FROM {records.PENDING_TABLE}").fetchone()
        assert pending is not None
        frozen = json.loads(pending[0])
    copying.pub = publisher.HostedRoomReplicationPublisher(copying.source)
    # New tasks/events on every retry must not move the frozen snapshot's anchor.
    for turn in range(frozen["history"]["seq"] + 2):
        add_task(copying.source, f"growing-{turn}")
        copying.pub._publish_one(KEY)
        if copying.records:
            break
    assert copying.records[0] == frozen
    assert copying.http.requests[0] == copying.http.requests[1]


def test_lost_record_ack_retries_identical_record_while_history_keeps_growing(copying):
    add_task(copying.source, "first")
    copying.pub._publish_one(KEY)
    copying.lose_record_ack = True
    add_task(copying.source, "second")
    copying.pub._publish_one(KEY)
    assert copying.records
    frozen = copying.records[-1]
    copying.pub = publisher.HostedRoomReplicationPublisher(copying.source)
    add_task(copying.source, "third")
    copying.pub._publish_one(KEY)
    assert copying.records[-1] == frozen
    assert copying.records[-2] == frozen
    assert copying.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(
        copying.source, room_id="room")["latest_seq"]
    add_task(copying.source, "fourth")
    copying.pub._publish_one(KEY)
    add_task(copying.source, "fifth")
    copying.pub._publish_one(KEY)
    assert copying.records[-1]["revision"] > frozen["revision"]


def test_quiet_history_suppresses_unchanged_records(copying):
    add_task(copying.source, "only")
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    assert copying.records
    delivered = list(copying.records)
    history = list(copying.http.requests)
    for _ in range(3):
        copying.pub._publish_one(KEY)
    assert copying.records == delivered
    assert copying.http.requests == history


def test_empty_group_history_is_created_before_first_record(copying):
    empty = copying.source.with_name("empty.db")
    members = rooms.room_state(copying.source, room_id="room")["members"]
    rooms.create_room(empty, room_id="room", name="Empty", members=members, authority_gateway_id=HOME)
    save_link(empty, permissions=("replicate", records.PERMISSION))
    copying.source = empty
    copying.pub = publisher.HostedRoomReplicationPublisher(empty)
    copying.pub._publish_one(KEY)
    assert copying.records == []
    assert copying.http.requests[-1][1]["page"]["cursor"] == 0
    copying.pub._publish_one(KEY)
    assert copying.records[-1]["history"]["seq"] == 0
    assert copying.pub.status()["work_records"][0]["status"] == "acked"


def test_unavailable_record_transport_does_not_stop_history(copying):
    copying.record_error = TimeoutError("record endpoint unavailable")
    for turn in range(5):
        add_task(copying.source, str(turn))
        copying.pub._publish_one(KEY)
    assert copying.records
    assert all(record == copying.records[0] for record in copying.records)
    assert copying.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(
        copying.source, room_id="room")["latest_seq"]


def record_profiles(pair):
    add_profile(pair)
    for member, profile in (("reviewer", "reviewer"), ("z-other", "default")):
        link = save_link(pair.source, member_id=member, profile=profile,
                         permissions=("replicate", records.PERMISSION))
        claims = peer.decode_room_grant(SECRET, link.grant, permission=records.PERMISSION)
        rooms.reserve_peer_room(pair.target, claims=claims, expires_at=claims["status_expires_at"])


@pytest.mark.parametrize("busy", [False, True], ids=["quiet-control", "continuous-history"])
def test_recovered_alternate_gets_a_work_attempt(pair, monkeypatch, busy):
    record_profiles(pair)
    recovered = False
    attempts, bodies = [], []

    def transport(request, *, timeout):
        if not request.full_url.endswith("/work-records"):
            return pair.http(request, timeout=timeout)
        token = request.get_header("Authorization").removeprefix("HermesRoom ")
        claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION)
        member = claims["member_id"]
        record = json.loads(request.data)["record"]
        attempts.append(member)
        bodies.append(record)
        if member == "reviewer" or not recovered:
            raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {},
                io.BytesIO(b'{"error":{"code":"unavailable"}}'))
        reply = records.ingest(pair.target, record=record, token=token, secret=SECRET,
                               target_install_id=TARGET, target_profile=claims["target_profile"])
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", transport)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    pub._publish_one(KEY)
    pub._publish_one(("room", "z-other"))
    assert attempts == ["reviewer", "z-other"]
    assert {row["work_record_status"] for row in pub.status()["routes"]} == {"unavailable"}
    frozen = bodies[0]
    recovered = True
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    start = len(attempts)
    for turn in range(6):
        if busy:
            append(pair.source, f"continuing-{turn}")
        pub._publish_one(("room", "z-other" if turn % 2 == 0 else "reviewer"))
        if pub.status()["work_records"][0]["status"] == "acked":
            break
    assert all(body == frozen for body in bodies)
    if busy:
        assert pair.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(
            pair.source, room_id="room")["latest_seq"]
    assert pub.status()["work_records"][0]["status"] == "acked", attempts[start:]


@pytest.mark.parametrize("busy", [False, True], ids=["quiet-control", "continuous-history"])
def test_recovered_work_route_with_stale_history_failure(pair, monkeypatch, busy):
    record_profiles(pair)
    add_task(pair.source, "anchored")
    recovered = False
    work_attempts, history_failures, bodies = [], [], []

    def unavailable(request):
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {},
            io.BytesIO(b'{"error":{"code":"unavailable"}}'))

    def transport(request, *, timeout):
        token = request.get_header("Authorization").removeprefix("HermesRoom ")
        is_work = request.full_url.endswith("/work-records")
        claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION if is_work else "replicate")
        member = claims["member_id"]
        if not is_work:
            if member == "z-other" and not recovered:
                history_failures.append(member)
                unavailable(request)
            return pair.http(request, timeout=timeout)
        record = json.loads(request.data)["record"]
        work_attempts.append(member)
        bodies.append(record)
        if member == "reviewer" or not recovered:
            unavailable(request)
        reply = records.ingest(pair.target, record=record, token=token, secret=SECRET,
                               target_install_id=TARGET, target_profile=claims["target_profile"])
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", transport)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    pub._publish_one(KEY)
    append(pair.source, "beta-history-failure")
    pub._publish_one(("room", "z-other"))
    assert work_attempts == ["reviewer", "z-other"]
    assert history_failures == ["z-other"]
    pub._publish_one(KEY)
    status = {row["member_id"]: row for row in pub.status()["routes"]}
    assert status["z-other"]["status"] == "unavailable"
    assert status["reviewer"]["status"] == "acked"
    assert {row["work_record_status"] for row in status.values()} == {"unavailable"}
    frozen = bodies[0]
    assert frozen["tasks"][0]["phase"] == "queued"
    assert pair.http.requests[-1][1]["page"]["cursor"] >= frozen["history"]["seq"]

    recovered = True
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    start = len(work_attempts)
    for turn in range(6):
        if busy:
            append(pair.source, f"busy-after-recovery-{turn}")
        pub._publish_one(("room", "z-other" if turn % 2 == 0 else "reviewer"))
        if pub.status()["work_records"][0]["status"] == "acked":
            break
    assert all(body == frozen for body in bodies)
    if busy:
        assert pair.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(
            pair.source, room_id="room")["latest_seq"]
    assert pub.status()["work_records"][0]["status"] == "acked", work_attempts[start:]


def test_unacknowledged_anchor_still_prefers_healthy_history_route(pair, monkeypatch):
    record_profiles(pair)
    for turn in range(4):
        add_task(pair.source, str(turn))
    failures = {"reviewer": 1, "z-other": 1}

    def transport(request, *, timeout):
        assert not request.full_url.endswith("/work-records")
        token = request.get_header("Authorization").removeprefix("HermesRoom ")
        member = peer.decode_room_grant(SECRET, token, permission="replicate")["member_id"]
        if failures[member]:
            failures[member] -= 1
            raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {},
                io.BytesIO(b'{"error":{"code":"unavailable"}}'))
        return pair.http(request, timeout=timeout)

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", transport)
    monkeypatch.setattr(publisher, "PAGE_LIMIT", 1)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    pub._publish_one(("room", "z-other"))
    assert failures == {"reviewer": 0, "z-other": 0}
    pub._publish_one(KEY)
    assert pair.http.requests[-1][1]["page"]["cursor"] == 1
    route = pub._select_route(pub._load_route(("room", "z-other")))
    assert route.key == KEY
    pub._publish_one(("room", "z-other"))
    assert pair.http.requests[-1][1]["page"]["cursor"] == 2


@pytest.mark.parametrize("bad_ack", [False, True], ids=["valid-work-ack", "blocked-work-ack"])
def test_no_deliverable_work_keeps_healthy_history_priority(pair, monkeypatch, bad_ack):
    add_profile(pair)  # Deliberately history-only alternate.
    link = save_link(pair.source, permissions=("replicate", records.PERMISSION))
    claims = peer.decode_room_grant(SECRET, link.grant, permission=records.PERMISSION)
    rooms.reserve_peer_room(pair.target, claims=claims, expires_at=claims["status_expires_at"])
    fail_first_primary_history = True
    history_attempts, work_attempts = [], []

    def transport(request, *, timeout):
        nonlocal fail_first_primary_history
        token = request.get_header("Authorization").removeprefix("HermesRoom ")
        is_work = request.full_url.endswith("/work-records")
        claims = peer.decode_room_grant(SECRET, token, permission=records.PERMISSION if is_work else "replicate")
        member = claims["member_id"]
        if not is_work:
            history_attempts.append(member)
            fail = member == "z-other" or fail_first_primary_history
            if member == "reviewer":
                fail_first_primary_history = False
            if fail:
                raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {},
                    io.BytesIO(b'{"error":{"code":"unavailable"}}'))
            return pair.http(request, timeout=timeout)
        work_attempts.append(member)
        record = json.loads(request.data)["record"]
        reply = records.ingest(pair.target, record=record, token=token, secret=SECRET,
                               target_install_id=TARGET, target_profile=claims["target_profile"])
        if bad_ack:
            reply["revision"] += 1
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", transport)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)  # Primary history fails once.
    pub._publish_one(("room", "z-other"))  # Alternate history failure is now durable.
    pub._publish_one(KEY)  # Primary catches up; alternate remains unavailable.
    assert history_attempts == ["reviewer", "z-other", "reviewer"]
    pub._publish_one(KEY)  # Valid or invalid work ACK, independently of healthy history.
    expected = "invalid_ack" if bad_ack else "acked"
    assert pub.status()["work_records"][0]["status"] == expected
    assert work_attempts == ["reviewer"]
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    start = len(history_attempts)
    for turn in range(4):
        append(pair.source, f"later-{turn}")
        pub._publish_one(("room", "z-other" if turn % 2 == 0 else "reviewer"))
    if bad_ack:
        assert work_attempts == ["reviewer"]  # Never retry the permanently refused generation.
    latest = rooms.room_state(pair.source, room_id="room")["latest_seq"]
    assert pair.http.requests[-1][1]["page"]["cursor"] == latest, history_attempts[start:]
