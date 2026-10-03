"""``groups.replicate`` / ``groups.promote`` / ``groups.demote`` stay wired behind one gate.

The exclusive-authority gate (``hosted_room_replicas.require_takeover``) refuses all three with a
typed reason; opening it is the only change needed to reach the implementations again. These RPCs
carry no proof, so what they promote or demote stays quarantined even through an open gate."""

from __future__ import annotations

import pytest

import tui_gateway.server as srv
from gateway import hosted_room_replicas as replicas
from tui_gateway import methods_groups

MEMBERS = [{"kind": "bot", "id": "planner"}]
LOCAL_MEMBERS = [
    {"member_id": "default", "profile": "default", "handle": "hermes"},
    {"member_id": "ops", "profile": "ops", "handle": "ops"},
]


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / ".hermes"
    path.mkdir()
    (path / "profiles" / "ops").mkdir(parents=True)
    (path / "profiles" / "ops" / "config.yaml").write_text("{}\n")  # identity marker: local roster
    monkeypatch.setenv("HERMES_HOME", str(path))
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


@pytest.fixture
def gate_open(monkeypatch):
    """Stand in for exclusive-authority recovery's verdict, to prove the methods stay wired."""
    monkeypatch.setattr(replicas, "takeover_enabled", lambda: True)


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def _error(envelope):
    assert "error" in envelope, envelope
    return envelope["error"]


def _authority_page(tmp_path, gateway_id="install:" + "a" * 32, n=3, room_id="room-1"):
    """Build a real room + log on a SEPARATE 'remote authority' DB and return
    its replay page, as a replicating client would fetch via groups.log."""
    from gateway import hosted_rooms as rooms

    db = tmp_path / "remote-authority.db"
    rooms.create_room(
        db,
        room_id=room_id,
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=gateway_id,
    )
    for index in range(n):
        rooms.append_event(
            db,
            room_id=room_id,
            event_id=f"e{index}",
            kind="message.user",
            actor={"kind": "user", "id": "tek"},
            payload={"text": f"msg {index}"},
            authority_gateway_id=gateway_id,
            authority_epoch=1,
        )
    return rooms.read_events(db, room_id=room_id, since_seq=0, limit=100)


def _replicate_params(page, room_id="room-1"):
    return {"room_id": room_id, "room_name": "Field Room", "members": MEMBERS, "page": page}


def test_capabilities_do_not_advertise_unverified_replication(home):
    assert (home / "profiles" / "ops" / "config.yaml").read_text() == "{}\n"
    capabilities = _result(srv._methods["groups.capabilities"](1, {}))
    assert {"log_replication", "authority_takeover"}.isdisjoint(capabilities["features"])
    # The methods stay registered so a caller gets the typed refusal, not "unknown method".
    assert {"groups.replicate", "groups.promote", "groups.demote"} <= set(capabilities["methods"])


def test_replicate_refuses_unverified_pages_without_storing_them(home, tmp_path):
    from gateway.hosted_rooms import default_db_path

    error = _error(srv._methods["groups.replicate"](1, _replicate_params(_authority_page(tmp_path))))
    assert error["code"] == 4116
    assert error["data"] == {"reason": "replica_provenance_required"}
    assert "verify that a page came from the room's authority" in error["message"]
    with pytest.raises(replicas.ReplicaError, match="not found"):
        replicas.replica_state(default_db_path(), room_id="room-1")


def test_promote_refuses_even_a_confirmed_complete_replica(home, tmp_path):
    from gateway.hosted_rooms import default_db_path

    page = _authority_page(tmp_path)
    replicas.ingest_page(default_db_path(), room_id="room-1", room_name="Field Room", members=MEMBERS, page=page)
    error = _error(srv._methods["groups.promote"](
        1, {"room_id": "room-1", "confirm": True, "reason": "planned-handover"}))
    assert error["code"] == 4118
    assert error["data"] == {"reason": "authority_takeover_disabled"}
    assert "globally exclusive authority" in error["message"]
    # Still a passive copy of the original authority; nothing was claimed locally.
    assert replicas.replica_state(default_db_path(), room_id="room-1")["authority"] == page["authority"]
    assert "error" in srv._methods["groups.log"](2, {"room_id": "room-1"})


def test_demote_refuses_and_keeps_the_local_authority(home):
    room = _result(srv._methods["groups.create"](
        1, {"room_id": "room-1", "name": "Local room", "members": LOCAL_MEMBERS}))["room"]
    error = _error(srv._methods["groups.demote"](2, {
        "room_id": "room-1", "observed_gateway_id": "install:" + "b" * 32, "observed_epoch": 2}))
    assert error["code"] == 4119
    assert error["data"] == {"reason": "authority_takeover_disabled"}
    state = _result(srv._methods["groups.state"](3, {"room_id": "room-1"}))["room"]
    assert (state["authority_gateway_id"], state["authority_epoch"]) == (
        room["authority_gateway_id"], room["authority_epoch"])


def test_open_gate_advertises_the_takeover_features(home, gate_open):
    capabilities = _result(srv._methods["groups.capabilities"](1, {}))
    assert {"log_replication", "authority_takeover"} <= set(capabilities["features"])


def test_open_gate_replicates_then_reports_state(home, tmp_path, gate_open):
    page = _authority_page(tmp_path)
    result = _result(srv._methods["groups.replicate"](1, _replicate_params(page)))
    assert result["ingested"] == 3
    state = _result(srv._methods["groups.replica_state"](2, {"room_id": "room-1"}))
    assert state["last_seq"] == 3
    assert state["authority"] == page["authority"]


def test_open_gate_promote_requires_confirm_and_takes_over(home, tmp_path, gate_open):
    page = _authority_page(tmp_path)
    _result(srv._methods["groups.replicate"](1, _replicate_params(page)))

    refused = _error(srv._methods["groups.promote"](2, {"room_id": "room-1"}))
    assert refused["code"] == 4118

    promoted = _result(srv._methods["groups.promote"](3, {"room_id": "room-1", "confirm": True}))
    assert promoted["authority_epoch"] == 2
    assert promoted["previous_gateway_id"] == page["authority"]["gateway_id"]

    # The room is now hosted locally with full history + claim event.
    log = _result(srv._methods["groups.log"](4, {"room_id": "room-1"}))
    kinds = [event["kind"] for event in log["events"]]
    assert kinds == ["message.user"] * 3 + ["authority.claimed"]
    assert log["authority"]["epoch"] == 2


def test_open_gate_quarantines_an_unmarked_promotion_but_not_a_marked_one(home, tmp_path, gate_open):
    """Opening the gate alone does not make a takeover trusted.

    ``groups.promote`` carries no proof, so the store records its promotion as unproven and
    quarantines the room. A promotion whose caller verified a certificate or an attestation is
    marked in the same transaction (``promote_replica(transition=...)``), the triggers in
    ``gateway/hosted_room_safety.py`` accept it, and the room stays writable.
    """
    from gateway.hosted_room_safety import transition_proof_digest
    from gateway.hosted_rooms import default_db_path, local_authority_gateway_id

    _result(srv._methods["groups.replicate"](1, _replicate_params(_authority_page(tmp_path))))
    assert _result(srv._methods["groups.promote"](2, {"room_id": "room-1", "confirm": True}))["authority_epoch"] == 2
    unmarked, = _result(srv._methods["groups.list"](3, {}))["rooms"]
    assert (unmarked["safety_status"], unmarked["safety_reason"]) == (
        "authority_quarantined", "unsafe_replica_promotion")
    refused = _error(srv._methods["groups.send"](4, {
        "room_id": "room-1", "event_id": "after-promotion", "payload": {"text": "continue", "thread_id": "thread-1"}}))
    assert refused["data"] == {"reason": "room_authority_quarantined"}

    other = tmp_path / "other-authority"
    other.mkdir()
    _result(srv._methods["groups.replicate"](
        5, _replicate_params(_authority_page(other, room_id="room-2"), room_id="room-2")))
    proof = {"room_id": "room-2", "from_epoch": 1, "to_epoch": 2, "successor_gateway_id": local_authority_gateway_id()}
    replicas.promote_replica(default_db_path(), room_id="room-2", transition={
        "proof_kind": "certified", "proof_digest": transition_proof_digest(proof), "proof": proof})
    marked = next(room for room in _result(srv._methods["groups.list"](6, {}))["rooms"] if room["room_id"] == "room-2")
    assert "safety_status" not in marked
    assert _result(srv._methods["groups.state"](7, {"room_id": "room-2"}))["room"]["authority_epoch"] == 2
    renamed = _result(srv._methods["groups.rename"](8, {"room_id": "room-2", "event_id": "rename", "name": "Kept"}))
    assert (renamed["room"]["name"], renamed["room"]["event"]["authority_epoch"]) == ("Kept", 2)


def test_open_gate_demote_fences_local_room_against_newer_epoch(home, gate_open):
    from gateway.hosted_rooms import local_authority_gateway_id

    _result(srv._methods["groups.create"](
        1, {"room_id": "room-1", "name": "Local room", "members": LOCAL_MEMBERS}))
    observed_gateway = "install:" + "b" * 32
    result = _result(srv._methods["groups.demote"](2, {
        "room_id": "room-1", "observed_gateway_id": observed_gateway, "observed_epoch": 2}))
    assert result["idempotent"] is False
    assert result["authority_gateway_id"] == observed_gateway

    # Local sends at the stale authority now fail.
    envelope = srv._methods["groups.send"](3, {
        "room_id": "room-1", "event_id": "stale-send", "actor": {"kind": "user", "id": "tek"},
        "payload": {"text": "should fence"}})
    assert "error" in envelope
    assert local_authority_gateway_id() != observed_gateway


def test_open_gate_replicate_rejects_gapped_page(home, tmp_path, gate_open):
    from gateway import hosted_rooms as rooms

    _authority_page(tmp_path, n=5)
    gapped = rooms.read_events(tmp_path / "remote-authority.db", room_id="room-1", since_seq=2, limit=100)
    assert _error(srv._methods["groups.replicate"](1, _replicate_params(gapped)))["code"] == 4116
