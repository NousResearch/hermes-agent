"""Tests for the ``groups.replicate`` / ``groups.promote`` / ``groups.demote``
JSON-RPC surface — cross-gateway room durability."""

from __future__ import annotations

import pytest

import tui_gateway.server as srv
from tui_gateway import methods_groups

MEMBERS = [{"kind": "bot", "id": "planner"}]


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / ".hermes"
    path.mkdir()
    (path / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(path))
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def _error(envelope):
    assert "error" in envelope, envelope
    return envelope["error"]


def _authority_page(tmp_path, gateway_id="install:" + "a" * 32, n=3):
    """Build a real room + log on a SEPARATE 'remote authority' DB and return
    its replay page, as a replicating client would fetch via groups.log."""
    from gateway import hosted_rooms as rooms

    db = tmp_path / "remote-authority.db"
    rooms.create_room(
        db,
        room_id="room-1",
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=gateway_id,
    )
    for index in range(n):
        rooms.append_event(
            db,
            room_id="room-1",
            event_id=f"e{index}",
            kind="message.user",
            actor={"kind": "user", "id": "tek"},
            payload={"text": f"msg {index}"},
            authority_gateway_id=gateway_id,
            authority_epoch=1,
        )
    return rooms.read_events(db, room_id="room-1", since_seq=0, limit=100)


def test_capabilities_advertise_replication(home):
    result = _result(srv._methods["groups.capabilities"](1, {}))
    assert "log_replication" in result["features"]
    assert "authority_takeover" in result["features"]
    for name in (
        "groups.replicate",
        "groups.replica_state",
        "groups.promote",
        "groups.demote",
    ):
        assert name in result["methods"]
        assert name in srv._LONG_HANDLERS


def test_replicate_then_state_roundtrip(home, tmp_path):
    page = _authority_page(tmp_path)
    result = _result(
        srv._methods["groups.replicate"](
            1,
            {
                "room_id": "room-1",
                "room_name": "Field Room",
                "members": MEMBERS,
                "page": page,
            },
        )
    )
    assert result["ingested"] == 3
    state = _result(srv._methods["groups.replica_state"](2, {"room_id": "room-1"}))
    assert state["last_seq"] == 3
    assert state["authority"] == page["authority"]


def test_promote_requires_confirm_and_takes_over(home, tmp_path):
    page = _authority_page(tmp_path)
    _result(
        srv._methods["groups.replicate"](
            1,
            {
                "room_id": "room-1",
                "room_name": "Field Room",
                "members": MEMBERS,
                "page": page,
            },
        )
    )

    refused = _error(srv._methods["groups.promote"](2, {"room_id": "room-1"}))
    assert refused["code"] == 4118

    promoted = _result(
        srv._methods["groups.promote"](3, {"room_id": "room-1", "confirm": True})
    )
    assert promoted["authority_epoch"] == 2
    assert promoted["previous_gateway_id"] == page["authority"]["gateway_id"]

    # The room is now hosted locally with full history + claim event.
    log = _result(srv._methods["groups.log"](4, {"room_id": "room-1"}))
    kinds = [event["kind"] for event in log["events"]]
    assert kinds == ["message.user"] * 3 + ["authority.claimed"]
    assert log["authority"]["epoch"] == 2


def test_demote_fences_local_room_against_newer_epoch(home):
    from gateway.hosted_rooms import local_authority_gateway_id

    _result(
        srv._methods["groups.create"](
            1,
            {
                "room_id": "room-1",
                "name": "Local room",
                "members": [
                    {
                        "member_id": "default",
                        "profile": "default",
                        "handle": "hermes",
                    },
                    {"member_id": "ops", "profile": "ops", "handle": "ops"},
                ],
            },
        )
    )
    observed_gateway = "install:" + "b" * 32
    result = _result(
        srv._methods["groups.demote"](
            2,
            {
                "room_id": "room-1",
                "observed_gateway_id": observed_gateway,
                "observed_epoch": 2,
            },
        )
    )
    assert result["idempotent"] is False
    assert result["authority_gateway_id"] == observed_gateway

    # Local sends at the stale authority now fail.
    envelope = srv._methods["groups.send"](
        3,
        {
            "room_id": "room-1",
            "event_id": "stale-send",
            "actor": {"kind": "user", "id": "tek"},
            "payload": {"text": "should fence"},
        },
    )
    assert "error" in envelope
    assert local_authority_gateway_id() != observed_gateway


def test_replicate_rejects_gapped_page(home, tmp_path):
    from gateway import hosted_rooms as rooms

    _authority_page(tmp_path, n=5)
    db = tmp_path / "remote-authority.db"
    gapped = rooms.read_events(db, room_id="room-1", since_seq=2, limit=100)
    envelope = srv._methods["groups.replicate"](
        1,
        {
            "room_id": "room-1",
            "room_name": "Field Room",
            "members": MEMBERS,
            "page": gapped,
        },
    )
    assert _error(envelope)["code"] == 4116
