"""A late renewal can affect only the grant it actually observed."""

import hashlib
import json
import time

import pytest

from gateway import hosted_room_link_records
from gateway import hosted_room_links, hosted_rooms
from gateway.hosted_room_peer import (
    HostedMemberDispatch,
    PROTOCOL_VERSION,
    issue_room_grant,
)
from tests.tui_gateway.test_hosted_room_grant_fingerprint import peers as peers
from tui_gateway.hosted_room_service import HostedRoomService
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _context(first):
    stored = hosted_room_links.load_room_link(
        first.db_path, room_id="room-1", member_id="member-peer"
    )
    home = hosted_rooms.local_authority_gateway_id()
    now = time.time()
    scope = dict(
        room_id="room-1",
        home_install_id=home,
        authority_gateway_id=home,
        authority_epoch=1,
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        execution_policy_digest=stored.catalog.execution_policy.policy_digest,
        ttl_seconds=3600,
        status_expires_at=now + 10000,
    )
    tokens = {
        name: issue_room_grant(b"s" * 32, grant_id=name, issued_at=now - age, **scope)
        for name, age in (("old", 3550), ("winner", 0), ("stale", 0))
    }
    dispatch = HostedMemberDispatch(
        protocol_version=PROTOCOL_VERSION,
        room_id="room-1",
        home_install_id=home,
        authority_gateway_id=home,
        authority_epoch=1,
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        task_id="task-1",
        execution_generation=1,
        source_event_seq=1,
        cancellation_scope_id="cancel-room",
        prompt="Review",
        prompt_digest=_hash("Review"),
        capability_digest=stored.catalog.catalog_digest,
        execution_policy_digest=stored.catalog.execution_policy.policy_digest,
        trace_id="trace-room",
    )
    return tokens, dispatch, json.loads(stored.as_record()["catalog_json"])


@pytest.mark.parametrize("hydrate", [False, True])
@pytest.mark.parametrize("status_write_fails", [False, True])
def test_late_refresh_preserves_winner_and_attempts_exact_cleanup(
    peers, monkeypatch, hydrate, status_write_fails
):
    first, _second, register = peers
    tokens, dispatch, catalog = _context(first)
    register(first, tokens["old"])
    second = HostedRoomService(first.server, db_path=first.db_path)

    class Peer:
        def __init__(self):
            self.exact = []
            self.dispatched = []

        def refresh_grant(self, **kwargs):
            assert kwargs["grant"] == tokens["old"]
            register(first, tokens["winner"], expected=_hash(tokens["old"]))
            if hydrate:
                second.status_with_grant_fingerprints("room-1")
            return {"grant": tokens["stale"], "catalog": catalog}

        def revoke_grant_exact(self, *, grant):
            self.exact.append(grant)
            return {"revoked": True}

        def dispatch(self, **kwargs):
            self.dispatched.append(kwargs["grant"])
            return {"status": "accepted"}

    if status_write_fails:

        def fail_status(*_args, **_kwargs):
            raise OSError("status storage unavailable")

        monkeypatch.setattr(hosted_room_links, "mark_room_link_status", fail_status)
    peer = Peer()
    tracked = second._tracked_peer_client("room-1", "member-peer", peer)
    with pytest.raises(hosted_rooms.HostedRoomError, match="changed during reconnect"):
        tracked.dispatch(dispatch=dispatch.as_mapping(), grant=tokens["old"])
    row = hosted_room_link_records.room_link_record(
        first.db_path, room_id="room-1", member_id="member-peer"
    )
    assert row["grant"] == tokens["winner"]
    assert row["status"] == "ready"
    assert peer.dispatched == []
    assert peer.exact == [tokens["stale"]]


def test_transport_reuses_its_successfully_published_renewal(peers):
    first, _second, register = peers
    tokens, dispatch, catalog = _context(first)
    register(first, tokens["old"])

    class Peer:
        def __init__(self):
            self.refreshed = 0
            self.dispatched = []

        def refresh_grant(self, **kwargs):
            self.refreshed += 1
            assert kwargs["grant"] == tokens["old"]
            return {"grant": tokens["winner"], "catalog": catalog}

        def dispatch(self, **kwargs):
            self.dispatched.append(kwargs["grant"])
            return {"status": "accepted"}

    peer = Peer()
    tracked = first._tracked_peer_client("room-1", "member-peer", peer)
    for _ in range(2):
        assert (
            tracked.dispatch(dispatch=dispatch.as_mapping(), grant=tokens["old"])[
                "status"
            ]
            == "accepted"
        )
    assert peer.refreshed == 1
    assert peer.dispatched == [tokens["winner"], tokens["winner"]]


@pytest.mark.parametrize("hydrate", [False, True])
@pytest.mark.parametrize("outcome", ["ready", "unavailable", "needs_reauthorization"])
def test_late_health_callback_cannot_change_reenrolled_grant_status(
    peers, hydrate, outcome
):
    first, _second, register = peers
    tokens, dispatch, _catalog = _context(first)
    old, winner = tokens["winner"], tokens["stale"]
    register(first, old)
    second = HostedRoomService(first.server, db_path=first.db_path)
    winner_status = "unavailable" if outcome == "ready" else "ready"
    cache = []

    class Peer:
        def dispatch(self, **kwargs):
            assert kwargs["grant"] == old
            register(first, winner, expected=_hash(old))
            first._set_route_status("room-1", "member-peer", winner_status)
            if hydrate:
                second.status_with_grant_fingerprints("room-1")
            cache.append(second._peer_route_status[("room-1", "member-peer")])
            if outcome != "ready":
                raise PeerRunsHTTPError(
                    "late response",
                    status_code=403 if outcome == "needs_reauthorization" else 503,
                    error_code="room_reauthorization_required"
                    if outcome == "needs_reauthorization"
                    else "offline",
                    not_admitted=True,
                )
            return {"status": "accepted"}

    tracked = second._tracked_peer_client("room-1", "member-peer", Peer())
    if outcome == "ready":
        assert (
            tracked.dispatch(dispatch=dispatch.as_mapping(), grant=old)["status"]
            == "accepted"
        )
    else:
        with pytest.raises(PeerRunsHTTPError):
            tracked.dispatch(dispatch=dispatch.as_mapping(), grant=old)
    row = hosted_room_link_records.room_link_record(
        first.db_path, room_id="room-1", member_id="member-peer"
    )
    assert row["grant"] == winner
    assert row["status"] == winner_status
    assert second._peer_route_status[("room-1", "member-peer")] == cache[0]
