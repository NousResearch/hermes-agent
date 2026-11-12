"""`hermes serve` peer routes: identity across re-registration, and Disband racing registration."""
from __future__ import annotations

import threading

import pytest

from hermes_agent.gateway.hosted_rooms import default_db_path as hosted_rooms_default_db_path
import hermes_agent.tui_gateway.server as srv
from hermes_agent.tui_gateway import methods_groups
from hermes_agent.tui_gateway.hosted_room_service import HostedRoomService

from tests.tui_gateway import test_groups_methods as groups_methods

_create_room, _result, home = groups_methods._create_room, groups_methods._result, groups_methods.home


class _RouteService(HostedRoomService):
    def __init__(self):
        self.peer_route_lock = threading.Lock()
        self._disbanding_rooms = set()


def test_reregistered_peer_route_keeps_its_trace_identity(home, monkeypatch):
    """A fresh grant for the same member must replay accepted work byte for byte: the target
    fingerprints the dispatch body, trace and cancellation ids included, under its key."""
    room, routes = _create_room(), []
    params, _ = _serve_peer_client(monkeypatch, room)
    catalog = params["catalog"]

    class FakeService(_RouteService):
        db_path = hosted_rooms_default_db_path()

        def register_peer_route(self, **kwargs):
            routes.append(kwargs["route"])

    monkeypatch.setattr(srv, "get_hosted_room_service", lambda: FakeService())
    for grant in ("first.room.grant", "second.room.grant"):
        _result(srv._methods["groups.peer.register"](3, params | {"grant": grant}))

    first, second = routes
    assert (first.grant, second.grant) == ("first.room.grant", "second.room.grant")
    assert (first.trace_id, first.cancellation_scope_id) == (second.trace_id, second.cancellation_scope_id)
    assert first.cancellation_scope_id == "cancel-room-1"

    # A route stored before the identity was derived (a random trace) keeps the one it has: the
    # member's gateway fingerprinted that body when it accepted the turn.
    from hermes_agent.gateway import hosted_room_links
    from hermes_agent.gateway.hosted_room_peer import GatewayRoomCatalog
    hosted_room_links.save_room_link(FakeService.db_path, hosted_room_links.make_stored_link(
        room_id="room-1", member_id="member-peer", target_url="https://peer.example.test",
        target_profile="reviewer", grant="older.room.grant", catalog=GatewayRoomCatalog.from_mapping(catalog),
        cancellation_scope_id="cancel-room-1", trace_id="trace-" + "ab" * 16))
    _result(srv._methods["groups.peer.register"](4, params | {"grant": "renewed.room.grant"}))
    assert (routes[-1].trace_id, routes[-1].cancellation_scope_id) == ("trace-" + "ab" * 16, "cancel-room-1")


def _serve_owed():
    """Revokes ``hermes serve`` still owes, as (room, member, target_url); the grant is not read."""
    import sqlite3
    with sqlite3.connect(hosted_rooms_default_db_path()) as conn:
        return conn.execute("SELECT room_id, member_id, target_url FROM hosted_room_owed_revokes "
                            "ORDER BY room_id, member_id").fetchall()


def _serve_peer_client(monkeypatch, room, *, during_probe=lambda: None):
    from hermes_agent.gateway.hosted_room_peer import catalog_mapping
    from hermes_agent.gateway.hosted_rooms import local_authority_gateway_id

    catalog = catalog_mapping(target_profile="default", installation_id="install-peer", persistent_process=True)
    probes = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def probe(self, *, grant):
            probes.append(grant)
            during_probe()
            return {"room_id": "room-1", "home_install_id": local_authority_gateway_id(),
                    "authority_gateway_id": room["authority_gateway_id"],
                    "authority_epoch": room["authority_epoch"], "member_id": "member-peer",
                    "target_profile": "reviewer", "catalog": catalog}

    monkeypatch.setattr("hermes_agent.tui_gateway.hosted_room_peer_http.PeerRunsHTTPClient", FakeClient)
    params = {"room_id": "room-1", "member_id": "member-peer", "target_url": "https://peer.example.test",
              "target_profile": "reviewer", "grant": _scoped_grant(room), "catalog": catalog}
    return params, probes


def _scoped_grant(room):
    """A grant whose (unverified) scope names this room: only such a grant is owed a revoke unprobed."""
    import base64
    import json

    from hermes_agent.gateway.hosted_rooms import local_authority_gateway_id

    claims = {"room_id": "room-1", "member_id": "member-peer", "home_install_id": local_authority_gateway_id(),
              "authority_gateway_id": room["authority_gateway_id"], "authority_epoch": room["authority_epoch"]}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{payload}.c2ln"


@pytest.mark.parametrize("disband_at", ["before", "probe", "publication"])
def test_serve_disband_and_register_never_leave_a_route_or_a_grant_behind(home, monkeypatch, disband_at):
    """`hermes serve` publishes a route only while its room is live. A grant handed in for a room
    that ended first is owed, never probed; one that Disband overtakes during the probe or before
    publication is refused and owed too."""
    from hermes_agent.gateway import hosted_room_links
    from hermes_agent.tui_gateway import hosted_room_peer_transport
    room = _create_room()

    def disband():
        _result(srv._methods["groups.disband"](9, {"room_id": "room-1"}))
    if disband_at == "before":
        disband()
    params, probes = _serve_peer_client(
        monkeypatch, room, during_probe=disband if disband_at == "probe" else lambda: None)
    if disband_at == "publication":
        identity = hosted_room_peer_transport.route_identity

        def identity_then_disband(**kwargs):
            result = identity(**kwargs)
            disband()
            return result
        monkeypatch.setattr(hosted_room_peer_transport, "route_identity", identity_then_disband)
    assert "error" in srv._methods["groups.peer.register"](3, params)
    assert probes == ([] if disband_at == "before" else [params["grant"]])
    assert hosted_room_links.load_room_links(hosted_rooms_default_db_path()) == ()
    assert methods_groups.get_hosted_room_service().peer_routes == {}
    assert _serve_owed() == [("room-1", "member-peer", "https://peer.example.test")]
