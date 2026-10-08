"""Authenticated target controls mint cleanup-only authority without reopening execution."""
import sqlite3
import pytest

from gateway import hosted_rooms
from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret
from tests.tui_gateway.test_groups_methods import home as home, _result
from tui_gateway import server as srv


def test_target_owner_retirement_invitation_preserves_live_reservation(home, monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    fields = dict(room_id="retirement-room", home_install_id="original-home", authority_gateway_id="original-home",
                  authority_epoch=1, member_id="peer-member")
    ordinary = _result(srv._methods["groups.peer.invite"](1, fields))
    with sqlite3.connect(hosted_rooms.default_db_path()) as conn:
        before = conn.execute("SELECT * FROM hosted_room_peer_reservations").fetchall()
    narrow = _result(srv._methods["groups.peer.invite"](2, {**fields, "retirement_only": True}))
    claims = decode_room_grant(gateway_room_grant_secret(), narrow["grant"], permission="retire")
    assert set(claims["permissions"]) == {"status", "retire"}
    with pytest.raises(ValueError):
        decode_room_grant(gateway_room_grant_secret(), narrow["grant"], permission="dispatch")
    assert decode_room_grant(gateway_room_grant_secret(), ordinary["grant"], permission="dispatch")
    with sqlite3.connect(hosted_rooms.default_db_path()) as conn:
        assert conn.execute("SELECT * FROM hosted_room_peer_reservations").fetchall() == before

    moved = {**fields, "home_install_id": "new-home", "authority_gateway_id": "new-home", "authority_epoch": 2}
    assert "error" in srv._methods["groups.peer.invite"](3, moved)
    with sqlite3.connect(hosted_rooms.default_db_path()) as conn:
        assert conn.execute("SELECT * FROM hosted_room_peer_reservations").fetchall() == before
    linked = _result(srv._methods["groups.peer.invite"](4, {**moved, "previous_authority": {
        key: fields[key] for key in ("home_install_id", "authority_gateway_id", "authority_epoch")}}))
    assert decode_room_grant(gateway_room_grant_secret(), linked["grant"], permission="dispatch")["home_install_id"] == "new-home"
