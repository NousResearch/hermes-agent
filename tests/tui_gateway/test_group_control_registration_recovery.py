"""Failed control registration preserves existing and concurrent access."""

import pytest

from gateway import hosted_room_controls as controls, hosted_rooms
from tests.tui_gateway.test_groups_methods import home, _result, srv


@pytest.fixture
def linked_room(home, monkeypatch):
    monkeypatch.setenv("HERMES_ROOM_LINK_URL", "https://home.example.test/hermes")
    monkeypatch.setattr(controls, "gateway_room_grant_secret", lambda: b"s" * 32)
    service = srv.get_hosted_room_service()
    service.create_room(room_id="review-room", name="Review", members=[
        {"member_id": "peer-member", "profile": "ops", "handle": "ops", "target": {
            "kind": "peer", "peer_id": "install-peer", "installation_id": "install-peer",
            "profile": "ops", "capability_digest": "a" * 64}},
        {"member_id": "default", "profile": "default", "handle": "home"},
    ])
    params = dict(room_id="review-room", member_id="peer-member", caller_install_id="install-peer", request_id="A")
    invitation = _result(srv._methods["groups.control.invite"](1, params))
    db = hosted_rooms.default_db_path()
    hosted_rooms.reserve_peer_room(db, claims={
        key: invitation[key] for key in ("room_id", "member_id", "authority_gateway_id", "authority_epoch")
    } | {"target_profile": "ops"}, expires_at=invitation["expires_at"])
    class Client:
        def __init__(self, link):
            self.link = link
        def summary(self):
            assert controls.verify_home_control_token(db, **{
                key: getattr(self.link, key) for key in (
                    "room_id", "member_id", "authority_gateway_id", "authority_epoch", "control_token")
            })
            return {"room": {key: getattr(self.link, key) for key in (
                "room_id", "authority_gateway_id", "authority_epoch")}}
        def revoke(self):
            controls.revoke_home_control_token_value(db, room_id=self.link.room_id,
                member_id=self.link.member_id, control_token=self.link.control_token)
    monkeypatch.setattr("gateway.hosted_room_control_client.RoomControlHTTPClient", Client)
    assert _result(srv._methods["groups.control.register"](2, invitation | {"profile": "ops"}))["registered"]
    return db, params, invitation, Client



@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("replacement", [False, True])
def test_failed_registration_compensates_only_its_new_exact_link(linked_room, monkeypatch, existing, replacement):
    db, params, invitation, client = linked_room
    if not existing:
        controls.delete_peer_control_links(db, room_id="review-room", member_id="peer-member")
    original_summary = client.summary
    winner = [invitation]

    def timeout(self):
        if replacement:
            controls.revoke_home_control_tokens(db, room_id="review-room", member_id="peer-member")
            winner[0] = _result(srv._methods["groups.control.invite"](5, {**params, "request_id": "winner"}))
            controls.save_peer_control_link(db, **{
                key: getattr(self.link, key) for key in (
                    "room_id", "member_id", "home_url", "authority_gateway_id",
                    "authority_epoch", "room_name", "member_count")
            }, control_token=winner[0]["control_token"], expires_at=winner[0]["expires_at"], allow_rotation=True)
        raise TimeoutError("synthetic summary timeout")

    monkeypatch.setattr(client, "summary", timeout)
    result = srv._methods["groups.control.register"](4, {**invitation, "profile": "ops"})
    assert result["error"]["code"] == 4151
    links = controls.load_peer_control_links(db, include_inactive=True).links
    if existing or replacement:
        assert len(links) == 1
        assert links[0].status == "active"
        assert links[0].control_token == winner[0]["control_token"]
        assert controls.verify_home_control_token(db, **{
            key: winner[0][key] for key in (
                "room_id", "member_id", "authority_gateway_id", "authority_epoch", "control_token")
        })
    else:
        assert links == ()
        monkeypatch.setattr(client, "summary", original_summary)
        assert _result(srv._methods["groups.control.register"](6, {**invitation, "profile": "ops"}))["registered"]
