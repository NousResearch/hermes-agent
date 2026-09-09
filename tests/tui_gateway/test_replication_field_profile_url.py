"""Replica profiles compose with field constructor scoping without rerouting."""

import io
from urllib.parse import quote

import pytest

from gateway.hosted_room_peer import HostedRoomPeerError
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient


@pytest.mark.parametrize("profile", ["default", "reviewer:west"])
@pytest.mark.parametrize("scoped_base", [False, True])
@pytest.mark.parametrize("matching_method", [False, True])
def test_replica_method_respects_constructor_profile(
    monkeypatch, profile, scoped_base, matching_method,
):
    captured = []
    root = "https://peer.example.test/hermes"
    suffix = "/p/" + quote(profile, safe="")

    def opened(request, **kwargs):
        captured.append((request.full_url, request.get_header("Authorization")))
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", opened)
    client = PeerRunsHTTPClient(
        base_url=root + (suffix if scoped_base else ""),
        api_key="", target_profile=profile,
    )
    fields = dict(
        grant="signed.room.grant", target_profile=profile if matching_method else "different",
        room_id="room", room_name="Workshop", members=[], page={},
    )
    if matching_method:
        assert client.replicate_page(**fields) == {"ok": True}
        assert captured == [(root + suffix + "/v1/room-members/replica", "HermesRoom signed.room.grant")]
    else:
        with pytest.raises(HostedRoomPeerError, match="does not match"):
            client.replicate_page(**fields)
        assert captured == []

    captured.clear()
    assert client.probe(grant="signed.room.grant") == {"ok": True}
    assert captured == [(root + suffix + "/v1/room-members/capabilities", "HermesRoom signed.room.grant")]
