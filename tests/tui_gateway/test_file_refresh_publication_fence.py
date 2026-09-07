"""File refresh conflicts retain the publication barrier, not just an error label."""

import pytest

from gateway import hosted_room_links, hosted_rooms
from tests.tui_gateway.test_hosted_room_file_route_fencing import routes  # noqa: F401
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient


@pytest.mark.parametrize("hydrate", [False, True])
@pytest.mark.parametrize("operation", [
    "read_artifact", "acknowledge_artifacts", "discard_artifacts", "stage_attachments",
])
def test_same_bearer_retarget_during_file_refresh_never_publishes_replacement(
    routes, monkeypatch, operation, hydrate,
):
    worker, key = routes.second, ("room-1", "member-peer")
    routes.register(routes.first, routes.tokens["aging"])
    worker._hydrate_persisted_peer_route(*key)
    stored = hosted_room_links.load_room_link(worker.db_path, room_id=key[0], member_id=key[1])
    target_url = "https://replacement.example.test"
    refreshed, cleaned, operated, published = [], [], [], []
    original_rotate = worker._rotate_route_grant

    def rotate(*args, **kwargs):
        published.append(True)
        return original_rotate(*args, **kwargs)

    monkeypatch.setattr(worker, "_rotate_route_grant", rotate)

    class Peer:
        base_url = stored.target_url

        def refresh_grant(self, **kwargs):
            assert kwargs["grant"] == stored.grant
            refreshed.append(True)
            routes.first.register_peer_route(
                room_id=key[0], member_id=key[1],
                route=routes.first.peer_routes[key],
                client=PeerRunsHTTPClient(base_url=target_url, api_key="", target_profile="reviewer"),
                target_url=target_url, catalog=routes.catalog,
            )
            if hydrate:
                worker.status_with_grant_fingerprints(key[0])
            return {"grant": routes.tokens["stale"]}

        def revoke_grant_exact(self, *, grant):
            cleaned.append(grant)
            return {"revoked": True}

        def revoke_grant(self, **kwargs):
            raise AssertionError("scope revocation cannot replace exact cleanup")

        def __getattr__(self, name):
            assert name == operation
            return lambda **kwargs: operated.append(kwargs)

    tracked = worker._tracked_peer_client(*key, Peer())
    with pytest.raises(Exception) as caught:
        getattr(tracked, operation)(grant=stored.grant)
    current = hosted_room_links.load_room_link(worker.db_path, room_id=key[0], member_id=key[1])
    assert refreshed == [True]
    assert published == operated == []
    assert cleaned == [routes.tokens["stale"]]
    assert current.grant == stored.grant
    assert current.target_url == target_url and current.status == "ready"
    assert isinstance(caught.value, hosted_rooms.HostedRoomError)
    assert "changed during reconnect" in str(caught.value)
