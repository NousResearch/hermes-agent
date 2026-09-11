"""A successful HTTP response must acknowledge revocation before retiring a route."""

import pytest

from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest.mark.parametrize("method", ["revoke_grant", "revoke_grant_exact"])
@pytest.mark.parametrize("response", [{}, {"revoked": False}, {"revoked": "true"}])
def test_unacknowledged_revocation_remains_retryable(monkeypatch, method, response):
    client = PeerRunsHTTPClient(base_url="https://peer.example", api_key="")
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    with pytest.raises(PeerRunsHTTPError) as caught:
        getattr(client, method)(grant="signed.test.grant")
    assert caught.value.retryable
    assert caught.value.status_code is None


@pytest.mark.parametrize("method", ["revoke_grant", "revoke_grant_exact"])
def test_acknowledged_revocation_is_returned(monkeypatch, method):
    client = PeerRunsHTTPClient(base_url="https://peer.example", api_key="")
    result = {"revoked": True}
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: result)
    assert getattr(client, method)(grant="signed.test.grant") is result
