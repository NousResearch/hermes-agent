"""Retired history is uncertainty; old peers retain their ordinary revoke contract."""
import io
import json
import time
import urllib.error

import pytest

from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest.mark.parametrize('path', ['/v1/runs', '/v1/runs/stop'])
def test_retired_history_never_claims_nonadmission(path):
    error = urllib.error.HTTPError('http://localhost' + path, 409, 'retired', {},
        io.BytesIO(json.dumps({'error': {'code': 'run_history_retired'}}).encode()))
    with pytest.raises(PeerRunsHTTPError) as caught:
        PeerRunsHTTPClient._raise_http_error(error, method='POST', path=path, deadline=time.monotonic() + 5)
    assert caught.value.ambiguous and not caught.value.not_admitted


@pytest.mark.parametrize('status,code', [(400, 'invalid_room_grant_revoke'), (403, 'room_retirement_not_granted')])
def test_old_endpoint_or_grant_falls_back_only_after_explicit_refusal(monkeypatch, status, code):
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:1', api_key='')
    bodies = []

    def request(path, grant, *, body):
        bodies.append(body)
        if body:
            raise PeerRunsHTTPError('unsupported', status_code=status, error_code=code)
        return {'revoked': True}

    monkeypatch.setattr(client, '_scoped_post', request)
    assert client.revoke_grant(grant='existing-scoped-grant', retire_authority=True)['revoked']
    assert bodies == [{'retire_authority': True}, {}]
