"""A renewal's requests: bounded by one budget, never redirected, never re-sending a refused grant."""
import io
import threading
import urllib.error
import urllib.request

import pytest

from tui_gateway import hosted_room_peer_http as http
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError, room_grant_request_budget


def _catalogs(**changes):
    from gateway.hosted_room_execution_policy import execution_policy_mapping
    from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
    policy = execution_policy_mapping(target_profile='default', config={'agent': {'max_turns': 20}})
    base = GatewayRoomCatalog.from_mapping(catalog_mapping(
        target_profile='default', installation_id='install-peer', persistent_process=True, execution_policy=policy))
    changed_policy = execution_policy_mapping(target_profile='default', config={'agent': {'max_turns': 21}})
    refreshed = catalog_mapping(
        target_profile='default', installation_id='install-peer', persistent_process=True,
        attachments=changes.get('attachments', False),
        execution_policy=changed_policy if changes.get('policy') else policy)
    return base, refreshed


@pytest.mark.parametrize('change,error_code', [
    ({'attachments': True}, 'room_capability_catalog_changed'),
    ({'policy': True}, 'room_execution_policy_changed'),
    ({'attachments': True, 'policy': True}, 'room_execution_policy_changed'),
])
def test_a_refreshed_grant_with_drifted_rights_is_refused_and_retired(change, error_code):
    base, refreshed = _catalogs(**change)
    client = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='')
    requests = []

    def request(path, **kwargs):
        requests.append((path, kwargs.get('room_grant')))
        if path == '/v1/room-members/grants/refresh':
            return {'grant': 'replacement.room.grant'}
        if path == '/v1/room-members/capabilities':
            return {'catalog': refreshed}
        assert path == '/v1/room-members/grants/revoke-exact'
        return {'revoked': True}
    client._request = request
    with pytest.raises(PeerRunsHTTPError) as caught:
        client.refresh_grant(grant='old.room.grant', capability_digest=base.catalog_digest,
                             execution_policy_digest=base.execution_policy.policy_digest)
    assert caught.value.error_code == error_code and caught.value.needs_reauthorization
    assert requests[-1] == ('/v1/room-members/grants/revoke-exact', 'replacement.room.grant')


def test_an_unchanged_refresh_keeps_its_catalog_and_retires_nothing():
    base, refreshed = _catalogs()
    client = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='')
    paths = []

    def request(path, **kwargs):
        paths.append(path)
        return {'grant': 'replacement.room.grant'} if path.endswith('/refresh') else {'catalog': refreshed}
    client._request = request
    result = client.refresh_grant(grant='old.room.grant', capability_digest=base.catalog_digest,
                                  execution_policy_digest=base.execution_policy.policy_digest)
    assert result['grant'] == 'replacement.room.grant'
    assert result['catalog']['catalog_digest'] == base.catalog_digest
    assert paths == ['/v1/room-members/grants/refresh', '/v1/room-members/capabilities']


def test_the_budget_covers_every_client_in_the_renewal_but_not_foreground_requests(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(http.time, 'monotonic', lambda: now[0])
    client = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='')
    ordinary = client.timeout_seconds
    foreground = []
    with room_grant_request_budget(2, clock=lambda: now[0]):
        cleanup = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='')
        assert client.timeout_seconds == cleanup.timeout_seconds == 1
        thread = threading.Thread(target=lambda: foreground.append(client.timeout_seconds))
        thread.start()
        thread.join(timeout=2)
        assert foreground == [ordinary]
        now[0] += 1.25
        with room_grant_request_budget(10, clock=lambda: now[0]):  # a nested budget never extends it
            assert client.timeout_seconds == cleanup.timeout_seconds == 0.75
        now[0] += 0.75
        with pytest.raises(PeerRunsHTTPError, match='budget exhausted'):
            cleanup.timeout_seconds
    assert client.timeout_seconds == cleanup.timeout_seconds == ordinary


def test_requests_and_response_reads_share_one_deadline(monkeypatch):
    from contextlib import contextmanager
    now, timeouts = [100.0], []
    monkeypatch.setattr(http.time, 'monotonic', lambda: now[0])

    @contextmanager
    def open_response(request, *, timeout, reject_redirects):
        assert reject_redirects is True
        timeouts.append(timeout)
        now[0] += min(0.75, timeout)
        response = io.BytesIO(b'{"ok": true}')
        response.headers = {}
        yield response
    monkeypatch.setattr(http, '_open_roomlink_url', open_response)
    with room_grant_request_budget(2, clock=lambda: now[0]):
        for _ in range(2):
            client = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='')
            assert client._request('/test') == {'ok': True}
        with pytest.raises(PeerRunsHTTPError, match='time budget'):
            client._request('/test')
        with pytest.raises(PeerRunsHTTPError, match='budget exhausted'):
            client._request('/test')
    assert timeouts == [1, 1, 0.5] and now[0] == 102


def test_a_renewal_refuses_redirects_but_keeps_the_installed_transport_policy(monkeypatch):
    from email.message import Message
    from urllib.response import addinfourl
    from hermes_cli import urllib_security
    requests = []

    class PolicyTransport(urllib.request.BaseHandler):
        handler_order = 1

        def https_open(self, request):
            requests.append(request)
            headers = Message()
            headers['Location'] = 'https://peer.example.test/redirected'
            response = addinfourl(io.BytesIO(b''), headers, request.full_url, 302)
            response.msg = 'Found'
            return response
    policy = urllib.request.build_opener(PolicyTransport())
    policy._hermes_initial_addheaders = [('X-Installed-Policy', 'present')]
    monkeypatch.setattr(urllib_security, '_secure_opener_from_installed_policy', lambda url: policy)
    client = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='')
    with room_grant_request_budget(2):
        with pytest.raises(PeerRunsHTTPError, match='refused an HTTP redirect'):
            client.probe(grant='synthetic.room.grant')
    assert len(requests) == 1
    assert requests[0].get_header('X-installed-policy') == 'present'
    assert requests[0].get_header('Authorization') == 'HermesRoom synthetic.room.grant'


def test_a_refused_grant_is_not_sent_to_the_same_check_again_for_a_minute(monkeypatch):
    now = [100.0]
    client = PeerRunsHTTPClient(base_url='https://peer.example.test', api_key='', clock=lambda: now[0])
    sent, answers = [], {}

    def respond(request, *, timeout, reject_redirects=False):
        path = request.full_url.removeprefix('https://peer.example.test')
        sent.append((request.get_method(), path, request.get_header('Authorization')))
        code = answers.get((path, request.get_header('Authorization')), 200)
        if code != 200:
            body = io.BytesIO(b'{"error": {"code": "room_reauthorization_required"}}' if code == 403 else b'{}')
            raise urllib.error.HTTPError(request.full_url, code, 'refused', {}, body)
        response = io.BytesIO(b'{"catalog": {}, "grant": "next"}')
        response.headers = {}
        return response
    monkeypatch.setattr(http, '_open_roomlink_url', respond)
    answers[('/v1/room-members/capabilities', 'HermesRoom refused')] = 403
    for _ in range(2):
        with pytest.raises(PeerRunsHTTPError) as caught:
            client.probe(grant='refused')
        assert caught.value.needs_reauthorization
    assert len(sent) == 1  # the second refusal came from memory
    client._scoped_post('/v1/room-members/grants/refresh', 'refused', body={})
    assert len(sent) == 2  # another check is asked on its own
    client._request('/v1/room-members/capabilities', room_grant='replacement')
    client._request('/v1/runs/run-1', room_grant='refused')  # reads and Stop are never held back
    assert len(sent) == 4
    answers[('/v1/room-members/capabilities', 'HermesRoom transient')] = 503
    for _ in range(2):
        with pytest.raises(PeerRunsHTTPError):
            client.probe(grant='transient')
    assert len(sent) == 6  # an outage is not a refusal: nothing is remembered
    now[0] += 60
    with pytest.raises(PeerRunsHTTPError):
        client.probe(grant='refused')
    assert len(sent) == 7
