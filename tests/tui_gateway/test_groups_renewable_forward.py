"""Renewable invitations traverse the standalone request validator and owning gateway."""

import asyncio
from types import SimpleNamespace

from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret
from tests.gateway.test_session_group_peers import gateway as gateway_fixture
from hermes_agent.tui_gateway import methods_groups_forward, owner_forward, server

gateway = gateway_fixture


def test_renewable_invitation_through_standalone_dispatch(gateway, monkeypatch):
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9')
    monkeypatch.setattr(methods_groups_forward, '_discover', lambda home: SimpleNamespace(state='ready', endpoint=object()))
    forwarded = []

    def forward(home, endpoint, method, params):
        forwarded.append((method, dict(params)))
        reply = asyncio.run(gateway.owner.dispatch({'id': 1, 'method': method, 'params': params}))
        assert 'result' in reply, reply
        return reply['result']

    monkeypatch.setattr(owner_forward, 'forward_rpc', forward)
    capabilities = server.handle_request({'id': 1, 'method': 'groups.capabilities', 'params': {}})['result']
    assert 'peer_renewal_horizon' in capabilities['features']
    params = dict(room_id='new-room', home_install_id='install:home', authority_gateway_id='install:home',
                  authority_epoch=1, member_id='peer', ttl_seconds=60, status_ttl_seconds=86400)
    reply = server.handle_request({'id': 2, 'method': 'groups.peer.invite', 'params': params})
    assert 'result' in reply, reply
    assert forwarded[-1] == ('groups.peer.invite', params)
    claims = decode_room_grant(gateway_room_grant_secret(), reply['result']['grant'], permission='dispatch')
    assert claims['status_expires_at'] - claims['issued_at'] == 86400

    monkeypatch.setattr(methods_groups_forward, '_discover', lambda home: SimpleNamespace(state='absent'))
    local = server.handle_request({'id': 3, 'method': 'groups.capabilities', 'params': {}})['result']
    assert 'peer_renewal_horizon' not in local['features']
