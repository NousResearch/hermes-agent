"""A client announces which client it is once per connection: ``client.capabilities {client}``.

Sent through the native socket's real dispatch (authority first, the sidecar handler as fallback).
The label is kept on the connection's transport, unknown kind removed; it is attribution only.
"""
from __future__ import annotations

import asyncio

import pytest

from tests.gateway.test_local_desktop_identity import _Transport, _connect
from tests.waiting import async_wait_until


def _peer(identity):
    """A real ``WSTransport`` (its own attributes) whose frames are collected instead of sent."""
    from tui_gateway.ws import WSTransport

    class Peer(WSTransport):
        def write(self, obj):
            self.frames.append(obj)
            return True

    peer = Peer(object(), asyncio.get_running_loop(), auth_identity=identity)
    peer.frames = []
    return peer


async def _hello_connection(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    conn = await _connect(tmp_path, monkeypatch)
    identity = dict(conn.connection._identity)
    peer = _peer(identity)
    return conn, AuthorityConnection(conn.authority, peer, identity), peer


async def _request(connection, transport, rid, params):
    from tui_gateway.ws import _dispatch_request
    request = {'jsonrpc': '2.0', 'id': rid, 'method': 'client.capabilities', 'params': params}
    reply = await _dispatch_request(connection, request, 'client.capabilities', transport)
    if reply is None:
        def written():
            return next((f for f in transport.frames if isinstance(f, dict) and f.get('id') == rid), None)
        await async_wait_until(written, timeout=20)
        reply = written()
    return reply


@pytest.mark.asyncio
async def test_a_hello_is_kept_on_the_connection_without_an_unknown_kind(tmp_path, monkeypatch):
    conn, connection, peer = await _hello_connection(tmp_path, monkeypatch)
    try:
        assert peer.client_identity is None
        reply = await _request(connection, peer, 1, {'client': {'name': 'hermes-desktop', 'version': '2.1.0',
                                                                'kind': 'watch'}})
        assert 'result' in reply, reply
        assert peer.client_identity == {'name': 'hermes-desktop', 'version': '2.1.0'}
        await _request(connection, peer, 2, {'client': {'name': 'hermes-desktop', 'kind': 'desktop'}})
        assert peer.client_identity == {'name': 'hermes-desktop', 'kind': 'desktop'}
    finally:
        await connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [{'name': 'Hermes'}, {'name': 'x', 'kind': 7}, {'name': 'x', 'version': 'v 1'}, None])
async def test_a_malformed_hello_is_refused_and_the_earlier_one_kept(tmp_path, monkeypatch, bad):
    conn, connection, peer = await _hello_connection(tmp_path, monkeypatch)
    try:
        await _request(connection, peer, 1, {'client': {'name': 'hermes-webapp', 'kind': 'webapp'}})
        refused = await _request(connection, peer, 2, {'client': bad})
        assert refused['error']['code'] == 4001, refused
        assert peer.client_identity == {'name': 'hermes-webapp', 'kind': 'webapp'}
        # A later call that names other fields only keeps the announced client.
        assert 'result' in await _request(connection, peer, 3, {'server_requests': True})
        assert peer.client_identity == {'name': 'hermes-webapp', 'kind': 'webapp'}
    finally:
        await connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
async def test_a_hello_sent_alone_changes_nothing_else_the_connection_declared(tmp_path, monkeypatch):
    from tui_gateway import server_requests
    conn, connection, peer = await _hello_connection(tmp_path, monkeypatch)
    try:
        reply = await _request(connection, peer, 1, {'client': {'name': 'hermes-desktop', 'kind': 'desktop'}})
        assert 'clarify' in reply['result']['server_requests']
        assert server_requests.answers_requests(peer) is False
        assert peer.desktop_local is False
        await _request(connection, peer, 2, {'server_requests': True})
        assert server_requests.answers_requests(peer) is True
    finally:
        await connection.close()
        await conn.connection.close()


@pytest.mark.asyncio
async def test_a_transport_that_keeps_no_identity_ignores_the_hello(tmp_path, monkeypatch):
    """stdio and other transports have no ``client_identity``: the call succeeds and stores nothing."""
    conn = await _connect(tmp_path, monkeypatch)
    try:
        assert isinstance(conn.transport, _Transport)
        reply = await _request(conn.connection, conn.transport, 1, {'client': {'name': 'hermes-desktop'}})
        assert 'result' in reply, reply
        assert not hasattr(conn.transport, 'client_identity')
    finally:
        await conn.connection.close()
