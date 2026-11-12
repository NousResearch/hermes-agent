"""A peer trickling headers cannot monopolize the room supervisor's revoke cycle."""

import socket
import threading
import time

import pytest

from tests.gateway.test_session_group_peers import gateway as gateway_fixture
from hermes_agent.tui_gateway.hosted_room_driver import HostedRoomBinding

gateway = gateway_fixture


def test_revoke_worker_has_a_total_header_deadline(gateway):
    stop = threading.Event()
    with socket.create_server(('127.0.0.1', 0)) as listener:
        listener.settimeout(10)

        def trickle():
            with listener.accept()[0] as connection:
                connection.settimeout(10)
                connection.recv(8192)
                try:
                    connection.sendall(b'HTTP/1.1 200 OK\r\n')
                    # Regular traffic avoids a socket's inactivity timeout.
                    for _ in range(60):
                        if stop.wait(0.1):
                            return
                        connection.sendall(b'X-Peer: still here\r\n')
                    connection.sendall(b'Content-Length: 2\r\n\r\n{}')
                except OSError:
                    pass  # The bounded client closes the connection while we are still writing.

        worker = threading.Thread(target=trickle)
        worker.start()
        service = gateway.service
        service.owe_grant_revoke(room_id='ended', member_id='peer', grant='opaque-grant',
                                 target_url=f'http://127.0.0.1:{listener.getsockname()[1]}')
        runtime, scheduled = service.runtime, []
        rooms = iter(((), (HostedRoomBinding('next-room', 'home', 1),)))
        runtime._rooms_provider = lambda: next(rooms)
        runtime.poll_interval_seconds = 0

        def run_room(binding):
            scheduled.append(binding.room_id)
            runtime._stop.set()

        runtime._run_room_once = run_room
        try:
            started = time.monotonic()
            runtime._worker_loop()
            assert scheduled == ['next-room']
            assert time.monotonic() - started < 4.5
        finally:
            stop.set()
            worker.join(10)
        assert not worker.is_alive()


@pytest.mark.parametrize('response', ['settles', 'redirect', 'oversized'])
def test_revoke_transport_without_optional_http_packages(gateway, monkeypatch, response):
    import builtins
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from hermes_agent.gateway import hosted_rooms

    from hermes_agent.tui_gateway import hosted_room_peer_http
    monkeypatch.setattr(hosted_room_peer_http, 'MAX_PEER_RESPONSE_BYTES', 32)
    seen = []

    class Peer(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.headers.get('Authorization'))
            body = b'{"revoked":true}' + (b' ' * 64 if response == 'oversized' else b'')
            self.send_response(302 if response == 'redirect' and self.path != '/redirected' else 200)
            self.send_header('Location', '/redirected')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST

        def log_message(self, *args):
            pass

    original = builtins.__import__

    def core_only(name, *args, **kwargs):
        if name == 'aiohttp' or name.startswith('aiohttp.'):
            raise ModuleNotFoundError('aiohttp is not installed')
        return original(name, *args, **kwargs)

    with ThreadingHTTPServer(('127.0.0.1', 0), Peer) as peer:
        worker = threading.Thread(target=peer.serve_forever)
        worker.start()
        try:
            monkeypatch.setattr(builtins, '__import__', core_only)
            service = gateway.service
            service.owe_grant_revoke(room_id='ended', member_id='peer', grant='opaque-grant',
                                     target_url=f'http://127.0.0.1:{peer.server_port}')
            assert service.retry_owed_revokes() == int(response == 'settles')
            assert seen == ['HermesRoom opaque-grant']
            remaining = hosted_rooms.due_room_grant_revokes(service.db_path, limit=10, now=service.runtime.clock() + 10)
            assert len(remaining) == int(response != 'settles')
        finally:
            peer.shutdown()
            worker.join(10)


def test_unexpected_transport_error_keeps_debt_on_backoff(gateway, monkeypatch):
    import sqlite3
    from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient

    def failed_transport(*args, **kwargs):
        raise TypeError('transport failed')

    service = gateway.service
    monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', failed_transport)
    service.owe_grant_revoke(room_id='ended', member_id='peer', grant='opaque-grant',
                             target_url='http://127.0.0.1:9')
    assert service.retry_owed_revokes() == 0
    with sqlite3.connect(service.db_path) as conn:
        attempts, retry_at = conn.execute('SELECT attempts, next_attempt_at FROM hosted_room_owed_revokes').fetchone()
    assert attempts == 1 and retry_at > service.runtime.clock()


@pytest.mark.parametrize('targets, limit', [(list(range(6)), 4), ([0] * 6, 1)])
def test_blocked_resolvers_bound_outstanding_attempts_and_drain(gateway, monkeypatch, targets, limit):
    import sqlite3
    from hermes_agent.tui_gateway import hosted_room_service

    service, release, workers, destinations = gateway.service, threading.Event(), [], []
    now = [service.runtime.clock() + 10]
    monkeypatch.setattr(service.runtime, 'clock', lambda: now[0])
    monkeypatch.setattr(hosted_room_service, '_OWED_REVOKE_TIMEOUT_SECONDS', 0.5)

    def blocked_lookup(host, port, *args, **kwargs):
        workers.append(threading.current_thread())
        destinations.append((host, port))
        assert release.wait(30)
        raise socket.gaierror('resolver unavailable')

    monkeypatch.setattr(socket, 'getaddrinfo', blocked_lookup)
    for index, target in enumerate(targets):
        service.owe_grant_revoke(room_id=f'ended-{index}', member_id='peer', grant=f'grant-{index}',
                                 target_url=f'http://127.0.0.1:{12000 + target}')
    try:
        for cycle in range(3):
            started = time.monotonic()
            for _ in targets:
                assert service.retry_owed_revokes() == 0
            assert len(workers) == limit
            assert len(set(destinations)) == limit
            assert all(worker.is_alive() for worker in workers)
            if cycle:
                assert time.monotonic() - started < 2  # Saturation refuses without another deadline wait.
            with sqlite3.connect(service.db_path) as conn:
                rows = conn.execute('SELECT attempts, next_attempt_at FROM hosted_room_owed_revokes').fetchall()
            assert len(rows) == len(targets)
            assert all(attempts == cycle + 1 and retry_at > now[0] for attempts, retry_at in rows)
            now[0] += 100
    finally:
        release.set()
        for worker in workers:
            worker.join(10)
        assert all(not worker.is_alive() for worker in workers)
    # Finishing the abandoned attempts releases their slots, even when DNS fails.
    assert service.retry_owed_revokes() == 0
    assert len(workers) == limit + 1
    workers[-1].join(10)
    assert not workers[-1].is_alive()
