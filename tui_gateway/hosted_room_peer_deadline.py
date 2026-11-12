"""Whole-request HTTP deadlines for the room supervisor's bounded maintenance."""

from __future__ import annotations

import http.client
import io
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from concurrent.futures import Future
from contextlib import suppress

_MAX_OUTSTANDING_ATTEMPTS = 4
_attempt_lock = threading.Lock()
_attempt_targets: set[tuple[str, int]] = set()


class _Budget:
    def __init__(self, timeout: float):
        self.deadline = time.monotonic() + timeout
        self.lock = threading.Lock()
        self.sockets = []
        self.closed = False

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if self.closed or remaining <= 0:
            raise TimeoutError('peer request deadline exceeded')
        return remaining

    def track(self, sock):
        with self.lock:
            if self.closed:
                sock.close()
                raise TimeoutError('peer request deadline exceeded')
            self.sockets.append(sock)
        return sock

    def connect(self, address, timeout=None, source_address=None):
        # A DNS lookup may outlive the caller's budget. Its eventual socket is closed before
        # sending credentials if the caller has already timed out.
        return self.track(socket.create_connection(address, self.remaining(), source_address))

    def close(self):
        with self.lock:
            self.closed = True
            for sock in self.sockets:
                # shutdown interrupts HTTPResponse's buffered read even after getresponse detached
                # the socket from its connection. close alone can leave that read trickling forever.
                with suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                sock.close()
            self.sockets.clear()


class _HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, budget, **kwargs):
        super().__init__(*args, **kwargs)
        self._budget = budget
        self._create_connection = budget.connect

    def connect(self):
        super().connect()
        self._budget.track(self.sock)  # TLS replaces the raw socket.


class _DeadlineHandler(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    def __init__(self, budget):
        super().__init__()
        self.budget = budget

    def http_open(self, request):
        def connection(*args, **kwargs):
            client = http.client.HTTPConnection(*args, **kwargs)
            client._create_connection = self.budget.connect
            return client
        return self.do_open(connection, request)

    def https_open(self, request):
        return self.do_open(lambda *a, **kw: _HTTPSConnection(*a, budget=self.budget, **kw), request)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch(request, budget):
    from agent.ssl_verify import install_truststore
    from tui_gateway.hosted_room_peer_http import (
        MAX_PEER_ERROR_RESPONSE_BYTES, MAX_PEER_RESPONSE_BYTES, _read_bounded_response)

    install_truststore()
    opener = urllib.request.build_opener(_DeadlineHandler(budget), _NoRedirect())
    try:
        response = opener.open(request, timeout=budget.remaining())
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        limit = MAX_PEER_ERROR_RESPONSE_BYTES if response.code >= 300 else MAX_PEER_RESPONSE_BYTES
        body = _read_bounded_response(response, max_bytes=limit, deadline=budget.deadline)
        stream = io.BytesIO(body)
        if response.code >= 300:
            raise urllib.error.HTTPError(request.full_url, response.code, 'peer request refused',
                                         response.headers, stream)
        return urllib.response.addinfourl(stream, response.headers, request.full_url, response.code)


def open_peer_url(request, *, timeout: float):
    """Bound the caller's wait and outstanding attempts; never follow redirects."""
    from agent.memory_provider import spawn_context_thread

    url = urllib.parse.urlsplit(request.full_url)
    target = (url.hostname, url.port or (443 if url.scheme == 'https' else 80))
    budget, result = _Budget(timeout), Future()

    def fetch():
        try:
            result.set_result(_fetch(request, budget))
        except Exception as exc:  # health: allow BLE001 -- Future re-raises the unchanged exception on the caller
            result.set_exception(exc)
        finally:
            # A timed-out resolver retains its slot until it actually returns.
            with _attempt_lock:
                _attempt_targets.remove(target)

    worker = spawn_context_thread(fetch, name='room-revoke-http')
    with _attempt_lock:
        if target in _attempt_targets or len(_attempt_targets) >= _MAX_OUTSTANDING_ATTEMPTS:
            raise TimeoutError('peer request capacity reached; retry later')
        _attempt_targets.add(target)
        try:
            worker.start()
        except RuntimeError:
            _attempt_targets.remove(target)
            raise
    try:
        return result.result(timeout=budget.remaining())
    finally:
        budget.close()
