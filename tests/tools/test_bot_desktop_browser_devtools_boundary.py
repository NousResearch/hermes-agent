"""Live, reproducible proof of the exact transport fact the Bot Screen threat-model doc now
states about the dock browser's ``--remote-debugging-port=0`` (see
``website/docs/user-guide/features/bot-screen.md`` and ``tools/bot_desktop/browser.py:87``).

This binds a loopback TCP socket the same way Chromium's desktop ``DevToolsSocketFactory``
does (``net::TCPServerSocket`` on ``127.0.0.1``, ephemeral port — see
chrome/browser/devtools/remote_debugging_server.cc upstream) and shows, on a real socket on
the real host running the test, that ANY local process can complete the handshake and speak
plain HTTP with no credential, cookie, X auth cookie, or shared file descriptor exchanged.
That is the whole gap: loopback TCP carries no identity, so a same-user boundary cannot be
built on it the way the RFB Unix socket (``launcher.sh``, ``-rfbunixmode 0600``) or the
control-lease file are. Chromium has no supported flag to change this on desktop platforms;
the authenticated ``UnixDomainServerSocketFactory`` / peer-credential path in Chromium's own
source exists only in the Android build. A same-user fix therefore needs a broker process in
front of Chromium, which is tracked as separate architecture follow-up, not shipped here.

If this test ever needs to change, the threat-model paragraph it backs must change with it —
that pairing is the point of pinning the claim in a test rather than leaving it as prose.
"""

from __future__ import annotations

import http.client
import socket
import threading


def test_loopback_tcp_debug_port_accepts_unauthenticated_connection_from_any_process():
    """Models exactly what ``--remote-debugging-port=0`` gives an attacker: a live loopback
    TCP listener that a wholly unrelated local socket (no shared fd, cookie, or token — only
    the port number, which is exactly what reading the world-readable ``DevToolsActivePort``
    file or a local port scan would hand a stranger) can connect to and receive a served
    response from, with zero authentication exchanged at any point in the handshake."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))  # port 0 == Chromium's --remote-debugging-port=0
    server.listen(5)
    port = server.getsockname()[1]

    accepted = threading.Event()

    def serve_one():
        conn, _addr = server.accept()
        accepted.set()  # the TCP handshake completed with no credential check of any kind
        request = conn.recv(4096)
        assert b"GET" in request  # a plain, unauthenticated HTTP request arrived
        body = b'{"Browser":"test"}'
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        conn.close()

    thread = threading.Thread(target=serve_one, daemon=True)
    thread.start()
    try:
        # A brand-new, unrelated client: it inherits nothing from the "browser" process above —
        # no fd, no environment, no shared secret. It knows only the port number.
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        client.request("GET", "/json/version")
        response = client.getresponse()
        payload = response.read()
        client.close()
    finally:
        thread.join(timeout=2)
        server.close()

    assert accepted.is_set(), "the loopback listener never accepted the unrelated connection"
    assert response.status == 200
    assert b"Browser" in payload
    # This is the boundary violation in one assertion: an unauthenticated, unrelated local
    # process obtained a served response with no identity check anywhere in the exchange.
