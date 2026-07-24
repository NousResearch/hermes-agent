#!/usr/bin/env python3
"""Self-contained, default-deny HTTP(S) forward proxy for the Docker sandbox.

This file is mounted read-only into a ``python:3.13-slim`` container and run with
``python /egress_proxy_server.py`` to provide domain-allowlisted egress for the
"allowlist" network mode (see :mod:`tools.environments.egress_proxy`).

Design constraints:
- **stdlib only.** It runs inside a vanilla python-slim image with no extra deps
  and cannot import the hermes package — everything it needs lives in this file.
- **default-deny.** Only hosts matching ``EGRESS_ALLOWLIST`` are reachable; every
  other CONNECT/absolute-URI request gets ``403``. An empty allowlist denies all.
- **CONNECT for HTTPS.** The proxy never terminates TLS; it allowlists by the
  CONNECT host (so no certificate interception) and blind-tunnels bytes.

The host-matching logic (:func:`host_allowed`) is the security-critical core and
is unit-tested from the hermes test suite by importing this module directly.
"""

from __future__ import annotations

import ipaddress
import os
import select
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def parse_allowlist(raw: str | None) -> list[str]:
    """Parse the comma/whitespace-separated ``EGRESS_ALLOWLIST`` env value."""
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for chunk in raw.replace("\n", ",").split(","):
        host = chunk.strip().lower()
        if host and host not in seen:
            seen.add(host)
            out.append(host)
    return out


def _is_ip_literal(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def host_allowed(host: str, allowlist: list[str]) -> bool:
    """Return True iff ``host`` is permitted by ``allowlist`` (default-deny).

    Matching is case-insensitive and supports:
    - exact match: ``github.com`` matches ``github.com``
    - leading-wildcard suffix: ``*.githubusercontent.com`` matches
      ``objects.githubusercontent.com`` but NOT the bare ``githubusercontent.com``
    - bare-domain subdomain match: an entry ``example.com`` also matches
      ``api.example.com`` (a parent domain authorizes its subdomains)
    """
    if not host or not allowlist:
        return False
    host = host.strip().lower().rstrip(".")
    for entry in allowlist:
        entry = entry.strip().lower().rstrip(".")
        if not entry:
            continue
        # Defense in depth: a literal IP must be listed exactly (no subdomain
        # or wildcard logic — "x.1.2.3.4" must not match an entry "1.2.3.4").
        if _is_ip_literal(entry):
            if host == entry:
                return True
            continue
        if entry.startswith("*."):
            suffix = entry[1:]  # ".githubusercontent.com"
            if host.endswith(suffix) and host != suffix.lstrip("."):
                return True
        elif host == entry or host.endswith("." + entry):
            return True
    return False


_ALLOWLIST = parse_allowlist(os.environ.get("EGRESS_ALLOWLIST"))
_PROXY_PORT = int(os.environ.get("PROXY_PORT", "8888"))
_TUNNEL_BUF = 65536


def _host_from_authority(authority: str) -> tuple[str, int]:
    """Split ``host:port`` (CONNECT target) into (host, port)."""
    if authority.startswith("["):  # IPv6 literal [::1]:443
        host, _, rest = authority[1:].partition("]")
        port = int(rest.lstrip(":")) if rest.lstrip(":") else 443
        return host, port
    host, _, port = authority.partition(":")
    return host, int(port) if port else 443


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter, line-per-request to stderr
        sys.stderr.write("egress-proxy: " + (fmt % args) + "\n")

    def _deny(self, host: str):
        self.send_response(403)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Connection", "close")
        body = f"egress denied by allowlist: {host}\n".encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_CONNECT(self):  # noqa: N802 — HTTPS tunneling
        host, port = _host_from_authority(self.path)
        if not host_allowed(host, _ALLOWLIST):
            self.log_message("DENY CONNECT %s", host)
            return self._deny(host)
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError as e:
            self.send_error(502, f"upstream connect failed: {e}")
            return
        self.send_response(200, "Connection Established")
        self.end_headers()
        self._tunnel(self.connection, upstream)

    def _forward_plain(self):
        """Forward an absolute-URI HTTP request (http://host/...)."""
        parts = urlsplit(self.path)
        host = parts.hostname or ""
        if not host_allowed(host, _ALLOWLIST):
            self.log_message("DENY %s %s", self.command, host)
            return self._deny(host)
        port = parts.port or 80
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError as e:
            self.send_error(502, f"upstream connect failed: {e}")
            return
        req = [f"{self.command} {path} HTTP/1.1", f"Host: {parts.netloc}"]
        for k, v in self.headers.items():
            # Host is re-derived from the absolute URI above; copying the
            # client's would send a duplicate Host header upstream.
            if k.lower() in ("proxy-connection", "connection", "host"):
                continue
            req.append(f"{k}: {v}")
        req.append("Connection: close")
        req.append("")
        req.append("")
        upstream.sendall("\r\n".join(req).encode() + body)
        self._tunnel(self.connection, upstream, half=True)

    # All non-CONNECT verbs route through the same absolute-URI forwarder.
    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_PATCH = do_OPTIONS = _forward_plain

    def _tunnel(self, client: socket.socket, upstream: socket.socket, half: bool = False):
        """Blind bidirectional byte tunnel until either side closes."""
        socks = [client, upstream]
        try:
            while True:
                readable, _, errored = select.select(socks, [], socks, 60)
                if errored:
                    break
                if not readable:
                    break
                for s in readable:
                    data = s.recv(_TUNNEL_BUF)
                    if not data:
                        return
                    (upstream if s is client else client).sendall(data)
        except OSError:
            pass
        finally:
            for s in (upstream,):
                try:
                    s.close()
                except OSError:
                    pass


def main() -> int:
    sys.stderr.write(
        f"egress-proxy: listening on :{_PROXY_PORT}, allowlist={_ALLOWLIST or 'DENY-ALL'}\n"
    )
    server = ThreadingHTTPServer(("0.0.0.0", _PROXY_PORT), _ProxyHandler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
