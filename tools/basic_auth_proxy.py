#!/usr/bin/env python3
"""Tiny localhost Basic Auth reverse proxy for the Hermes dashboard."""

from __future__ import annotations

import argparse
import base64
import hmac
import http.client
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=9120)
    parser.add_argument("--target", default="http://127.0.0.1:9119")
    parser.add_argument("--user", default=os.environ.get("HERMES_PROXY_USER", "claudio"))
    parser.add_argument("--password", default=os.environ.get("HERMES_PROXY_PASSWORD"))
    return parser.parse_args()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    target = "http://127.0.0.1:9119"
    expected_auth = ""
    session_cookie = ""

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_PATCH(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    def _authorized(self) -> bool:
        auth_header = self.headers.get("Authorization", "")
        if hmac.compare_digest(auth_header, self.expected_auth):
            return True
        cookie_header = self.headers.get("Cookie", "")
        cookies = [part.strip() for part in cookie_header.split(";")]
        return any(hmac.compare_digest(cookie, self.session_cookie) for cookie in cookies)

    def _reject(self) -> None:
        body = b"Authentication required\n"
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Hermes Dashboard"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        if not self._authorized():
            self._reject()
            return
        fresh_basic_auth = hmac.compare_digest(self.headers.get("Authorization", ""), self.expected_auth)

        target = urlsplit(self.target)
        path = self.path
        body_length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(body_length) if body_length else None

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "authorization"
        }
        headers["Host"] = target.netloc
        headers["X-Forwarded-Proto"] = "https"
        headers["X-Forwarded-Host"] = self.headers.get("Host", target.netloc)

        conn = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=120)
        try:
            conn.request(self.command, path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
        finally:
            conn.close()

        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            if key.lower() == "content-length":
                continue
            self.send_header(key, value)
        if fresh_basic_auth:
            self.send_header(
                "Set-Cookie",
                f"{self.session_cookie}; HttpOnly; Secure; SameSite=Lax; Path=/",
            )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> int:
    args = parse_args()
    if not args.password:
        raise SystemExit("Set HERMES_PROXY_PASSWORD or pass --password")
    token = base64.b64encode(f"{args.user}:{args.password}".encode()).decode()
    ProxyHandler.target = args.target.rstrip("/")
    ProxyHandler.expected_auth = f"Basic {token}"
    ProxyHandler.session_cookie = f"hermes_proxy_session={secrets.token_urlsafe(32)}"
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    print(f"Basic Auth proxy listening on http://{args.listen_host}:{args.listen_port} -> {ProxyHandler.target}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
