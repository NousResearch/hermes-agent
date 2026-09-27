"""Shared local stand-in for ``open.feishu.cn`` used by the Feishu user-grant tests (#11540).

Only the host table is redirected, so the tests drive real ``httpx`` against a real HTTP server:
form encoding, query strings, status codes and JSON bodies are the production ones.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterable, Tuple
from urllib.parse import parse_qs, urlparse

from tools import feishu_user_auth


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def configure_app(monkeypatch, *, domain: str = "feishu") -> None:
    """A configured Feishu bot app in this profile's secrets."""
    monkeypatch.setenv("FEISHU_APP_ID", "cli_app123")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret456")
    monkeypatch.setenv("FEISHU_DOMAIN", domain)


def point_open_host(monkeypatch, server: "FakeFeishu", domain: str = "feishu") -> None:
    """Redirect the ``open.*`` host for *domain* at the local scripted server."""
    monkeypatch.setitem(feishu_user_auth.OPEN_BASE_URLS, domain, server.base_url)


def token_payload(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "code": 0, "access_token": "u-access-1", "refresh_token": "u-refresh-1",
        "expires_in": 7200, "token_type": "Bearer",
        "scope": "offline_access search:message im:message:readonly",
    }
    payload.update(overrides)
    return payload


def store_grant(*, access_token: str = "u-access-1", refresh_token: str = "u-refresh-1",
                expires_at: str = "2099-01-01T00:00:00+00:00") -> None:
    """Persist a grant directly, for tests that start from "already signed in"."""
    feishu_user_auth.save_state({
        "client_id": "cli_app123", "domain": "feishu",
        "redirect_uri": "http://127.0.0.1:43829/feishu/callback",
        "scope": feishu_user_auth.scope_string(), "granted_scope": feishu_user_auth.scope_string(),
        "token_type": "Bearer", "access_token": access_token, "refresh_token": refresh_token,
        "expires_at": expires_at, "expires_in": 7200,
        "auth_type": "oauth_authorization_code_pkce",
    })


class FakeFeishu:
    """Serves a scripted list of ``(status, payload)`` responses and logs every request."""

    def __init__(self, responses: Iterable[Tuple[int, Dict[str, Any]]]):
        self.responses = list(responses)
        self.requests: list[Dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _respond(self):
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                outer.requests.append({
                    "method": self.command, "path": parsed.path,
                    "query": {k: v[0] for k, v in parse_qs(parsed.query).items()},
                    "headers": dict(self.headers), "raw": raw.decode("utf-8"),
                })
                status, payload = (outer.responses.pop(0) if outer.responses
                                   else (500, {"error": "no scripted response"}))
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET = _respond

            def log_message(self, *_args):
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeFeishu":
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def form(self, index: int) -> Dict[str, str]:
        return {k: v[0] for k, v in parse_qs(self.requests[index]["raw"]).items()}

    def json_body(self, index: int) -> Dict[str, Any]:
        return json.loads(self.requests[index]["raw"])
