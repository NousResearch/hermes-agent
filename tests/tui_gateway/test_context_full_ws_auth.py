"""Integration-style auth-boundary tests for raw session.context_full over /api/ws.

The existing dashboard-auth suite documents a pre-existing Starlette TestClient
WebSocket-upgrade regression, so these tests drive the real route coroutine with
a WebSocket-shaped fake. This still exercises the production /api/ws auth gate
(`_ws_auth_ok` + request boundary) and then the real tui_gateway.ws.handle_ws
JSON-RPC loop on the accepted path.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

from hermes_cli import mcp_startup, web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests, mint_ticket
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
from tui_gateway import server
from tui_gateway import ws as ws_mod


SECRET = "SECRET_TOKEN=ws123"


class _QueryParams:
    def __init__(self, data: dict[str, str]):
        self._data = data

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(key, default)


class _FakeWebSocket:
    def __init__(self, *, query: dict[str, str], host: str, client_host: str, request: dict | None = None):
        self.query_params = _QueryParams(query)
        self.headers = {"host": host}
        self.client = SimpleNamespace(host=client_host, port=54321)
        self.url = SimpleNamespace(path="/api/ws")
        self.scope = {}
        self.sent: list[str] = []
        self.accepted = False
        self.closed_code = None
        self._request = request
        self._delivered = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, line: str):
        self.sent.append(line)

    async def receive_text(self):
        if self._request is not None and not self._delivered:
            self._delivered = True
            return json.dumps(self._request)
        raise ws_mod._WebSocketDisconnect()

    async def close(self, code=None):
        self.closed_code = code


def _agent():
    return SimpleNamespace(
        model="test-model",
        tools=[],
        _cached_system_prompt="cached system",
        ephemeral_system_prompt=None,
        _memory_store=None,
        _memory_enabled=True,
        _user_profile_enabled=True,
        context_compressor=SimpleNamespace(context_length=4096, last_prompt_tokens=0),
    )


def _configure_gated_app():
    _reset_for_tests()
    clear_providers()
    register_provider(StubAuthProvider())
    prev = (
        getattr(web_server.app.state, "bound_host", None),
        getattr(web_server.app.state, "bound_port", None),
        getattr(web_server.app.state, "auth_required", None),
    )
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    return prev


def _restore_app(prev):
    clear_providers()
    _reset_for_tests()
    web_server.app.state.bound_host = prev[0]
    web_server.app.state.bound_port = prev[1]
    web_server.app.state.auth_required = prev[2]


def test_unauthenticated_api_ws_cannot_invoke_context_full():
    prev = _configure_gated_app()
    fake = _FakeWebSocket(query={}, host="fly-app.fly.dev", client_host="203.0.113.10")
    try:
        asyncio.run(web_server.gateway_ws(fake))
    finally:
        _restore_app(prev)

    assert fake.accepted is False
    assert fake.closed_code == 4401
    assert fake.sent == []
    assert SECRET not in json.dumps(fake.sent)


def test_authenticated_api_ws_returns_raw_context_only_in_response_frame(monkeypatch):
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **kwargs: None)
    prev = _configure_gated_app()
    ticket = mint_ticket(user_id="u1", provider="stub")
    sid = "ctx-full-ws"
    server._sessions[sid] = {
        "agent": _agent(),
        "history": [{"role": "user", "content": f"hello {SECRET}"}],
        "history_lock": threading.RLock(),
    }
    request = {"jsonrpc": "2.0", "id": "ctx", "method": "session.context_full", "params": {"session_id": sid}}
    fake = _FakeWebSocket(
        query={"ticket": ticket},
        host="fly-app.fly.dev",
        client_host="203.0.113.10",
        request=request,
    )
    try:
        asyncio.run(web_server.gateway_ws(fake))
    finally:
        server._sessions.pop(sid, None)
        _restore_app(prev)

    frames = [json.loads(line) for line in fake.sent]
    event_frames = [frame for frame in frames if frame.get("method") == "event"]
    response_frames = [frame for frame in frames if frame.get("id") == "ctx"]

    assert fake.accepted is True
    assert response_frames
    assert SECRET in json.dumps(response_frames)
    assert SECRET not in json.dumps(event_frames)
