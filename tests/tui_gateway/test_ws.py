"""Tests for tui_gateway.ws message-size guards."""

from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _FakeDisconnect(Exception):
    def __init__(self, code: int = 1000, reason: str = "bye") -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


class _FakeWebSocket:
    def __init__(self, incoming: list[object]) -> None:
        self._incoming = list(incoming)
        self.client = SimpleNamespace(host="127.0.0.1", port=43124)
        self.accepted = False
        self.sent_text: list[str] = []
        self.closed_calls: list[dict[str, object | None]] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    async def receive_text(self) -> str:
        if not self._incoming:
            raise _FakeDisconnect()
        item = self._incoming.pop(0)
        if isinstance(item, BaseException):
            raise item
        return str(item)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed_calls.append({"code": code, "reason": reason})


@pytest.fixture()
def ws_mod():
    sys.modules.pop("tui_gateway.ws", None)
    sys.modules.pop("tui_gateway.server", None)
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        mod = importlib.import_module("tui_gateway.ws")
        yield mod


@pytest.mark.asyncio
async def test_oversized_message_closes_with_1009(ws_mod, monkeypatch, caplog):
    oversized = "x" * 9
    ws = _FakeWebSocket([oversized])

    monkeypatch.setattr(ws_mod, "_MAX_WS_MESSAGE_BYTES", 8)
    monkeypatch.setattr(ws_mod, "_WebSocketDisconnect", _FakeDisconnect)
    monkeypatch.setattr(ws_mod, "_disable_nagle", lambda _ws: None)
    monkeypatch.setattr(ws_mod.server, "resolve_skin", lambda: "test-skin")
    monkeypatch.setattr(ws_mod.server, "dispatch", lambda req, transport: {"unexpected": True})
    monkeypatch.setattr(ws_mod.server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))

    with caplog.at_level("WARNING"):
        await ws_mod.handle_ws(ws)

    assert ws.accepted is True
    assert any(close["code"] == 1009 and close["reason"] == "Message too large" for close in ws.closed_calls)
    assert "ws oversized message rejected" in caplog.text
    assert len(ws.sent_text) == 1
    assert json.loads(ws.sent_text[0])["params"]["type"] == "gateway.ready"


@pytest.mark.asyncio
async def test_boundary_message_dispatches(ws_mod, monkeypatch):
    request = {"jsonrpc": "2.0", "id": "r1", "method": "ping"}
    raw = json.dumps(request)
    responses: list[dict[str, object]] = []
    ws = _FakeWebSocket([raw, _FakeDisconnect()])

    monkeypatch.setattr(ws_mod, "_MAX_WS_MESSAGE_BYTES", len(raw.encode("utf-8")))
    monkeypatch.setattr(ws_mod, "_WebSocketDisconnect", _FakeDisconnect)
    monkeypatch.setattr(ws_mod, "_disable_nagle", lambda _ws: None)
    monkeypatch.setattr(ws_mod.server, "resolve_skin", lambda: "test-skin")
    monkeypatch.setattr(
        ws_mod.server,
        "dispatch",
        lambda req, transport: responses.append(req) or {"jsonrpc": "2.0", "id": req["id"], "result": {"ok": True}},
    )
    monkeypatch.setattr(ws_mod.server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))

    await ws_mod.handle_ws(ws)

    assert responses == [request]
    assert any(json.loads(text).get("result") == {"ok": True} for text in ws.sent_text)
    assert not any(close["code"] == 1009 for close in ws.closed_calls)
