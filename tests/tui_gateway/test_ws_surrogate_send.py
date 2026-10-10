"""Lone UTF-16 surrogates must not tear down the Desktop WebSocket (#97288)."""

from __future__ import annotations

import asyncio

from tui_gateway.ws import WSTransport, _sanitize_ws_text


LONE_SURROGATE = "\ud83d"


def test_sanitize_ws_text_makes_utf8_encodable() -> None:
    dirty = f"gateway.ready {LONE_SURROGATE} payload"
    out = _sanitize_ws_text(dirty)
    out.encode("utf-8")
    assert LONE_SURROGATE not in out


def test_sanitize_ws_text_leaves_valid_text_unchanged() -> None:
    clean = '{"type":"gateway.ready","ok":true}'
    assert _sanitize_ws_text(clean) is clean or _sanitize_ws_text(clean) == clean


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.raise_on: str | None = None

    async def send_text(self, line: str) -> None:
        line.encode("utf-8")
        if self.raise_on is not None and self.raise_on in line:
            raise UnicodeEncodeError("utf-8", line, 0, 1, "surrogates not allowed")
        self.sent.append(line)


def test_safe_send_sanitizes_surrogate_and_keeps_connection() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        ws = _FakeWS()
        transport = WSTransport(ws, loop, peer="127.0.0.1:1")
        dirty = f'{{"type":"gateway.ready","x":"{LONE_SURROGATE}"}}'
        await transport._safe_send_many(["first", dirty, "third"])
        assert transport.closed is False
        assert ws.sent[0] == "first"
        assert ws.sent[-1] == "third"
        assert LONE_SURROGATE not in "".join(ws.sent)
        assert len(ws.sent) == 3

    asyncio.run(_run())


def test_unicode_encode_error_does_not_close_socket() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        ws = _FakeWS()
        ws.raise_on = "BOOM"
        transport = WSTransport(ws, loop, peer="127.0.0.1:1")
        await transport._safe_send_many(["ok-a", "BOOM-frame", "ok-b"])
        assert transport.closed is False
        assert ws.sent == ["ok-a", "ok-b"]

    asyncio.run(_run())
