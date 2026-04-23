"""WebSocket transport for the tui_gateway JSON-RPC server."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from tui_gateway import server

_log = logging.getLogger(__name__)
_WS_WRITE_TIMEOUT_S = 10.0

try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:  # pragma: no cover
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


class WSTransport:
    def __init__(self, ws: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._ws = ws
        self._loop = loop
        self._closed = False

    def write(self, obj: dict) -> bool:
        if self._closed:
            return False

        line = json.dumps(obj, ensure_ascii=False)

        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False

        if on_loop:
            self._loop.create_task(self._safe_send(line))
            return True

        try:
            fut = asyncio.run_coroutine_threadsafe(self._safe_send(line), self._loop)
            fut.result(timeout=_WS_WRITE_TIMEOUT_S)
            return not self._closed
        except Exception as exc:
            self._closed = True
            _log.debug("ws write failed: %s", exc)
            return False

    async def write_async(self, obj: dict) -> bool:
        if self._closed:
            return False
        await self._safe_send(json.dumps(obj, ensure_ascii=False))
        return not self._closed

    async def _safe_send(self, line: str) -> None:
        try:
            await self._ws.send_text(line)
        except Exception as exc:
            self._closed = True
            _log.debug("ws send failed: %s", exc)

    def close(self) -> None:
        self._closed = True


async def handle_ws(ws: Any) -> None:
    await ws.accept()

    transport = WSTransport(ws, asyncio.get_running_loop())

    await transport.write_async(
        {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": "gateway.ready",
                "payload": {"skin": server.resolve_skin()},
            },
        }
    )

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except _WebSocketDisconnect:
                break

            line = raw.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                ok = await transport.write_async(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "parse error"},
                        "id": None,
                    }
                )
                if not ok:
                    break
                continue

            resp = await asyncio.to_thread(server.dispatch, req, transport)
            if resp is not None and not await transport.write_async(resp):
                break
    finally:
        transport.close()
        for _, sess in list(server._sessions.items()):
            if sess.get("transport") is transport:
                sess["transport"] = server._stdio_transport
        try:
            await ws.close()
        except Exception:
            pass
