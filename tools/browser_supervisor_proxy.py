"""A task-owned CDP view for clients that otherwise auto-select foreign tabs."""

from __future__ import annotations

import asyncio
import json
import secrets
from typing import Any

from websockets.asyncio.client import connect
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed


class OwnedPageProxy:
    """Keep agent-browser discovery and attachment on one immutable page target.

    Runs on the supervisor loop. Each client retains its own upstream connection,
    flattened CDP sessions, request IDs and page events, including child frames.
    This is a routing boundary for the browser driver, not a general CDP sandbox.
    """

    def __init__(self, cdp_url: str, target_id: str) -> None:
        self.cdp_url = cdp_url
        self.target_id = target_id
        self._path = "/" + secrets.token_urlsafe(32)
        self._server: Server | None = None
        self.url = ""
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> str:
        async with self._lifecycle_lock:
            if self._server is None:
                self._server = await serve(
                    self._handle, "127.0.0.1", 0, origins=[None],
                    max_size=50 * 1024 * 1024, close_timeout=1,
                )
                port = next(iter(self._server.sockets)).getsockname()[1]
                self.url = f"ws://127.0.0.1:{port}{self._path}"
            return self.url

    async def close(self) -> None:
        async with self._lifecycle_lock:
            server, self._server = self._server, None
            if server is not None:
                server.close()
                await server.wait_closed()

    async def _handle(self, client: ServerConnection) -> None:
        if client.request is None or client.request.path != self._path:
            await client.close(code=1008, reason="Unknown page endpoint")
            return
        requests: dict[int, str] = {}
        sessions: set[str] = set()
        try:
            async with connect(self.cdp_url, max_size=50 * 1024 * 1024,
                               open_timeout=5, close_timeout=1) as browser:
                async def to_browser() -> None:
                    async for raw in client:
                        message = json.loads(raw)
                        method = message.get("method", "")
                        params = message.get("params") or {}
                        target_id = params.get("targetId", self.target_id)
                        session_id = message.get("sessionId")
                        forbidden = (
                            target_id != self.target_id
                            or (session_id is not None and session_id not in sessions)
                            or method in {"Target.createTarget", "Target.createBrowserContext",
                                          "Target.disposeBrowserContext", "Target.attachToBrowserTarget", "Browser.close"}
                            or (method == "Target.setAutoAttach" and session_id is None)
                        )
                        if forbidden:
                            await client.send(json.dumps({"id": message["id"], "error": {
                                "code": -32000, "message": "Command is outside the task-owned page",
                            }}))
                            continue
                        if method == "Target.getTargetInfo":
                            message["params"] = {**params, "targetId": self.target_id}
                        requests[message["id"]] = method
                        await browser.send(json.dumps(message))

                async def to_client() -> None:
                    async for raw in browser:
                        message = json.loads(raw)
                        if "id" in message:
                            method = requests.pop(message["id"], "")
                            result = message.get("result") or {}
                            if method == "Target.getTargets":
                                result["targetInfos"] = [target for target in result.get("targetInfos", [])
                                                         if target.get("targetId") == self.target_id]
                            if method == "Target.attachToTarget" and result.get("sessionId"):
                                sessions.add(result["sessionId"])
                        elif not self._owned_event(message, sessions):
                            continue
                        await client.send(json.dumps(message))

                tasks = [asyncio.create_task(to_browser()), asyncio.create_task(to_client())]
                try:
                    completed, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in completed:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except ConnectionClosed:
            pass
        except Exception as exc:
            # Upstream WebSocket exceptions may contain endpoint credentials.
            await client.close(code=1011, reason=f"Page connection failed: {type(exc).__name__}")

    def _owned_event(self, message: dict[str, Any], sessions: set[str]) -> bool:
        method = message.get("method", "")
        params = message.get("params") or {}
        parent_session = message.get("sessionId")
        if parent_session is not None and parent_session not in sessions:
            return False
        if method == "Target.attachedToTarget":
            target = params.get("targetInfo") or {}
            owned = target.get("targetId") == self.target_id
            child = parent_session in sessions and target.get("type") not in {"page", "webview"}
            if not (owned or child):
                return False
            sessions.add(params["sessionId"])
        elif method == "Target.detachedFromTarget":
            session_id = params.get("sessionId")
            if session_id not in sessions:
                return False
            sessions.discard(session_id)
        elif method.startswith("Target."):
            target_id = (params.get("targetInfo") or {}).get("targetId", params.get("targetId"))
            if target_id is not None and target_id != self.target_id:
                return False
        return True
