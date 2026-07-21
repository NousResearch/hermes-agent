"""Small authenticated client for Home Assistant's official REST and WebSocket APIs."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


class HomeAssistantClientError(RuntimeError):
    """Sanitized Home Assistant API failure."""


class HomeAssistantClient:
    def __init__(self, base_url: str, token: str, *, timeout_seconds: int = 15):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Home Assistant URL must use http or https")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Home Assistant URL cannot contain credentials, query, or fragment")
        if not token:
            raise ValueError("Home Assistant token is required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    async def rest(self, method: str, path: str, payload: Any = None) -> Any:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=self._headers,
                    json=payload,
                    allow_redirects=False,
                ) as response:
                    if response.status < 200 or response.status >= 300:
                        raise HomeAssistantClientError(
                            f"Home Assistant returned HTTP {response.status}"
                        )
                    if response.status == 204:
                        return None
                    return await response.json(content_type=None)
        except HomeAssistantClientError:
            raise
        except Exception as exc:
            raise HomeAssistantClientError(
                f"Home Assistant request failed: {type(exc).__name__}"
            ) from exc

    async def websocket(self, command: dict[str, Any]) -> Any:
        import aiohttp

        ws_url = self.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(f"{ws_url}/api/websocket") as socket:
                    hello = await socket.receive_json()
                    if hello.get("type") != "auth_required":
                        raise HomeAssistantClientError("unexpected Home Assistant WebSocket handshake")
                    await socket.send_json({"type": "auth", "access_token": self.token})
                    auth = await socket.receive_json()
                    if auth.get("type") != "auth_ok":
                        raise HomeAssistantClientError("Home Assistant WebSocket authentication failed")
                    request = {"id": 1, **command}
                    await socket.send_json(request)
                    result = await socket.receive_json()
                    if result.get("type") != "result" or not result.get("success"):
                        error = result.get("error", {})
                        code = error.get("code", "unknown_error")
                        raise HomeAssistantClientError(f"Home Assistant WebSocket error: {code}")
                    return result.get("result")
        except HomeAssistantClientError:
            raise
        except Exception as exc:
            raise HomeAssistantClientError(
                f"Home Assistant WebSocket request failed: {type(exc).__name__}"
            ) from exc
