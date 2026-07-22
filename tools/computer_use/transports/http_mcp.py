"""HTTP transport for a remotely hosted cua-driver MCP endpoint."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

from tools.computer_use.transports.base import CuaToolTransport


class HttpMcpTransport(CuaToolTransport):
    """Small JSON-RPC-over-HTTP transport suitable for fleet-backed CUA."""

    def __init__(self, endpoint: str, *, headers: Mapping[str, str] | None = None, timeout: float = 30):
        self.endpoint = endpoint.rstrip("/")
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._started = False
        self._request_id = 0

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def is_alive(self) -> bool:
        if not self._started:
            return False
        try:
            self._post({"jsonrpc": "2.0", "id": 0, "method": "ping", "params": {}})
        except (OSError, URLError, ValueError):
            return False
        return True

    def list_tools(self) -> list[Mapping[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": dict(arguments)})

    def _request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._started:
            self.start()
        self._request_id += 1
        response = self._post({"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params})
        if "error" in response:
            error = response["error"]
            raise RuntimeError(error.get("message", str(error)) if isinstance(error, Mapping) else str(error))
        result = response.get("result", {})
        return result if isinstance(result, Mapping) else {"result": result}

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        request = Request(self.endpoint, data=data, headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, Mapping):
            raise ValueError("MCP endpoint returned a non-object response")
        return parsed
