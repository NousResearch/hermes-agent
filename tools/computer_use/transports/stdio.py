"""Stdio transport for the existing ``cua-driver mcp`` process."""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any, Mapping, Sequence

from tools.computer_use.transports.base import CuaToolTransport


class StdioMcpTransport(CuaToolTransport):
    """Spawn cua-driver and exchange simple JSON-RPC messages over stdio."""

    def __init__(self, command: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None):
        self.command = list(command or ("cua-driver", "mcp"))
        self.env = dict(env) if env is not None else None
        self.process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._request_id = 0

    def start(self) -> None:
        if self.is_alive():
            return
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self.env,
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def list_tools(self) -> list[Mapping[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": dict(arguments)})

    def _request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.is_alive():
            self.start()
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("cua-driver stdio transport is unavailable")
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        if not line:
            raise RuntimeError("cua-driver closed its MCP stdio stream")
        response = json.loads(line)
        if "error" in response:
            error = response["error"]
            raise RuntimeError(error.get("message", str(error)) if isinstance(error, Mapping) else str(error))
        result = response.get("result", {})
        return result if isinstance(result, Mapping) else {"result": result}
