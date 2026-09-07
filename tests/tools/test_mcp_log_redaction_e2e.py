"""Exercise redaction across actual stdio JSON-RPC and the installed MCP SDK."""
import asyncio
import json
import logging
import sys

import pytest

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_config import _resolve_mcp_server_config


@pytest.mark.asyncio
@pytest.mark.parametrize("retire_file", ["rotate", "delete"])
async def test_stdio_notifications_keep_resolving_generation(
    tmp_path, monkeypatch, caplog, retire_file
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    secret = "opaque-wire-prefix-7391-middle-suffix-8427"
    env_file = tmp_path / "server.env"
    env_file.write_text(f"WIRE_VALUE={secret}\n", encoding="utf-8")
    child = tmp_path / "server.py"
    child.write_text(
        '''import json, os, sys

def send(message):
    print(json.dumps(message), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if "id" not in request:
        continue
    if method == "server/discover":
        send({"jsonrpc": "2.0", "id": request["id"], "error": {
            "code": -32601, "message": "wire-fallback " + os.environ["WIRE_VALUE"]}})
        continue
    if method == "initialize":
        result = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}, "logging": {}},
            "serverInfo": {"name": "redaction-wire-fixture", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{"name": "emit_log", "description": "Emit synthetic logs",
                             "inputSchema": {"type": "object", "properties": {}}}]}
    elif method == "tools/call":
        secret = os.environ["WIRE_VALUE"]
        for data in ("wire-full " + secret,
                     "x" * 1990 + secret + "y" * 100,
                     {"wire-nested": [secret]}):
            send({"jsonrpc": "2.0", "method": "notifications/message", "params": {
                "level": "warning", "logger": "wire-logger/" + secret, "data": data}})
        result = {"content": [{"type": "text", "text": "wire-complete"}]}
    else:
        result = {}
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})
''',
        encoding="utf-8",
    )
    config = _resolve_mcp_server_config({
        "command": sys.executable,
        "args": [str(child)],
        "env_file": str(env_file),
        "env": {"WIRE_VALUE": "${WIRE_VALUE}"},
        "protocol": "stateless",
        "connect_timeout": 10,
        "sampling": {"enabled": False},
        "elicitation": {"enabled": False},
    })
    if retire_file == "rotate":
        env_file.write_text("WIRE_VALUE=rotated-wire-value-9999\n", encoding="utf-8")
    else:
        env_file.unlink()

    received = asyncio.Event()

    class LogReceipt(logging.Handler):
        def __init__(self):
            super().__init__()
            self.count = 0

        def emit(self, record):
            if record.msg == "MCP server log [%s]: %s":
                self.count += 1
                if self.count == 3:
                    received.set()

    receipt = LogReceipt()
    logger = logging.getLogger("tools.mcp_tool")
    logger.addHandler(receipt)
    server = MCPServerTask("wire-redaction")
    try:
        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            await server.start(config)
            assert server.session is not None
            result = await server.session.call_tool("emit_log", {})
            await asyncio.wait_for(received.wait(), timeout=5)
            assert result.content[0].text == "wire-complete"
    finally:
        await server.shutdown()
        logger.removeHandler(receipt)

    records = [r for r in caplog.records if r.name == "tools.mcp_tool"]
    rendered = "\n".join(r.getMessage() for r in records)
    raw_args = json.dumps([r.args for r in records], default=str)
    for fragment in (secret, secret[:10], secret[-10:]):
        assert fragment not in rendered
        assert fragment not in raw_args
    assert "wire-full [REDACTED]" in rendered
    assert "wire-fallback [REDACTED]" in rendered
    assert "falling back to the legacy handshake" in rendered
    assert "wire-logger/[REDACTED]" in rendered
    assert "[truncated]" in rendered
    assert receipt.count == 3
