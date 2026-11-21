"""An OAuth rejection of a ``tools/call`` reaches the model as an auth problem.

mcp 2.x runs the OAuth flow inside the transport's request task. When a ``tools/call`` is answered
with a 401 that needs a new browser sign-in, or a 403 ``insufficient_scope`` step-up, the flow raises
there, the session is torn down, and the pending call only sees ``MCPError: Connection closed``. The
model was then told the transport session expired and the write's outcome was UNKNOWN, with no hint
to sign in again. These tests serve a real MCP server (``mcp.server.MCPServer``) behind a bearer
gate and an OAuth stub, and drive Hermes' real discovery and tool dispatch against it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from tests.waiting import wait_until

pytest.importorskip("mcp.server", reason="MCP SDK required")

_SERVER = r'''
import asyncio, json, socket, sys
import uvicorn
from contextlib import AsyncExitStack
from mcp.server import MCPServer

port_path, mode = sys.argv[1], sys.argv[2]
server = MCPServer("gated")

def ping() -> str:
    return "PONG"

server.add_tool(ping, name="ping", description="ping")
mcp_app = server.streamable_http_app()

async def send_json(send, status, body, headers=()):
    raw = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), *headers]})
    await send({"type": "http.response.body", "body": raw})

async def app(scope, receive, send):
    if scope["type"] != "http":
        return await mcp_app(scope, receive, send)
    base = f"http://127.0.0.1:{scope['server'][1]}"
    path = scope["path"]
    if path.startswith("/.well-known/oauth-protected-resource"):
        return await send_json(send, 200, {"resource": base + "/mcp", "authorization_servers": [base]})
    if path.startswith("/.well-known/"):
        return await send_json(send, 200, {"issuer": base, "authorization_endpoint": base + "/authorize",
                                           "token_endpoint": base + "/token", "response_types_supported": ["code"],
                                           "code_challenge_methods_supported": ["S256"]})
    if path == "/token":
        return await send_json(send, 400, {"error": "invalid_grant"})
    body, more = b"", True
    while more:
        msg = await receive()
        body, more = body + msg.get("body", b""), msg.get("more_body", False)
    method = json.loads(body).get("method") if body else None
    www = f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"'
    if method == "tools/call" and mode == "403":
        return await send_json(send, 403, {"error": "insufficient_scope"}, [(b"www-authenticate", (
            www + ', error="insufficient_scope", scope="tools:write"').encode())])
    if method == "tools/call" and mode == "401":
        return await send_json(send, 401, {"error": "TOKEN_EXPIRED"}, [(b"www-authenticate", www.encode())])
    replayed = []
    async def replay():
        if not replayed:
            replayed.append(1)
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()
    return await mcp_app(scope, replay, send)

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
uv = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))

async def main():
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(server.session_manager.run())
        task = asyncio.create_task(uv.serve(sockets=[sock]))
        while not uv.started:
            await asyncio.sleep(0.01)
        open(port_path, "w").write(str(sock.getsockname()[1]))
        await task

asyncio.run(main())
'''


@pytest.fixture
def gated_server(tmp_path, monkeypatch):
    """Start the gated server in *mode* with a signed-in token store; yields the HERMES_HOME."""
    procs = []

    def start(mode: str):
        home = tmp_path / "home"
        (home / "mcp-tokens").mkdir(parents=True)
        script, port_file = tmp_path / "gated.py", tmp_path / "port"
        script.write_text(_SERVER, encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, str(script), str(port_file), mode]))
        wait_until(lambda: port_file.exists() and port_file.read_text(encoding="utf-8"), timeout=30,
                   what="gated MCP server to bind")
        base = f"http://127.0.0.1:{port_file.read_text(encoding='utf-8')}"
        tokens = home / "mcp-tokens"
        (tokens / "gated.json").write_text(json.dumps({
            "access_token": "AT", "token_type": "Bearer", "refresh_token": "RT", "expires_in": 3600,
            "expires_at": time.time() + 3600}), encoding="utf-8")
        (tokens / "gated.client.json").write_text(json.dumps({
            "client_id": "cid", "redirect_uris": ["http://127.0.0.1:8421/callback"], "issuer": base,
            "token_endpoint_auth_method": "none"}), encoding="utf-8")
        (tokens / "gated.meta.json").write_text(json.dumps({
            "issuer": base, "authorization_endpoint": base + "/authorize", "token_endpoint": base + "/token",
            "response_types_supported": ["code"]}), encoding="utf-8")
        (home / "config.yaml").write_text(
            f"mcp_servers:\n  gated:\n    url: {base}/mcp\n    auth: oauth\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))

    yield start
    from tools import mcp_tool_loop as _loop
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    shutdown_mcp_servers()
    _loop._stop_mcp_loop()
    for proc in procs:
        proc.terminate()
        proc.wait(10)


def _call_ping() -> dict:
    from tools import mcp_tool_discovery as disc
    from tools.registry import registry

    assert "mcp__gated__ping" in disc.discover_mcp_tools()
    return json.loads(registry.dispatch("mcp__gated__ping", {}))


def test_insufficient_scope_on_tools_call_asks_for_a_new_sign_in(gated_server):
    gated_server("403")

    result = _call_ping()

    assert result.get("needs_reauth") is True and result.get("insufficient_scope") is True, result
    assert "tools:write" in result["error"] and "hermes mcp login gated" in result["error"], result
    assert "outcome_uncertain" not in result, result


def test_rejected_token_on_tools_call_asks_for_a_new_sign_in(gated_server):
    gated_server("401")

    result = _call_ping()

    assert result.get("needs_reauth") is True, result
    assert "hermes mcp login gated" in result["error"], result
    assert "outcome_uncertain" not in result and "insufficient_scope" not in result, result
