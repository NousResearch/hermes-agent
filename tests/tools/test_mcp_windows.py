"""Real sockets/children and MCP handshakes; only the cross-OS executable is
substituted on Linux. The actual PowerShell byte bridge runs on native Windows.
"""

import asyncio
import json
import queue
import ssl
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools import mcp_windows

# A byte-compatible stand-in for the Windows process boundary, not an HTTP or
# MCP mock. The real Unix socket, subprocess pipes, TLS, HTTPX and SDK all run.
_BYTE_BRIDGE = r'''
import os, socket, sys, threading
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
sys.stdout.buffer.write(b"\0"); sys.stdout.buffer.flush()
def receive():
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk: break
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
    except (OSError, BrokenPipeError):
        pass
threading.Thread(target=receive, daemon=True).start()
try:
    while True:
        chunk = os.read(0, 65536)
        if not chunk: break
        s.sendall(chunk)
finally:
    s.close()
'''


@contextmanager
def _endpoint(*, tls=None, sse=False):
    requests = []
    messages = queue.Queue()
    stopping = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def reply(self, status, body=b"", content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)
                self.wfile.flush()

        def do_HEAD(self):
            requests.append(("HEAD", self.path, dict(self.headers)))
            self.reply(200)

        def do_GET(self):
            requests.append(("GET", self.path, dict(self.headers)))
            if self.path.startswith("/sse") and sse:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"event: endpoint\ndata: /messages/\n\n")
                self.wfile.flush()
                while not stopping.is_set():
                    try:
                        message = messages.get(timeout=0.2)
                    except queue.Empty:
                        message = None
                    payload = b": keepalive\n\n" if message is None else b"event: message\ndata: " + message + b"\n\n"
                    try:
                        self.wfile.write(payload)
                        self.wfile.flush()
                    except OSError:
                        break
                return
            if self.path.startswith("/events"):
                self.reply(200, b"event: message\ndata: {\"ok\":true}\n\n", "text/event-stream")
            elif self.path.startswith("/mcp"):
                self.reply(405)
            else:
                self.reply(200, b'{"direct":true}')

        def do_POST(self):
            requests.append(("POST", self.path, dict(self.headers)))
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if "id" not in request:
                self.reply(202)
                return
            method = request["method"]
            results = {
                "initialize": {"protocolVersion": request.get("params", {}).get("protocolVersion"),
                               "serverInfo": {"name": "unity-fixture", "version": "1"}, "capabilities": {"tools": {}, "prompts": {}, "resources": {}}},
                "tools/list": {"tools": [{"name": "unity_ping", "description": "fixture", "inputSchema": {"type": "object"}}]},
                "tools/call": {"content": [{"type": "text", "text": "pong"}]},
                "prompts/list": {"prompts": [{"name": "scene"}]},
                "prompts/get": {"messages": [{"role": "user", "content": {"type": "text", "text": "scene prompt"}}]},
                "resources/list": {"resources": [{"name": "scene", "uri": "scene://active"}]},
                "resources/templates/list": {"resourceTemplates": []},
                "resources/read": {"contents": [{"uri": "scene://active", "text": "scene data"}]},
                "ping": {},
            }
            response = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": results[method]}).encode()
            if sse:
                self.reply(202)
                messages.put(response)
            else:
                self.reply(200, response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if tls:
        server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        stopping.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _tls_contexts(tmp_path):
    from datetime import datetime, timedelta, timezone
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                   .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1))
                   .not_valid_after(now + timedelta(days=1))
                   .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
                   .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_path, key_path)
    client = ssl.create_default_context(cafile=str(cert_path))
    names = []
    server.set_servername_callback(lambda _socket, name, _ctx: names.append(name))
    return server, client, names


@pytest.mark.linux_only
@pytest.mark.asyncio
@pytest.mark.parametrize("httpx_name", ["httpx", "httpx2"])
@pytest.mark.parametrize("scheme", ["http", "https"])
async def test_bridge_preserves_origin_tls_streaming_and_never_captures_another_port(monkeypatch, tmp_path, httpx_name, scheme):
    httpx = pytest.importorskip(httpx_name)
    tls, verify, sni = _tls_contexts(tmp_path) if scheme == "https" else (None, True, [])
    children = []
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        process = await real_spawn(*args, **kwargs)
        children.append(process)
        assert "TEST_API_KEY" not in kwargs["env"]
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setenv("TEST_API_KEY", "must-not-reach-the-bridge")
    with _endpoint(tls=tls) as (port, requests), _endpoint(tls=tls) as (other_port, other_requests):
        monkeypatch.setattr(mcp_windows, "_windows_tunnel_command",
                            lambda *_: [sys.executable, "-u", "-c", _BYTE_BRIDGE, str(port)])
        # Default ports are intentional: HTTPX's mount matcher treats :80/:443
        # as wildcard ports, so this catches a real cross-origin routing leak.
        url = f"{scheme}://localhost/mcp?project=unity"
        async with mcp_windows.windows_loopback_route(url) as route:
            socket_path = Path(route.socket_path)
            assert socket_path.parent.stat().st_mode & 0o077 == 0
            async with httpx.AsyncClient(verify=verify, trust_env=False,
                                        **route.client_options(httpx, verify=verify)) as client:
                response = await client.head(url, headers={"Authorization": "Bearer fixture"})
                assert response.status_code == 200
                async with client.stream("GET", f"{scheme}://localhost/events") as response:
                    assert b'data: {"ok":true}' in b"".join([chunk async for chunk in response.aiter_bytes()])
                other = await client.get(f"{scheme}://localhost:{other_port}/direct")
                assert other.json() == {"direct": True}
            assert requests[0][1] == "/mcp?project=unity"
            assert requests[0][2]["Host"] == "localhost"
            assert requests[0][2]["Authorization"] == "Bearer fixture"
            assert all(path != "/direct" for _, path, _ in requests)
            assert other_requests[0][1] == "/direct"
        assert not socket_path.exists()
        assert children and all(process.returncode is not None for process in children)
        if scheme == "https":
            assert sni and set(sni) == {"localhost"}


@pytest.mark.linux_only
@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "sse"])
@pytest.mark.parametrize("network", ["auto", "windows"])
async def test_real_mcp_probe_and_runtime_use_the_same_windows_route(monkeypatch, transport, network):
    pytest.importorskip("mcp")
    from tools.mcp_tool import MCPServerTask

    with _endpoint(sse=transport == "sse") as (port, requests):
        monkeypatch.setattr(mcp_windows, "is_wsl", lambda: True)
        monkeypatch.setattr(mcp_windows, "_windows_tunnel_command",
                            lambda *_: [sys.executable, "-u", "-c", _BYTE_BRIDGE, str(port)])
        task = MCPServerTask("windows-fixture")
        path = "/sse" if transport == "sse" else "/mcp"
        config = {"url": f"http://localhost:1{path}", "transport": transport, "protocol": "legacy",
                  "network": network, "headers": {"Authorization": "Bearer fixture"}, "connect_timeout": 3}
        try:
            await asyncio.wait_for(task.start(config), timeout=10)
            assert [tool.name for tool in task._tools] == ["unity_ping"]
            result = await asyncio.wait_for(task.session.call_tool("unity_ping", {}), timeout=5)
            assert result.content[0].text == "pong"
            prompts = await task.session.list_prompts()
            assert prompts.prompts[0].name == "scene"
            prompt = await task.session.get_prompt("scene")
            assert prompt.messages[0].content.text == "scene prompt"
            resources = await task.session.list_resources()
            resource = await task.session.read_resource(str(resources.resources[0].uri))
            assert resource.contents[0].text == "scene data"
            if transport == "http":
                assert any(method == "HEAD" for method, _, _ in requests)
            assert all(headers["Host"] == "localhost:1" for _, _, headers in requests)
            assert all(headers.get("Authorization") == "Bearer fixture" for _, _, headers in requests)
        finally:
            await task.shutdown()


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_legacy_omission_never_opens_windows_bridge_or_sends_authorization(monkeypatch):
    """A stopped WSL server cannot transfer authority to a Windows canary.

    The canary is the substituted Windows namespace at the same advertised
    endpoint. If a bridge is opened, its byte relay reaches this HTTP server and
    records the secret-bearing request. Omitted ``network`` must do neither.
    """
    pytest.importorskip("mcp")
    from tools.mcp_tool import MCPServerTask

    bridge_calls = []
    with _endpoint() as (canary_port, canary_requests):
        monkeypatch.setattr(mcp_windows, "is_wsl", lambda: True)

        def windows_canary(host, advertised_port):
            bridge_calls.append((host, advertised_port))
            return [sys.executable, "-u", "-c", _BYTE_BRIDGE, str(canary_port)]

        monkeypatch.setattr(mcp_windows, "_windows_tunnel_command", windows_canary)
        task = MCPServerTask("legacy-local")
        config = {"url": "http://localhost:1/mcp",
                  "headers": {"Authorization": "Bearer must-not-cross"},
                  "connect_timeout": 0.2}
        assert await task._prepare_run(config)
        assert bridge_calls == []
        assert canary_requests == []


@pytest.mark.linux_only
@pytest.mark.asyncio
@pytest.mark.parametrize("network", ["auto", "windows"])
async def test_explicit_cross_namespace_choice_reaches_authorized_canary(monkeypatch, network):
    """Positive control for the same preflight path and Authorization header."""
    pytest.importorskip("mcp")
    from tools.mcp_tool import MCPServerTask

    bridge_calls = []
    with _endpoint() as (canary_port, canary_requests):
        monkeypatch.setattr(mcp_windows, "is_wsl", lambda: True)

        def windows_canary(host, advertised_port):
            bridge_calls.append((host, advertised_port))
            return [sys.executable, "-u", "-c", _BYTE_BRIDGE, str(canary_port)]

        monkeypatch.setattr(mcp_windows, "_windows_tunnel_command", windows_canary)
        task = MCPServerTask("explicit-crossing")
        config = {"url": "http://localhost:1/mcp", "network": network,
                  "headers": {"Authorization": "Bearer allowed-to-cross"},
                  "connect_timeout": 2}
        assert await task._prepare_run(config)
        assert bridge_calls == [("localhost", 1)]
        assert canary_requests
        assert canary_requests[0][2]["Authorization"] == "Bearer allowed-to-cross"


@pytest.mark.windows_only
@pytest.mark.asyncio
@pytest.mark.parametrize("host,bind_host", [("127.0.0.1", "127.0.0.1"), ("localhost", "127.0.0.1"),
                                          ("::1", "::1"), ("localhost", "::1")])
async def test_native_powershell_bridge_carries_binary_bytes_and_exits_on_stdin_eof(host, bind_host):
    import socket
    import socketserver

    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            while data := self.request.recv(65536):
                self.request.sendall(data)

    class EchoServer(socketserver.ThreadingTCPServer):
        address_family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET

    with EchoServer((bind_host, 0), Echo) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        process = await asyncio.create_subprocess_exec(
            *mcp_windows._windows_tunnel_command(host, server.server_address[1]),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            assert await asyncio.wait_for(process.stdout.readexactly(1), timeout=10) == b"\0"
            payload = bytes(range(256)) * 512
            receive = asyncio.create_task(process.stdout.readexactly(len(payload)))
            process.stdin.write(payload)
            await asyncio.wait_for(process.stdin.drain(), timeout=10)
            assert await asyncio.wait_for(receive, timeout=10) == payload
            process.stdin.close()
            await asyncio.wait_for(process.wait(), timeout=10)
            assert process.returncode == 0, (await process.stderr.read()).decode(errors="replace")
        finally:
            await mcp_windows._close_child(process)
            server.shutdown()
            thread.join(timeout=5)
