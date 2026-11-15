"""A Streamable HTTP server that rejects the configured ``Authorization`` header parks once and stays
quiet until the credentials change.

Regression for #148455: mcp >= 2.0 folds a 401/403 on the ``initialize`` POST into an ``MCPError``
without ``.response``, so the failure read as transient. Every attempt ran the initial-connect ladder
(plus an SSE fallback on the SDK's opaque -32603 for it), and the parked self-probe and discovery
nudges replayed the same rejected token every ``_PARKED_RETRY_INTERVAL``, all day. Each test serves
a real MCP server from a subprocess behind a header check that journals every request to its MCP
path (other local clients probing the loopback port must not count).
"""

from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
import sys

import pytest

import hermes_yaml as yaml
from hermes_constants import get_hermes_home
from tests.waiting import async_wait_until, wait_until
from tools import mcp_tool
from tools.mcp_tool import MCPServerTask

_SERVER = """
import asyncio, json, socket, sys
import uvicorn
from mcp.server import MCPServer
accepted, variant, status, journal_path, port_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
server = MCPServer("guarded")

def noop() -> str:
    return "ok"

server.add_tool(noop, name="noop", description="noop")
inner = server.streamable_http_app()
journal = open(journal_path, "a", encoding="utf-8")
if variant == "json":
    reject_type = b"application/json"
    reject_body = json.dumps({"jsonrpc": "2.0", "id": None,
                              "error": {"code": -32001, "message": "Unauthorized"}}).encode()
else:
    reject_type, reject_body = b"text/plain", b"Unauthorized"

async def app(scope, receive, send):
    if scope["type"] != "http":
        return await inner(scope, receive, send)
    granted = dict(scope["headers"]).get(b"authorization", b"").decode() == f"Bearer {accepted}"
    if scope["path"] == "/mcp":
        journal.write(f"{scope['method']} {'ok' if granted else status}\\n")
        journal.flush()
    if granted:
        return await inner(scope, receive, send)
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", reject_type), (b"content-length", str(len(reject_body)).encode())]})
    await send({"type": "http.response.body", "body": reject_body})

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
uv = uvicorn.Server(uvicorn.Config(app, log_level="warning"))

async def main():
    task = asyncio.create_task(uv.serve(sockets=[sock]))
    while not uv.started:
        await asyncio.sleep(0.01)
    open(port_path, "w", encoding="utf-8").write(str(sock.getsockname()[1]))
    await task

asyncio.run(main())
"""

_EXPIRED = "tok-expired-1"
_FRESH = "tok-fresh-000002"  # another length: the config cache keys on the file signature


@pytest.fixture
def guarded_server(tmp_path):
    """Start a header-checked MCP server ``(variant, status)``; yields ``(url, journal path)``."""
    procs = []

    def start(variant: str, status: int):
        script, port_file, journal = tmp_path / "guarded_server.py", tmp_path / "port", tmp_path / "journal"
        script.write_text(_SERVER, encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, str(script), _FRESH, variant, str(status),
                                       str(journal), str(port_file)]))
        wait_until(lambda: port_file.exists() and port_file.read_text(encoding="utf-8"), timeout=30,
                   what="MCP test server to bind")
        return f"http://127.0.0.1:{port_file.read_text(encoding='utf-8')}/mcp", journal

    yield start
    for proc in procs:
        proc.terminate()
        proc.wait(10)


def _write_config(entry: dict) -> None:
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"srv": entry}}),
                                                   encoding="utf-8")


def _count(journal, line: str) -> int:
    return journal.read_text(encoding="utf-8").splitlines().count(line) if journal.exists() else 0


def _starts_with(journal, prefix: str) -> int:
    lines = journal.read_text(encoding="utf-8").splitlines() if journal.exists() else []
    return sum(1 for line in lines if line.startswith(prefix))


class _Task(MCPServerTask):
    def __init__(self, name: str, state: dict):
        super().__init__(name)
        self.state = state

    async def _backoff_sleep(self, budget):
        return None

    async def _wait_for_reconnect_or_shutdown(self, timeout=None):
        outcome = await super()._wait_for_reconnect_or_shutdown(timeout)
        if outcome == "self-probe":
            self.state["wakes"] += 1
        return outcome


class _CountingTask(_Task):
    async def _run_http(self, config):
        self.state["attempts"] += 1
        return await super()._run_http(config)


@pytest.mark.parametrize(("variant", "status"), [("json", 401), ("text", 401), ("text", 403)])
def test_rejected_header_credentials_park_until_they_change(guarded_server, monkeypatch, caplog, variant, status):
    from tools.mcp_tool_config import _load_mcp_config
    from tools.mcp_tool_discovery import _select_new_servers
    from tools.mcp_tool_loop import _signal_reconnect

    url, journal = guarded_server(variant, status)
    _write_config({"url": url, "headers": {"Authorization": f"Bearer {_EXPIRED}"}})
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)
    state = {"wakes": 0}
    rejected = f"POST {status}"

    async def wakes_beyond(n: int, what: str):
        target = state["wakes"] + n
        await async_wait_until(lambda: state["wakes"] >= target, what=what)

    async def scenario():
        cfg = _load_mcp_config()["srv"]
        task = _Task("srv", state)
        monkeypatch.setitem(mcp_tool._servers, "srv", task)
        run = asyncio.ensure_future(task.run(cfg))
        try:
            await async_wait_until(lambda: task._was_parked, what="the server to park")
            await async_wait_until(lambda: state["wakes"] >= 3, what="three parked wakes")
            assert _count(journal, rejected) == 1, journal.read_text(encoding="utf-8")
            assert _starts_with(journal, "GET") == 0, journal.read_text(encoding="utf-8")

            for _ in range(3):
                _select_new_servers({"srv": cfg})
            await wakes_beyond(2, "wakes after discovery nudges")
            assert _count(journal, rejected) == 1, journal.read_text(encoding="utf-8")

            _signal_reconnect(task)
            await async_wait_until(lambda: _count(journal, rejected) == 2, what="the explicit reconnect")
            await wakes_beyond(2, "wakes after the explicit reconnect")
            assert _count(journal, rejected) == 2, journal.read_text(encoding="utf-8")

            _write_config({"url": url, "headers": {"Authorization": f"Bearer {_FRESH}"}})
            await async_wait_until(lambda: task.session is not None and task._ready.is_set(),
                                   what="revival with the fresh token")
            assert _count(journal, "POST ok") >= 1
        finally:
            await task.shutdown()
            await asyncio.wait_for(run, 10)

    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        asyncio.run(scenario())

    park_lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
                  and f"credentials rejected (HTTP {status} from POST" in r.getMessage()
                  and "/reload-mcp" in r.getMessage()]
    assert len(park_lines) == 1, park_lines
    assert _EXPIRED not in caplog.text


def test_network_park_keeps_probing_on_schedule(monkeypatch):
    from tools.mcp_tool_config import _load_mcp_config
    from tools.mcp_tool_discovery import _select_new_servers

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()
    _write_config({"url": f"http://127.0.0.1:{closed_port}/mcp", "headers": {"Authorization": "Bearer tok"}})
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.05)
    state = {"wakes": 0, "attempts": 0}

    async def scenario():
        cfg = _load_mcp_config()["srv"]
        task = _CountingTask("srv", state)
        monkeypatch.setitem(mcp_tool._servers, "srv", task)
        run = asyncio.ensure_future(task.run(cfg))
        try:
            await async_wait_until(lambda: task._was_parked, what="the server to park")
            n = state["attempts"]
            await async_wait_until(lambda: state["wakes"] >= 2 and state["attempts"] > n,
                                   what="a scheduled probe while parked")
            before = state["attempts"]
            _select_new_servers({"srv": cfg})
            await async_wait_until(lambda: state["attempts"] > before, what="a probe after a discovery nudge")
        finally:
            await task.shutdown()
            await asyncio.wait_for(run, 10)

    asyncio.run(scenario())
