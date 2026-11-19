"""Same-server parallel admission must reach the Streamable HTTP wire.

Real SDK server, real MCPServerTask and registry handlers. The server waits for a
second request before completing the first, with a bounded escape so the serial
control still returns two successful, correctly associated results.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import uvicorn
from mcp.server import MCPServer

import hermes_yaml as yaml
from hermes_constants import get_hermes_home
from tests.waiting import wait_until


class _HttpPair:
    def __init__(self):
        self.active = self.peak = 0
        self.received = []
        self.pair = asyncio.Event()
        self.release = asyncio.Event()
        self.hold = False
        self.loop = None
        server = MCPServer("parallel-probe")

        @server.tool()
        async def probe(label: str) -> str:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.received.append(label)
            if self.active == 2:
                self.pair.set()
            try:
                if self.hold:
                    await self.release.wait()
                else:
                    try:
                        await asyncio.wait_for(self.pair.wait(), 2)
                    except TimeoutError:
                        pass  # serial controls must also finish successfully
                return f"answer:{label}"
            finally:
                self.active -= 1

        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}/mcp"
        self.uv = uvicorn.Server(uvicorn.Config(
            server.streamable_http_app(stateless_http=True, json_response=True),
            log_level="error", timeout_graceful_shutdown=2))

    def start(self):
        async def run():
            self.loop = asyncio.get_running_loop()
            await self.uv.serve(sockets=[self.sock])
        self.thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
        self.thread.start()
        wait_until(lambda: self.uv.started, timeout=15, what="loopback MCP server startup")
        return self

    def stop(self):
        self.loop.call_soon_threadsafe(self.release.set)
        self.uv.should_exit = True
        self.thread.join(10)
        self.sock.close()
        assert not self.thread.is_alive()


@pytest.fixture
def connected():
    from tools import mcp_tool as core
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry

    http = _HttpPair().start()

    def connect(*, opt_in=True, elicitation=False, sampling=False):
        cfg = {"url": http.url, "elicitation": {"enabled": elicitation},
               "sampling": {"enabled": sampling}, "keepalive_interval": 0}
        if opt_in is not None:
            cfg["supports_parallel_tool_calls"] = opt_in
        (get_hermes_home() / "config.yaml").write_text(
            yaml.safe_dump({"mcp_servers": {"parallel-probe": cfg}}), encoding="utf-8")
        from tools.mcp_tool_discovery import discover_mcp_tools
        names = discover_mcp_tools()
        assert "mcp__parallel_probe__probe" in names, names
        return core._servers["parallel-probe"], lambda args: registry.dispatch("mcp__parallel_probe__probe", args)

    yield http, connect
    shutdown_mcp_servers()
    http.stop()


def _pair(handler):
    barrier = threading.Barrier(2)
    def call(label):
        barrier.wait(5)
        return json.loads(handler({"label": label}))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, label) for label in ("a", "b")]
        return [future.result(15) for future in futures]


def test_opted_in_streamable_http_calls_overlap_on_server(connected):
    http, connect = connected
    server, handler = connect()
    assert _pair(handler) == [{"result": "answer:a"}, {"result": "answer:b"}]
    assert sorted(http.received) == ["a", "b"]
    assert http.peak == 2, "successful calls to one opted-in HTTP server were serialized"
    assert not server._inflight_tasks


@pytest.mark.parametrize("options", [
    {"opt_in": None}, {"opt_in": False}, {"elicitation": True}, {"sampling": True},
])
def test_serial_controls_preserve_results(connected, options):
    http, connect = connected
    server, handler = connect(**options)
    assert _pair(handler) == [{"result": "answer:a"}, {"result": "answer:b"}]
    assert http.peak == 1
    assert not server._inflight_tasks


@pytest.mark.parametrize("bridge", [False, True], ids=["native", "peeled-tool-call"])
def test_agent_batch_reaches_same_server_in_parallel(connected, bridge):
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

    http, connect = connected
    connect()
    name = "mcp__parallel_probe__probe"
    calls = [(name, {"label": label}) for label in ("a", "b")]
    if bridge:
        calls = [("tool_call", {"name": n, "arguments": a}) for n, a in calls]
    script = [ToolCall(*calls[0], parallel=[calls[1]]), Text("done")]
    with FakeLLMServer(script) as llm:
        agent = AIAgent(model="fake-model", provider="custom", api_mode="chat_completions",
                        api_key="test-key", base_url=llm.base_url, quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=4,
                        enabled_toolsets=["mcp-parallel-probe"])
        result = agent.run_conversation("Run both probes")
        assert result["final_response"] == "done", result
        messages = llm.main_requests()[-1]["messages"]
        outputs = [m["content"] for m in messages if m.get("role") == "tool"]
        assert len(outputs) == 2, outputs
        assert "answer:a" in outputs[0] and "answer:b" in outputs[1], outputs
    assert http.peak == 2
    assert sorted(http.received) == ["a", "b"]
