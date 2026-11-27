"""Same-server parallel admission must reach the Streamable HTTP wire.

Real SDK server, real MCPServerTask and registry handlers. The server waits for a
second request before completing the first, with a bounded escape so the serial
control still returns two successful, correctly associated results.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

import pytest
import uvicorn
from mcp.server import MCPServer
from mcp.server.mcpserver import Context

import hermes_yaml as yaml
from hermes_constants import get_hermes_home
from tests.waiting import wait_until


class _HttpPair:
    def __init__(self, transport="http", *, json_response=True):
        self.active = self.peak = 0
        self.received = []
        self.pair = asyncio.Event()
        self.release = asyncio.Event()
        self.hold = False
        self.loop = None
        server = MCPServer("parallel-probe")

        @server.tool()
        async def probe(label: str, ctx: Context) -> str:
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
                await ctx.report_progress(1, 1, label)
                return f"answer:{label}"
            finally:
                self.active -= 1

        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        path = "mcp" if transport == "http" else "sse"
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}/{path}"
        app = (server.streamable_http_app(stateless_http=True, json_response=json_response)
               if transport == "http" else server.sse_app())
        self.uv = uvicorn.Server(uvicorn.Config(
            app, log_level="error", timeout_graceful_shutdown=2))

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
def connected(request):
    from tools import mcp_tool as core
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry

    transport = getattr(request, "param", "http")
    http = _HttpPair("http" if transport == "http-sse-response" else transport,
                     json_response=transport != "http-sse-response").start()

    def connect(*, opt_in=True, elicitation=False, sampling=False, tool_search="auto"):
        cfg = {"url": http.url, "elicitation": {"enabled": elicitation},
               "sampling": {"enabled": sampling}, "keepalive_interval": 0}
        if transport == "sse":
            cfg["transport"] = "sse"
        if opt_in is not None:
            cfg["supports_parallel_tool_calls"] = opt_in
        (get_hermes_home() / "config.yaml").write_text(
            yaml.safe_dump({"mcp_servers": {"parallel-probe": cfg},
                            "tools": {"tool_search": {"enabled": tool_search}}}), encoding="utf-8")
        from tools.mcp_tool_discovery import discover_mcp_tools
        names = discover_mcp_tools()
        assert "mcp__parallel_probe__probe" in names, names
        from tools.mcp_tool_scope import _resolve_server_key
        return core._servers[_resolve_server_key("parallel-probe")], lambda args: registry.dispatch("mcp__parallel_probe__probe", args)

    yield http, connect
    shutdown_mcp_servers()
    http.stop()


def _pair(handler, *, per_call=None):
    barrier = threading.Barrier(2)
    def call(label):
        barrier.wait(5)
        with per_call(label) if per_call else nullcontext():
            return json.loads(handler({"label": label}))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(contextvars.copy_context().run, call, label) for label in ("a", "b")]
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
    connect(tool_search="on" if bridge else "off")
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
        requests = llm.main_requests()
        names = {t["function"]["name"] for t in requests[0]["tools"]}
        assert ("tool_call" if bridge else name) in names
        if bridge:
            assert name not in names
        messages = requests[-1]["messages"]
        outputs = [m["content"] for m in messages if m.get("role") == "tool"]
        assert len(outputs) == 2, outputs
        assert "answer:a" in outputs[0] and "answer:b" in outputs[1], outputs
    assert http.peak == 2
    assert sorted(http.received) == ["a", "b"]


@pytest.mark.parametrize("connected", ["sse", "fallback"], indirect=True)
def test_legacy_sse_remains_serial(connected):
    http, connect = connected
    server, handler = connect()
    assert _pair(handler) == [{"result": "answer:a"}, {"result": "answer:b"}]
    assert http.peak == 1
    assert server._sse_fallback is (server._config.get("transport") != "sse")


@pytest.mark.parametrize("owner_parallel", [True, False])
def test_adopted_connection_uses_callers_parallel_policy(connected, tmp_path, monkeypatch, owner_parallel):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_tool as core
    from tools.mcp_tool_scope import _resolve_server_key

    http, connect = connected
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    homes = {name: tmp_path / name for name in ("owner", "adopter")}
    for home in homes.values():
        home.mkdir()
    @contextmanager
    def profile(name):
        token = set_hermes_home_override(homes[name])
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    with profile("owner"):
        owner, handler = connect(opt_in=owner_parallel)
        owner_key = _resolve_server_key("parallel-probe")
    with profile("adopter"):
        adopted, handler = connect(opt_in=not owner_parallel)
        assert adopted is owner
        assert _resolve_server_key("parallel-probe") == owner_key
        assert _pair(handler) == [{"result": "answer:a"}, {"result": "answer:b"}]
        assert http.peak == (1 if owner_parallel else 2)
    with profile("owner"):
        http.peak = 0
        assert _pair(handler) == [{"result": "answer:a"}, {"result": "answer:b"}]
        assert http.peak == (2 if owner_parallel else 1)
    assert len(core._servers) == 1


@pytest.mark.parametrize("connected", ["http-sse-response"], indirect=True)
def test_overlapping_progress_stays_with_its_caller(connected):
    from agent.tool_progress_relay import bound_tool_progress

    http, connect = connected
    _, handler = connect()
    seen = []
    @contextmanager
    def progress(label):
        with bound_tool_progress(lambda *a, **kw: seen.append((label, a[2], kw["tool_call_id"])),
                                 "mcp__parallel_probe__probe", label):
            yield
    assert _pair(handler, per_call=progress) == [{"result": "answer:a"}, {"result": "answer:b"}]
    assert http.peak == 2
    assert sorted(seen) == [("a", "a", "a"), ("b", "b", "b")]


def test_serial_elicitation_keeps_each_callers_context_and_callback(connected, monkeypatch):
    from tools import terminal_tool

    http, connect = connected
    server, handler = connect(elicitation=True)
    marker = contextvars.ContextVar("caller-marker")
    seen = []
    real_call = server.session.call_tool

    async def call_tool(name, arguments, **kwargs):
        # Simulate the SDK receive task, which has none of the sending task's context.
        async def receive():
            def consent():
                callback = terminal_tool._get_approval_callback()
                seen.append((arguments["label"], marker.get(), callback()))
            await asyncio.to_thread(server._elicitation._on_turn_thread(consent))
        await asyncio.create_task(receive(), context=contextvars.Context())
        return await real_call(name, arguments=arguments, **kwargs)
    monkeypatch.setattr(server.session, "call_tool", call_tool)

    @contextmanager
    def caller(label):
        token = marker.set(label)
        previous = terminal_tool._get_approval_callback()
        terminal_tool.set_approval_callback(lambda: label)
        try:
            yield
        finally:
            terminal_tool.set_approval_callback(previous)
            marker.reset(token)
    assert _pair(handler, per_call=caller) == [{"result": "answer:a"}, {"result": "answer:b"}]
    assert http.peak == 1
    assert sorted(seen) == [("a", "a", "a"), ("b", "b", "b")]
    assert server._pending_call_context is None
    assert server._pending_call_callbacks == ()


@pytest.mark.parametrize("event", ["reconnect", "shutdown"])
def test_lifecycle_settles_all_overlapping_http_calls(connected, event):
    from tools import mcp_tool as core
    from tools.mcp_tool_loop import _run_on_mcp_loop

    http, connect = connected
    server, handler = connect()
    http.hold = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(handler, {"label": label}) for label in ("a", "b")]
        try:
            wait_until(lambda: http.active == 2, timeout=10, what="two HTTP calls on the server")
            async def teardown():
                assert len(server._inflight_tasks) == 2
                assert not server._rpc_lock.locked()
                getattr(server, f"_{event}_event").set()
            _run_on_mcp_loop(teardown)
            results = [json.loads(f.result(15)) for f in futures]
            assert all("error" in result and "verify" in result["error"] for result in results), results
            wait_until(lambda: not server._inflight_tasks, timeout=10, what="all tracked calls to settle")
            # A canceled write may already have executed. It must never be replayed.
            assert sorted(http.received) == ["a", "b"]
        finally:
            http.loop.call_soon_threadsafe(http.release.set)
    if event == "reconnect":
        wait_until(lambda: server.session is not None and server._ready.is_set(), timeout=15,
                   what="fresh MCP session")
        http.hold = False
        assert json.loads(handler({"label": "after"})) == {"result": "answer:after"}
    assert core._mcp_loop.is_running()


def test_a_call_during_a_slow_reconnect_teardown_waits_for_the_fresh_session(connected, monkeypatch):
    """The closed session must be withdrawn before the transport exit, which can take seconds: a call
    arriving meanwhile waits for the fresh session instead of failing on the closed one (#145292)."""
    from tools import mcp_tool

    class _SlowTransportExit(mcp_tool.ClientSession):
        async def __aexit__(self, *exc):
            out = await super().__aexit__(*exc)
            await asyncio.sleep(1.0)  # the transport exit after the session closed: a dead socket, a slow DELETE
            return out

    monkeypatch.setattr(mcp_tool, "ClientSession", _SlowTransportExit)
    test_lifecycle_settles_all_overlapping_http_calls(connected, "reconnect")


def test_overlapping_timeout_cleans_up_every_call(connected):
    from tools.mcp_tool_handlers import _make_tool_handler

    http, connect = connected
    server, _ = connect()
    http.hold = True
    handler = _make_tool_handler("parallel-probe", "probe", 2)
    try:
        results = _pair(handler)
        assert http.peak == 2
        assert all("timed out" in result["error"] for result in results), results
        wait_until(lambda: not server._inflight_tasks, timeout=10, what="timeout cleanup")
        assert sorted(http.received) == ["a", "b"]
    finally:
        http.loop.call_soon_threadsafe(http.release.set)


_STDIO = """
import asyncio, json
from mcp.server import MCPServer
server = MCPServer("serial-probe")
active = peak = 0
pair = asyncio.Event()
@server.tool()
async def probe(label: str) -> str:
    global active, peak
    active += 1
    peak = max(peak, active)
    if active == 2:
        pair.set()
    try:
        await asyncio.wait_for(pair.wait(), 2)
    except TimeoutError:
        pass
    active -= 1
    return label
@server.tool()
def stats() -> str:
    return str(peak)
server.run("stdio")
"""


def test_opted_in_stdio_keeps_the_single_stream_serial(tmp_path):
    from tools.mcp_tool_discovery import discover_mcp_tools
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry

    script = tmp_path / "serial_probe.py"
    script.write_text(_STDIO, encoding="utf-8")
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {
        "serial-probe": {"command": sys.executable, "args": [str(script)],
                         "supports_parallel_tool_calls": True,
                         "sampling": {"enabled": False}, "elicitation": {"enabled": False}}}}), encoding="utf-8")
    try:
        assert "mcp__serial_probe__probe" in discover_mcp_tools()
        assert _pair(lambda a: registry.dispatch("mcp__serial_probe__probe", a)) == [
            {"result": "a"}, {"result": "b"}]
        assert json.loads(registry.dispatch("mcp__serial_probe__stats", {})) == {"result": "1"}
    finally:
        shutdown_mcp_servers()
