"""Same-server parallel admission must reach the stdio wire.

Real SDK stdio server subprocess, real MCPServerTask and registry handlers. The server
waits until ``group`` calls are active before answering, with a bounded escape so the
serial controls still return successful, correctly associated results.
"""

from __future__ import annotations

import contextvars
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import hermes_yaml as yaml
from hermes_constants import get_hermes_home
from tests.waiting import wait_until

_SERVER = """
import asyncio, os, sys
from mcp.server import MCPServer
server = MCPServer("stdio-probe")
LOG = sys.argv[1]
active = peak = 0
gate = asyncio.Event()

@server.tool()
async def probe(label: str, group: int = 2, hold: float = 0, exit_when_grouped: bool = False) -> str:
    global active, peak
    active += 1
    peak = max(peak, active)
    with open(LOG, "a") as log:
        log.write(label + "\\n")
    if active >= group:
        gate.set()
    try:
        try:
            await asyncio.wait_for(gate.wait(), 2)
        except TimeoutError:
            pass  # serial controls must also finish successfully
        if exit_when_grouped and gate.is_set():
            os._exit(1)
        await asyncio.sleep(hold)
        return f"answer:{label}"
    finally:
        active -= 1
        if active == 0:
            gate.clear()

@server.tool()
def stats() -> str:
    global peak
    seen, peak = peak, 0
    return str(seen)

server.run("stdio")
"""


@pytest.fixture
def connected(tmp_path):
    from tools import mcp_tool as core
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.registry import registry

    script = tmp_path / "stdio_probe.py"
    script.write_text(_SERVER, encoding="utf-8")
    log = tmp_path / "received.log"
    log.write_text("", encoding="utf-8")

    def connect(*, opt_in=True, elicitation=False, sampling=False, **extra):
        cfg = {"command": sys.executable, "args": [str(script), str(log)],
               "elicitation": {"enabled": elicitation}, "sampling": {"enabled": sampling}, **extra}
        if opt_in is not None:
            cfg["supports_parallel_tool_calls"] = opt_in
        (get_hermes_home() / "config.yaml").write_text(
            yaml.safe_dump({"mcp_servers": {"stdio-probe": cfg}}), encoding="utf-8")
        from tools.mcp_tool_discovery import discover_mcp_tools
        names = discover_mcp_tools()
        assert "mcp__stdio_probe__probe" in names, names
        from tools.mcp_tool_scope import _resolve_server_key
        server = core._servers[_resolve_server_key("stdio-probe")]
        return server, lambda args: registry.dispatch("mcp__stdio_probe__probe", args)

    def peak():
        return int(json.loads(registry.dispatch("mcp__stdio_probe__stats", {}))["result"])

    def received():
        return sorted(line for line in log.read_text(encoding="utf-8-sig").splitlines() if line)

    yield SimpleNamespace(connect=connect, peak=peak, received=received)
    shutdown_mcp_servers()


def _together(handler, labels=("a", "b"), **args):
    barrier = threading.Barrier(len(labels))

    def call(label):
        barrier.wait(10)
        return json.loads(handler({"label": label, "group": len(labels), **args}))
    with ThreadPoolExecutor(max_workers=len(labels)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, call, label) for label in labels]
        return [future.result(60) for future in futures]


def _answers(labels=("a", "b")):
    return [{"result": f"answer:{label}"} for label in labels]


def test_opted_in_stdio_calls_overlap_on_server(connected):
    server, handler = connected.connect()
    assert _together(handler) == _answers()
    assert connected.peak() == 2, "successful calls to one opted-in stdio server were serialized"
    assert not server._inflight_tasks
    assert not server._rpc_lock.locked()


@pytest.mark.parametrize("options", [
    {"opt_in": None}, {"opt_in": False}, {"elicitation": True}, {"sampling": True},
])
def test_serial_controls_preserve_results(connected, options):
    server, handler = connected.connect(**options)
    assert _together(handler) == _answers()
    assert connected.peak() == 1
    assert not server._inflight_tasks


def test_many_overlapping_calls_each_get_their_own_result(connected):
    server, handler = connected.connect()
    labels = [f"c{i}" for i in range(8)]
    assert _together(handler, labels) == _answers(labels)
    assert connected.peak() == 8
    assert connected.received() == sorted(labels)
    assert not server._inflight_tasks


def test_one_overlapping_call_times_out_and_the_other_still_answers(connected):
    from tools.mcp_tool_handlers import _make_tool_handler

    server, handler = connected.connect()
    timed = _make_tool_handler("stdio-probe", "probe", 2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(contextvars.copy_context().run, timed, {"label": "slow", "hold": 30})
        fast = pool.submit(contextvars.copy_context().run, handler, {"label": "fast", "hold": 3})
        assert "timed out" in json.loads(slow.result(30))["error"]
        assert json.loads(fast.result(30)) == {"result": "answer:fast"}
    wait_until(lambda: not server._inflight_tasks, timeout=10, what="timeout cleanup")
    assert connected.peak() == 2


@pytest.mark.parametrize("event", ["reconnect", "shutdown"])
def test_lifecycle_settles_every_overlapping_call_without_replay(connected, event):
    from tools.mcp_tool_loop import _run_on_mcp_loop

    server, handler = connected.connect()
    old_session = server.session
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(handler, {"label": label, "hold": 60}) for label in ("a", "b")]
        wait_until(lambda: len(connected.received()) == 2, timeout=10, what="two calls on the server")

        async def teardown():
            assert len(server._inflight_tasks) == 2
            assert not server._rpc_lock.locked()
            getattr(server, f"_{event}_event").set()
        _run_on_mcp_loop(teardown)
        results = [json.loads(f.result(20)) for f in futures]
    assert all("verify" in result["error"] for result in results), results
    wait_until(lambda: not server._inflight_tasks, timeout=10, what="all tracked calls to settle")
    assert connected.received() == ["a", "b"], "a call was replayed"
    if event == "reconnect":
        wait_until(lambda: server.session not in (None, old_session) and server._ready.is_set(),
                   timeout=20, what="fresh stdio session")
        assert json.loads(handler({"label": "after", "group": 1})) == {"result": "answer:after"}


def test_child_exit_settles_every_overlapping_call_without_replay(connected):
    server, handler = connected.connect()
    results = _together(handler, exit_when_grouped=True)
    for result in results:
        assert result.get("outcome_uncertain") is True, results
        assert "did not replay" in result["error"], results
    wait_until(lambda: not server._inflight_tasks, timeout=10, what="all tracked calls to settle")
    assert connected.received() == ["a", "b"], "a call was replayed"
    wait_until(lambda: server.session is not None and server._ready.is_set(), timeout=20,
               what="respawned stdio session")
    assert json.loads(handler({"label": "after", "group": 1})) == {"result": "answer:after"}


@pytest.mark.parametrize("limit", ["idle_timeout_seconds", "max_lifetime_seconds"])
def test_recycle_waits_for_overlapping_calls(connected, limit):
    server, handler = connected.connect(**{limit: 0.5})
    # Each call outlives the limit; neither holds the lock while it runs.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pair = pool.submit(_together, handler, hold=3)
        wait_until(lambda: len(server._inflight_tasks) == 2, timeout=10, what="two calls in flight")
        past_limit = time.monotonic() + 1.0
        wait_until(lambda: time.monotonic() >= past_limit, timeout=5, what="the limit to pass")
        assert len(server._inflight_tasks) == 2
        assert server._recycled_reason is None, "recycled while overlapping calls were running"
        assert pair.result(30) == _answers()
    wait_until(lambda: server._recycled_reason == limit, timeout=10,
               what="recycle after the calls finished")
    assert json.loads(handler({"label": "after", "group": 1})) == {"result": "answer:after"}
