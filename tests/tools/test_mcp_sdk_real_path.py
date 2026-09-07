"""Behavioral proof through a real in-memory MCP SDK peer and Hermes handler."""

import asyncio
import json
import threading

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession
from mcp.server import MCPServer
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, TextContent

from tools import mcp_tool
from tools import mcp_tool_loop as _mcp_loop
from tools.mcp_tool_handlers import _make_tool_handler


@pytest.fixture
def real_sdk_peer(monkeypatch, tmp_path):
    """Connect the real SDK client/server streams on Hermes' background MCP loop."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    peer = MCPServer("hermes-real-path-test")
    calls = {"timeout": 0}

    @peer.tool(structured_output=False)
    async def domain_error():
        return CallToolResult(
            content=[TextContent(type="text", text="Readable validation failure")],
            structured_content={"error": {"kind": "invalid_filter"}},
            is_error=True,
            meta={"com.example/error": "safe", "mcp.io/internal": "drop"},
        )

    @peer.tool(structured_output=False)
    async def distinct_result():
        return CallToolResult(
            content=[TextContent(type="text", text="2 records")],
            structured_content={"records": [{"id": "INC-1"}, {"id": "INC-2"}]},
        )

    identical_payload = {"items": [{"enabled": True, "count": 1}]}

    @peer.tool(structured_output=False)
    async def identical_result():
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(identical_payload))],
            structured_content=identical_payload,
        )

    nested_payload = {
        "incident": {
            "id": "INC-9007199254740993",
            "alerts": [{"id": 9007199254740993, "deviceIds": ["aad-device-1"]}],
        }
    }

    @peer.tool(structured_output=False)
    async def nested_result():
        return CallToolResult(content=[], structured_content=nested_payload)

    @peer.tool(structured_output=False)
    async def transport_timeout():
        calls["timeout"] += 1
        await asyncio.sleep(1)
        return CallToolResult(content=[TextContent(type="text", text="too late")])

    state = {}
    ready = threading.Event()

    async def _run_peer():
        try:
            async with create_client_server_memory_streams() as (
                client_streams,
                server_streams,
            ):
                server_task = asyncio.create_task(
                    peer._lowlevel_server.run(
                        server_streams[0],
                        server_streams[1],
                        peer._lowlevel_server.create_initialization_options(),
                    )
                )
                try:
                    async with ClientSession(
                        client_streams[0], client_streams[1]
                    ) as session:
                        initialize_result = await session.initialize()
                        await session.list_tools()
                        hermes_server = mcp_tool.MCPServerTask("sdk-real-peer")
                        hermes_server.session = session
                        hermes_server.initialize_result = initialize_result
                        hermes_server._rpc_lock = asyncio.Lock()
                        hermes_server._ready.set()
                        hermes_server._session_proven = True
                        state.update(server=hermes_server, stop=asyncio.Event())
                        with mcp_tool._lock:
                            mcp_tool._servers["sdk-real-peer"] = hermes_server
                        ready.set()
                        await state["stop"].wait()
                finally:
                    server_task.cancel()
                    await asyncio.gather(server_task, return_exceptions=True)
        except BaseException as exc:
            state["error"] = exc
            ready.set()

    _mcp_loop._ensure_mcp_loop()
    loop = mcp_tool._mcp_loop
    assert loop is not None
    peer_future = asyncio.run_coroutine_threadsafe(_run_peer(), loop)
    assert ready.wait(5), "in-memory MCP SDK peer did not become ready"
    if "error" in state:
        raise state["error"]

    state.update(calls=calls, identical=identical_payload, nested=nested_payload)
    try:
        yield state
    finally:
        stop = state.get("stop")
        if stop is not None and mcp_tool._mcp_loop is not None:
            mcp_tool._mcp_loop.call_soon_threadsafe(stop.set)
        peer_future.result(timeout=5)
        with mcp_tool._lock:
            mcp_tool._servers.pop("sdk-real-peer", None)
        mcp_tool._server_error_counts.pop("sdk-real-peer", None)
        mcp_tool._server_breaker_opened_at.pop("sdk-real-peer", None)
        _mcp_loop._stop_mcp_loop(only_if_idle=True)


def test_real_sdk_results_and_transport_breaker(real_sdk_peer):
    """Exercise SDK serialization, client parsing, Hermes rendering and breaker accounting."""
    mcp_tool._server_error_counts["sdk-real-peer"] = 2

    error = json.loads(_make_tool_handler("sdk-real-peer", "domain_error", 2)({}))
    assert error == {
        "error": "Readable validation failure",
        "structuredContent": {"error": {"kind": "invalid_filter"}},
        "_meta": {"com.example/error": "safe"},
    }
    assert mcp_tool._server_error_counts["sdk-real-peer"] == 0

    distinct = json.loads(_make_tool_handler("sdk-real-peer", "distinct_result", 2)({}))
    assert distinct == {
        "result": "2 records",
        "structuredContent": {"records": [{"id": "INC-1"}, {"id": "INC-2"}]},
    }

    identical = json.loads(
        _make_tool_handler("sdk-real-peer", "identical_result", 2)({})
    )
    assert identical == {"result": json.dumps(real_sdk_peer["identical"])}

    nested = json.loads(_make_tool_handler("sdk-real-peer", "nested_result", 2)({}))
    assert nested == {"result": real_sdk_peer["nested"]}

    timed_out = json.loads(
        _make_tool_handler("sdk-real-peer", "transport_timeout", 0.05)({})
    )
    assert "error" in timed_out
    assert "timed out" in timed_out["error"].lower()
    assert real_sdk_peer["calls"]["timeout"] == 1
    assert mcp_tool._server_error_counts["sdk-real-peer"] == 1
