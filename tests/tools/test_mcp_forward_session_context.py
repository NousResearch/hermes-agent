"""MCP request origin follows the calling task, never a shared connection or model args."""

import asyncio
from types import SimpleNamespace

from gateway.session_context import clear_session_vars, set_session_vars
from tools.mcp_tool_handlers import _call_tool_racing_stdio_death
from tools.mcp_tool_registration import _connection_identity


def test_forwarded_context_is_task_local_and_opt_in():
    async def exercise():
        calls = []
        arrived = asyncio.Event()

        async def call_tool(name, *, arguments, **kwargs):
            calls.append((arguments, kwargs))
            if len(calls) == 2:
                arrived.set()
            await asyncio.wait_for(arrived.wait(), timeout=5)
            return "ok"

        server = SimpleNamespace(
            _config={"forward_session_context": True},
            session=SimpleNamespace(call_tool=call_tool),
        )

        async def call(run):
            tokens = set_session_vars(platform="api_server", session_key=run, session_id="shared-session", request_session_id=run + "-alias")
            try:
                from gateway.session_context import scoped_current_session_id
                with scoped_current_session_id(run + "-rotated"):
                    return await _call_tool_racing_stdio_death(
                        server, "harness", "inspect", {"session_key": "model-supplied"})
            finally:
                clear_session_vars(tokens)

        assert await asyncio.gather(call("run_a"), call("run_b")) == ["ok", "ok"]
        assert {entry[1]["meta"]["com.nousresearch.hermes/session_key"] for entry in calls} == {"run_a", "run_b"}
        for arguments, kwargs in calls:
            assert arguments == {"session_key": "model-supplied"}
            assert kwargs["meta"]["com.nousresearch.hermes/platform"] == "api_server"
            run = kwargs["meta"]["com.nousresearch.hermes/session_key"]
            assert kwargs["meta"]["com.nousresearch.hermes/session_id"] == run + "-rotated"
            assert kwargs["meta"]["com.nousresearch.hermes/request_session_id"] == run + "-alias"

        for config in ({}, {"forward_session_context": False}, {"forward_session_context": "true"}):
            server._config = config
            await call("private-run")
            assert calls[-1][1] == {}

    asyncio.run(exercise())


def test_connection_sharing_cannot_enable_context_for_another_profile():
    base = {"url": "https://example.com/mcp"}
    assert _connection_identity(base) == _connection_identity({**base, "forward_session_context": False})
    assert _connection_identity(base) != _connection_identity({**base, "forward_session_context": True})


def test_real_mcp_protocol_carries_runtime_origin():
    import anyio
    from mcp import ClientSession, types
    from mcp.server.lowlevel import Server
    from mcp.shared.memory import create_client_server_memory_streams

    async def exercise():
        received = []

        async def list_tools(ctx, params):
            return types.ListToolsResult(tools=[types.Tool(name="inspect", input_schema={"type": "object"})])

        async def inspect_tool(ctx, params):
            received.append(params.meta)
            return types.CallToolResult(content=[types.TextContent(type="text", text="recorded")])

        remote = Server("origin-proof", on_list_tools=list_tools, on_call_tool=inspect_tool)

        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as group:
                group.start_soon(remote.run, *server_streams, remote.create_initialization_options())
                async with ClientSession(*client_streams) as session:
                    await session.initialize()
                    server = SimpleNamespace(_config={"forward_session_context": True}, session=session)
                    tokens = set_session_vars(platform="api_server", session_key="run_origin", session_id="conversation", request_session_id="original")
                    try:
                        result = await _call_tool_racing_stdio_death(server, "harness", "inspect", {})
                    finally:
                        clear_session_vars(tokens)
                    assert not result.is_error
                    assert received == [{
                        "com.nousresearch.hermes/platform": "api_server",
                        "com.nousresearch.hermes/session_key": "run_origin",
                        "com.nousresearch.hermes/session_id": "conversation",
                        "com.nousresearch.hermes/request_session_id": "original"}]
                group.cancel_scope.cancel()

    asyncio.run(exercise())


def test_registry_handler_keeps_origin_across_mcp_loop_thread(monkeypatch):
    import json
    import threading
    from unittest.mock import MagicMock

    from mcp import types
    from tools import mcp_tool, mcp_tool_loop
    from tools.mcp_tool_handlers import _make_tool_handler

    received = []

    async def call_tool(name, *, arguments, meta):
        received.append(meta)
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

    server = MagicMock()
    server._config = {"forward_session_context": True}
    server.session.call_tool = call_tool
    server._ready = threading.Event()
    server._ready.set()
    server._is_recycled_stdio.return_value = False
    server._stdio_children_dead.return_value = False
    monkeypatch.setitem(mcp_tool._servers, "origin-proof", server)
    mcp_tool_loop._ensure_mcp_loop()
    tokens = set_session_vars(platform="api_server", session_key="run_thread", session_id="thread-session", request_session_id="thread-alias")
    try:
        result = json.loads(_make_tool_handler("origin-proof", "inspect", 5)({}))
        assert result["result"] == "ok"
    finally:
        clear_session_vars(tokens)
    assert received == [{
        "com.nousresearch.hermes/platform": "api_server",
        "com.nousresearch.hermes/session_key": "run_thread",
        "com.nousresearch.hermes/session_id": "thread-session",
        "com.nousresearch.hermes/request_session_id": "thread-alias"}]


def test_no_ambient_or_non_api_identity_is_forwarded(monkeypatch):
    import contextvars
    from gateway.session_context import reset_session_vars

    for name, value in {"PLATFORM": "api_server", "KEY": "ambient-run", "ID": "ambient-session", "REQUEST_ID": "ambient-alias"}.items():
        monkeypatch.setenv(f"HERMES_SESSION_{name}", value)

    async def call_tool(name, *, arguments):
        return "no metadata"

    server = SimpleNamespace(_config={"forward_session_context": True}, session=SimpleNamespace(call_tool=call_tool))

    async def exercise():
        reset_session_vars()
        assert await _call_tool_racing_stdio_death(server, "harness", "inspect", {}) == "no metadata"
        for platform, key, session in (("telegram", "chat-key", "session"), ("api_server", "run", ""), ("api_server", "run", "live-without-request")):
            tokens = set_session_vars(platform=platform, session_key=key, session_id=session)
            try:
                assert await _call_tool_racing_stdio_death(server, "harness", "inspect", {}) == "no metadata"
            finally:
                clear_session_vars(tokens)

    contextvars.Context().run(asyncio.run, exercise())
