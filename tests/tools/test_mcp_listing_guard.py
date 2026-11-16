"""One tool the MCP SDK rejects costs that tool, not the server's whole catalog (#101669).

Drives a real ``ClientSession`` over in-memory streams against a fake server; no subprocess."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest

from tools.mcp_listing_guard import guard_session_streams

pytest.importorskip("mcp_types.methods")

GOOD = {"name": "good", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}
BROKEN = {"name": "broken", "description": "no inputSchema"}  # rejected by every protocol era
CUSTOM_METHOD = "x/tools_like"
CUSTOM_RESULT = {"tools": [copy.deepcopy(BROKEN)], "note": "extension payload"}


async def _fake_server(recv, send):
    from mcp import types
    from mcp.shared.message import SessionMessage

    async with recv, send:
        async for item in recv:
            msg = item.message
            if getattr(msg, "id", None) is None:
                continue
            result = {
                "initialize": {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}},
                               "serverInfo": {"name": "fake", "version": "0"}},
                "tools/list": {"tools": [copy.deepcopy(GOOD), copy.deepcopy(BROKEN)]},
                CUSTOM_METHOD: copy.deepcopy(CUSTOM_RESULT),
            }.get(msg.method, {})
            await send.send(SessionMessage(message=types.JSONRPCResponse(jsonrpc="2.0", id=msg.id, result=result)))


async def _run(use, *, guarded):
    import anyio
    from mcp.client.session import ClientSession

    client_send, server_recv = anyio.create_memory_object_stream(64)
    server_send, client_recv = anyio.create_memory_object_stream(64)
    server = asyncio.create_task(_fake_server(server_recv, server_send))
    dropped = []
    read, write = client_recv, client_send
    if guarded:
        read, write = guard_session_streams(client_recv, client_send, server_name="fake", on_drop=dropped.extend)
    try:
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=5)
            return await asyncio.wait_for(use(session), timeout=5), dropped
    finally:
        server.cancel()


async def _tool_names(session):
    return [tool.name for tool in (await session.list_tools()).tools]


async def _custom(session):
    from mcp import types
    from pydantic import TypeAdapter

    return await session.send_request(types.Request(method=CUSTOM_METHOD, params=None), TypeAdapter(dict[str, Any]))


def test_invalid_tool_is_dropped_by_name_and_the_rest_load():
    with pytest.raises(BaseException) as excinfo:  # the SDK alone rejects the whole page
        asyncio.run(_run(_tool_names, guarded=False))
    leaves, stack = [], [excinfo.value]
    while stack:
        exc = stack.pop()
        if hasattr(exc, "exceptions"):
            stack.extend(exc.exceptions)
        else:
            leaves.append(exc)
    assert any("inputSchema" in str(leaf) for leaf in leaves), leaves

    names, dropped = asyncio.run(_run(_tool_names, guarded=True))
    assert names == ["good"]
    assert [name for name, _ in dropped] == ["broken"]


def test_custom_method_result_carrying_a_tools_list_is_untouched():
    unguarded, _ = asyncio.run(_run(_custom, guarded=False))
    guarded, dropped = asyncio.run(_run(_custom, guarded=True))
    assert guarded == unguarded == CUSTOM_RESULT
    assert dropped == []
