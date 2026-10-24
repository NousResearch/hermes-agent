"""Exercise closed-connection recovery with the real MCP SDK, without a native driver."""

from contextlib import asynccontextmanager
import subprocess
from typing import Any

import anyio
import anyio.lowlevel
import pytest
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from mcp.types import CONNECTION_CLOSED, JSONRPCResponse

from tools.computer_use import cua_backend, cua_backend_driver
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession
from tools.environments import local


class _MemoryPeer:
    def __init__(self):
        self.starts = 0
        self.calls = []
        self.drop_next_call = False
        self.send = None

    @asynccontextmanager
    async def stdio_client(self, params):
        self.starts += 1
        send, read = anyio.create_memory_object_stream(16)
        write, receive = anyio.create_memory_object_stream(16)
        self.send = send

        async def serve():
            async with receive:
                async for envelope in receive:
                    request = envelope.message
                    if not hasattr(request, "id"):
                        continue
                    if request.method == "initialize":
                        result = {
                            "protocolVersion": request.params["protocolVersion"],
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "memory-peer", "version": "1"},
                        }
                    elif request.method == "tools/list":
                        result = {"tools": [
                            {"name": name, "inputSchema": {"type": "object"}}
                            for name in ("list_windows", "click", "start_session")
                        ]}
                    elif request.method == "tools/call":
                        self.calls.append((request.params["name"], request.params["arguments"]))
                        if self.drop_next_call:
                            # The action reached the peer, but its response was lost.
                            self.drop_next_call = False
                            await send.aclose()
                            continue
                        result = {"content": [], "isError": False}
                    else:
                        raise AssertionError(f"Unexpected MCP method: {request.method}")
                    await send.send(SessionMessage(JSONRPCResponse(
                        jsonrpc="2.0", id=request.id, result=result,
                    )))

        async with anyio.create_task_group() as group:
            group.start_soon(serve)
            try:
                yield read, write
            finally:
                group.cancel_scope.cancel()
                await send.aclose()
                await write.aclose()

    async def eof(self, session):
        assert self.send is not None
        await self.send.aclose()
        with anyio.fail_after(2):
            while not session._session._dispatcher._closed:
                await anyio.lowlevel.checkpoint()
        # EOF finishes the dispatcher, not the owning lifecycle coroutine.
        assert session._started
        assert not session._lifecycle_future.done()


@pytest.fixture
def memory_session(monkeypatch):
    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("Native subprocesses are forbidden in this memory-only test")

    peer = _MemoryPeer()
    monkeypatch.setattr(subprocess, "Popen", forbidden_spawn)
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: "memory-only")
    monkeypatch.setattr(cua_backend_driver, "_resolve_mcp_invocation", lambda cmd: (cmd, []))
    monkeypatch.setattr(cua_backend, "cua_driver_child_env", dict)
    monkeypatch.setattr(local, "_sanitize_subprocess_env", lambda env: {})
    monkeypatch.setattr("mcp.client.stdio.stdio_client", peer.stdio_client)
    bridge = _AsyncBridge()
    session = _CuaDriverSession(bridge)
    resets = []
    session.set_transport_reset_callback(lambda: resets.append("reset"))
    try:
        session.start()
        yield session, peer, resets
    finally:
        session.stop()
        bridge.stop()


@pytest.mark.parametrize("name", ["list_windows", "click"])
@pytest.mark.parametrize("eof_during_call", [False, True], ids=["idle-eof", "lost-response"])
def test_sdk_eof_recovers_without_replaying_mutations(memory_session, name, eof_during_call):
    session, peer, resets = memory_session
    label = {"session": "recovery-label"}
    session.call_tool("start_session", label)
    if eof_during_call:
        peer.drop_next_call = True
    else:
        session._bridge.run(peer.eof(session))

    args = {"x": 1, "y": 1} if name == "click" else {}
    result = session.call_tool(name, args)
    expected_calls: list[tuple[str, dict[str, Any]]] = [("start_session", label)]
    if eof_during_call:
        expected_calls.append((name, args))
    expected_calls.append(("start_session", label))
    if name == "click":
        assert result["isError"] is True
        assert result["structuredContent"]["code"] == "transport_outcome_unknown"
        assert result["structuredContent"]["next_step"] == "fresh_state"
    else:
        assert result["isError"] is False
        expected_calls.append((name, args))
    assert peer.calls == expected_calls
    assert peer.starts == 2
    assert resets == ["reset"]

    # Subsequent reads reuse the replacement rather than the closed connection.
    assert session.call_tool("list_windows", {})["isError"] is False
    assert peer.calls == expected_calls + [("list_windows", {})]
    assert peer.starts == 2
    assert resets == ["reset"]


@pytest.mark.parametrize("error, closed", [
    (MCPError(CONNECTION_CLOSED, "localized closure message"), True),
    (MCPError(-32603, "Connection closed"), False),
    (MCPError(-32602, "Invalid parameters"), False),
    (RuntimeError("Connection closed"), False),
])
def test_only_sdk_connection_closed_enters_recovery(memory_session, monkeypatch, error, closed):
    session, peer, resets = memory_session
    assert session._is_closed_session_error(error) is closed
    if closed:
        return  # Protocol-level recovery is exercised by the real SDK EOF test above.

    async def fail_call(name, args):
        raise error

    monkeypatch.setattr(session, "_call_tool_async", fail_call)
    with pytest.raises(type(error)) as caught:
        session.call_tool("list_windows", {})
    assert caught.value is error
    assert peer.starts == 1
    assert peer.calls == []
    assert resets == []
