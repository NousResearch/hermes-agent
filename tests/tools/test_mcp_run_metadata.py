import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.mcp_run_context import mcp_run_metadata, read_mcp_run_metadata
from tools import mcp_tool


class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _ToolResult:
    isError = False
    structuredContent = None

    def __init__(self, text: str):
        self.content = [_TextBlock(text)]


def _run_direct(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory

    async def _with_lock():
        for server in mcp_tool._servers.values():
            if getattr(server, "_rpc_lock", None) is None:
                server._rpc_lock = asyncio.Lock()
        return await coro

    return asyncio.run(_with_lock())


def _server(session, *, forward: bool):
    return SimpleNamespace(
        session=session,
        _rpc_lock=None,
        _config={"forward_run_metadata": forward},
        _pending_call_context=None,
    )


def test_opted_in_server_receives_metadata_outside_model_arguments():
    metadata = {
        "magic.employee": {
            "binding": "signed-secret-binding",
            "crisp_session_id": "session-1",
        }
    }
    session = SimpleNamespace(
        call_tool=AsyncMock(return_value=_ToolResult("ok")),
    )
    server = _server(session, forward=True)

    with patch.dict(mcp_tool._servers, {"employee": server}, clear=False), patch(
        "tools.mcp_tool._run_on_mcp_loop", side_effect=_run_direct
    ), mcp_run_metadata(metadata):
        handler = mcp_tool._make_tool_handler("employee", "schedule", 30)
        assert json.loads(handler({"visible": "value"})) == {"result": "ok"}

    session.call_tool.assert_awaited_once_with(
        "schedule",
        arguments={"visible": "value"},
        meta=metadata,
    )
    assert "signed-secret-binding" not in json.dumps({"visible": "value"})


def test_metadata_never_changes_the_model_facing_tool_schema():
    metadata = {"private": "schema-secret"}
    listed_tool = SimpleNamespace(
        name="schedule",
        description="Read the schedule",
        inputSchema={
            "type": "object",
            "properties": {"day": {"type": "string"}},
            "required": ["day"],
        },
    )

    with mcp_run_metadata(metadata):
        schema = mcp_tool._convert_mcp_schema("employee", listed_tool)

    assert schema["parameters"] == listed_tool.inputSchema
    assert "schema-secret" not in json.dumps(schema)


def test_server_without_opt_in_receives_no_run_metadata():
    session = SimpleNamespace(
        call_tool=AsyncMock(return_value=_ToolResult("ok")),
    )
    server = _server(session, forward=False)

    with patch.dict(mcp_tool._servers, {"regular": server}, clear=False), patch(
        "tools.mcp_tool._run_on_mcp_loop", side_effect=_run_direct
    ), mcp_run_metadata({"private": "must-not-leak"}):
        handler = mcp_tool._make_tool_handler("regular", "lookup", 30)
        handler({"query": "public"})

    session.call_tool.assert_awaited_once_with(
        "lookup",
        arguments={"query": "public"},
    )


def test_mcp_loop_propagates_metadata_without_leaking_after_scope():
    mcp_tool._ensure_mcp_loop()
    try:
        async def _read():
            return read_mcp_run_metadata()

        with mcp_run_metadata({"request": "one"}):
            assert mcp_tool._run_on_mcp_loop(_read, timeout=10) == {"request": "one"}
        assert mcp_tool._run_on_mcp_loop(_read, timeout=10) is None
    finally:
        mcp_tool._stop_mcp_loop()


def test_two_concurrent_runs_keep_metadata_isolated():
    mcp_tool._ensure_mcp_loop()
    barrier = threading.Barrier(2)
    results = {}

    async def _read_after_peer_started():
        await asyncio.sleep(0.05)
        return read_mcp_run_metadata()

    def _run(key: str):
        with mcp_run_metadata({"request": key}):
            barrier.wait(timeout=5)
            results[key] = mcp_tool._run_on_mcp_loop(_read_after_peer_started, timeout=10)

    threads = [threading.Thread(target=_run, args=(key,)) for key in ("a", "b")]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
    finally:
        mcp_tool._stop_mcp_loop()

    assert results == {
        "a": {"request": "a"},
        "b": {"request": "b"},
    }
