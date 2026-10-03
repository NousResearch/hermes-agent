"""A tool call's timeout bounds its own round-trip, not its wait for the server's turn.

One RPC runs per MCP server at a time (``_rpc_lock``). Two sessions sharing a server — two cron
routines firing on the same slot — queue behind each other, and the clock used to start when the
call was scheduled: a 31-second query timed out "after 300s" because another session held the
server for five minutes. Real MCP loop, real handler, a session whose ``call_tool`` takes a fixed time.
"""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.mcp_tool as mcp_mod
from tools.mcp_tool import MCPServerTask, _servers
from tools.mcp_tool_handlers import _make_tool_handler

RPC_S = 0.2  # each call's own round-trip


@pytest.fixture
def server():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    old_loop, old_thread = mcp_mod._mcp_loop, mcp_mod._mcp_thread
    mcp_mod._mcp_loop, mcp_mod._mcp_thread = loop, thread

    async def call_tool(name, arguments):
        await asyncio.sleep(arguments["seconds"])
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], isError=False)

    srv = MCPServerTask("shared")
    srv.session = MagicMock()
    srv.session.call_tool = call_tool
    _servers["shared"] = srv
    try:
        yield srv
    finally:
        _servers.pop("shared", None)
        mcp_mod._mcp_loop, mcp_mod._mcp_thread = old_loop, old_thread
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def _call(timeout: float, seconds: float) -> dict:
    return json.loads(_make_tool_handler("shared", "query", timeout)({"seconds": seconds}))


def test_a_call_queued_behind_another_session_is_not_charged_for_the_wait(server):
    # Two sessions at once, each within its own budget; together they exceed one budget.
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: _call(timeout=1.5 * RPC_S, seconds=RPC_S), range(2)))
    assert results == [{"result": "ok"}, {"result": "ok"}]


def test_a_call_that_itself_overruns_still_times_out(server):
    result = _call(timeout=RPC_S, seconds=10 * RPC_S)
    assert "MCP call timed out after 0.2s (configured timeout: 0.2s)" in result["error"]
