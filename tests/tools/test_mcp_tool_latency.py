import asyncio

from tools.environments.base import set_activity_callback
from tools import mcp_tool


def test_run_on_mcp_loop_emits_activity_while_waiting_for_slow_mcp_call():
    """A slow MCP RPC should keep the gateway activity lease alive while blocked.

    This tickles the same shape as the observed hang: the synchronous caller is
    waiting on a coroutine scheduled onto the MCP event loop, and no subprocess
    wait loop is involved to emit normal terminal heartbeats.
    """
    mcp_tool._ensure_mcp_loop()
    activity = []
    set_activity_callback(activity.append)

    async def slow_result():
        await asyncio.sleep(0.15)
        return "ok"

    try:
        result = mcp_tool._run_on_mcp_loop(
            slow_result(),
            timeout=1,
            activity_label="calling MCP forge/digest",
            activity_interval=0.01,
        )
    finally:
        set_activity_callback(None)

    assert result == "ok"
    assert any("calling MCP forge/digest" in entry for entry in activity)
    assert any("elapsed" in entry for entry in activity)
