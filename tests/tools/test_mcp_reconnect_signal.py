"""Tests for the MCPServerTask reconnect signal.

When the OAuth layer cannot recover in-place (e.g., external refresh of a
single-use refresh_token made the SDK's in-memory refresh fail), the tool
handler signals MCPServerTask to tear down the current MCP session and
reconnect with fresh credentials. This file exercises the signal plumbing
in isolation from the full stdio/http transport machinery.
"""
import asyncio

import pytest


@pytest.mark.asyncio
async def test_reconnect_event_attribute_exists():
    """MCPServerTask has a _reconnect_event alongside _shutdown_event."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")
    assert hasattr(task, "_reconnect_event")
    assert isinstance(task._reconnect_event, asyncio.Event)
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_returns_reconnect():
    """When _reconnect_event fires, helper returns 'reconnect' and clears it."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "reconnect"
    # Should have cleared so the next cycle starts fresh
    assert not task._reconnect_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_returns_shutdown():
    """When _shutdown_event fires, helper returns 'shutdown'."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


@pytest.mark.asyncio
async def test_wait_for_lifecycle_event_shutdown_wins_when_both_set():
    """If both events are set simultaneously, shutdown takes precedence."""
    from tools.mcp_tool import MCPServerTask
    task = MCPServerTask("test")

    task._shutdown_event.set()
    task._reconnect_event.set()
    reason = await task._wait_for_lifecycle_event()
    assert reason == "shutdown"


@pytest.mark.asyncio
async def test_run_keepalive_once_acquires_rpc_lock():
    """``_run_keepalive_once`` must hold ``_rpc_lock`` around ``list_tools``
    so the keepalive frame doesn't interleave with concurrent
    ``call_tool`` frames over the shared JSON-RPC session.

    Pre-fix the keepalive called ``self.session.list_tools()`` directly
    without the lock, while every other RPC site in this file
    (``_refresh_tools``, ``call_tool`` paths) acquired ``_rpc_lock``.
    Under concurrent traffic the unlocked keepalive frame could
    interleave with a tool-call frame and wedge stdio streams or
    misroute response IDs on streamable-HTTP, which the reconnect path
    then masked as 'transient'."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test_lock")
    lock_held_during_call = {"value": False}
    sentinel_result = SimpleNamespace(tools=[])

    async def fake_list_tools():
        lock_held_during_call["value"] = task._rpc_lock.locked()
        return sentinel_result

    task.session = SimpleNamespace(list_tools=AsyncMock(side_effect=fake_list_tools))

    result = await task._run_keepalive_once()
    assert result is sentinel_result
    assert lock_held_during_call["value"] is True, (
        "_run_keepalive_once must hold _rpc_lock around list_tools to "
        "serialize JSON-RPC frames with concurrent tool calls."
    )
    assert not task._rpc_lock.locked(), "Lock must be released after success."


@pytest.mark.asyncio
async def test_run_keepalive_once_releases_lock_on_failure():
    """If ``list_tools`` raises, the async-with-managed lock must release
    before the exception propagates — otherwise the reconnect path
    deadlocks because every reconnect grabs ``_rpc_lock`` to re-init
    the session."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test_lock_release")
    task.session = SimpleNamespace(
        list_tools=AsyncMock(side_effect=RuntimeError("connection dead"))
    )

    with pytest.raises(RuntimeError, match="connection dead"):
        await task._run_keepalive_once()
    assert not task._rpc_lock.locked()


@pytest.mark.asyncio
async def test_successful_keepalive_passes_prefetched_to_refresh():
    """After a successful keepalive, ``_schedule_tools_refresh`` is called
    with the keepalive's ``tools_result`` as ``prefetched=``.

    Passing the prefetched result avoids a second ``list_tools()`` wire
    call per cycle AND prevents the second frame from racing the
    keepalive's frame on the shared session. Tests the contract
    (keepalive's result feeds the refresh) without patching
    ``asyncio.wait`` semantics — by stubbing ``_run_keepalive_once``
    directly."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test_passthrough")
    task.session = SimpleNamespace()
    sentinel_result = SimpleNamespace(tools=[])

    async def fake_keepalive():
        return sentinel_result

    iteration = {"n": 0}
    real_wait = asyncio.wait

    async def fast_wait(aws, timeout=None, return_when=None):
        iteration["n"] += 1
        if iteration["n"] == 1:
            # First call: simulate keepalive timeout (no events fired).
            return set(), set(aws)
        # Second call: shutdown so the loop exits cleanly. Setting
        # before delegating to real_wait guarantees shutdown_task
        # appears in `done` and the loop breaks on this iteration.
        task._shutdown_event.set()
        return await real_wait(aws, timeout=1.0, return_when=return_when)

    with patch.object(MCPServerTask, "_run_keepalive_once", side_effect=fake_keepalive), \
         patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule, \
         patch("tools.mcp_tool.asyncio.wait", side_effect=fast_wait):

        reason = await task._wait_for_lifecycle_event()

    assert reason == "shutdown"
    mock_schedule.assert_called_once()
    assert mock_schedule.call_args.kwargs.get("prefetched") is sentinel_result, (
        "_schedule_tools_refresh must receive the keepalive's tools_result "
        "as prefetched= to avoid a second list_tools() round-trip."
    )


@pytest.mark.asyncio
async def test_failed_keepalive_does_not_schedule_refresh():
    """If ``_run_keepalive_once`` raises, ``_schedule_tools_refresh`` must
    not be called — the reconnect path rebuilds the registry from scratch
    and scheduling against a broken session would surface a misleading
    error.

    Stronger than the structural assertion: verify the ordering by
    patching the helper to raise (which forces the except branch) and
    asserting schedule was never called."""
    from unittest.mock import patch
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test_keepalive_fail")
    task.session = object()  # truthy; not touched (stub raises before use)

    async def failing_keepalive():
        raise RuntimeError("connection dead")

    iteration = {"n": 0}
    real_wait = asyncio.wait

    async def fast_wait(aws, timeout=None, return_when=None):
        iteration["n"] += 1
        if iteration["n"] == 1:
            # First call: simulate keepalive timeout (no events fired).
            return set(), set(aws)
        # Second call: shutdown so the loop exits cleanly. Setting
        # before delegating to real_wait guarantees shutdown_task
        # appears in `done` and the loop breaks on this iteration.
        task._shutdown_event.set()
        return await real_wait(aws, timeout=1.0, return_when=return_when)

    with patch.object(MCPServerTask, "_run_keepalive_once", side_effect=failing_keepalive), \
         patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule, \
         patch("tools.mcp_tool.asyncio.wait", side_effect=fast_wait):
        reason = await task._wait_for_lifecycle_event()

    assert reason == "reconnect"
    assert mock_schedule.call_count == 0


@pytest.mark.asyncio
async def test_refresh_on_keepalive_opt_out_via_config():
    """``mcp_servers.<name>.refresh_on_keepalive: false`` keeps the
    keepalive itself (connection health check) but skips the registry
    refresh. For long-running sessions where stable catalogs are
    preferred to converging-but-occasionally-noisy ones."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.mcp_tool import MCPServerTask

    task = MCPServerTask("test_opt_out")
    task._config = {"refresh_on_keepalive": False}
    task.session = SimpleNamespace()

    async def fake_keepalive():
        return SimpleNamespace(tools=[])

    iteration = {"n": 0}
    real_wait = asyncio.wait

    async def fast_wait(aws, timeout=None, return_when=None):
        iteration["n"] += 1
        if iteration["n"] == 1:
            # First call: simulate keepalive timeout (no events fired).
            return set(), set(aws)
        # Second call: shutdown so the loop exits cleanly. Setting
        # before delegating to real_wait guarantees shutdown_task
        # appears in `done` and the loop breaks on this iteration.
        task._shutdown_event.set()
        return await real_wait(aws, timeout=1.0, return_when=return_when)

    with patch.object(MCPServerTask, "_run_keepalive_once", side_effect=fake_keepalive) as mock_keepalive, \
         patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule, \
         patch("tools.mcp_tool.asyncio.wait", side_effect=fast_wait):

        await task._wait_for_lifecycle_event()

    assert mock_keepalive.call_count == 1, "Keepalive must still run for connection-health detection."
    assert mock_schedule.call_count == 0, (
        "refresh_on_keepalive=false must suppress the registry refresh."
    )


@pytest.mark.asyncio
async def test_refresh_tools_skips_inner_list_tools_when_prefetched():
    """When a ``prefetched`` ``ListToolsResult`` is passed to
    ``_refresh_tools``, the inner ``self.session.list_tools()`` call
    must be skipped — that's the whole point of the prefetched param.
    Otherwise the keepalive path would do two list_tools round-trips
    per cycle."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from tools.mcp_tool import MCPServerTask
    from tools.registry import ToolRegistry

    task = MCPServerTask("test_prefetched")
    task._refresh_lock = asyncio.Lock()
    task._config = {}
    task._registered_tool_names = []

    inner_list_tools = AsyncMock(side_effect=AssertionError(
        "inner list_tools must NOT be called when prefetched is provided"
    ))
    task.session = SimpleNamespace(list_tools=inner_list_tools)

    prefetched = SimpleNamespace(tools=[])

    mock_registry = ToolRegistry()
    with patch("tools.registry.registry", mock_registry):
        await task._refresh_tools(prefetched=prefetched)

    inner_list_tools.assert_not_called()
