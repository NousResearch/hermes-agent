"""Tests for RuntimeError handling during MCP server shutdown.

When the event loop closes while MCPServerTask.run() is awaiting
(asyncio.sleep or _wait_for_reconnect_or_shutdown), the awaitable
raises RuntimeError("Event loop is closed"). The four guard points
must catch this and return cleanly instead of propagating the error.

Reproduction: on Hermes /exit, asyncio.run() tears down the loop
while MCP server tasks may still be mid-await in the reconnect/
parked backoff loop. Without the guards, this produces a noisy
"Exception ignored in: <coroutine object MCPServerTask.run>" traceback.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_session(tools=None):
    """Create a mock MCP session with initialize and list_tools."""
    mock_tools = tools or []
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(
        return_value=SimpleNamespace(tools=mock_tools)
    )
    return mock_session


def _mock_stdio_and_session(session):
    """Return patches for stdio_client and ClientSession as async CMs."""
    mock_read, mock_write = MagicMock(), MagicMock()
    mock_stdio_cm = MagicMock()
    mock_stdio_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
    mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)
    mock_cs_cm = MagicMock()
    mock_cs_cm.__aenter__ = AsyncMock(return_value=session)
    mock_cs_cm.__aexit__ = AsyncMock(return_value=False)
    return (
        patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm),
        patch("tools.mcp_tool.ClientSession", return_value=mock_cs_cm),
        mock_read, mock_write,
    )


class TestRuntimeErrorOnShutdown:
    """RuntimeError from a closed event loop is caught at all four guard points
    in MCPServerTask.run(), causing a clean return instead of an unhandled traceback."""

    # --- Guard 1: initial-connection backoff asyncio.sleep ---

    def test_initial_backoff_sleep_raises_runtimeerror(self):
        """asyncio.sleep(backoff) in initial-connection retry raises RuntimeError
        (event loop closed) — run() must catch and return, not propagate."""
        from tools.mcp_tool import MCPServerTask

        real_sleep = asyncio.sleep
        sleep_call_count = {"n": 0}

        async def _fake_sleep(delay):
            # Only raise on the first call (the backoff sleep in run())
            # to avoid hitting the OSV malware check's asyncio.to_thread
            # internals that also await sleep internally.
            sleep_call_count["n"] += 1
            if sleep_call_count["n"] == 1:
                raise RuntimeError("Event loop is closed")
            await real_sleep(delay)

        mock_session = _make_mock_session()

        async def _test():
            p_stdio, p_cs, _, _ = _mock_stdio_and_session(mock_session)
            with patch("tools.mcp_tool.StdioServerParameters"), \
                 p_stdio, p_cs, \
                 patch("tools.mcp_tool._kill_orphaned_mcp_children"), \
                 patch("tools.mcp_tool._snapshot_child_pids", return_value=set()), \
                 patch("tools.mcp_tool._filter_mcp_children", return_value=set()), \
                 patch("tools.mcp_tool._OSV_MALWARE_CHECK_TIMEOUT_S", 0.01), \
                 patch("asyncio.sleep", side_effect=_fake_sleep):

                server = MCPServerTask("test_srv")
                # Make _run_stdio fail on first attempt so we enter initial-retry path
                with patch.object(MCPServerTask, "_run_stdio", side_effect=ConnectionRefusedError("fail")):
                    await server.run({"command": "test"})

                # If RuntimeError propagated, this line is never reached
                assert True

        asyncio.run(_test())

    # --- Guard 2: parked initial-connect _wait_for_reconnect_or_shutdown ---

    def test_parked_initial_wait_raises_runtimeerror(self):
        """_wait_for_reconnect_or_shutdown in initial-connection parking
        raises RuntimeError (event loop closed) — run() must catch and return."""
        from tools.mcp_tool import MCPServerTask

        mock_session = _make_mock_session()

        async def _wait_raises(self, timeout=None):
            raise RuntimeError("Event loop is closed")

        async def _test():
            p_stdio, p_cs, _, _ = _mock_stdio_and_session(mock_session)
            with patch("tools.mcp_tool.StdioServerParameters"), \
                 p_stdio, p_cs, \
                 patch("tools.mcp_tool._kill_orphaned_mcp_children"), \
                 patch("tools.mcp_tool._snapshot_child_pids", return_value=set()), \
                 patch("tools.mcp_tool._filter_mcp_children", return_value=set()), \
                 patch("tools.mcp_tool._OSV_MALWARE_CHECK_TIMEOUT_S", 0.01), \
                 patch.object(MCPServerTask, "_wait_for_reconnect_or_shutdown", _wait_raises), \
                 patch.object(MCPServerTask, "_run_stdio", side_effect=ConnectionRefusedError("fail")), \
                 patch("tools.mcp_tool._MAX_INITIAL_CONNECT_RETRIES", 0):

                server = MCPServerTask("test_srv")
                await server.run({"command": "test"})
                assert True  # reached = RuntimeError was caught

        asyncio.run(_test())

    # --- Guard 3: parked reconnect _wait_for_reconnect_or_shutdown ---

    def test_parked_reconnect_wait_raises_runtimeerror(self):
        """_wait_for_reconnect_or_shutdown in post-reconnect-budget parking
        raises RuntimeError (event loop closed) — run() must catch and return."""
        from tools.mcp_tool import MCPServerTask

        mock_session = _make_mock_session()

        async def _wait_raises(self, timeout=None):
            raise RuntimeError("Event loop is closed")

        async def _test():
            p_stdio, p_cs, _, _ = _mock_stdio_and_session(mock_session)
            with patch("tools.mcp_tool.StdioServerParameters"), \
                 p_stdio, p_cs, \
                 patch("tools.mcp_tool._kill_orphaned_mcp_children"), \
                 patch("tools.mcp_tool._snapshot_child_pids", return_value=set()), \
                 patch("tools.mcp_tool._filter_mcp_children", return_value=set()), \
                 patch("tools.mcp_tool._OSV_MALWARE_CHECK_TIMEOUT_S", 0.01), \
                 patch.object(MCPServerTask, "_wait_for_reconnect_or_shutdown", _wait_raises), \
                 patch.object(MCPServerTask, "_run_stdio", side_effect=ConnectionRefusedError("fail")), \
                 patch("tools.mcp_tool._MAX_RECONNECT_RETRIES", 0):

                server = MCPServerTask("test_srv")
                # With _ready not set, this enters the initial-connect path first,
                # not the reconnect path. We need _ready set and _reconnect_retries
                # > _MAX_RECONNECT_RETRIES to enter the parked-reconnect path.
                # Easiest: set _ready so the exception is treated as a reconnect failure.
                server._ready.set()
                server._reconnect_retries = 1  # > _MAX_RECONNECT_RETRIES (0)

                await server.run({"command": "test"})
                assert True  # reached = RuntimeError was caught

        asyncio.run(_test())

    # --- Guard 4: reconnect backoff asyncio.sleep ---

    def test_reconnect_backoff_sleep_raises_runtimeerror(self):
        """asyncio.sleep(backoff) in reconnect retry raises RuntimeError
        (event loop closed) — run() must catch and return, not propagate."""
        from tools.mcp_tool import MCPServerTask

        real_sleep = asyncio.sleep
        sleep_call_count = {"n": 0}

        async def _fake_sleep(delay):
            sleep_call_count["n"] += 1
            # Raise RuntimeError on the reconnect backoff sleep
            # (not on earlier sleeps from OSV check etc.)
            if sleep_call_count["n"] == 1:
                raise RuntimeError("Event loop is closed")
            await real_sleep(delay)

        mock_session = _make_mock_session()

        async def _test():
            p_stdio, p_cs, _, _ = _mock_stdio_and_session(mock_session)
            with patch("tools.mcp_tool.StdioServerParameters"), \
                 p_stdio, p_cs, \
                 patch("tools.mcp_tool._kill_orphaned_mcp_children"), \
                 patch("tools.mcp_tool._snapshot_child_pids", return_value=set()), \
                 patch("tools.mcp_tool._filter_mcp_children", return_value=set()), \
                 patch("tools.mcp_tool._OSV_MALWARE_CHECK_TIMEOUT_S", 0.01), \
                 patch("asyncio.sleep", side_effect=_fake_sleep), \
                 patch.object(MCPServerTask, "_run_stdio", side_effect=ConnectionRefusedError("fail")):

                server = MCPServerTask("test_srv")
                # Set _ready so the failure is treated as reconnect, not initial-connect
                server._ready.set()
                server._reconnect_retries = 1  # below _MAX_RECONNECT_RETRIES

                await server.run({"command": "test"})
                assert True  # reached = RuntimeError was caught

        asyncio.run(_test())

    # --- Regression: normal shutdown still works ---

    def test_normal_shutdown_still_works(self):
        """After adding RuntimeError guards, the normal shutdown path
        (shutdown event set, task exits cleanly) still works."""
        from tools.mcp_tool import MCPServerTask

        mock_tools = [SimpleNamespace(
            name="echo", description="Echo", inputSchema={"type": "object"}
        )]
        mock_session = _make_mock_session(mock_tools)

        async def _test():
            p_stdio, p_cs, _, _ = _mock_stdio_and_session(mock_session)
            with patch("tools.mcp_tool.StdioServerParameters"), \
                 p_stdio, p_cs:
                server = MCPServerTask("test_srv")
                await server.start({"command": "npx"})

                assert server.session is not None
                assert not server._task.done()

                await server.shutdown()

                assert server.session is None
                assert server._task.done()

        asyncio.run(_test())

    # --- RuntimeError from _run_stdio is handled as a connection failure ---

    def test_runtime_error_from_transport_is_connection_failure(self):
        """A RuntimeError raised inside _run_stdio (NOT from the four guarded
        await points) is caught by the 'except Exception' block in run() and
        treated as a connection failure — not swallowed by our RuntimeError
        guards, which only protect asyncio.sleep and _wait_for_reconnect_or_shutdown."""
        from tools.mcp_tool import MCPServerTask

        mock_session = _make_mock_session()

        # _run_stdio raises RuntimeError -> enters except Exception -> treated as
        # connection failure -> hits reconnect path -> _shutdown_event is set
        # so it returns immediately instead of sleeping.
        async def _test():
            p_stdio, p_cs, _, _ = _mock_stdio_and_session(mock_session)
            with patch("tools.mcp_tool.StdioServerParameters"), \
                 p_stdio, p_cs, \
                 patch("tools.mcp_tool._kill_orphaned_mcp_children"), \
                 patch("tools.mcp_tool._snapshot_child_pids", return_value=set()), \
                 patch("tools.mcp_tool._filter_mcp_children", return_value=set()), \
                 patch("tools.mcp_tool._OSV_MALWARE_CHECK_TIMEOUT_S", 0.01), \
                 patch.object(MCPServerTask, "_run_stdio", side_effect=RuntimeError("bug in transport")):

                server = MCPServerTask("test_srv")
                # Set shutdown so the reconnect path returns immediately
                # instead of trying to sleep/backoff.
                server._shutdown_event.set()

                # This should NOT raise — RuntimeError from _run_stdio enters
                # the except Exception block, then the shutdown check at the
                # top of the while loop causes a clean return.
                await server.run({"command": "test"})
                assert True  # reached = clean exit

        asyncio.run(_test())

    # --- RuntimeError from t.cancel() in finally blocks ---
    #
    # t.cancel() internally calls loop.call_soon(callback), which checks
    # loop.is_closed() and raises RuntimeError("Event loop is closed").
    # asyncio.Task is a C extension type so we can't monkey-patch cancel().
    # Instead, we test the guard by verifying that the production code
    # has try/except RuntimeError around every t.cancel() call in the
    # finally blocks of _wait_for_lifecycle_event and
    # _wait_for_reconnect_or_shutdown.

    def test_wait_for_lifecycle_event_cancel_is_guarded(self):
        """Verify _wait_for_lifecycle_event's finally block wraps t.cancel()
        in try/except RuntimeError to handle closed event loops."""
        import inspect
        from tools.mcp_tool import MCPServerTask

        source = inspect.getsource(MCPServerTask._wait_for_lifecycle_event)
        # Find the finally block and verify every t.cancel() is guarded
        # by try/except RuntimeError.
        lines = source.split('\n')
        in_finally = False
        guarded_cancel_count = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'finally:':
                in_finally = True
            if in_finally and 't.cancel()' in stripped and not stripped.startswith('#'):
                # The line before should be 'try:', and within the try block
                # t.cancel() should be followed by 'except RuntimeError'
                # Check: there's a try: before this cancel, and except RuntimeError after
                prev_lines = [lines[j].strip() for j in range(max(0, i-2), i)]
                next_lines = [lines[j].strip() for j in range(i+1, min(len(lines), i+4))]
                assert 'try:' in prev_lines, (
                    f"t.cancel() at line {i} not guarded by try: "
                    f"(prev lines: {prev_lines})"
                )
                assert any('RuntimeError' in nl for nl in next_lines), (
                    f"t.cancel() at line {i} not followed by except RuntimeError: "
                    f"(next lines: {next_lines})"
                )
                guarded_cancel_count += 1
            if in_finally and stripped and not stripped.startswith(('for', 'if', 'try', 'except', 'pass', '#', 't.', 'await')) and 'finally' not in stripped and guarded_cancel_count > 0:
                # We've left the finally block pattern
                in_finally = False

        assert guarded_cancel_count >= 1, (
            f"Expected at least 1 guarded t.cancel() call in "
            f"_wait_for_lifecycle_event's finally block, found {guarded_cancel_count}"
        )

    def test_wait_for_reconnect_cancel_is_guarded(self):
        """Verify _wait_for_reconnect_or_shutdown's finally block wraps
        t.cancel() in try/except RuntimeError to handle closed event loops."""
        import inspect
        from tools.mcp_tool import MCPServerTask

        source = inspect.getsource(
            MCPServerTask._wait_for_reconnect_or_shutdown
        )
        lines = source.split('\n')
        in_finally = False
        guarded_cancel_count = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'finally:':
                in_finally = True
            if in_finally and 't.cancel()' in stripped and not stripped.startswith('#'):
                prev_lines = [lines[j].strip() for j in range(max(0, i-2), i)]
                next_lines = [lines[j].strip() for j in range(i+1, min(len(lines), i+4))]
                assert 'try:' in prev_lines, (
                    f"t.cancel() at line {i} not guarded by try: "
                    f"(prev lines: {prev_lines})"
                )
                assert any('RuntimeError' in nl for nl in next_lines), (
                    f"t.cancel() at line {i} not followed by except RuntimeError: "
                    f"(next lines: {next_lines})"
                )
                guarded_cancel_count += 1
            if in_finally and stripped and not stripped.startswith(('for', 'if', 'try', 'except', 'pass', '#', 't.', 'await')) and 'finally' not in stripped and guarded_cancel_count > 0:
                in_finally = False

        assert guarded_cancel_count >= 1, (
            f"Expected at least 1 guarded t.cancel() call in "
            f"_wait_for_reconnect_or_shutdown's finally block, found {guarded_cancel_count}"
        )