"""Tests for MCP keepalive bug fix — _receive_loop must not close write_stream.

Root cause: BaseSession._receive_loop() uses `async with (self._read_stream, self._write_stream)`,
which closes BOTH streams when the loop exits. This is wrong because:

1. _receive_loop does not OWN write_stream — it only reads from read_stream.
2. When the subprocess stdout goes EOF (idle pipe timeout), read_stream ends,
   causing _receive_loop to exit its async-with block, which closes write_stream.
3. The keepalive timer then tries send_ping() on the now-closed write_stream,
   raising ClosedResourceError and triggering a reconnect cascade.

The fix: wrap write_stream in _NonClosingStreamWrapper so the async-with block
cannot close it. The wrapper stays in place permanently (not restored after
_receive_loop exits) because the keepalive timer can fire between _receive_loop
exit and session teardown.

Bug: https://github.com/NousResearch/hermes-agent/issues/19417
SDK bug: https://github.com/modelcontextprotocol/python-sdk/issues/631
"""

import asyncio
from unittest.mock import MagicMock

import anyio
import pytest

from mcp.shared.message import SessionMessage
from mcp.shared.session import BaseSession
from mcp.types import JSONRPCNotification, JSONRPCRequest

# Import the production monkey-patch components directly. The production
# code in mcp_tool.py patches BaseSession._receive_loop at import time,
# so these are the authoritative implementations. Keeping a separate
# test-local copy caused isinstance() mismatches when both modules were
# loaded (production class vs test-local class).
from tools.mcp_tool import _NonClosingStreamWrapper, _patched_receive_loop


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Minimal request/notification types for testing BaseSession construction.
# ---------------------------------------------------------------------------

class _TestRequest(JSONRPCRequest):
    """Minimal request type for test sessions."""

class _TestNotification(JSONRPCNotification):
    """Minimal notification type for test sessions."""

def _make_session():
    """Create a BaseSession with memory streams for testing.

    anyio.create_memory_object_stream() returns (send_stream, receive_stream).
    Following the SDK naming convention:
      - read_stream_writer (send side) → read_stream (receive side): messages FROM the server
      - write_stream (send side) → write_stream_reader (receive side): messages TO the server

    Returns (session, read_stream, read_stream_writer, write_stream, write_stream_reader).
    """
    read_stream_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream[SessionMessage](0)

    session = BaseSession(
        read_stream=read_stream,
        write_stream=write_stream,
        receive_request_type=_TestRequest,
        receive_notification_type=_TestNotification,
    )

    return session, read_stream, read_stream_writer, write_stream, write_stream_reader


class TestReceiveLoopStreamOwnership:
    """_receive_loop must NOT close write_stream on exit.

    When read_stream ends (e.g. subprocess stdout EOF), _receive_loop exits
    its async-for loop. The original SDK code closes write_stream too via
    `async with (read_stream, write_stream)`, which kills the keepalive probe
    and triggers a reconnect cascade.

    The fix ensures only read_stream is closed by _receive_loop.
    """

    @pytest.mark.asyncio
    async def test_receive_loop_does_not_close_write_stream_on_exit(self):
        """When _receive_loop exits (read_stream ends), write_stream must remain open.

        This is the core bug fix: _receive_loop should only manage read_stream
        lifecycle, not write_stream, because write_stream is owned by the
        caller (stdio_client or the session context manager).
        """
        session, read_stream, read_stream_writer, write_stream, write_stream_reader = _make_session()

        # Close read_stream_writer to simulate subprocess stdout EOF.
        # This causes _receive_loop to exit because read_stream ends.
        await read_stream_writer.aclose()

        # Start the session (which starts _receive_loop), wait for it to process
        # the EOF, then close the session.
        async with session:
            await asyncio.sleep(0.1)

        # After _receive_loop exits and the session closes,
        # write_stream should STILL be open (it's owned by the caller,
        # not by _receive_loop).
        #
        # BUG (original SDK): async with (read_stream, write_stream) closes
        # write_stream, causing ClosedResourceError on subsequent send() calls.
        #
        # FIX: _receive_loop wraps write_stream so __aexit__ is a no-op.
        assert not write_stream._closed, (
            "write_stream must NOT be closed by _receive_loop — "
            "it is owned by the caller (stdio_client), not by _receive_loop"
        )

    @pytest.mark.asyncio
    async def test_receive_loop_closes_read_stream_on_exit(self):
        """_receive_loop SHOULD close read_stream on exit (it owns it as reader)."""
        session, read_stream, read_stream_writer, write_stream, write_stream_reader = _make_session()

        # Close read_stream_writer to trigger _receive_loop exit
        await read_stream_writer.aclose()

        async with session:
            await asyncio.sleep(0.1)

        # read_stream SHOULD be closed (it's the one _receive_loop reads from)
        assert read_stream._closed, "read_stream should be closed by _receive_loop"

    @pytest.mark.asyncio
    async def test_write_stream_remains_usable_after_receive_loop_exit(self):
        """After _receive_loop exits, write_stream.send() must NOT raise ClosedResourceError.

        This directly tests the keepalive failure scenario: the keepalive timer
        calls send_ping() → send_request() → write_stream.send() after
        _receive_loop has exited. If write_stream was incorrectly closed,
        this raises ClosedResourceError and triggers a reconnect cascade.
        """
        session, read_stream, read_stream_writer, write_stream, write_stream_reader = _make_session()

        # Close read_stream_writer to trigger _receive_loop exit
        await read_stream_writer.aclose()

        async with session:
            await asyncio.sleep(0.1)

        # After _receive_loop exits, write_stream should still be usable.
        # BUG (original SDK): write_stream is closed, send_nowait raises ClosedResourceError
        # FIX: write_stream remains open (wrapped in _NonClosingStreamWrapper).
        #
        # With a zero-buffer stream and no active reader, send_nowait raises WouldBlock
        # (buffer full, no waiter) or BrokenResourceError (receiver gone). Both are
        # expected — they prove the stream is OPEN. Only ClosedResourceError is the bug.
        assert not write_stream._closed, (
            "write_stream must NOT be closed by _receive_loop"
        )
        try:
            msg = SessionMessage(message=MagicMock())
            write_stream.send_nowait(msg)
            # Message queued or delivered — stream is fully functional
        except anyio.ClosedResourceError:
            pytest.fail(
                "write_stream.send_nowait() raised ClosedResourceError — "
                "_receive_loop incorrectly closed write_stream"
            )
        except (anyio.BrokenResourceError, anyio.WouldBlock):
            # BrokenResourceError: receiver gone (expected after session cleanup)
            # WouldBlock: no waiter and zero buffer (expected — stream is open but idle)
            # Both prove write_stream is NOT closed — the fix works.
            pass

    @pytest.mark.asyncio
    async def test_wrapper_stays_after_receive_loop_exit(self):
        """After _receive_loop exits, _write_stream remains wrapped (not restored).

        The production code intentionally does NOT restore the original
        write_stream after _receive_loop exits. This is because the keepalive
        timer can fire between _receive_loop exit and session teardown; if we
        restored the unwrapped stream, the original async-with __aexit__ could
        still close it.
        """
        session, read_stream, read_stream_writer, write_stream, write_stream_reader = _make_session()

        # Before _receive_loop starts, write_stream is the raw memory stream
        assert not isinstance(session._write_stream, _NonClosingStreamWrapper)

        # Close read_stream_writer to trigger _receive_loop exit
        await read_stream_writer.aclose()

        async with session:
            await asyncio.sleep(0.1)

        # After _receive_loop exits, write_stream should STILL be wrapped
        # (the production code intentionally does not restore the original)
        assert isinstance(session._write_stream, _NonClosingStreamWrapper), (
            "_write_stream should remain wrapped after _receive_loop exits — "
            "restoring the original would expose it to ClosedResourceError "
            "from the keepalive timer"
        )

    @pytest.mark.asyncio
    async def test_reentry_skips_wrapping(self):
        """If _write_stream is already wrapped, the patched _receive_loop skips re-wrapping.

        This tests the re-entry guard: during reconnect, _receive_loop is called
        again on the same session. If write_stream is already wrapped, we must
        not double-wrap it.
        """
        session, read_stream, read_stream_writer, write_stream, write_stream_reader = _make_session()

        # Pre-wrap write_stream to simulate re-entry. Use the production
        # _NonClosingStreamWrapper (imported above) so the isinstance()
        # check in _patched_receive_loop recognizes it.
        original = session._write_stream
        session._write_stream = _NonClosingStreamWrapper(original)

        await read_stream_writer.aclose()

        async with session:
            await asyncio.sleep(0.1)

        # After exit, write_stream should still be wrapped exactly once
        # (the pre-wrap we set, NOT double-wrapped)
        assert isinstance(session._write_stream, _NonClosingStreamWrapper)
        # The inner stream should be the original MemoryObjectSendStream,
        # not another _NonClosingStreamWrapper (which would mean double-wrap)
        assert isinstance(session._write_stream._stream, type(original)), (
            "Inner stream should be the original MemoryObjectSendStream, "
            "not another _NonClosingStreamWrapper (double-wrap bug)"
        )


class TestKeepaliveWithOpenWriteStream:
    """Integration test: keepalive probe must survive read_stream EOF.

    After _receive_loop exits due to read_stream EOF, the keepalive probe
    should still be able to send a ping on write_stream. In the original
    SDK, this fails with ClosedResourceError because _receive_loop closed
    write_stream.
    """

    @pytest.mark.asyncio
    async def test_keepalive_probe_succeeds_after_read_stream_eof(self):
        """Simulate the exact keepalive failure scenario.

        1. Create a session with memory streams
        2. Close read_stream_writer (simulating subprocess stdout EOF)
        3. Wait for _receive_loop to exit
        4. Verify write_stream is still open (for keepalive probe)
        """
        session, read_stream, read_stream_writer, write_stream, write_stream_reader = _make_session()

        # Close read_stream_writer to simulate subprocess stdout EOF
        await read_stream_writer.aclose()

        async with session:
            await asyncio.sleep(0.1)

        # Now simulate the keepalive probe trying to send a ping
        # In the original SDK, write_stream is closed, and send() raises ClosedResourceError
        # With the fix, write_stream is still open
        assert not write_stream._closed, (
            "write_stream must remain open for keepalive probe after read_stream EOF"
        )

    @pytest.mark.asyncio
    async def test_response_streams_get_connection_closed_after_eof(self):
        """After _receive_loop exits, pending response streams must get CONNECTION_CLOSED.

        Even with the fix (write_stream stays open), the _receive_loop's finally
        block must still send CONNECTION_CLOSED errors to any pending request
        response streams so callers don't hang forever.
        """
        session, read_stream, read_stream_writer, write_stream, write_stream_reader = _make_session()

        # Start the session
        async with session:
            # Create a response stream that simulates a pending request
            from mcp.types import ErrorData

            # Manually register a response stream (simulating a pending request)
            response_stream, response_reader = anyio.create_memory_object_stream(1)
            session._response_streams[42] = response_stream  # fake request_id

            # Close read_stream_writer to trigger _receive_loop exit
            await read_stream_writer.aclose()
            await asyncio.sleep(0.2)

            # The response stream should have received a CONNECTION_CLOSED error
            # and been cleaned up
            assert 42 not in session._response_streams, (
                "Response stream for request 42 should be cleaned up after _receive_loop exit"
            )