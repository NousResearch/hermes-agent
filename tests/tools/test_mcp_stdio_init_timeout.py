"""Regression test for the stdio-MCP subprocess/FD leak (#59349).

A stdio MCP server that never completes ``initialize`` (e.g. emits a
non-JSON-RPC frame and then blocks on stdin) used to hang ``_run_stdio``
forever on the background event loop: ``connect_timeout`` bounded only the
*caller's* ``.result()`` wait, not the coroutine itself. Because the connect
never unwound, the cleanup ``finally`` in ``_run_stdio`` never ran, so the
spawned child process and its stdio pipes / pidfd leaked on *every* discovery
retry — unbounded, until the gateway hit EMFILE.

The fix wraps ``session.initialize()`` in
``asyncio.wait_for(..., timeout=connect_timeout)`` so a stalled handshake fails
instead of hanging, which lets the existing ``finally`` reap the child.

This test drives the *real* ``_run_stdio`` with a fake transport whose
``initialize()`` hangs, and asserts the connect is bounded by
``connect_timeout`` rather than blocking forever. It is fully hermetic — no real
subprocess, no network (the drain-to-zero behaviour was additionally verified
manually against the reporter's live repro).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")


class _HangingSession:
    """Stand-in ClientSession whose handshake never completes."""

    async def initialize(self):
        await asyncio.sleep(3600)


class _FakeAsyncCM:
    """Minimal async context manager yielding a fixed value; spawns nothing."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_exc):
        return False


def _fake_stdio_client(*_args, **_kwargs):
    # `async with stdio_client(...) as (read, write)` — no subprocess spawned.
    return _FakeAsyncCM((object(), object()))


def _fake_client_session(*_args, **_kwargs):
    # `async with ClientSession(...) as session` -> a session that hangs.
    return _FakeAsyncCM(_HangingSession())


class TestStdioInitializeTimeout:
    def test_hanging_initialize_is_bounded_not_leaked(self):
        """A stdio server that hangs at ``initialize`` must fail within
        ``connect_timeout`` — not block ``_run_stdio`` forever (#59349)."""
        from tools import mcp_tool

        server = mcp_tool.MCPServerTask("leak-guard")
        # "npx" (not e.g. "fake-mcp") so this stays green regardless of
        # whether the (opt-in) tools.mcp_command_guard.validate_stdio_command
        # allowlist is enabled in the environment running this test —
        # otherwise it would reject the command before _run_stdio reaches
        # the initialize() hang this test is exercising.
        config = {"command": "npx", "args": [], "connect_timeout": 0.2}

        async def drive():
            with patch.object(mcp_tool, "stdio_client", _fake_stdio_client), \
                 patch.object(mcp_tool, "ClientSession", _fake_client_session), \
                 patch.object(mcp_tool, "_resolve_stdio_command", lambda c, e: (c, e)), \
                 patch.object(mcp_tool, "_write_stderr_log_header", lambda *_a, **_k: None), \
                 patch.object(mcp_tool, "_get_mcp_stderr_log", lambda: None), \
                 patch("tools.osv_check.check_package_for_malware",
                       lambda *_a, **_k: None):
                start = time.monotonic()
                # The outer 5s guard exists ONLY so a regression can't hang the
                # whole suite. With the fix, the inner connect_timeout (0.2s)
                # fires first; the elapsed assertion below is what actually
                # distinguishes "bounded" (fixed) from "hung" (regressed).
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(server._run_stdio(config), timeout=5.0)
                return time.monotonic() - start

        elapsed = asyncio.run(drive())
        assert elapsed < 2.0, (
            f"_run_stdio blocked {elapsed:.1f}s on a hanging initialize() — the "
            f"connect_timeout ({config['connect_timeout']}s) bound was not applied; "
            f"the #59349 subprocess/FD leak has regressed."
        )
