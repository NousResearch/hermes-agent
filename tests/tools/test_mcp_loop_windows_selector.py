"""Regression: the MCP event loop must not be a ProactorEventLoop on Windows.

The MCP stdio client spawns servers as subprocesses and reads their stdout
pipe. Windows' ProactorEventLoop hangs those subprocess-pipe reads during the
`initialize` handshake, so discovery times out and no MCP tools register.
`_ensure_mcp_loop` therefore builds an explicit SelectorEventLoop on win32,
matching the win32 SelectorEventLoop handling in cli.py and web_server.py.
"""
import asyncio
import sys


def test_mcp_loop_is_selector_on_windows():
    import tools.mcp_tool as mcp_tool

    mcp_tool._ensure_mcp_loop()
    try:
        loop = mcp_tool._mcp_loop
        assert loop is not None
        if sys.platform == "win32":
            proactor = getattr(asyncio, "ProactorEventLoop", None)
            assert proactor is None or not isinstance(loop, proactor)
            assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        mcp_tool._stop_mcp_loop()
