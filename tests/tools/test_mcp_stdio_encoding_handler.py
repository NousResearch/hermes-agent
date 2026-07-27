"""Tests for MCP stdio encoding error handler fix (issue #46099).

On Windows, pipe I/O can produce non-UTF-8 bytes at chunk boundaries,
causing UnicodeDecodeError when the MCP SDK's TextReceiveStream uses
errors="strict". This test verifies that StdioServerParameters is created
with encoding_error_handler="replace".
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _MCP_AVAILABLE

pytestmark = pytest.mark.skipif(not _MCP_AVAILABLE, reason="MCP SDK not installed")


class TestStdioEncodingErrorHandler:
    """Verify that _run_stdio passes encoding_error_handler='replace'."""

    def test_stdio_server_params_uses_replace_encoding_handler(self):
        """StdioServerParameters must use encoding_error_handler='replace'.

        On Windows, pipe chunk boundaries can split multi-byte UTF-8 sequences,
        producing bytes that fail with errors='strict'. Using 'replace' ensures
        undecodable bytes become U+FFFD instead of crashing.
        """
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

        mock_stdio_cm = MagicMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(object(), object()))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        async def _test():
            with (
                patch("tools.mcp_tool.StdioServerParameters") as mock_params,
                patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm),
                patch("tools.mcp_tool.ClientSession", return_value=mock_session_cm),
                patch("tools.mcp_tool._snapshot_child_pids", return_value=set()),
                patch("tools.mcp_tool._write_stderr_log_header"),
                patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None),
            ):
                server = MCPServerTask("test-encoding")
                await server.start({
                    "command": "echo",
                    "args": ["hello"],
                })

                call_kwargs = mock_params.call_args.kwargs
                assert call_kwargs["encoding_error_handler"] == "replace", (
                    f"Expected encoding_error_handler='replace', "
                    f"got '{call_kwargs.get('encoding_error_handler')}'"
                )

                await server.shutdown()

        asyncio.run(_test())

    def test_stdio_server_params_defaults_encoding_utf8(self):
        """Verify that the default encoding (utf-8) is not overridden."""
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

        mock_stdio_cm = MagicMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(object(), object()))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        async def _test():
            with (
                patch("tools.mcp_tool.StdioServerParameters") as mock_params,
                patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm),
                patch("tools.mcp_tool.ClientSession", return_value=mock_session_cm),
                patch("tools.mcp_tool._snapshot_child_pids", return_value=set()),
                patch("tools.mcp_tool._write_stderr_log_header"),
                patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None),
            ):
                server = MCPServerTask("test-encoding")
                await server.start({
                    "command": "echo",
                    "args": ["hello"],
                })

                call_kwargs = mock_params.call_args.kwargs
                # encoding is not explicitly set — StdioServerParameters defaults to utf-8
                # We just verify we don't override it
                assert "encoding" not in call_kwargs or call_kwargs.get("encoding") == "utf-8"

                await server.shutdown()

        asyncio.run(_test())
