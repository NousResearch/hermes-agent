"""End-to-end coverage for stdio MCP subprocess working directories."""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest
pytest.importorskip("mcp.server")


def test_real_stdio_server_uses_configured_workdir(tmp_path, monkeypatch):
    """The configured workdir reaches the real MCP child process."""
    from tools import mcp_tool

    server_script = tmp_path / "cwd_probe_server.py"
    server_script.write_text(
        textwrap.dedent(
            """
            import os

            try:
                from mcp.server.fastmcp import FastMCP as _Server
            except ImportError:
                from mcp.server import MCPServer as _Server

            mcp = _Server("cwd-probe")

            @mcp.tool()
            def report_cwd() -> str:
                return os.getcwd()

            if __name__ == "__main__":
                mcp.run(transport="stdio")
            """
        ),
        encoding="utf-8",
    )
    configured_cwd = tmp_path / "configured-cwd"
    configured_cwd.mkdir()

    # Keep this regression hermetic: the subprocess and MCP transport are real,
    # while network malware checks and process-wide orphan cleanup are not part
    # of the behavior under test.
    monkeypatch.setattr(
        "tools.osv_check.check_package_for_malware",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.mcp_tool_lifecycle._kill_orphaned_mcp_children",
        lambda *_args, **_kwargs: None,
    )

    async def drive_server():
        server = mcp_tool.MCPServerTask("cwd-probe")
        try:
            await asyncio.wait_for(
                server.start(
                    {
                        "command": sys.executable,
                        "args": [str(server_script)],
                        "workdir": str(configured_cwd),
                        "connect_timeout": 30,
                    }
                ),
                timeout=60,
            )
            result = await asyncio.wait_for(
                server.session.call_tool("report_cwd", {}),
                timeout=30,
            )
            assert result.content[0].text == str(configured_cwd)
        finally:
            await server.shutdown()

    asyncio.run(drive_server())
