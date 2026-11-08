"""Minimal MCP stdio server used by the MCP/UTCP coexistence test.

Started as a subprocess by ``tests/tools/test_utcp_mcp_coexistence.py`` so the
test exercises Hermes' real MCP client path (``tools/mcp_tool.py``) next to the
real UTCP client path, rather than two mocked registry entries.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("mock-mcp")


@server.tool()
def mcp_echo(message: str) -> str:
    """Echo a message through the MCP mock server."""
    return f"mcp:{message}"


@server.tool()
def mcp_add(a: int, b: int) -> int:
    """Add two integers on the MCP side."""
    return a + b


if __name__ == "__main__":
    server.run()
